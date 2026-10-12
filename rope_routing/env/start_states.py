"""Environment-generated start states for the RL start-state curriculum (no expert, no demonstrations).

From a standard reset state (rope laid out, robot at home, fingers open) the robot is driven through the environment's
OWN action interface: ``env._apply_action`` (TCP target step of at most 2 cm per axis and within 4 cm of the TCP,
workspace clamps, yaw step of at most 0.15 rad, finger target with the release hysteresis), followed by the 10 physics
substeps of one 0.1 s control step. The starts are:

* "hover": the TCP 3-6 cm above the episode's grasp point, fingers open and straddling the rope, arm settled;
* "grasped": through the hover point, descend to the rope, close until the contact-gated weld forms, lift to about
  3 cm;
* "roam" (:func:`setup_roam`): a "full" start from a random gripper position above the object, used by
  ``GenericSurfaceEnv`` in training to explore where to grasp.

Every command is a legal policy action, so the start states are reachable states of the MDP. This is a scripted reset
of the environment, like a test fixture: nothing is recorded, nothing is used as a training label, and the scripted
expert is never called.

Scene properties that shape the motion (measured on ``SceneCfg()``, the "C1-nc" scene of notes C;
``scripts/test_start_states.py``):

* The OSC's integral term (``i_corr``) winds up during long moves, and the TCP then lurches up to 5 cm past the goal.
  The plan is therefore fed forward through it: commanded target = plan - i_corr, so the controller's effective target
  IS the plan. "Settled" requires the TCP to sit at its commanded target, so that the state stays put under zero
  actions.
* The arm lags a 1.5 cm/step plan by about one step. A descent planned straight to the closing height overshoots it by
  about 5 mm, the opening fingertips (which sweep about 3 mm below the pinch point) strike the table, the hand bounces
  up to about 13 mm and the fingers close over the rope (5% of first attempts). The last 2.5 cm are therefore approached
  at 5 mm/step, and the plan stops AT the closing height.
* The rope is calm after the scene's adaptive settle (below 2 mm/s, no hops), but the fingers nudge it, so the goals
  track a low-pass of the grasp node's CURRENT position. Every phase has a step budget, so a stuck arm (the legacy
  scene's FR3 joint Coulomb "freezes", now removed) aborts instead of blocking a reset.

Usage (inside a reset)::

    obs, _ = env.reset(seed=seed, options={"layout_seed": layout_seed, "no_prefix": True})
    ok, info = setup_start(env, mode, rng)      # mode in {"full", "hover", "grasped"}
    obs = info["obs"]                           # observation of the new start state (bookkeeping refreshed)

On failure (``ok`` False) the environment is still consistent (bookkeeping refreshed; after a failed grasp the fingers
are open above the rope) unless ``info["finite"]`` is False (a non-finite state or a solver divergence,
``info["diverged"]``); callers then reset again or keep the state as a "full" start.

Positions are in metres in the robot base frame, angles in radians, durations in control steps of 0.1 s.
"""

import time

import numpy as np

from ..deformation import measures
from .grasp_route_env import STEP_POS, STEP_YAW, wrap_half, wrap_pi

Z_TRANSIT = 0.10  # travel height: the open fingers clear the 7 cm pegs [m]
NEAR_XY = 0.03  # leave the travel height only within this horizontal distance of the hover point [m]
HOVER_H = (0.03, 0.06)  # hover height range above the grasp point [m]
HOVER_XY, HOVER_YAW = 0.005, 0.1  # hover randomisation, uniform +- [m], [rad]
LIFT_H = (0.025, 0.035)  # carry height range after the grasp [m] (the grasped potential peaks at 3 cm)
V_FINE = 0.015  # planned-point speed near the rope [m per control step]
V_LAND, LAND_H = 0.005, 0.025  # final approach: planned-point speed within LAND_H above the closing height [m]
CLOSE_BELOW, CLOSE_Z = 0.011, 0.008  # descend to 8 mm, close once the pinch point is below 11 mm, hold 8 mm [m].
# The closing fingertips sweep about 3 mm below the pinch point: pinch points near 0 jam them on the table and lift the
# hand (closing over the rope); at 13 mm they close on nothing. 8 mm puts the fingertips at the rope centre (5 mm).
TRACK = 0.3  # low-pass gain on the (possibly hopping) grasp node position and direction, per control step
YAW_MAX = 2.9  # the environment's clamp on the yaw target [rad]
Q7_MAX = 2.6  # keep the implied wrist angle (q7 ~ azimuth - yaw, tool down) inside the FR3 limit of 3.016 rad
TOL_POS, TOL_YAW, TOL_V, TOL_REST = 0.015, 0.15, 0.01, 0.005  # settled: [m], [rad], [m/s], |target - TCP| [m]
BUDGET = dict(transit=40, hover=20, descend=20, close=12, lift=12)  # step budget per phase [control steps]
MAX_STEPS = dict(hover=50, grasped=80, roam=50)  # whole setup (used <= 40 / <= 54 of 64 layouts); a stuck arm aborts
ROAM_XY, ROAM_Z = 0.06, (0.06, 0.12)  # roam start: xy offset around a random object point (+-), start height [m]


def _yaw_goal(yaw_rope, ref, point_xy):
    """Closing-axis yaw equivalent to ``yaw_rope`` (mod pi) that is nearest to ``ref`` and inside the yaw clamp.

    The yaw is flipped by pi when the implied wrist angle (q7 ~ base azimuth - yaw) would leave the joint range.

    Parameters
    ----------
    yaw_rope : float
        Desired closing-axis yaw [rad], defined modulo pi.
    ref : float
        Reference yaw, e.g. the current yaw target [rad].
    point_xy : np.ndarray, shape (2,)
        Horizontal position of the goal, which sets the base azimuth [m].

    Returns
    -------
    float
        Yaw goal [rad].

    Examples
    --------
    >>> round(_yaw_goal(np.pi + 0.1, 0.0, np.array([0.5, 0.0])), 6)
    0.1
    """
    yaw = ref + wrap_half(yaw_rope - ref)
    if abs(yaw) > YAW_MAX or abs(wrap_pi(np.arctan2(point_xy[1], point_xy[0]) - yaw)) > Q7_MAX:
        flipped = yaw - np.pi * np.sign(yaw)
        if abs(flipped) <= YAW_MAX:
            yaw = flipped
    return float(yaw)


class _Driver:
    """Issues 0.1 s control steps through ``env._apply_action`` and keeps the setup statistics.

    The planned point ``want`` moves straight toward the goal by at most ``max_step`` per axis per step and is the
    OSC's effective target: the commanded target is want - i_corr, so the integral term causes no windup overshoot.

    Parameters
    ----------
    env : SurfaceRouteEnv
        The environment, freshly reset.

    Attributes
    ----------
    env, sim
        The environment and its ``GraspScene``.
    n_steps : int
        Control steps issued.
    max_strain, max_seg_strain : float
        Largest sample strain of the environment's material graph and largest segment strain seen.
    tcp : np.ndarray, shape (3,)
        TCP position after the last finite step [m].
    speed : float
        TCP speed over the last finite step [m/s].
    finite, diverged : bool
        Whether the state is finite (and not diverged), and whether a solver divergence was seen.
    nodes : np.ndarray, shape (N, 3)
        Rope nodes after the last finite step [m].
    want : np.ndarray, shape (3,)
        Planned point, the OSC's effective target [m].
    """

    def __init__(self, env):
        self.env, self.sim = env, env.sim
        self.n_steps, self.max_strain, self.max_seg_strain = 0, 0.0, 0.0
        self.tcp, self.speed, self.finite, self.diverged = self.sim.tcp(), 0.0, True, False
        self.nodes = self.sim.rope_nodes()
        self.want = self.sim.target_pos + self.sim.i_corr

    def step(self, goal, yaw, grip, max_step=STEP_POS):
        """Issue one control step toward ``goal`` [m] and ``yaw`` [rad] with the given grip command.

        Parameters
        ----------
        goal : array_like, shape (3,)
            Goal of the planned point [m].
        yaw : float
            Yaw goal [rad].
        grip : float
            Grip action in [-1, 1].
        max_step : float
            Largest move of the planned point per axis [m per control step].

        Returns
        -------
        bool
            Whether the state is finite and the solver did not diverge.
        """
        env, sim = self.env, self.sim
        delta = np.asarray(goal, float) - self.want
        self.want = self.want + delta * min(1.0, max_step / max(np.abs(delta).max(), 1e-9))
        target = self.want - sim.i_corr
        act = np.zeros(5)
        act[:3] = np.clip((target - sim.target_pos) / STEP_POS, -1.0, 1.0)
        act[3] = np.clip(wrap_pi(yaw - sim.target_yaw) / STEP_YAW, -1.0, 1.0)
        act[4] = grip
        env._apply_action(act)
        for _ in range(env.substeps):
            sim.substep()
            # A diverged Newton solve makes Mochi zero the rod's displacements (the rope jumps back to its straight
            # rest shape, through the robot): env.step flags this as "bad"; here it makes the state unusable, like a
            # non-finite one.
            self.diverged |= not np.any(sim.rope.get_displacements())
        self.n_steps += 1
        tcp, nodes = sim.tcp(), sim.rope_nodes()
        self.finite = bool(np.all(np.isfinite(nodes)) and np.all(np.isfinite(tcp))) and not self.diverged
        if self.finite:
            self.speed = float(np.linalg.norm(tcp - self.tcp)) / (env.substeps * sim.dt)
            self.tcp, self.nodes = tcp, nodes
            seg_strain = np.linalg.norm(np.diff(nodes, axis=0), axis=1) / sim.seg - 1.0
            self.max_seg_strain = max(self.max_seg_strain, float(seg_strain.max()))
            strain = measures.node_strain(env.graph, env.graph.samples(nodes))  # the environment's own 21-sample strain
            self.max_strain = max(self.max_strain, float(strain.max()))
        return self.finite

    def settled(self, goal, yaw):
        """Whether the TCP is near the goal and slow and the controller is at rest.

        "At rest" means the TCP sits at its commanded target, so the OSC integral is not moving and the state stays
        put under zero actions after the hand-over to the policy.
        """
        sim = self.sim
        return bool(np.linalg.norm(self.tcp - goal) < TOL_POS and abs(wrap_pi(sim.tool_yaw() - yaw)) < TOL_YAW
                    and self.speed < TOL_V and np.linalg.norm(sim.target_pos - self.tcp) < TOL_REST)

    def out_of_time(self, mode):
        """Whether the setup has used its step budget for ``mode``."""
        return self.n_steps >= MAX_STEPS[mode]


class _Rope:
    """Low-pass filter of the grasp node's position and closing-axis yaw (the rope creeps and hops after a reset).

    Parameters
    ----------
    env : SurfaceRouteEnv
        The environment.
    drv : _Driver
        The driver, whose ``nodes`` hold the latest rope state.
    node : int
        The tracked rope node.
    """

    def __init__(self, env, drv, node):
        self.env, self.drv, self.node = env, drv, node
        self.position = drv.nodes[node].copy()
        self.yaw = env._rope_dir_yaw(drv.nodes, node)

    def update(self):
        """Advance the filter by one control step; returns the filtered position [m] and yaw [rad]."""
        nodes = self.drv.nodes
        self.position += TRACK * (nodes[self.node] - self.position)
        self.yaw += TRACK * wrap_half(self.env._rope_dir_yaw(nodes, self.node) - self.yaw)
        return self.position, self.yaw


def setup_start(env, mode: str, rng: np.random.Generator):
    """Drive a freshly reset environment into a "full" (untouched), "hover" or "grasped" start state.

    Parameters
    ----------
    env : SurfaceRouteEnv
        The environment, just reset.
    mode : str
        "full", "hover" or "grasped".
    rng : np.random.Generator
        Randomness of the hover offset, height, yaw and lift height.

    Returns
    -------
    ok : bool
        Whether the requested state was reached.
    info : dict
        "obs" (observation of the resulting state, bookkeeping refreshed as by ``reset``), "reached" ("full",
        "hover", "grasped" or "none"), "finite", "diverged", the setup statistics (errors, strains, steps,
        "sim_time" [s], "wall" [s]) and, for "grasped", the attempt log.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.

    Examples
    --------
    >>> from rope_routing.env.surface_route_env import make_env  # doctest: +SKIP
    >>> env = make_env(rank=0, seed=0)  # doctest: +SKIP
    >>> _ = env.reset(seed=0, options={"layout_seed": 9000021})  # doctest: +SKIP
    >>> ok, info = setup_start(env, "hover", np.random.default_rng(0))  # doctest: +SKIP
    >>> ok, info["reached"]  # doctest: +SKIP
    (True, 'hover')
    """
    if mode == "full":
        return True, dict(obs=env._last_obs, mode=mode, reached="full", ok=True, wall=0.0, sim_time=0.0)
    if mode not in ("hover", "grasped"):
        raise ValueError(f"unknown start mode {mode!r}")
    start_time = time.perf_counter()
    sim = env.sim
    driver = _Driver(env)
    grasp_node = env._grasp_point(driver.nodes)  # chosen once per episode (cached in env._gi)
    tracker = _Rope(env, driver, grasp_node)
    offset, height, dyaw = (rng.uniform(-HOVER_XY, HOVER_XY, 2), rng.uniform(*HOVER_H),
                            rng.uniform(-HOVER_YAW, HOVER_YAW))
    info = dict(mode=mode, gi=grasp_node, attempts=0, weld_node=-1, hover_h=height, attempt_log=[])
    yaw = sim.target_yaw

    def hover_goal():
        """Current hover goal [m] above the tracked grasp node, and its yaw [rad]."""
        position, rope_yaw = tracker.update()
        return np.r_[position[:2] + offset, position[2] + height], _yaw_goal(rope_yaw + dyaw, yaw, position[:2])

    # 1) Full-speed travel at the travel height to above the grasp point.
    ok = False
    for _ in range(BUDGET["transit"]):
        goal, yaw = hover_goal()
        if driver.out_of_time(mode) or not driver.step(np.r_[goal[:2], max(goal[2], Z_TRANSIT)], yaw, -1.0):
            break
        if np.linalg.norm(driver.tcp[:2] - goal[:2]) < NEAR_XY:
            ok = True
            break
    # 2) Slow descent to the hover height: settle there in "hover" mode, pass through in "grasped" mode.
    if ok:
        ok = False
        for _ in range(BUDGET["hover"]):
            goal, yaw = hover_goal()
            if driver.out_of_time(mode) or not driver.step(goal, yaw, -1.0, V_FINE):
                break
            if driver.settled(goal, yaw) or (mode == "grasped" and np.linalg.norm(driver.tcp - goal) < TOL_POS):
                ok = True
                break
    reached = "hover" if ok else "none"
    # 3) Grasp and lift.
    if ok and mode == "grasped":
        ok, goal = _grasp(env, driver, tracker, offset, yaw, rng, info)
        reached = "grasped" if ok else "hover"
    info.update(reached=reached, ok=ok, finite=driver.finite, diverged=driver.diverged,
                tcp_err=float(np.linalg.norm(driver.tcp - goal)),
                yaw_err=float(abs(wrap_half(sim.tool_yaw() - env._rope_dir_yaw(driver.nodes, grasp_node)))),
                tcp_speed=driver.speed, rest_err=float(np.linalg.norm(sim.target_pos - driver.tcp)),
                max_strain=driver.max_strain, max_seg_strain=driver.max_seg_strain, steps=driver.n_steps,
                sim_time=driver.n_steps * env.substeps * sim.dt)
    info["obs"] = refresh_bookkeeping(env) if driver.finite else env._last_obs
    info["wall"] = time.perf_counter() - start_time
    return ok, info


def setup_roam(env, rng: np.random.Generator):
    """A "full" start from a RANDOM gripper position, for exploring WHERE to grasp (notes B).

    With the fingers open, the TCP travels at the travel height to a random point within ROAM_XY of a uniformly
    random point of the object, settles at a random height in ROAM_Z with a random closing-axis yaw, and the episode
    starts there. Nothing is grasped, and no grasp point is chosen for the policy.

    Parameters
    ----------
    env : GenericSurfaceEnv
        The environment, just reset.
    rng : np.random.Generator
        Randomness of the start point, height and yaw.

    Returns
    -------
    ok : bool
        Whether the start position was reached and settled.
    info : dict
        As for :func:`setup_start`, with "goal" the start position [m].
    """
    start_time = time.perf_counter()
    driver = _Driver(env)
    point = driver.nodes[int(rng.integers(len(driver.nodes)))]
    goal = np.r_[point[:2] + rng.uniform(-ROAM_XY, ROAM_XY, 2), rng.uniform(*ROAM_Z)]
    yaw = _yaw_goal(env.sim.target_yaw + rng.uniform(-0.5 * np.pi, 0.5 * np.pi), env.sim.target_yaw, goal[:2])
    # 1) Travel at the travel height to above the start point.
    ok = False
    for _ in range(BUDGET["transit"]):
        if driver.out_of_time("roam") or not driver.step(np.r_[goal[:2], max(goal[2], Z_TRANSIT)], yaw, -1.0):
            break
        if np.linalg.norm(driver.tcp[:2] - goal[:2]) < NEAR_XY:
            ok = True
            break
    # 2) Settle at the start height (the state then stays put under zero actions).
    if ok:
        ok = False
        for _ in range(BUDGET["hover"]):
            if driver.out_of_time("roam") or not driver.step(goal, yaw, -1.0, V_FINE):
                break
            if driver.settled(goal, yaw):
                ok = True
                break
    info = dict(mode="roam", reached="roam" if ok else "none", ok=ok, finite=driver.finite, diverged=driver.diverged,
                goal=goal.tolist(), steps=driver.n_steps, sim_time=driver.n_steps * env.substeps * env.sim.dt)
    info["obs"] = refresh_bookkeeping(env) if driver.finite else env._last_obs
    info["wall"] = time.perf_counter() - start_time
    return ok, info


def _grasp(env, driver, tracker, offset, yaw, rng, info):
    """Descend with open fingers onto the tracked grasp node, close until the weld forms, and lift.

    One retry, re-centred (no offset) on the grasp node's current position and re-aligned. The fingers never close in
    mid-air: an attempt whose descent stalls above the grasp height is abandoned. On failure the fingers re-open above
    the rope.

    Returns
    -------
    ok : bool
        Whether the rope is held at the carry height.
    goal : np.ndarray, shape (3,)
        Last goal of the TCP (the lift goal, or the TCP position on failure) [m].
    """
    sim = env.sim
    for attempt in range(2):
        if driver.out_of_time("grasped"):
            break
        info["attempts"] = attempt + 1
        if attempt == 1:  # re-open, back off by 2 cm, re-centre
            offset = np.zeros(2)
            for _ in range(8):
                if not driver.step(np.r_[driver.tcp[:2], env.z_rope + 0.02], yaw, -1.0, V_FINE):
                    return False, driver.tcp
                if sim.theta_cmd < 0.05 and driver.tcp[2] > env.z_rope + 0.012:
                    break
        # Descend: the pinch point down to the closing height, slowing down for the last LAND_H.
        low, at_rope = None, False
        for _ in range(BUDGET["descend"]):
            position, rope_yaw = tracker.update()
            if attempt:
                yaw = _yaw_goal(rope_yaw, yaw, position[:2])
            low = np.r_[position[:2] + offset, CLOSE_Z]
            max_step = V_LAND if driver.want[2] < CLOSE_Z + LAND_H + 1e-6 else V_FINE
            if driver.out_of_time("grasped") or not driver.step(low, yaw, -1.0, max_step):
                break
            if driver.tcp[2] < CLOSE_BELOW:
                at_rope = True
                break
        log = dict(reached_rope=at_rope, z_close=float(driver.tcp[2]), z_lo=float(driver.tcp[2]),
                   z_hi=float(driver.tcp[2]), pad_max=0.0, gap=float(sim.pad_gap()), weld=False)
        info["attempt_log"].append(log)
        if not (at_rope and driver.finite):
            continue
        # Close with the pinch point held at 8 mm until both pads press and the weld forms.
        hold = np.r_[low[:2], CLOSE_Z]
        for _ in range(BUDGET["close"]):
            ok = driver.step(hold, yaw, 1.0, V_FINE)
            log.update(z_lo=min(log["z_lo"], float(driver.tcp[2])), z_hi=max(log["z_hi"], float(driver.tcp[2])),
                       pad_max=max(log["pad_max"], min(sim.pad_forces())))
            if not ok or (sim.grasped and env._grasp_state()[0]):
                break
        log.update(gap=float(sim.pad_gap()), weld=bool(sim.grasped))
        # Lift to the carry height.
        if driver.finite and sim.grasped and env._grasp_state()[0]:
            info["weld_node"] = int(sim.grasp_node)
            lift = np.r_[driver.tcp[:2], rng.uniform(*LIFT_H)]
            for _ in range(BUDGET["lift"]):
                if not driver.step(lift, yaw, 1.0, V_FINE) or driver.settled(lift, yaw):
                    break
            return bool(driver.finite and env._grasp_state()[0] and driver.tcp[2] > 0.015), lift
    # Failed: open and back off, so that the state is a clean not-grasped one.
    for _ in range(10):
        if not driver.finite or not driver.step(np.r_[driver.tcp[:2], env.z_rope + 0.03], yaw, -1.0, V_FINE):
            break
    return False, driver.tcp


def refresh_bookkeeping(env):
    """Make the environment's episode state consistent with the current physical state, as ``reset`` initialises it.

    Sets zero elapsed steps, no waypoint progress, a zero previous action and the potential of the CURRENT state, so
    the first ``env.step`` after a scripted start carries no stale-potential reward spike. The episode's grasp point
    (``env._gi``) is kept: the start state was built for it.

    Returns
    -------
    np.ndarray of float32
        The fresh observation.
    """
    env.elapsed_steps, env.wp_i = 0, 0
    env.prev_action = np.zeros(5)
    env.grasped_steps, env.drops, env.max_stage = 0, 0, 0
    env._route0 = None
    nodes, samples, feats, wind, tcp = env._measure()
    env.prev_x = samples
    grasped, *_ = env._grasp_state()
    env.was_grasped = grasped
    env.phi, _, _ = env._potential(tcp, nodes, samples, wind, grasped)
    if env._route0 is None:  # grasped start: freeze the pre-grasp route term now, as reset() does at t = 0
        env._potential(tcp, nodes, samples, wind, False)
    return env._make_obs(nodes, samples, feats, wind, tcp)
