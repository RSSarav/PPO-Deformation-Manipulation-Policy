"""GraspRouteEnv: the first rope-routing environment, kept as the base class of the later environments.

The FR3 with a Robotiq 2F-85 gripper must pick up an anchored soft rope lying in a random layout and wrap it around
three pegs into a triangle, in SuperDex physics throughout (``sim/grasp_scene.py``).

Action (5-D, each component in [-1, 1]):

    dx, dy, dz   TCP target displacement, up to 2 cm per step and axis
    dyaw         tool yaw change, up to 0.15 rad per step
    grip         finger closure target theta = 0.8 (grip + 1) / 2 (-1 open, +1 closed; values in between pre-shape)

Control: a 10 Hz policy with 10 physics substeps of 10 ms per step (the OSC and the implicit gripper PD run every
substep).

Observation (flat vector): robot proprioception, gripper state and pad contact forces, the generic per-sample
deformation features of the rope (positions and velocities relative to the TCP, edge strain, Laplacian bending,
tautness to the anchor and to the TCP; ``deformation/measures.py``), the per-sample goal points, the grasp target, the
windings and the waypoint progress.

Reward: staged potential-based shaping, penalties and a success bonus; see ``_potential`` and ``step``.

Positions are in metres in the robot base frame; angles are in radians.

Role in the system: ``SurfaceRouteEnv``, and through it the final ``GenericSurfaceEnv``, inherits this module's action
map (``_apply_action``), grasp state, helpers and constants, and replaces the observation, the reward and the reset.
The scripted expert (``ScriptedExpert``) exists only in this environment; the pure-RL trainer forbids its use.
"""

import gymnasium as gym
import numpy as np

from ..deformation import measures, shape_distance
from ..deformation.material_graph import from_polyline
from ..sim.grasp_scene import THETA_CLOSED, GraspScene, SceneCfg
from ..task.geometry import TriangleTask, winding_angles
from ..task.rope_init import random_layout

PEGS = np.array([[0.40, -0.09], [0.58, -0.09], [0.49, 0.066]])  # peg centres A, B, C (counter-clockwise) [m]
STEP_POS, STEP_YAW = 0.02, 0.15  # largest TCP target step per axis [m] and yaw step [rad] per control step
Z_MIN, Z_MAX = 0.003, 0.25  # TCP target height limits [m]
GRASP_FROM_END = 0.08  # preferred grasp point: 8 cm from the free end [m]
SUCCESS_SHAPE = 0.05  # success: mean distance of the samples from their goal points below this [m]
SUCCESS_WIND_FRAC = 0.85  # success: every peg wrapped by at least this fraction of the route's winding
GAMMA = 0.99  # RL discount that the shaping is designed for
GRASP_BONUS = 2.0  # potential jump when the rope is grasped (x10 reward scale -> +20)


def make_task():
    """The routing task shared by all environments.

    Returns
    -------
    TriangleTask
        Pegs A, B, C (``PEGS``), 7 cm peg height, 5 mm rope radius, and legacy waypoints on an arc of radius 18 cm
        around the centroid (10 cm clearance otherwise).

    Examples
    --------
    >>> task = make_task()
    >>> round(task.route_length, 4), len(task.waypoints)
    (0.6149, 15)
    """
    return TriangleTask(pegs=PEGS.copy(), peg_height=0.07, rope_radius=0.005, waypoint_clearance=0.10, arc_radius=0.18)


def wrap_pi(angle):
    """Wrap angles to [-pi, pi) [rad].

    Examples
    --------
    >>> round(float(wrap_pi(3 * np.pi / 2)), 6)
    -1.570796
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_half(angle):
    """Wrap angles to [-pi/2, pi/2) [rad]: the gripper's closing axis is symmetric under a rotation by pi.

    Examples
    --------
    >>> round(float(wrap_half(2.0)), 6)
    -1.141593
    """
    return (angle + np.pi / 2) % np.pi - np.pi / 2


class GraspRouteEnv(gym.Env):
    """Legacy rope-routing environment with point-graph deformation features and a scripted expert.

    Parameters
    ----------
    num_samples : int
        Number of material samples along the rope.
    horizon : int
        Episode length limit [control steps].
    substeps : int
        Physics substeps per control step.
    scene_cfg : SceneCfg, optional
        Physics configuration; ``SceneCfg()`` if omitted.

    Attributes
    ----------
    task : TriangleTask
        Pegs, anchor, route and legacy waypoints.
    sim : GraspScene
        The physics world.
    graph : MaterialGraph
        Material samples along the rope.
    num_samples : int
        Number of material samples K.
    sample_s : np.ndarray, shape (K,)
        Rest arclength of each sample from the anchored end [m].
    z_rope : float
        Height of the rope's centreline on the table (the rope radius) [m].
    sample_goals : np.ndarray, shape (K, 3)
        Goal point of each sample on the route [m].
    goal_mask : np.ndarray of bool, shape (K,)
        Samples that have a goal (those within the route length).
    target_wind : np.ndarray, shape (3,)
        Winding angle of the route around each peg [rad].
    pegs3 : np.ndarray, shape (3, 3)
        Peg centres at z = 0 [m].
    horizon, substeps : int
        Episode length limit and physics substeps per control step.
    wp_remaining : np.ndarray, shape (n_waypoints,)
        Route length remaining after each waypoint, measured along the waypoints [m].
    grasp_node_pref : int
        Default grasp node, ``GRASP_FROM_END`` from the free end.
    prefix_prob : float
        Probability of starting an episode with an expert prefix (reverse curriculum).
    expert : ScriptedExpert
        The privileged scripted expert.
    elapsed_steps : int
        Control steps since the start of the episode.
    wp_i : int
        Index of the next waypoint.
    phi : float
        Potential of the current state.
    prev_action : np.ndarray, shape (5,)
        Previous (clipped) action.
    prev_x : np.ndarray, shape (K, 3)
        Sample positions before the current control step [m] (for the velocities).
    grasped_steps, drops, max_stage : int
        Episode statistics: steps with the rope held, number of drops or releases, highest stage reached.
    was_grasped : bool
        Whether the rope was held after the previous step.

    Examples
    --------
    >>> env = GraspRouteEnv()  # doctest: +SKIP
    >>> obs, info = env.reset(seed=0, options={"layout_seed": 9000000})  # doctest: +SKIP
    >>> obs.shape, env.observation_space.shape  # doctest: +SKIP
    ((379,), (379,))
    """

    metadata = {"render_modes": []}

    def __init__(self, num_samples: int = 21, horizon: int = 500, substeps: int = 10,
                 scene_cfg: SceneCfg | None = None):
        super().__init__()
        self.task = make_task()
        self.sim = GraspScene(self.task, scene_cfg or SceneCfg())
        self.graph = from_polyline(self.sim.rest_nodes, num_samples)
        self.num_samples = num_samples
        self.sample_s = self.graph.geodesic[:, 0]
        self.z_rope = self.sim.rope_radius
        self.sample_goals, self.goal_mask = self.task.targets(self.sample_s, self.z_rope)
        self.target_wind = self.task.target_windings(self.z_rope)
        self.pegs3 = np.c_[self.task.pegs, np.zeros(3)]
        self.horizon, self.substeps = horizon, substeps
        waypoints = self.task.waypoints
        self.wp_remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)[::-1])[::-1], 0.0]
        self.grasp_node_pref = self.sim.num_nodes - 1 - int(round(GRASP_FROM_END / self.sim.seg))
        self.prefix_prob = 0.0
        self.expert = ScriptedExpert(self)
        obs_dim = len(self.reset(seed=0)[0])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)

    # ================================================================ helpers
    def set_prefix_prob(self, prob):
        """Set the probability of starting an episode with an expert prefix (reverse curriculum)."""
        self.prefix_prob = float(prob)

    def _grasp_point(self, nodes):
        """Rope node to grasp, chosen once per episode (task information, also given to the policy).

        The node nearest the free end (6-14 cm from it) where the rope is locally straight and both OPEN fingertips
        (4.5 cm to either side along the closing axis) land at least 2.5 cm from any other part of the rope and from
        the pegs; failing that, the candidate with the largest clearance.

        Parameters
        ----------
        nodes : np.ndarray, shape (N, 3)
            Rope node positions [m].

        Returns
        -------
        int
            Index of the grasp node.
        """
        if getattr(self, "_gi", None) is not None:
            return self._gi
        num_nodes, seg = self.sim.num_nodes, self.sim.seg
        best, best_clear = self.grasp_node_pref, -1.0
        for dist_from_end in np.arange(0.06, 0.145, 0.01):
            node = num_nodes - 1 - int(round(dist_from_end / seg))
            direction = nodes[min(node + 3, num_nodes - 1)] - nodes[max(node - 3, 0)]
            direction = direction[:2] / max(np.linalg.norm(direction[:2]), 1e-9)
            axis = np.array([-direction[1], direction[0]])
            tips = [nodes[node, :2] + 0.045 * axis, nodes[node, :2] - 0.045 * axis]
            # Clearance of the open fingertips from the rest of the rope and from the peg surfaces.
            far = np.abs(np.arange(num_nodes) - node) > 4
            clear = min(np.linalg.norm(nodes[far, :2] - tip, axis=1).min() for tip in tips)
            clear = min(clear, min(np.linalg.norm(self.task.pegs - tip, axis=1).min() - self.task.peg_radius
                                   for tip in tips))
            straight = np.linalg.norm(nodes[min(node + 3, num_nodes - 1), :2] - nodes[max(node - 3, 0), :2]) / (6 * seg)
            score = clear if straight > 0.9 else clear - 0.05
            if score >= 0.025:
                best = node
                break
            if score > best_clear:
                best, best_clear = node, score
        self._gi = best
        return best

    def _rope_dir_yaw(self, nodes, node):
        """Yaw of the closing axis perpendicular to the rope at a node (from the chord over 3 nodes each side) [rad]."""
        direction = nodes[min(node + 3, self.sim.num_nodes - 1)] - nodes[max(node - 3, 0)]
        axis = np.array([-direction[1], direction[0]])
        return float(np.arctan2(axis[1], axis[0]))  # closing axis perpendicular to the rope

    def _grasp_state(self):
        """Grasp state: (held with the fingers commanded closed, left pad force [N], right pad force [N], rope node
        nearest the TCP)."""
        force_left, force_right = self.sim.pad_forces()
        nodes = self.sim.rope_nodes()
        nearest = int(np.argmin(np.linalg.norm(nodes - self.sim.tcp(), axis=1)))
        grasped = bool(self.sim.grasped and self.sim.theta_target > 0.5 * THETA_CLOSED)
        return grasped, force_left, force_right, nearest

    def _measure(self):
        """Measure the current state: (rope nodes, sample positions, deformation features, winding angles, TCP)."""
        nodes = self.sim.rope_nodes()
        samples = self.graph.samples(nodes)
        tcp = self.sim.tcp()
        # Tautness toward the TCP uses the sample nearest to the pads (generically: the "constrained point").
        nearest = int(np.argmin(np.linalg.norm(samples - tcp, axis=1)))
        feats = measures.node_features(self.graph, samples, grasp_idx=nearest, anchor_idx=0)
        wind = winding_angles(samples, self.pegs3, self.task.peg_height)
        return nodes, samples, feats, wind, tcp

    # ================================================================ potential (reward shaping)
    def _potential(self, tcp, nodes, samples, wind, grasped):
        """Staged potential phi with a CONTINUOUS "total distance to go" across the grasp.

        Picking the rope up is rewarded (+GRASP_BONUS) and dropping it is penalised by the same amount; being
        potential-based, the shaping does not change the optimal policy. With D the distance to go:

        * not grasped: D = reach (TCP to the grasp point; hover first, then descend) + route length from the grasp
          point, and phi = -3 D + common - 0.3 (yaw misalignment);
        * grasped: D = route length still to go from the TCP (along the waypoints around B, C, A), and
          phi = GRASP_BONUS - 3 D + common - 2 |z - 3 cm|;

        where common = 2 sum(winding fractions) - 3 shape_error.

        Returns
        -------
        phi : float
            The potential.
        shape : float
            Mean distance of the samples from their goal points [m].
        wind_frac : np.ndarray, shape (3,)
            Winding of the rope around each peg as a fraction of the route's, clipped to [0, 1].
        """
        grasp_node = self._grasp_point(nodes)
        grasp_pos = nodes[grasp_node]
        shape = shape_distance.ordered(samples, self.sample_goals, self.goal_mask)
        wind_frac = np.clip(wind / self.target_wind, 0.0, 1.0)
        waypoints = self.task.waypoints
        wp_index = min(self.wp_i, len(waypoints) - 1)
        remaining = self.wp_remaining[wp_index] if self.wp_i < len(waypoints) else 0.0
        common = 2.0 * float(wind_frac.sum()) - 3.0 * shape
        if grasped:
            togo = (np.linalg.norm(tcp[:2] - waypoints[wp_index]) + remaining) if self.wp_i < len(waypoints) else 0.0
            phi = GRASP_BONUS - 3.0 * togo + common - 2.0 * abs(tcp[2] - 0.03)
        else:
            d_xy = float(np.linalg.norm(tcp[:2] - grasp_pos[:2]))
            z_goal = self.z_rope if d_xy < 0.015 else 0.08
            reach = d_xy + abs(tcp[2] - z_goal) * (0.5 if d_xy >= 0.015 else 1.0)
            # FROZEN at the episode start: when it depended on the rope's current position, PPO learned to push the
            # rope toward the route start with the fingers instead of grasping it (grasp rate fell from 96% to 4% in
            # run v3).
            if getattr(self, "_route0", None) is None:
                self._route0 = ((np.linalg.norm(grasp_pos[:2] - waypoints[wp_index]) + remaining)
                                if self.wp_i < len(waypoints) else 0.0)
            route_from_grasp = self._route0
            yaw_err = abs(wrap_half(self.sim.tool_yaw() - self._rope_dir_yaw(nodes, grasp_node)))
            phi = -3.0 * (reach + route_from_grasp) - 0.3 * yaw_err + common
        return phi, shape, wind_frac

    # ================================================================ observation
    def _obs_vec(self, dummy=False):
        """The last observation, or zeros of the expected size if ``dummy``."""
        if dummy:
            return np.zeros(self._obs_dim_guess(), np.float32)
        return self._last_obs

    def _obs_dim_guess(self):
        """Size of the observation vector."""
        return (7 + 7 + 3 + 2 + 3 + 3 + 3 + self.num_samples * (3 + 3 + 5 + 3 + 1) + 3 + 3 + 6 + 3
                + len(self.task.waypoints) + 1 + 5)

    def _make_obs(self, nodes, samples, feats, wind, tcp):
        """Assemble the float32 observation (layout in the module docstring) and store it in ``_last_obs``."""
        joint_pos, joint_vel = self.sim.joint_state()
        yaw = self.sim.tool_yaw()
        grasped, force_left, force_right, _ = self._grasp_state()
        grasp_node = self._grasp_point(nodes)
        grasp_pos = nodes[grasp_node]
        grasp_yaw = self._rope_dir_yaw(nodes, grasp_node)
        velocity = (samples - self.prev_x) / (self.substeps * self.sim.dt)
        features = np.stack([10 * feats["strain"], 0.05 * feats["bending"], feats["taut_grasp"], feats["taut_anchor"],
                             np.clip(feats["overstretch"], -1, 1)], 1)
        wp_onehot = np.zeros(len(self.task.waypoints))
        if self.wp_i < len(wp_onehot):
            wp_onehot[self.wp_i] = 1.0
        obs = np.concatenate([
            joint_pos[:7], 0.1 * joint_vel[:7],
            tcp, [np.sin(2 * yaw), np.cos(2 * yaw)], self.sim.target_pos - tcp,
            [self.sim.theta_cmd / THETA_CLOSED, self.sim.pad_gap() / 0.085, float(grasped)],
            [min(force_left, 30.0) / 10.0, min(force_right, 30.0) / 10.0, 0.0],
            (samples - tcp).ravel(), (0.1 * velocity).ravel(), features.ravel(), (self.sample_goals - tcp).ravel(),
            self.goal_mask.astype(float),
            grasp_pos - tcp, [np.sin(2 * (grasp_yaw - yaw)), np.cos(2 * (grasp_yaw - yaw)), 0.0],
            (self.pegs3 - tcp)[:, :2].ravel(), wind / self.target_wind, wp_onehot,
            [self.elapsed_steps / self.horizon], self.prev_action,
        ])
        self._last_obs = obs.astype(np.float32)
        return self._last_obs

    # ================================================================ gym API
    def reset(self, seed=None, options=None):
        """Start an episode on a new random layout.

        Parameters
        ----------
        seed : int, optional
            Seed of the environment's random number generator.
        options : dict, optional
            "layout_seed" fixes the layout; "no_prefix" disables the expert prefix of the reverse curriculum.

        Returns
        -------
        obs : np.ndarray of float32
            Initial observation.
        info : dict
            Empty.
        """
        super().reset(seed=seed)
        options = options or {}
        layout_seed = options.get("layout_seed", None)
        rng = np.random.default_rng(layout_seed) if layout_seed is not None else self.np_random
        layout = random_layout(rng, self.task.anchor, self.task.pegs, self.sim.rope_length, self.sim.num_nodes,
                               self.task.peg_radius, self.sim.rope_radius)
        self.sim.reset(layout)
        self._gi = None
        self._route0 = None
        self.elapsed_steps, self.wp_i = 0, 0
        self.prev_action = np.zeros(5)
        self.grasped_steps, self.drops, self.max_stage = 0, 0, 0
        self.was_grasped = False
        self.expert.reset()
        nodes, samples, feats, wind, tcp = self._measure()
        self.prev_x = samples
        grasped, *_ = self._grasp_state()
        self.phi, _, _ = self._potential(tcp, nodes, samples, wind, grasped)
        obs = self._make_obs(nodes, samples, feats, wind, tcp)
        # Reverse curriculum: play the expert for a random number of steps.
        if not options.get("no_prefix", False) and self.prefix_prob > 0 and self.np_random.random() < self.prefix_prob:
            prefix_steps = int(self.np_random.integers(20, 220))
            for _ in range(prefix_steps):
                obs, reward, terminated, truncated, info = self.step(self.expert.act())
                if terminated or truncated:
                    return self.reset(seed=None, options={"no_prefix": True})
            self.elapsed_steps = 0
        return obs, {}

    def _apply_action(self, act):
        """Map an action to the robot's targets: TCP target step, yaw step and finger closure.

        The TCP target moves by up to STEP_POS per axis, is clamped to Z_MIN..Z_MAX in height and to 0.25-0.78 m from
        the base horizontally, and is kept within 4 cm of the TCP per axis; the yaw target moves by up to STEP_YAW and
        is clamped to +-2.9 rad. While the rope is held, any grip command above -0.5 keeps the fingers closed.

        Parameters
        ----------
        act : np.ndarray, shape (5,)
            Clipped action (dx, dy, dz, dyaw, grip).
        """
        tcp = self.sim.tcp()
        target = self.sim.target_pos + act[:3] * STEP_POS
        target[2] = np.clip(target[2], Z_MIN, Z_MAX)
        radius = np.linalg.norm(target[:2])
        if radius < 0.25:
            target[:2] *= 0.25 / max(radius, 1e-6)
        elif radius > 0.78:
            target[:2] *= 0.78 / radius
        self.sim.target_pos = tcp + np.clip(target - tcp, -0.04, 0.04)
        self.sim.target_yaw = float(np.clip(self.sim.target_yaw + act[3] * STEP_YAW, -2.9, 2.9))
        grip = float(np.clip(act[4], -1, 1))
        if self.sim.grasped and grip > -0.5:
            # Holding the rope: the fingers open only on a deliberate release command (hysteresis, as in a real
            # gripper interface), so small exploration noise cannot drop the rope.
            self.sim.theta_target = THETA_CLOSED
        else:
            self.sim.theta_target = float(THETA_CLOSED * (grip + 1.0) / 2.0)  # continuous closure / pre-shaping

    def step(self, action):
        """Apply one action for one control step (``substeps`` physics substeps).

        Returns
        -------
        obs : np.ndarray of float32
            Next observation.
        reward : float
            Shaping 10 (GAMMA phi' - phi), strain, overstretch, smoothness, effort, table and time penalties, +20 on
            success, -10 on a simulator failure.
        terminated : bool
            Success.
        truncated : bool
            Simulator failure or the horizon reached.
        info : dict
            "is_success", "routed", "shape", "grasped", "stage", "max_stage", "wind_frac", "max_strain", "dropped",
            "t" (elapsed steps), "pad_force", "wp".
        """
        act = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.prev_x = self.graph.samples(self.sim.rope_nodes())
        self._apply_action(act)
        for _ in range(self.substeps):
            self.sim.substep()
        self.elapsed_steps += 1
        nodes, samples, feats, wind, tcp = self._measure()
        grasped, force_left, force_right, nearest = self._grasp_state()
        if grasped:
            self.grasped_steps += 1
        dropped = self.was_grasped and not grasped and self.sim.theta_target > 0.5 * THETA_CLOSED
        released = self.was_grasped and not grasped and not dropped
        if dropped or released:
            self.drops += 1
        self.was_grasped = grasped
        if (grasped and self.wp_i < len(self.task.waypoints)
                and np.linalg.norm(tcp[:2] - self.task.waypoints[self.wp_i]) < 0.035):
            self.wp_i += 1

        # Reward terms.
        phi, shape, wind_frac = self._potential(tcp, nodes, samples, wind, grasped)
        r_shaping = 10.0 * (GAMMA * phi - self.phi)
        self.phi = phi
        strain = feats["strain"]
        r_strain = -50.0 * float(np.sum(np.maximum(0.0, strain - 0.02) ** 2))
        r_overstretch = -5.0 * float(np.sum(np.maximum(0.0, feats["overstretch"]) ** 2))
        r_smooth = -0.02 * float(np.sum((act - self.prev_action) ** 2))
        r_action = -0.005 * float(np.sum(act[:4] ** 2))
        # Contact penalty: the gripper body pressing on the table.
        r_table = -0.5 * float(tcp[2] < 0.0015)
        r_time = -0.01
        routed = bool(np.all(wind_frac >= SUCCESS_WIND_FRAC))
        success = bool(routed and shape < SUCCESS_SHAPE and grasped)
        stage = 0 if not grasped else (1 + int(np.sum(wind_frac >= SUCCESS_WIND_FRAC)))
        self.max_stage = max(self.max_stage, stage)
        reward = (r_shaping + r_strain + r_overstretch + r_smooth + r_action + r_table + r_time
                  + (20.0 if success else 0.0))
        bad = (not np.all(np.isfinite(nodes))) or strain.max() > 0.3 or (not np.all(np.isfinite(tcp)))
        if bad:
            reward -= 10.0
        self.prev_action = act
        terminated = success
        truncated = bool(bad or self.elapsed_steps >= self.horizon)
        obs = self._make_obs(nodes, samples, feats, wind, tcp)
        info = dict(is_success=success, routed=routed, shape=shape, grasped=grasped, stage=stage,
                    max_stage=self.max_stage, wind_frac=wind_frac, max_strain=float(strain.max()),
                    dropped=int(dropped), t=self.elapsed_steps, pad_force=(force_left, force_right), wp=self.wp_i)
        return obs, float(reward), terminated, truncated, info

    # ================================================================ verification helper
    def release_and_hold(self, steps=15):
        """Open the gripper, move up and away, and check that the rope stays wrapped (no policy involved).

        Parameters
        ----------
        steps : int
            Control steps; the TCP target rises by 6 mm per step.

        Returns
        -------
        dict
            "held" (all pegs still wrapped and the shape error below 1.5 SUCCESS_SHAPE), "shape" [m] and
            "wind_frac".
        """
        tcp = self.sim.tcp()
        self.sim.theta_target = 0.0
        for step_index in range(steps):
            self.sim.target_pos = tcp + np.array([0.0, 0.0, 0.006 * (step_index + 1)])
            for _ in range(self.substeps):
                self.sim.substep()
        nodes = self.sim.rope_nodes()
        samples = self.graph.samples(nodes)
        wind = winding_angles(samples, self.pegs3, self.task.peg_height)
        wind_frac = np.clip(wind / self.target_wind, 0, 1)
        shape = shape_distance.ordered(samples, self.sample_goals, self.goal_mask)
        return dict(held=bool(np.all(wind_frac >= SUCCESS_WIND_FRAC) and shape < 1.5 * SUCCESS_SHAPE), shape=shape,
                    wind_frac=wind_frac)

    def expert_action(self):
        """Action of the scripted expert in the current state (not available to pure-RL training)."""
        return self.expert.act()


class ScriptedExpert:
    """Privileged closed-loop expert that uses the exact rope state; produces actions in the policy's action space.

    A state machine: approach above the grasp node, descend, close, lift to the carry height, route through the
    waypoints, then tension the rope with short pulls.

    Parameters
    ----------
    env : GraspRouteEnv
        The environment it acts in.
    """

    def __init__(self, env: GraspRouteEnv):
        self.env = env
        self.reset()

    def reset(self):
        """Return to the approach state at the start of an episode."""
        self.state = "approach"
        self.n_in_state = 0
        self.grasp_offset = 0
        self.close_wait = 0
        self.pull = 0

    def _goto(self, goal, yaw_goal, grip):
        """Action that moves the TCP target toward ``goal`` [m] and the yaw target toward ``yaw_goal`` [rad]."""
        sim = self.env.sim
        delta = (goal - sim.target_pos) / STEP_POS
        command = np.zeros(5)
        command[:3] = np.clip(delta, -1, 1)
        command[3] = np.clip(wrap_half(yaw_goal - sim.target_yaw) / STEP_YAW, -1, 1)
        command[4] = grip
        return command

    def act(self):
        """Expert action for the current state, shape (5,)."""
        env, sim = self.env, self.env.sim
        self.n_in_state += 1
        nodes = sim.rope_nodes()
        tcp = sim.tcp()
        node = int(np.clip(env._grasp_point(nodes) - self.grasp_offset, 3, sim.num_nodes - 4))
        yaw_rope = env._rope_dir_yaw(nodes, node)
        yaw_goal = sim.target_yaw + wrap_half(yaw_rope - sim.target_yaw)
        grip_open = -1.0  # fingers open while approaching and descending
        if self.state == "approach":
            command = self._goto(np.r_[nodes[node, :2], 0.10], yaw_goal, grip_open)
            if (np.linalg.norm(tcp[:2] - nodes[node, :2]) < 0.012 and abs(wrap_half(sim.tool_yaw() - yaw_rope)) < 0.08
                    and tcp[2] < 0.115):
                self._go("descend")
            elif self.n_in_state > 80:
                self._retry()
            return command
        if self.state == "descend":
            correction = nodes[node, :2] - tcp[:2]  # centre the pinch point on the rope
            height = max(sim.target_pos[2] - 0.012, env.z_rope + 0.0005)
            command = self._goto(np.r_[sim.target_pos[:2] + 0.7 * correction, height], yaw_goal, grip_open)
            if tcp[2] < env.z_rope + 0.006:
                self._go("close")
            elif self.n_in_state > 40:
                self._retry()
            return command
        if self.state == "close":
            command = self._goto(sim.target_pos.copy(), sim.target_yaw, 1.0)
            self.close_wait += 1
            grasped, force_left, force_right, _ = env._grasp_state()
            if grasped and self.close_wait > 8:
                self._go("lift")
            elif self.close_wait > 18:
                self._retry()
            return command
        if self.state == "lift":
            command = self._goto(np.r_[sim.target_pos[:2], 0.03], sim.target_yaw, 1.0)
            if tcp[2] > 0.022:
                self._go("route")
            return command
        if self.state == "route":
            waypoints = env.task.waypoints
            if env.wp_i < len(waypoints):
                return self._goto(np.r_[waypoints[env.wp_i], 0.03], sim.target_yaw, 1.0)
            self._go("tension")
        if self.state == "tension":
            # Gentle tightening: at most 6 short pulls, stopping as soon as the shape is good or the rope strains.
            self.pull += 1
            samples = env.graph.samples(nodes)
            shape = shape_distance.ordered(samples, env.sample_goals, env.goal_mask)
            strain = measures.node_strain(env.graph, samples).max()
            direction = env.task.waypoints[-1] - env.task.pegs[0]
            direction = direction / np.linalg.norm(direction)
            pull = self.pull <= 6 and shape > 0.045 and strain < 0.01
            goal = sim.target_pos + (np.r_[direction * 0.008, 0.0] if pull else 0.0)
            return self._goto(goal, sim.target_yaw, 1.0)
        return np.zeros(5)

    def _go(self, state):
        """Switch to ``state`` and reset the state counters."""
        self.state, self.n_in_state, self.close_wait = state, 0, 0

    def _retry(self):
        """Restart from the approach with the grasp point shifted by 2 more nodes (cycling through 0, 2, 4, 6)."""
        self.grasp_offset = (self.grasp_offset + 2) % 8
        self.state, self.n_in_state, self.close_wait = "approach", 0, 0
