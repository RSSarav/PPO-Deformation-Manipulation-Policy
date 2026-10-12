"""SurfaceRouteEnv: rope routing with the deformation observed and rewarded through the first and second fundamental
forms of the rope's boundary surface (surface area and curvature, notes/A_deformation_math.md). Built for pure
reinforcement learning.

The FR3 with a Robotiq 2F-85 picks up the anchored rope and wraps it counter-clockwise around pegs B, C and A. The
scene, the action space (5-D: dx, dy, dz, dyaw, grip in [-1, 1]), the 10 Hz control, the horizon (500 steps), the
success definition and its thresholds are those of ``GraspRouteEnv``. The differences:

* Deformation: ``TubeSurfaceDescriptor`` (``deformation/surface_forms.py``) is evaluated every step on the tube surface
  around the simulated rope: the first fundamental form gives the areal stretch, the second the mean, Gaussian,
  principal and vector curvature of every material sample. The point-graph features (chord strain, MLS deformation
  gradient, Laplacian bending, tautness, overstretch) are not computed: the chord strain reads every bend as
  compression and cannot see real stretch on a peg wrap (notes A).
* Potential: phi = staged potential + phi_kappa. The staged potential is ``GraspRouteEnv``'s except for its reach term
  (see ``_potential``): the parent switches the reach goal from the hover height (8 cm) to rope height at
  d_xy = 1.5 cm, a jump of up to about 0.45 in phi (+-4.5 of shaping); here the goal height and the vertical weight
  blend continuously (smoothstep over d_xy in [1, 3] cm). phi_kappa = -W_KAPPA L_KAPPA (masked mean over samples of
  |kappa_i - kappa*_i|) rewards matching the route's curvature profile.
* Penalties: the chord-strain and overstretch terms are replaced by an areal-STRETCH term on area_ratio_i - 1, one-sided
  like the chord-strain term it replaces (a rope cannot carry compression).
* No scripted expert exists in this environment: there is no expert prefix in ``reset``, and ``expert_action`` and
  ``set_prefix_prob(> 0)`` raise.
* Release debounce: while the rope is held, the fingers open only after RELEASE_STEPS consecutive commands with
  grip < -0.5 (releasing is never needed for the task; this keeps exploration noise from dropping the rope).
* Start-state curriculum: "full" (untouched layout), "hover" (fingers open 3-6 cm above the grasp point) or "grasped"
  (rope held about 3 cm above the table), produced by driving the robot through the environment's OWN action
  interface (``env/start_states.py``: legal policy actions, nothing recorded, no expert, no demonstrations).
* SuperDex divergence: when the Newton solve diverges, Mochi's explosion control zeroes the rod's displacements (the
  rope jumps to its straight rest shape through the robot base; 7-15% of policy episodes, notes B). It is detected
  exactly (all displacements are 0 after a substep) and handled like a non-finite state: "bad" (-10, truncated).

Folds (notes A). The 12-gon tube is exact while every node turns by less than about 106 degrees; sharper hairpins fold
its prisms through each other. The descriptor flags those samples (valid = 0) and gives them the smooth tube's closed
forms, which equal the mesh values for area_ratio and curvature_vector wherever both exist. area_ratio - 1 is therefore
the true axial stretch and the curvature the centreline's window mean on EVERY sample, and phi_kappa and the area
penalty use them as they are. Two mesh quantities are still not trusted near the validity limit, and the environment
corrects them:

* r^2 K_min is floored at the smooth tube's -9 (r kappa <= 0.9, the clamp of the closed forms): at sharp turns the mesh
  K_min of valid samples runs away (r^2 K_min -2.2 instead of -1.35 at 89 degrees per node, the sharpest turn the rope
  reaches; -5.6 instead of -1.8 at 100 degrees; -22 instead of -2.2 at 106 degrees). K_max needs no clip (mesh
  r^2 K_max <= 0.73 on valid samples of uniform bends, <= 0.9 at an isolated valid 110-degree kink; closed form
  <= 0.47), and K_abs tracks the smooth tube within 3% up to the limit: it is the clean unsigned-bending feature.
* The Willmore global is ALWAYS the smooth tube's closed form (``descriptor.smooth_tube_forms``): the descriptor
  switches the whole-rope value from the mesh to the closed form at the first fold (a jump), and the mesh value
  under-resolves bending. Both equal willmore_rest at rest by construction.

Observation (flat float32, 516 entries for K = 21; names and ranges in ``self.obs_index``, deformation columns in
DEF_FEATURES):

    robot       [0, 33)     q (7), 0.1 qd (7), TCP (3), sin/cos 2 yaw (2), TCP target - TCP (3), [finger command,
                            pad gap / 8.5 cm, grasped] (3), pad forces / 10 N (2), previous action (5), elapsed steps /
                            horizon (1)
    deformation [33, 390)   per material sample (21 x 17, sample-major, DEF_FEATURES): position - TCP (3),
                            0.1 velocity (3), 10 (area_ratio - 1), 10 r (H - H_rest), r^2 K_mean, r^2 K_abs,
                            max(r^2 K_min, -9), r^2 K_max, L_KAPPA curvature_vector (3, world frame),
                            L_KAPPA signed_planar_curvature, valid (1 = unfolded tube)
                [390, 392)  globals: 10 (area_ratio_total - 1), willmore_smooth / willmore_rest - 1
    potential   [392, 393)  clip(phi / 10, -5, 5): the shaping potential itself (the critic's target is V - 10 phi, a
                            nonlinear function of the rest of the observation; notes B)
    goal        [393, 516)  target point - TCP (3K), target mask (K), L_KAPPA target signed curvature (K), grasp
                            node - TCP (3), sin/cos 2 (rope yaw - tool yaw) (2), next waypoint at carry height - TCP
                            (3), waypoint progress (1), winding fraction per peg (3), pegs - TCP (xy, 6)

Here r is the rope radius (5 mm) and L_KAPPA = 3 cm (a peg wrap reads about 1). Non-finite entries are zeroed and
everything is clipped to +-OBS_LIM.

Reward per step:

    10 (shaping_gamma phi' - phi)                     with shaping_gamma = 1 (progress reward; see __init__)
    - 50 sum_i max(0, area_ratio_i - 1 - 0.02)^2      areal stretch (replaces chord strain and overstretch)
    - 0.02 |a - a_prev|^2 - 0.005 |a_xyz,yaw|^2 - 0.5 [TCP below 1.5 mm] - 0.01 + 20 [success]
    - 10, and truncation, if "bad": a non-finite state, any |area_ratio_i - 1| > 0.3, or a solver divergence.

``info["r_terms"]`` holds the parts (they sum to the reward), ``info["phi"]`` and ``info["phi_kappa"]`` the potentials,
``info["n_invalid"]`` and ``info["fold_max_deg"]`` the folded-sample count and the sharpest node turn (for logging). A
"bad" step is a simulator failure, not a task state: a diverged rope back at its rest shape, or an exploded one (smoke
run: true segment stretch of +79% to +311% one step after < 2.2%, mostly with the gripper 12-21 cm above the table). It
pays the -10 and the action terms only (no shaping into the invalid state and no area penalty on it, which read up to
-589 on exploded ropes) and returns the LAST VALID observation, so the truncation bootstraps from a state the critic was
trained on (as ``rl/async_ppo.py`` does for environment exceptions) instead of extrapolating to a rope it never saw.

Positions are in metres in the robot base frame; angles are in radians; curvatures are in 1/m.

Usage::

    env = make_env(rank, seed, eval=False)
    env.set_start_probs({"full": 0.5, "hover": 0.2, "grasped": 0.3})
    obs, info = env.reset(seed=seed, options={"layout_seed": layout_seed, "no_prefix": True, "start_mode": "grasped"})

Role in the system: the predecessor of the final ``GenericSurfaceEnv``, which subclasses it for the robot interface,
the stretch penalty, the bad-step handling, the release debounce and the start-state curriculum, and replaces the
observation, the potential and the success test. The best previous policy (``data/runs/surf_ppo2/best_2M.pt``) was trained
in this environment.
"""

import time

import gymnasium as gym
import numpy as np

from ..deformation import shape_distance
from ..deformation.material_graph import from_polyline
from ..deformation.surface_forms import TubeSurfaceDescriptor, polyline_window_curvature
from ..sim.grasp_scene import THETA_CLOSED, GraspScene, SceneCfg
from ..task.geometry import winding_angles
from ..task.rope_init import random_layout
from .grasp_route_env import (GAMMA, GRASP_BONUS, GRASP_FROM_END, SUCCESS_SHAPE, SUCCESS_WIND_FRAC, GraspRouteEnv,
                              make_task, wrap_half)
from .start_states import setup_start

MODES = ("full", "hover", "grasped")  # start modes, in the order of the start probabilities
FALLBACK = {"grasped": ("grasped", "hover", "full"), "hover": ("hover", "full"), "full": ("full",)}  # if setup fails
L_KAPPA = 0.03  # curvature length scale [m]: observation scaling and phi_kappa weight (peg-wrap window mean ~33 1/m)
W_KAPPA = 1.0  # phi_kappa = -W_KAPPA * L_KAPPA * masked mean |kappa - kappa*|
AREA_TOL, W_AREA, AREA_BAD = 0.02, 50.0, 0.3  # areal-stretch dead band, penalty weight, "bad" on |area_ratio - 1|
RELEASE_STEPS, RELEASE_GRIP = 3, -0.5  # release debounce: consecutive grip < -0.5 commands needed to open while holding
CARRY_Z = 0.03  # carry height of the grasped potential (GraspRouteEnv._potential): z of the waypoint goal [m]
HOVER_Z = 0.08  # reach goal height far from the grasp point (GraspRouteEnv._potential) [m]
REACH_BLEND = (0.01, 0.03)  # d_xy range [m] over which the reach goal blends from rope height (below) to hover (above)
R2K_MIN = -0.9 / (1.0 - 0.9)  # floor of r^2 K_min: the smooth-tube closed form at its r kappa clamp of 0.9 (= -9)
PHI_OBS_SCALE, PHI_OBS_LIM = 0.1, 5.0  # potential observation: clip(phi / 10, -5, 5)
OBS_LIM = 100.0  # observation clip
DEF_FEATURES = ("dx", "dy", "dz", "vx", "vy", "vz", "area_strain", "dH", "K_mean", "K_abs", "K_min", "K_max",
                "kappa_x", "kappa_y", "kappa_z", "kappa_signed", "valid")  # per-sample deformation columns


class SurfaceRouteEnv(GraspRouteEnv):
    """Rope routing with fundamental-form deformation features, a stretch penalty and a start-state curriculum.

    Parameters
    ----------
    num_samples : int
        Number of material samples K along the rope.
    horizon : int
        Episode length limit [control steps].
    substeps : int
        Physics substeps per control step.
    scene_cfg : SceneCfg or dict, optional
        Physics configuration (a dict is passed to ``SceneCfg``); ``SceneCfg()`` if omitted.
    start_probs : dict or sequence, optional
        Start-state distribution (see :meth:`set_start_probs`); only "full" starts if omitted.
    eval_mode : bool
        Evaluation: "full" starts unless a reset explicitly asks for another start mode.
    seed : int, optional
        Seed of the first reset (which the constructor performs).
    shaping_gamma : float
        Discount in the progress shaping 10 (shaping_gamma phi' - phi); 1 rewards progress itself.

    Attributes
    ----------
    desc : TubeSurfaceDescriptor
        Fundamental-form descriptor of the rope's tube surface.
    gamma : float
        RL discount this environment is designed for; the trainer checks that it matches.
    shaping_gamma : float
        Discount of the progress shaping.
    kappa_target : np.ndarray, shape (K,)
        Target signed planar curvature of each sample [1/m] (0 for samples without a goal).
    eval_mode : bool
        Evaluation mode.
    start_probs : np.ndarray, shape (3,)
        Probabilities of the "full", "hover" and "grasped" starts.
    start_mode : str
        Start mode of the current episode.
    obs_index : dict of str to (int, int)
        Index range of every observation block.

    The remaining attributes are those of :class:`GraspRouteEnv`, without the expert (``expert`` is None).

    Examples
    --------
    >>> env = make_env(rank=0, seed=0)  # doctest: +SKIP
    >>> env.observation_space.shape, env.obs_index["deform"]  # doctest: +SKIP
    ((516,), (33, 390))
    """

    metadata = {"render_modes": []}

    def __init__(self, num_samples: int = 21, horizon: int = 500, substeps: int = 10,
                 scene_cfg: SceneCfg | dict | None = None, start_probs=None, eval_mode: bool = False,
                 seed: int | None = None, shaping_gamma: float = 1.0):
        # GraspRouteEnv.__init__ is not called: it builds a ScriptedExpert and resets through the old observation. The
        # task, scene and sample setup below is the same; no expert object exists in this environment.
        gym.Env.__init__(self)
        if isinstance(scene_cfg, dict):
            scene_cfg = SceneCfg(**scene_cfg)
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
        self.expert = None
        self.gamma = GAMMA  # the RL discount this environment is designed for; the trainer asserts that it matches
        # Progress shaping 10 (shaping_gamma phi' - phi). With shaping_gamma = gamma it is policy-invariant (Ng 1999):
        # each step then pays -10 (1 - gamma) phi, so moving forward and staying there is worth exactly nothing in
        # discounted return, and only the sparse +20 success counts. Run surf_ppo1 showed the critic learning exactly
        # that (explained variance 0.998) and no progress in 1.9M steps; holding still in a "grasped" start earned 38,
        # against 48 for routing the rope past all 15 waypoints. shaping_gamma = 1 rewards progress itself: the sum over
        # an episode is 10 (phi_T - phi_0).
        self.shaping_gamma = float(shaping_gamma)

        # The fundamental-form descriptor of the rope's tube surface and the target curvature profile.
        self.desc = TubeSurfaceDescriptor(self.sim.rest_nodes, self.sim.rope_radius, self.sample_s)
        # Closed-form (smooth-tube) Willmore energy for the Willmore global: continuous through folds (module
        # docstring). Bound to this descriptor instance, so it is unaffected if self.desc is swapped for a wrapper.
        self._smooth_forms, self._willmore_rest = self.desc.smooth_tube_forms, self.desc.forms.willmore_rest
        # Target = window mean of the route's curvature over each sample's rest-arclength window, (T(hi) - T(lo)) /
        # length: the estimator the descriptor applies to the rope (a plain average of the vertex curvature would read
        # up to 14% higher on peg wraps, where a 4 cm window turns by about 110 degrees; notes A).
        _, kappa_route = polyline_window_curvature(self.task.route, self.desc.sample_window)
        self.kappa_target = np.where(self.goal_mask, kappa_route, 0.0)
        self._mask_idx = np.flatnonzero(self.goal_mask)
        self._surf_nodes = self._surf = None

        # Start-state curriculum, release debounce, episode state and observation layout.
        self.eval_mode = bool(eval_mode)
        self.start_probs = np.array([1.0, 0.0, 0.0])
        if start_probs is not None:
            self.set_start_probs(start_probs)
        self.start_mode, self._release_votes = "full", 0
        blocks = self._obs_blocks()
        ends = np.cumsum([size for _, size in blocks])
        self.obs_index = {name: (int(end - size), int(end)) for (name, size), end in zip(blocks, ends)}
        obs, _ = self.reset(seed=seed, options={"start_mode": "full"})
        assert len(obs) == ends[-1], (len(obs), ends[-1])
        self.observation_space = gym.spaces.Box(-OBS_LIM, OBS_LIM, (len(obs),), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)

    def _obs_blocks(self):
        """(name, size) of the observation blocks, in order; subclasses with another observation override this."""
        return [("q", 7), ("qd", 7), ("tcp", 3), ("yaw", 2), ("target_offset", 3), ("gripper", 3), ("pad_force", 2),
                ("prev_action", 5), ("time", 1), ("deform", self.num_samples * len(DEF_FEATURES)),
                ("deform_global", 2), ("potential", 1), ("target_pos", 3 * self.num_samples),
                ("target_mask", self.num_samples), ("target_curvature", self.num_samples), ("grasp_pos", 3),
                ("grasp_yaw", 2), ("waypoint", 3), ("waypoint_progress", 1), ("wind_frac", 3), ("pegs", 6)]

    # ================================================================ curriculum / no-expert API
    def set_start_probs(self, probs):
        """Set the start-state distribution.

        Parameters
        ----------
        probs : dict or sequence of float
            A dict over "full", "hover" and "grasped" (missing modes get 0), or a (full, hover, grasped) triple;
            normalised to sum to 1.

        Raises
        ------
        ValueError
            For unknown modes, a wrong shape, or negative, non-finite or all-zero probabilities.
        """
        if isinstance(probs, dict):
            unknown = set(probs) - set(MODES)
            if unknown:
                raise ValueError(f"unknown start modes {sorted(unknown)}")
            weights = np.array([float(probs.get(mode, 0.0)) for mode in MODES])
        else:
            weights = np.asarray(probs, dtype=np.float64).ravel()
            if weights.shape != (3,):
                raise ValueError(f"expected (full, hover, grasped) probabilities, got {probs!r}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"invalid start probabilities {probs!r}")
        self.start_probs = weights / weights.sum()

    def set_prefix_prob(self, prob):
        """Reject the expert-prefix curriculum: only a probability of 0 is accepted (pure RL)."""
        if float(prob) != 0.0:
            raise ValueError("SurfaceRouteEnv is pure RL: the expert-prefix curriculum does not exist here")

    def expert_action(self):
        """Always raises: this environment has no scripted expert (pure RL)."""
        raise RuntimeError("SurfaceRouteEnv has no scripted expert (pure RL)")

    # ================================================================ deformation measurement
    def _surface(self, nodes):
        """``TubeSurfaceDescriptor`` output for these rope nodes, plus "willmore_smooth" (the smooth tube's closed-form
        Willmore energy).

        Cached: ``_measure``, ``_potential`` and ``_make_obs`` of one state share one evaluation (about 1 ms).
        """
        if self._surf_nodes is None or not np.array_equal(self._surf_nodes, nodes):
            self._surf = dict(self.desc(nodes))
            self._surf["willmore_smooth"] = self._smooth_forms(nodes)["willmore"]
            self._surf_nodes = np.array(nodes, dtype=np.float64)
        return self._surf

    def _measure(self):
        """Measure the current state: (rope nodes, sample positions, surface measures, winding angles, TCP)."""
        nodes = self.sim.rope_nodes()
        samples = self.graph.samples(nodes)
        wind = winding_angles(samples, self.pegs3, self.task.peg_height)
        return nodes, samples, self._surface(nodes), wind, self.sim.tcp()

    def curvature_match(self, nodes):
        """Masked mean |kappa_i - kappa*_i| of the measured and target signed planar curvature [1/m]."""
        kappa = self._surface(nodes)["signed_planar_curvature"]
        return float(np.mean(np.abs(kappa - self.kappa_target)[self._mask_idx]))

    def reach_blend(self, d_xy):
        """Weight in [0, 1] of the "descend to the rope" reach goal at horizontal distance ``d_xy`` [m] from the
        grasp point: 1 within 1 cm, 0 beyond 3 cm, a smoothstep in between (C1, so the shaping never jumps).

        Examples
        --------
        >>> [env.reach_blend(d_xy) for d_xy in (0.0, 0.01, 0.02, 0.03, 0.2)]  # doctest: +SKIP
        [1.0, 1.0, 0.5, 0.0, 0.0]
        """
        lo, hi = REACH_BLEND
        ramp = float(np.clip((hi - d_xy) / (hi - lo), 0.0, 1.0))
        return ramp * ramp * (3.0 - 2.0 * ramp)

    def _potential(self, tcp, nodes, samples, wind, grasped):
        """Staged potential plus phi_kappa (curvature-profile matching).

        Everything that sets ``self.phi`` (``reset``, ``start_states.refresh_bookkeeping``, ``step``) goes through
        here, so phi always includes both. The staged part is ``GraspRouteEnv._potential`` term for term, except the
        reach: the parent jumps from d_xy + 0.5 |z - 8 cm| to d_xy + |z - z_rope| at d_xy = 1.5 cm (a step of up to
        about 0.45 in phi, i.e. +-4.5 of shaping in one step, which a policy can farm by hovering at the boundary).
        Here the goal height and the vertical weight blend with w = reach_blend(d_xy):
        reach = d_xy + (0.5 + 0.5 w) |z - (w z_rope + (1 - w) 8 cm)|. The potential remains a function of the state
        (and of the episode's frozen grasp point and route term) only.

        Returns
        -------
        phi : float
            The potential.
        shape : float
            Mean distance of the samples from their goal points [m].
        wind_frac : np.ndarray, shape (3,)
            Winding fraction per peg, clipped to [0, 1].
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
            phi = GRASP_BONUS - 3.0 * togo + common - 2.0 * abs(tcp[2] - CARRY_Z)
        else:
            d_xy = float(np.linalg.norm(tcp[:2] - grasp_pos[:2]))
            blend = self.reach_blend(d_xy)
            z_goal = blend * self.z_rope + (1.0 - blend) * HOVER_Z
            reach = d_xy + abs(tcp[2] - z_goal) * (0.5 + 0.5 * blend)
            # FROZEN at the episode start, as in the parent: a route term that follows the rope taught PPO to push the
            # rope toward the route start instead of grasping it.
            if getattr(self, "_route0", None) is None:
                self._route0 = ((np.linalg.norm(grasp_pos[:2] - waypoints[wp_index]) + remaining)
                                if self.wp_i < len(waypoints) else 0.0)
            yaw_err = abs(wrap_half(self.sim.tool_yaw() - self._rope_dir_yaw(nodes, grasp_node)))
            phi = -3.0 * (reach + self._route0) - 0.3 * yaw_err + common
        return phi - W_KAPPA * L_KAPPA * self.curvature_match(nodes), shape, wind_frac

    # ================================================================ observation
    def _make_obs(self, nodes, samples, surf, wind, tcp):
        """Assemble the float32 observation (layout in the module docstring) and store it in ``_last_obs``."""
        sim, radius = self.sim, self.sim.rope_radius
        joint_pos, joint_vel = sim.joint_state()
        yaw = sim.tool_yaw()
        grasped, force_left, force_right, _ = self._grasp_state()
        grasp_node = self._grasp_point(nodes)
        grasp_yaw = self._rope_dir_yaw(nodes, grasp_node)
        velocity = (samples - self.prev_x) / (self.substeps * sim.dt)
        # Per-sample deformation features (DEF_FEATURES), scaled to order 1.
        deform = np.column_stack([samples - tcp, 0.1 * velocity, 10.0 * (surf["area_ratio"] - 1.0),
                                  10.0 * radius * (surf["H"] - surf["H_rest"]),
                                  radius * radius * surf["K_mean"], radius * radius * surf["K_abs"],
                                  np.maximum(radius * radius * surf["K_min"], R2K_MIN), radius * radius * surf["K_max"],
                                  L_KAPPA * surf["curvature_vector"], L_KAPPA * surf["signed_planar_curvature"],
                                  np.asarray(surf["valid"], dtype=np.float64)])
        num_waypoints = len(self.task.waypoints)
        waypoint = self.task.waypoints[min(self.wp_i, num_waypoints - 1)]
        obs = np.concatenate([
            joint_pos[:7], 0.1 * joint_vel[:7], tcp, [np.sin(2 * yaw), np.cos(2 * yaw)], sim.target_pos - tcp,
            [sim.theta_cmd / THETA_CLOSED, sim.pad_gap() / 0.085, float(grasped)],
            [min(force_left, 30.0) / 10.0, min(force_right, 30.0) / 10.0],
            self.prev_action, [self.elapsed_steps / self.horizon],
            deform.ravel(),
            [10.0 * (surf["area_ratio_total"] - 1.0), surf["willmore_smooth"] / self._willmore_rest - 1.0],
            [np.clip(PHI_OBS_SCALE * self.phi, -PHI_OBS_LIM, PHI_OBS_LIM)],
            (self.sample_goals - tcp).ravel(), self.goal_mask, L_KAPPA * self.kappa_target,
            nodes[grasp_node] - tcp, [np.sin(2 * (grasp_yaw - yaw)), np.cos(2 * (grasp_yaw - yaw))],
            np.r_[waypoint, CARRY_Z] - tcp, [self.wp_i / num_waypoints], wind / self.target_wind,
            (self.pegs3 - tcp)[:, :2].ravel()])
        obs = np.nan_to_num(obs, nan=0.0, posinf=OBS_LIM, neginf=-OBS_LIM)
        self._last_obs = np.clip(obs, -OBS_LIM, OBS_LIM).astype(np.float32)
        return self._last_obs

    def deformation_block(self, obs):
        """Per-sample deformation features, shape (K, len(DEF_FEATURES)), and the 2 deformation globals of an
        observation."""
        start, stop = self.obs_index["deform"]
        global_start, global_stop = self.obs_index["deform_global"]
        return (np.asarray(obs)[start:stop].reshape(self.num_samples, len(DEF_FEATURES)),
                np.asarray(obs)[global_start:global_stop])

    # ================================================================ gym API
    def _begin_episode(self):
        """Initialise the episode bookkeeping for the current physical state, as ``GraspRouteEnv.reset`` does, and
        return the observation."""
        self._gi = None
        self._route0 = None
        self.elapsed_steps, self.wp_i = 0, 0
        self.prev_action = np.zeros(5)
        self.grasped_steps, self.drops, self.max_stage = 0, 0, 0
        self.was_grasped = False
        self._release_votes = 0
        nodes, samples, surf, wind, tcp = self._measure()
        self.prev_x = samples
        grasped, *_ = self._grasp_state()
        self.phi, _, _ = self._potential(tcp, nodes, samples, wind, grasped)
        return self._make_obs(nodes, samples, surf, wind, tcp)

    def _pick_mode(self, options):
        """Start mode of a reset: ``options["start_mode"]`` if given, else "full" in evaluation, else a draw from the
        start probabilities."""
        mode = options.get("start_mode")
        if mode is not None:
            if mode not in MODES:
                raise ValueError(f"unknown start mode {mode!r}")
            return mode
        if self.eval_mode or self.start_probs[0] >= 1.0:
            return "full"
        return MODES[int(self.np_random.choice(len(MODES), p=self.start_probs))]

    def reset(self, seed=None, options=None):
        """Start an episode: a new random layout, then the start state.

        A failed setup restores the untouched layout and falls back from "grasped" to "hover" to "full".

        Parameters
        ----------
        seed : int, optional
            Seed of the environment's random number generator.
        options : dict, optional
            "layout_seed" fixes the layout; "start_mode" forces the start mode.

        Returns
        -------
        obs : np.ndarray of float32
            Initial observation.
        info : dict
            "start_mode" (the mode actually used), "start_requested", "start_tried" (one record per attempted
            setup), "start_wall" [s] and "phi".
        """
        gym.Env.reset(self, seed=seed)
        options = options or {}
        layout_seed = options.get("layout_seed", None)
        rng = np.random.default_rng(layout_seed) if layout_seed is not None else self.np_random
        self._layout = random_layout(rng, self.task.anchor, self.task.pegs, self.sim.rope_length, self.sim.num_nodes,
                                     self.task.peg_radius, self.sim.rope_radius)
        self.sim.reset(self._layout)
        obs = self._begin_episode()
        requested = self._pick_mode(options)
        self.start_mode, tried, start_time = "full", [], time.perf_counter()
        for mode in FALLBACK[requested]:
            if mode == "full":
                break
            ok, start_info = setup_start(self, mode, self.np_random)
            tried.append(dict(mode=mode, ok=bool(ok), reached=start_info.get("reached"),
                              steps=int(start_info.get("steps", 0))))
            if ok:
                self.start_mode, obs = mode, start_info["obs"]
                self._release_votes = 0
                break
            self.sim.reset(self._layout)  # back to the untouched layout for the next mode in the chain
            obs = self._begin_episode()
        info = dict(start_mode=self.start_mode, start_requested=requested, start_tried=tried,
                    start_wall=time.perf_counter() - start_time, phi=float(self.phi))
        return obs, info

    def _apply_action(self, act):
        """``GraspRouteEnv``'s action map with the release debounce: while the rope is held, grip < -0.5 opens the
        fingers only on the RELEASE_STEPS-th consecutive such command (the parent opens on the first)."""
        super()._apply_action(act)
        grip = float(np.clip(act[4], -1.0, 1.0))
        if self.sim.grasped:
            self._release_votes = self._release_votes + 1 if grip < RELEASE_GRIP else 0
            self.sim.theta_target = (THETA_CLOSED if self._release_votes < RELEASE_STEPS
                                     else float(THETA_CLOSED * (grip + 1.0) / 2.0))
        else:
            self._release_votes = 0

    def _simulate(self):
        """Run the control step's physics substeps; returns True if Mochi's explosion control fired.

        On a diverged Newton solve the rod's displacements AND velocities are zeroed (mochi_rod.cpp), which never
        happens otherwise (the anchored node alone is about 1 mm off its rest position).
        """
        diverged = False
        for _ in range(self.substeps):
            self.sim.substep()
            diverged |= not np.any(self.sim.rope.get_displacements())
        return diverged

    def step(self, action):
        """Apply one action for one control step.

        Returns
        -------
        obs : np.ndarray of float32
            Next observation (the last valid one after a "bad" step).
        reward : float
            Sum of ``info["r_terms"]`` (see the module docstring).
        terminated : bool
            Success.
        truncated : bool
            A "bad" step or the horizon reached.
        info : dict
            "is_success", "start_mode", "routed", "shape", "grasped", "stage", "max_stage", "wind_frac",
            "area_strain_max", "max_strain", "curvature_match", "phi", "phi_kappa", "n_invalid", "fold_max_deg",
            "sim_diverged", "dropped", "t" (elapsed steps), "pad_force", "wp" and "r_terms".
        """
        # NaN -> 0: a NaN TCP target would poison the simulator until the next reset.
        act = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64).ravel()), -1, 1)
        self.prev_x = self.graph.samples(self.sim.rope_nodes())
        self._apply_action(act)
        diverged = self._simulate()
        self.elapsed_steps += 1
        nodes, samples, surf, wind, tcp = self._measure()
        grasped, force_left, force_right, _ = self._grasp_state()
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
        curv_match = self.curvature_match(nodes)
        stretch = surf["area_ratio"] - 1.0
        area_dev = np.abs(stretch)
        terms = dict(
            shaping=10.0 * (self.shaping_gamma * phi - self.phi),  # self.gamma: the discount the trainer checks
            # Stretch only: a rope cannot carry compression (area_ratio is the exact axial stretch, folds included).
            area=-W_AREA * float(np.sum(np.maximum(0.0, stretch - AREA_TOL) ** 2)),
            smooth=-0.02 * float(np.sum((act - self.prev_action) ** 2)),
            action=-0.005 * float(np.sum(act[:4] ** 2)),
            table=-0.5 * float(tcp[2] < 0.0015),  # gripper pressing into the table
            time=-0.01)
        routed = bool(np.all(wind_frac >= SUCCESS_WIND_FRAC))
        success = bool(routed and shape < SUCCESS_SHAPE and grasped)
        stage = 0 if not grasped else (1 + int(np.sum(wind_frac >= SUCCESS_WIND_FRAC)))
        self.max_stage = max(self.max_stage, stage)
        terms["success"] = 20.0 if success else 0.0

        # Simulator failure: no potential difference into, and no area penalty on, the invalid state.
        finite = bool(np.all(np.isfinite(nodes)) and np.all(np.isfinite(tcp)) and np.isfinite(phi)
                      and np.all(np.isfinite(area_dev)))
        bad = (not finite) or diverged or bool(area_dev.max() > AREA_BAD)
        if bad:
            success = False
            terms.update(shaping=0.0, area=0.0, success=0.0)
        terms = {key: (float(value) if np.isfinite(value) else 0.0) for key, value in terms.items()}
        terms["bad"] = -10.0 if bad else 0.0
        reward = float(sum(terms.values()))
        if np.isfinite(phi) and not bad:
            self.phi = phi
        self.prev_action = act
        terminated = success
        fold = np.asarray(surf["fold_max"], dtype=np.float64)
        fold = fold[np.isfinite(fold)]
        truncated = bool(bad or self.elapsed_steps >= self.horizon)
        # A bad step returns the LAST VALID observation: the truncation bootstraps from it (module docstring).
        obs = self._last_obs.copy() if bad else self._make_obs(nodes, samples, surf, wind, tcp)
        info = dict(is_success=success, start_mode=self.start_mode, routed=routed, shape=shape, grasped=grasped,
                    stage=stage, max_stage=self.max_stage, wind_frac=wind_frac,
                    area_strain_max=float(area_dev.max()), max_strain=float(np.max(surf["area_ratio"] - 1.0)),
                    curvature_match=curv_match, phi=float(phi), phi_kappa=-W_KAPPA * L_KAPPA * curv_match,
                    n_invalid=int(np.count_nonzero(~np.asarray(surf["valid"], bool))),
                    fold_max_deg=float(np.degrees(np.max(fold, initial=0.0))),
                    sim_diverged=bool(diverged), dropped=int(dropped), t=self.elapsed_steps,
                    pad_force=(force_left, force_right), wp=self.wp_i, r_terms=terms)
        return obs, reward, terminated, truncated, info


def make_env(rank=0, seed=0, eval=False, **kwargs):
    """Environment factory for ``rl/async_ppo.py`` and the verification scripts.

    Parameters
    ----------
    rank : int
        Worker rank, stored as ``env.rank``.
    seed : int
        Seed of the environment's random number generator and action space.
    eval : bool
        Evaluation mode: "full" starts unless a reset explicitly asks for another start mode.
    **kwargs
        Further ``SurfaceRouteEnv`` arguments.

    Returns
    -------
    SurfaceRouteEnv
        The environment, already reset once.
    """
    env = SurfaceRouteEnv(eval_mode=eval, seed=int(seed), **kwargs)
    env.action_space.seed(int(seed))
    env.rank = rank
    return env
