"""GenericSurfaceEnv: the final, object-agnostic rope-routing environment (notes/B_rl_environment.md).

The goal and the reward are expressed ONLY through the deformable object's boundary surface and generic obstacle
geometry. Nothing names a point of the object, a peg, a grasp node or a gripper path:

goal
    A TARGET SURFACE of the object (here the tube surface of the rope lying along the routing path), given as a fixed
    farthest-point sample of its vertices with their target Gaussian curvature (second fundamental form). Any object
    whose boundary is a triangle mesh (cloth, soft body, rope tube) is specified this way.
shape term
    CORRESPONDENCE-FREE surface coverage (Berenson 2013): the mean, over the target-surface vertices, of the distance to
    the nearest vertex of the CURRENT surface, plus the curvature mismatch |r^2 K_target - r^2 K_current| at that
    nearest vertex. There are no point labels: the object may cover the target with any of its parts.
wrap term
    For every obstacle in the scene (a list of vertical cylinders, unlabelled), the angle the object winds around it
    divided by the angle the TARGET surface winds around it; obstacles the target does not wrap are ignored. This is
    generic winding geometry, not "peg B".
stretch
    The surface-area stretch penalty (first fundamental form), as in ``SurfaceRouteEnv``.
reach
    Before the object is held: distance from the gripper's pinch point to the nearest point of the object's FAR REGION
    (the generic grasp rule below), plus a hover/descend height blend.
hold
    +GRASP_BONUS while the object is held (generic "object in hand").

Removed from ``SurfaceRouteEnv``'s observation and reward: the per-sample target positions and masks, the per-sample
target curvature, the hand-picked grasp node and its yaw, the 15 hand-placed gripper waypoints and the waypoint
progress, and the per-peg winding targets of the task file (target wraps are computed from the target surface).

Kept: the robot state, the fundamental-form deformation descriptor of the CURRENT object (per material sample:
position, velocity, area stretch, mean, Gaussian and principal curvature, curvature vector, validity), the potential
phi, the progress shaping 10 (phi' - phi), the stretch penalty, the bad-step handling, the release debounce and the
start-state curriculum. For the "hover" and "grasped" start states the grasp point is a RANDOM material node, re-drawn
every episode at least 25% of the object's length from its pinned end, not a chosen one.

Success: the object is held, every obstacle that the target wraps is wrapped by at least 85% of the target's angle,
and the target surface is covered within COVER_SUCCESS (mean distance).

Observation (602 entries; ranges in ``self.obs_index``): the robot block and the deformation block of
``SurfaceRouteEnv`` ([0, 392)), the potential (392), the nearest far-region point relative to the TCP (3), the
N_TARGET target-surface samples relative to the TCP with their r^2 K* (4 N_TARGET), the coverage and curvature
mismatch (2), each obstacle's position relative to the TCP (2 per obstacle), and each obstacle's wrap fraction with the
"target wraps it" flag (2 per obstacle).

Positions are in metres in the robot base frame, angles in radians, curvatures in 1/m and 1/m^2.

Role in the system: the environment of the final policy (``data/runs/generic_ppo4/best.pt``), created through
:func:`make_env` by ``rl/async_ppo.py`` and the scripts. It subclasses ``SurfaceRouteEnv`` for the robot interface, the
stretch penalty, the bad-step handling, the release debounce and the start-state curriculum.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..task.geometry import winding_angles
from .grasp_route_env import GRASP_BONUS, SUCCESS_WIND_FRAC
from .start_states import setup_roam
from .surface_route_env import (L_KAPPA, OBS_LIM, PHI_OBS_LIM, PHI_OBS_SCALE, R2K_MIN, THETA_CLOSED, DEF_FEATURES,
                                HOVER_Z, SurfaceRouteEnv)

N_TARGET = 48  # farthest-point samples of the target surface shown to the policy
W_COVER, W_CURV, W_WRAP = 10.0, 0.3, 2.0  # potential weights: coverage [1/m], curvature mismatch, wrap per obstacle
COVER_SUCCESS = 0.025  # mean coverage distance for success [m] (successes of best_2M scored 0.009-0.020; notes B)
GRASP_MIN_FRAC = 0.25  # random start-state grasp nodes: at least this fraction of the length from the pinned end
# Generic grasp rule (notes B): before grasping, the reach term pulls toward the nearest point of the
# FAR REGION of the object, the surface points whose rest geodesic distance from the object's fixed (pinned) material
# is in the top GRASP_FAR fraction. It needs only the rest surface mesh and which material is fixed: no peg, target or
# point label.
GRASP_FAR = 0.15  # 0.30 let the policy grasp at ~0.6 of the length, short of the ~0.75 the route needs (5M diagnosis)


def farthest_point_sample(points, count, start=0):
    """Indices of a farthest-point sample: an even, object-agnostic sampling of a surface's vertices.

    Starting from ``start``, each next point is the one farthest from all points chosen so far.

    Parameters
    ----------
    points : np.ndarray, shape (n, 3)
        Points to sample from [m].
    count : int
        Number of points to choose.
    start : int
        Index of the first point.

    Returns
    -------
    np.ndarray of int, shape (count,)
        Indices of the chosen points, in the order they were chosen.

    Examples
    --------
    >>> farthest_point_sample(np.c_[np.arange(11.0), np.zeros(11), np.zeros(11)], 3).tolist()
    [0, 10, 5]
    """
    chosen = [int(start)]
    distance = np.linalg.norm(points - points[start], axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.linalg.norm(points - points[index], axis=1))
    return np.array(chosen)


class GenericSurfaceEnv(SurfaceRouteEnv):
    """Object-agnostic surface environment: a target surface, correspondence-free coverage, generic wraps and grasp rule.

    Parameters
    ----------
    *args, **kwargs
        ``SurfaceRouteEnv`` arguments (``num_samples``, ``horizon``, ``substeps``, ``scene_cfg``, ``start_probs``,
        ``eval_mode``, ``seed``, ``shaping_gamma``).
    roam_prob : float
        Probability that a training "full" start begins from a random gripper position instead of home
        (``start_states.setup_roam``), so the policy experiences grasping everywhere on the object and learns WHERE to
        grasp from the outcomes. Never used in evaluation mode, which always starts from home (notes B).
    reward_terms : str
        "full" (default): the reward of notes B (grasp bonus, wraps, coverage, curvature, reach with the grasp rule).
        "surface": an ablation whose task reward uses only the surface terms. The potential is
        phi = -W_COVER coverage - W_CURV curv_mismatch, without the grasp bonus, the wrap term or the reach term;
        success is coverage below COVER_SUCCESS on a valid step, whether or not the rope is held or the obstacles are
        wrapped; and the "nearest_surface" observation block holds the nearest point of the object's surface instead
        of the nearest point of the far region. The stretch penalty (also a surface term) and the small action,
        smoothness, table and time costs of ``SurfaceRouteEnv`` are unchanged.

    Attributes
    ----------
    target_nodes : np.ndarray, shape (N, 3)
        Rope nodes of the target configuration: laid along the route from the pinned end, the remainder continuing
        straight [m].
    target_V : np.ndarray, shape (n_target, 3)
        Vertices of the target surface (the part of the target tube laid on the route) [m].
    target_K : np.ndarray, shape (n_target,)
        Target Gaussian curvature at those vertices [1/m^2].
    target_pts, target_pts_K : np.ndarray, shapes (N_TARGET, 3) and (N_TARGET,)
        Farthest-point sample of the target surface shown to the policy, and its Gaussian curvature.
    obstacles : np.ndarray, shape (n_obstacles, 3)
        Obstacle (peg) centres at z = 0, unlabelled [m].
    target_wrap : np.ndarray, shape (n_obstacles,)
        Absolute winding angle of the target around each obstacle [rad].
    roam_prob : float
        Probability of a roam start in training.

    Examples
    --------
    >>> env = make_env(rank=0, seed=0)  # doctest: +SKIP
    >>> env.observation_space.shape, env.target_pts.shape, env._wrap_mask.tolist()  # doctest: +SKIP
    ((602,), (48, 3), [True, True, True])
    >>> np.degrees(env.target_wrap).round(1).tolist()  # winding of the target around pegs A, B, C  # doctest: +SKIP
    [233.0, 354.3, 345.0]
    >>> env.coverage(env._surface(env.target_nodes))  # the target covers itself exactly  # doctest: +SKIP
    (0.0, 0.0)
    """

    def __init__(self, *args, roam_prob: float = 0.0, reward_terms: str = "full", **kwargs):
        if reward_terms not in ("full", "surface"):
            raise ValueError(f"unknown reward_terms {reward_terms!r}")
        self._target_ready = False
        self.roam_prob = float(roam_prob)
        self.reward_terms = reward_terms
        super().__init__(*args, **kwargs)  # obs_index and observation_space use this class's _obs_blocks/_make_obs

    # ================================================================ target surface (built once)
    def _build_target(self):
        """Build the target surface, the obstacle list with the target's wraps, and the far region.

        The target object configuration is the rope laid along the routing path from its pinned end, with the remainder
        continuing straight on. Only its SURFACE (vertices and Gaussian curvature) is used afterwards.
        """
        sim, radius = self.sim, self.sim.rope_radius
        route = np.c_[self.task.route, np.full(len(self.task.route), radius)]
        route_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        node_arc = np.arange(sim.num_nodes) * sim.seg
        end_dir = route[-1] - route[-2]
        end_dir /= np.linalg.norm(end_dir)
        nodes = np.array([np.r_[[np.interp(arc, route_arc, route[:, axis]) for axis in range(3)]]
                          if arc <= route_arc[-1] else route[-1] + (arc - route_arc[-1]) * end_dir for arc in node_arc])
        self.desc(nodes)  # the descriptor's own tube mesh of the target, so target and current K share one estimator
        vertices, gauss_curv = self.desc.last_V, np.asarray(self.desc.last_out["K_v"])
        # The target surface is the part laid on the route; the remainder is free (coverage is one-sided).
        _, nearest = cKDTree(nodes).query(vertices)
        keep = node_arc[nearest] <= route_arc[-1] + 0.5 * sim.seg
        self.target_nodes, self.target_V, self.target_K = nodes, vertices[keep].copy(), gauss_curv[keep].copy()
        target_vertices, target_curv = self.target_V, self.target_K
        chosen = farthest_point_sample(target_vertices, N_TARGET)
        self.target_pts, self.target_pts_K = target_vertices[chosen], target_curv[chosen]
        # Obstacles are unlabelled vertical cylinders; the target wrap of each is the TARGET object's winding around it.
        self.obstacles = np.c_[self.task.pegs, np.zeros(len(self.task.pegs))]
        on_target = node_arc <= route_arc[-1] + 0.5 * sim.seg  # the routed part (the rest of the object is free)
        target_wrap = np.abs(winding_angles(nodes[on_target], self.obstacles, self.task.peg_height))
        self.target_wrap = target_wrap
        self._wrap_mask = target_wrap > np.deg2rad(30.0)  # obstacles that the target actually wraps
        self._far_idx = self._far_region()
        self._target_ready = True

    def _far_region(self):
        """Surface vertices in the top GRASP_FAR fraction of rest geodesic distance from the fixed material.

        The geodesic distance is computed with Dijkstra on the rest mesh edges, from the vertices of the pinned node
        (node 0's ring, held by the anchor constraint).

        Returns
        -------
        np.ndarray of int
            Indices of the far-region vertices.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        faces, rest_vertices = self.desc.faces, self.desc.forms.V_rest
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
                          axis=0)
        lengths = np.linalg.norm(rest_vertices[edges[:, 0]] - rest_vertices[edges[:, 1]], axis=1)
        graph = coo_matrix((np.r_[lengths, lengths],
                            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(rest_vertices), len(rest_vertices)))
        fixed = np.flatnonzero(self.desc.ring_of_vertex == 0)  # pinned material (anchor constraint on node 0)
        distance = dijkstra(graph.tocsr(), indices=fixed, min_only=True)
        return np.flatnonzero(distance >= (1.0 - GRASP_FAR) * distance.max())

    def _far_point(self, surf, point):
        """Far-region surface point nearest to ``point`` in the current configuration [m]."""
        far_vertices = surf["V"][self._far_idx]
        return far_vertices[int(np.argmin(np.linalg.norm(far_vertices - point, axis=1)))]

    # ================================================================ measurement
    def _surface(self, nodes):
        """``SurfaceRouteEnv._surface`` plus the current surface's vertices ("V"), their Gaussian curvature ("K_v")
        and a k-d tree of the vertices ("tree"), added once per state."""
        new = self._surf_nodes is None or not np.array_equal(self._surf_nodes, nodes)
        surf = super()._surface(nodes)
        if new:
            surf["V"], surf["K_v"] = self.desc.last_V.copy(), np.asarray(self.desc.last_out["K_v"]).copy()
            surf["tree"] = cKDTree(surf["V"])
        return surf

    def coverage(self, surf):
        """Correspondence-free coverage of the target surface by the current surface.

        For every target-surface vertex, the nearest current-surface vertex gives a distance and a curvature error
        |r^2 K* - r^2 K| (clipped to 10). Memoised in the surface dict (one per state), so ``step``, the potential and
        the observation share one query.

        Returns
        -------
        coverage : float
            Mean distance [m].
        curv_mismatch : float
            Mean clipped curvature error (dimensionless).
        """
        if "cov" not in surf:
            radius_sq = self.sim.rope_radius ** 2
            distance, nearest = surf["tree"].query(self.target_V)
            curv_err = np.abs(radius_sq * self.target_K - radius_sq * surf["K_v"][nearest])
            surf["cov"] = (float(distance.mean()), float(np.clip(curv_err, 0, 10).mean()))
        return surf["cov"]

    def wrap_frac(self, nodes):
        """Wrap fraction per obstacle: |winding of the object| / |winding of the target|, clipped to [0, 1]; 1 for
        obstacles that the target does not wrap.

        Parameters
        ----------
        nodes : np.ndarray, shape (N, 3)
            Rope node positions [m].

        Returns
        -------
        np.ndarray, shape (n_obstacles,)
            Wrap fractions.
        """
        winding = np.abs(winding_angles(nodes, self.obstacles, self.task.peg_height))
        return np.where(self._wrap_mask, np.clip(winding / np.maximum(self.target_wrap, 1e-9), 0.0, 1.0), 1.0)

    def _measure(self):
        """Measure the current state: (rope nodes, sample positions, surface measures, wrap fractions, TCP)."""
        if not self._target_ready:
            self._build_target()
        nodes = self.sim.rope_nodes()
        samples = self.graph.samples(nodes)
        return nodes, samples, self._surface(nodes), self.wrap_frac(nodes), self.sim.tcp()

    # ================================================================ start-state grasp point (random, generic)
    def _grasp_point(self, nodes):
        """Grasp node of the "hover" and "grasped" start states: a random node at least GRASP_MIN_FRAC of the length
        from the pinned end, drawn once per episode."""
        if getattr(self, "_gi", None) is None:
            lowest = int(np.ceil(GRASP_MIN_FRAC * (self.sim.num_nodes - 1)))
            self._gi = int(self.np_random.integers(lowest, self.sim.num_nodes))
        return self._gi

    # ================================================================ potential
    def curvature_match(self, nodes):
        """Curvature mismatch of the coverage (see :meth:`coverage`)."""
        return self.coverage(self._surface(nodes))[1]

    def _potential(self, tcp, nodes, samples, wind_frac, grasped):
        """Potential phi of the state.

        holding:      phi = GRASP_BONUS + common
        not holding:  phi = -3 reach + common

        with common = W_WRAP sum(wrap fractions of the wrapped obstacles) - W_COVER coverage - W_CURV curv_mismatch,
        and reach the distance to the far region with the hover/descend height blend of ``SurfaceRouteEnv``.

        Returns
        -------
        phi : float
            The potential.
        coverage : float
            Coverage distance [m]; the parent's ``step`` uses it as "shape" for success and info.
        wind_frac : np.ndarray, shape (n_obstacles,)
            Wrap fraction per obstacle.
        """
        surf = self._surface(nodes)
        cover, curv_err = self.coverage(surf)
        wrap = np.asarray(wind_frac, dtype=np.float64)
        if self.reward_terms == "surface":  # ablation: surface terms only (no grasp bonus, wraps or reach)
            return -W_COVER * cover - W_CURV * curv_err, cover, wrap
        common = W_WRAP * float(wrap[self._wrap_mask].sum()) - W_COVER * cover - W_CURV * curv_err
        if grasped:
            phi = GRASP_BONUS + common
        else:
            d_xy = float(np.linalg.norm(tcp[:2] - self._far_point(surf, tcp)[:2]))  # generic grasp rule
            blend = self.reach_blend(d_xy)
            z_goal = blend * self.z_rope + (1.0 - blend) * HOVER_Z
            reach = d_xy + abs(tcp[2] - z_goal) * (0.5 + 0.5 * blend)
            phi = -3.0 * reach + common
        # The parent's step uses "shape" for success and info: here it is the coverage distance.
        return phi, cover, wrap

    # ================================================================ observation
    def _obs_blocks(self):
        """(name, size) of the observation blocks, in order."""
        num_features = len(DEF_FEATURES)
        num_obstacles = len(self.task.pegs)
        return [("q", 7), ("qd", 7), ("tcp", 3), ("yaw", 2), ("target_offset", 3), ("gripper", 3), ("pad_force", 2),
                ("prev_action", 5), ("time", 1), ("deform", self.num_samples * num_features), ("deform_global", 2),
                ("potential", 1), ("nearest_surface", 3), ("target_surface", 4 * N_TARGET), ("coverage", 2),
                ("obstacles", 2 * num_obstacles), ("wrap", 2 * num_obstacles)]

    def _make_obs(self, nodes, samples, surf, wind_frac, tcp):
        """Assemble the float32 observation (layout in the module docstring) and store it in ``_last_obs``."""
        sim, radius = self.sim, self.sim.rope_radius
        joint_pos, joint_vel = sim.joint_state()
        yaw = sim.tool_yaw()
        grasped, force_left, force_right, _ = self._grasp_state()
        velocity = (samples - self.prev_x) / (self.substeps * sim.dt)
        # Per-sample deformation features (DEF_FEATURES), scaled to order 1, as in SurfaceRouteEnv.
        deform = np.column_stack([samples - tcp, 0.1 * velocity, 10.0 * (surf["area_ratio"] - 1.0),
                                  10.0 * radius * (surf["H"] - surf["H_rest"]),
                                  radius * radius * surf["K_mean"], radius * radius * surf["K_abs"],
                                  np.maximum(radius * radius * surf["K_min"], R2K_MIN), radius * radius * surf["K_max"],
                                  L_KAPPA * surf["curvature_vector"], L_KAPPA * surf["signed_planar_curvature"],
                                  np.asarray(surf["valid"], dtype=np.float64)])
        if self.reward_terms == "surface":  # ablation: the nearest point of the whole surface, no grasp-rule hint
            far_point = surf["V"][surf["tree"].query(tcp)[1]]
        else:  # "nearest_surface" block: the nearest point of the far region
            far_point = self._far_point(surf, tcp)
        cover, curv_err = self.coverage(surf)
        wrap = np.asarray(wind_frac, dtype=np.float64)
        obs = np.concatenate([
            joint_pos[:7], 0.1 * joint_vel[:7], tcp, [np.sin(2 * yaw), np.cos(2 * yaw)], sim.target_pos - tcp,
            [sim.theta_cmd / THETA_CLOSED, sim.pad_gap() / 0.085, float(grasped)],
            [min(force_left, 30.0) / 10.0, min(force_right, 30.0) / 10.0],
            self.prev_action, [self.elapsed_steps / self.horizon],
            deform.ravel(),
            [10.0 * (surf["area_ratio_total"] - 1.0), surf["willmore_smooth"] / self._willmore_rest - 1.0],
            [np.clip(PHI_OBS_SCALE * self.phi, -PHI_OBS_LIM, PHI_OBS_LIM)],
            far_point - tcp,
            np.c_[self.target_pts - tcp, np.clip(radius * radius * self.target_pts_K, R2K_MIN, 10.0)].ravel(),
            [10.0 * cover, curv_err],
            (self.obstacles - tcp)[:, :2].ravel(), np.c_[wrap, self._wrap_mask.astype(float)].ravel()])
        obs = np.nan_to_num(obs, nan=0.0, posinf=OBS_LIM, neginf=-OBS_LIM)
        self._last_obs = np.clip(obs, -OBS_LIM, OBS_LIM).astype(np.float32)
        return self._last_obs

    def reset(self, seed=None, options=None):
        """``SurfaceRouteEnv.reset``, followed in training "full" starts by a roam start with probability
        ``roam_prob``; ``info["roam"]`` then records whether the roam setup succeeded."""
        obs, info = super().reset(seed=seed, options=options)
        opts = options or {}
        if (self.roam_prob > 0.0 and self.start_mode == "full" and not self.eval_mode and "start_mode" not in opts
                and self.np_random.random() < self.roam_prob):
            ok, start_info = setup_roam(self, self.np_random)
            if ok:
                obs = start_info["obs"]
                self._release_votes = 0
            else:  # back to the untouched layout: a plain "full" start from home
                self.sim.reset(self._layout)
                obs = self._begin_episode()
            info.update(roam=bool(ok), phi=float(self.phi))
        return obs, info

    def step(self, action):
        """``SurfaceRouteEnv.step`` with this environment's success test: held, every wrapped obstacle at
        SUCCESS_WIND_FRAC or more, coverage below COVER_SUCCESS, and not a bad step.

        Adds "coverage" and "curv_mismatch" to the info dict and re-judges "is_success", "routed" and the success
        bonus in ``info["r_terms"]``.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        cover, curv_err = self.coverage(self._surface(self.sim.rope_nodes()))
        wrap = np.asarray(info["wind_frac"])
        routed = bool(np.all(wrap[self._wrap_mask] >= SUCCESS_WIND_FRAC))
        success_full = bool(routed and cover < COVER_SUCCESS and info["grasped"] and info["r_terms"]["bad"] == 0.0)
        if self.reward_terms == "surface":  # ablation: the surface alone decides success
            success = bool(cover < COVER_SUCCESS and info["r_terms"]["bad"] == 0.0)
        else:
            success = success_full
        # The parent judged success with its own threshold on "shape" (= coverage here): re-judge with this one's.
        bonus = 20.0 if success else 0.0
        reward += bonus - info["r_terms"]["success"]
        info["r_terms"]["success"] = bonus
        terminated = success
        info.update(is_success=success, routed=routed, coverage=cover, curv_mismatch=curv_err,
                    success_full=success_full)
        return obs, reward, terminated, truncated, info

    def release_and_hold(self, steps=15):
        """Open the gripper, move up and away, and check that the rope stays wrapped (no policy involved).

        Parameters
        ----------
        steps : int
            Control steps; the TCP target rises by 6 mm per step.

        Returns
        -------
        dict
            "held" (every wrapped obstacle still at SUCCESS_WIND_FRAC or more and coverage below 1.5 COVER_SUCCESS),
            "shape" (coverage [m]) and "wind_frac".
        """
        tcp = self.sim.tcp()
        self.sim.theta_target = 0.0
        for step_index in range(steps):
            self.sim.target_pos = tcp + np.array([0.0, 0.0, 0.006 * (step_index + 1)])
            for _ in range(self.substeps):
                self.sim.substep()
        nodes = self.sim.rope_nodes()
        cover, _ = self.coverage(self._surface(nodes))
        wrap = self.wrap_frac(nodes)
        held = bool(np.all(wrap[self._wrap_mask] >= SUCCESS_WIND_FRAC) and cover < 1.5 * COVER_SUCCESS)
        return dict(held=held, shape=cover, wind_frac=wrap)


def make_env(rank=0, seed=0, eval=False, **kwargs):
    """Environment factory for ``rl/async_ppo.py`` and the scripts ("rope_routing.env.generic_surface_env:make_env").

    Parameters
    ----------
    rank : int
        Worker rank, stored as ``env.rank``.
    seed : int
        Seed of the environment's random number generator and action space.
    eval : bool
        Evaluation mode: "full" starts from home unless a reset explicitly asks for another start mode.
    **kwargs
        Further ``GenericSurfaceEnv`` arguments, e.g. ``roam_prob`` or ``scene_cfg``.

    Returns
    -------
    GenericSurfaceEnv
        The environment, already reset once.
    """
    env = GenericSurfaceEnv(eval_mode=eval, seed=int(seed), **kwargs)
    env.action_space.seed(int(seed))
    env.rank = rank
    return env
