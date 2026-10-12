"""SuperDex scene for rope routing: an FR3 arm with a Robotiq 2F-85 gripper, an anchored soft rope, three pegs and a
table.

``GraspScene`` builds the physics world once and resets it every episode; every environment (``GraspRouteEnv``,
``SurfaceRouteEnv`` and the final ``GenericSurfaceEnv``) drives the robot through it. ``SceneCfg()`` holds the default
configuration, the "C1-nc" scene of notes/C_physics_and_infrastructure.md (section 1), which reproduces the rope of
SuperDex's "Rope Braid" demo. ``SceneCfg.legacy_v3()`` and ``SceneCfg.tube_t0()`` are alternative configurations kept
for tests and comparisons.

Units are SI (m, s, kg, N, rad). Positions are in the world frame, which is the robot base frame (x forward, y left,
z up; the table surface is z = 0). The tool centre point (TCP) is the pinch point near the finger tips,
``SceneCfg.tcp_offset`` below the fr3_link8 flange along its z axis.

Physics choices (notes/C_physics_and_infrastructure.md):

* Rope: a discrete elastic rod (stretch, bending and twist) with soft-rope parameters and damping. Contact uses the
  centreline with the rope radius as contact padding on every collider (table, pegs, gripper links), plus a
  POINT_CLOUD collider of the same radius for rigid bodies that hit the rope. The result is a round rope of 5 mm
  radius at about 1000 steps/s; contact on the tube mesh was 13 times slower. For self-contact the point cloud (3
  samples per segment) collides with the rope's own centreline padded by r - 0.5 mm, so strands touch at 2r and never
  pass through each other. A friction fall-off velocity of 1e-3 m/s with 8 Newton iterations reproduces Coulomb
  friction on wraps (capstan slip at an effective coefficient of 0.60 for a nominal 0.55); 1e-4 m/s is numerically
  sticky.
* Reset: frame-consistent placement (lift by translation, morph in the air with node position constraints, lower by
  translation; see :func:`frame_consistent_place`). A boundary-condition jump into a curved layout skips the parallel
  transport of the rod's element frames, which then remain those of the straight rope and corrupt the bending for the
  whole episode (hops, single-node kinks, arm stalls, solves twice as slow).
* Solver: the sparse direct LDLT linear solver (the default iterative solver did not converge on the soft rope) with
  the ARMIJO_OR_RESIDUAL_NORM line search (no sticky sliding, cheaper solves).
* Arm: the FR3 under SuperDex's BASIC_OSC_PD operational-space controller on the TCP, plus a null-space
  joint-posture term. Link gravity is disabled, which models the real FR3's internal gravity compensation.
* Gripper: the two 2F-85 knuckles follow a rate-limited closure command (close to the real gripper's speed) through
  an implicit PD solved inside the physics step (default) or BASIC_JSC_PD torques; the passive four-bar joints follow
  through the closed kinematic loops.
* Grasp: the fingers physically close on the rope. When both pads press on the rope with at least
  ``SceneCfg.grasp_min_force``, the contact is upgraded to a no-slip grasp: a node-to-rigid constraint to the gripper
  base at the current relative position, coupled in both directions. It is released when the fingers open.
"""

from dataclasses import dataclass

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset

from ..task.geometry import TriangleTask

# Robot asset and DOF layout of the articulated FR3 + 2F-85 actor: 7 arm joints followed by 6 gripper joints.
BOT_PATH = "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot"
ARM_DOFS = np.arange(7)
GRIP_DOFS = np.arange(7, 13)
KNUCKLE_DOFS = np.array([8, 11])  # the two actuated knuckle joints
GRIP_SIGN = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])  # gripper DOF values = GRIP_SIGN * theta (knuckle closure)
THETA_OPEN, THETA_CLOSED = 0.0, 0.8  # knuckle closure of the open and the closed gripper [rad]
GRASP_MAX_FORCE = 60.0  # friction-limited holding force of the closed 2F-85 on the rope [N]
PAD_L = np.array([-0.0253, -0.0008, 0.0283])  # left pad centre in the left finger-tip link frame [m]
PAD_R = np.array([0.0253, 0.0001, 0.0283])  # right pad centre in the right finger-tip link frame [m]
HOME = np.array([0.0, -0.35, 0.0, -2.35, 0.0, 2.0, np.pi / 4])  # home configuration of the arm joints [rad]
DELTA = 0.0005  # penalty activation distance beyond each collider's padding [m]

_initialized = False


def _init():
    """Initialise the SuperDex physics runtime once per process, with Mochi single-threaded (no worker threads)."""
    global _initialized
    if not _initialized:
        physics.initialize(num_worker_threads=0)
        _initialized = True


def tf(translation=(0, 0, 0), rotation=None):
    """Build a SuperDex rigid transform.

    Parameters
    ----------
    translation : sequence of float, length 3
        Translation [m]. SuperDex stores it in single precision.
    rotation : physics.Quaternion, optional
        Rotation; the identity if omitted.

    Returns
    -------
    physics.TransformRT
        The transform.

    Examples
    --------
    >>> np.asarray(tf([0.0, 0.0, 0.16]).translation).tolist()
    [0.0, 0.0, 0.1599999964237213]
    """
    transform = physics.TransformRT()
    transform.translation = [float(value) for value in translation]
    if rotation is not None:
        transform.rotation = rotation
    return transform


def quat_down(yaw):
    """Orientation of fr3_link8 with the tool pointing straight down and the finger closing axis (link x) along the
    world direction (cos yaw, sin yaw).

    Parameters
    ----------
    yaw : float
        Yaw of the closing axis about world z [rad].

    Returns
    -------
    physics.Quaternion
        The rotation Rz(yaw) Rx(pi).

    Examples
    --------
    >>> (rot(quat_down(0.0)).round(6) + 0.0).tolist()
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    """
    return physics.Quaternion.rotation_z(float(yaw)) * physics.Quaternion.rotation_x(np.pi)


def rot(quaternion) -> np.ndarray:
    """Rotation matrix of a unit quaternion stored in (x, y, z, w) order.

    Parameters
    ----------
    quaternion : array_like, shape (4,)
        Unit quaternion (x, y, z, w), for example a SuperDex ``Quaternion``.

    Returns
    -------
    np.ndarray, shape (3, 3)
        The rotation matrix.

    Examples
    --------
    A rotation by 90 degrees about z:

    >>> rot([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]).round(6).tolist()
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    """
    qx, qy, qz, qw = [float(value) for value in np.asarray(quaternion)]
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def cylinder_mesh(radius, height, segments=24):
    """Closed triangle mesh of a vertical cylinder standing on z = 0 (the peg shape).
    SuperDex has no built-in cylinder shape for collisions.

    Parameters
    ----------
    radius, height : float
        Cylinder radius and height [m].
    segments : int
        Number of sides of the polygonal cross-section.

    Returns
    -------
    vertices : np.ndarray of float32, shape (2 * segments + 2, 3)
        Bottom ring, top ring, then the bottom and top centres [m].
    faces : np.ndarray of int32, shape (4 * segments, 3)
        Triangles with outward normals: two per side, and one bottom and one top fan triangle per side.

    Examples
    --------
    >>> vertices, faces = cylinder_mesh(0.015, 0.07)
    >>> vertices.shape, faces.shape
    ((50, 3), (96, 3))
    """
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], 1)
    vertices = np.concatenate([np.c_[ring, np.zeros(segments)], np.c_[ring, np.full(segments, height)],
                               [[0, 0, 0], [0, 0, height]]])
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces += [[index, following, segments + following], [index, segments + following, segments + index],
                  [2 * segments, following, index], [2 * segments + 1, segments + index, segments + following]]
    return vertices.astype(np.float32), np.array(faces, dtype=np.int32)


@dataclass
class SceneCfg:
    """Physics configuration of the scene. The defaults are the "C1-nc" scene of notes C (section 1).

    The field names appear in run configurations (``scene_cfg``) and checkpoints and are part of the interface.

    Attributes
    ----------
    dt : float
        Physics substep [s].
    rope_radius : float
        Rope radius [m]; the defaults model a braided cotton rope about 10 mm thick.
    rope_nodes : int
        Number of rod nodes.
    rope_tail : float
        Rope length beyond the target route [m].
    rope_linear_density : float
        Mass per length [kg/m].
    rope_axial_stiffness : float
        Axial stiffness EA [N].
    rope_bending_stiffness : float
        Bending stiffness EI [N m^2].
    rope_torsional_stiffness : float
        Torsional stiffness GJ [N m^2].
    rope_mass_damping, rope_stiffness_damping : float
        Rayleigh damping coefficients, proportional to mass [1/s] and to stiffness [s].
    rope_friction, table_friction, peg_friction, pad_friction : float
        Coulomb friction coefficients of the rope, the table, the pegs and the finger pads.
    peg_height : float
        Peg height [m].
    tcp_offset : float
        Distance of the TCP (the pinch point near the finger tips) below fr3_link8 [m].
    osc_gains : tuple of float
        Gains (kp_p, kd_p, kp_r, kd_r) of BASIC_OSC_PD: position and rotation stiffness and damping.
    tcp_ki : float
        Integral gain on the TCP position error [1/s]; it cancels the steady-state error left by joint friction.
    tcp_i_max : float
        Bound of the integral correction per axis [m].
    posture_kp, posture_kd : float
        Stiffness [N m/rad] and damping [N m s/rad] of the null-space joint-posture term.
    grip_kp, grip_kd : float
        Knuckle stiffness [N m/rad] and damping [N m s/rad].
    grip_sat : float
        Maximum knuckle torque [N m] (about 30 N of pad force).
    grip_speed : float
        Rate limit of the knuckle closure command [rad/s] (about 150 mm/s at the pads).
    grasp_min_force : float
        Contact force each pad must reach to count as a grasp [N].
    grasp_mode : str
        "attach" (contact-gated no-slip constraint) or "friction" (contact friction only).
    grip_controller : str
        "implicit" (PD on the knuckles solved inside the physics step) or "jsc" (BASIC_JSC_PD torques).
    nonlinear_iters : int
        Newton iterations per substep. With 8 the friction law with a fall-off velocity of 1e-3 m/s is followed; 4
        makes wraps sticky (notes C).
    line_search : str
        Name of a ``physics.LineSearchType``; "" keeps the SuperDex default (RESIDUAL_NORM).
    implicit_normal_force : bool
        ``experimental_eval.implicit_normal_force_for_dissipation`` (frictional point-cloud contact).
    joint_coulomb_friction : bool
        Keep the FR3 joint Coulomb friction of the robot prefab. Its regularised stick (1e-3 rad/s) can freeze the
        whole arm for 1-4 s under tens of newtons of command (scripts/test_start_states.py, notes C), which a
        torque-controlled FR3 does not do, and it forces extra Newton iterations. False zeroes the Coulomb term and
        keeps the viscous joint damping.
    contact_penalty : float
        Penalty coefficient of the rope, table, peg and gripper-link contacts.
    rope_pc_radius : float
        Radius of the rope's point-cloud collider as seen by rigid bodies [m]; 0 uses the rope radius.
    rope_contact : str
        "centerline" (the rope radius is carried as collider padding) or "tube" (contact on the tube surface
        vertices).
    rope_self_contact : bool
        Whether the rope's point cloud also collides with the rope itself.
    rope_pc_quad : str
        Point-cloud samples per segment (a ``physics.ActorSegmentElementType`` name); "" places one sample per node.
    rope_pc_exclusion : float
        ``self_contact_exclusion_ratio`` of the point cloud, evaluated on the straight rest shape.
    rope_pad : float or None
        The rope's own padding in rope-rope contact [m]; None uses r - DELTA, so strands touch at 2r.
    rope_smoothing : float
        Penalty smoothing half distance of the rope [m]; the SuperDex default of 5 mm is too soft.
    rope_friction_falloff : float
        Friction fall-off velocity of the rope, also in rope-table and rope-peg contact [m/s]; the default 1e-2
        creeps.
    placement : str
        "frames" (frame-consistent placement, notes C) or "bc" (legacy one-step boundary-condition jump).
    place_steps : int
        Substeps of the morph; 10 more substeps follow to converge.
    place_z : float
        Height of the morph [m]: above the 7 cm pegs and below the gripper at home.
    place_k : float
        Stiffness of the morph's node position constraints [N/m].
    layout_z0 : float
        Height of the placed rope above its radius [m].
    settle : str
        "adaptive" (settle until the rope is calm) or "fixed" (``settle_steps`` substeps).
    settle_steps : int
        Substeps of the fixed settle.
    settle_v, settle_calm, settle_min, settle_max : float, int, int, int
        Adaptive settle: the largest node speed must stay below ``settle_v`` [m/s] for ``settle_calm`` consecutive
        substeps, after at least ``settle_min`` and at most ``settle_max`` substeps.

    Examples
    --------
    >>> SceneCfg().rope_nodes, SceneCfg.legacy_v3().nonlinear_iters, SceneCfg.tube_t0(nonlinear_iters=8).nonlinear_iters
    (61, 4, 8)
    """

    dt: float = 0.01
    # Rope.
    rope_radius: float = 0.005
    rope_nodes: int = 61
    rope_tail: float = 0.20
    rope_linear_density: float = 0.06
    rope_axial_stiffness: float = 5e3
    rope_bending_stiffness: float = 3e-4
    rope_torsional_stiffness: float = 2e-4
    rope_mass_damping: float = 1.0
    rope_stiffness_damping: float = 0.01
    rope_friction: float = 0.6
    table_friction: float = 0.6
    peg_friction: float = 0.5
    pad_friction: float = 2.0
    peg_height: float = 0.07
    # Robot.
    tcp_offset: float = 0.160
    osc_gains: tuple = (900.0, 75.0, 30.0, 3.0)
    tcp_ki: float = 3.0
    tcp_i_max: float = 0.03
    posture_kp: float = 8.0
    posture_kd: float = 1.0
    grip_kp: float = 200.0
    grip_kd: float = 0.1
    grip_sat: float = 2.0
    grip_speed: float = 1.4
    grasp_min_force: float = 1.0
    grasp_mode: str = "attach"
    grip_controller: str = "implicit"
    # Solver.
    nonlinear_iters: int = 8
    line_search: str = "ARMIJO_OR_RESIDUAL_NORM"
    implicit_normal_force: bool = True
    joint_coulomb_friction: bool = False
    # Contact.
    contact_penalty: float = 1e10
    rope_pc_radius: float = 0.0
    rope_contact: str = "centerline"
    rope_self_contact: bool = True
    rope_pc_quad: str = "P1Q3"
    rope_pc_exclusion: float = 1.5
    rope_pad: float | None = None
    rope_smoothing: float = 1e-3
    rope_friction_falloff: float = 1e-3
    # Reset: rope placement and settling.
    placement: str = "frames"
    place_steps: int = 30
    place_z: float = 0.10
    place_k: float = 1e4
    layout_z0: float = 3e-4
    settle: str = "adaptive"
    settle_steps: int = 40
    settle_v: float = 0.002
    settle_calm: int = 10
    settle_min: int = 20
    settle_max: int = 400

    @classmethod
    def legacy_v3(cls, **overrides):
        """The scene before C1-nc ("v3"), kept for comparison.

        Boundary-condition-jump placement (stale rod frames), no self-contact, the default rope contact (fall-off
        1e-2 m/s, 5 mm smoothing), 4 Newton iterations with RESIDUAL_NORM, FR3 joint Coulomb friction and a fixed
        settle of 40 substeps.

        Parameters
        ----------
        **overrides
            Fields to set on top of the legacy configuration.

        Returns
        -------
        SceneCfg
            The configuration.
        """
        return cls(**{**dict(nonlinear_iters=4, line_search="", implicit_normal_force=False,
                             joint_coulomb_friction=True, rope_self_contact=False, rope_pc_quad="", rope_pad=0.0,
                             rope_smoothing=5e-3, rope_friction_falloff=1e-2, placement="bc", settle="fixed"),
                      **overrides})

    @classmethod
    def tube_t0(cls, **overrides):
        """The "T0-nc" scene with contact on the tube surface, for evaluation and videos.

        Tube contact can roll and twist the rope (the best plaits); penalty 1e11, no padding, fall-off 1e-4 m/s with
        4 Newton iterations. It costs about 2.8 times the default scene (notes C, remaining gaps).

        Parameters
        ----------
        **overrides
            Fields to set on top of the tube-contact configuration.

        Returns
        -------
        SceneCfg
            The configuration.
        """
        return cls(**{**dict(rope_contact="tube", contact_penalty=1e11, rope_pad=0.0, rope_friction_falloff=1e-4,
                             nonlinear_iters=4), **overrides})


def collider_pad(cfg: SceneCfg) -> float:
    """Contact padding of the table, the pegs and the gripper links.

    With centreline contact the rope is a curve, so its radius (less the penalty activation distance DELTA) is carried
    by the padding of every collider it touches; tube contact needs no padding.

    Parameters
    ----------
    cfg : SceneCfg
        Scene configuration.

    Returns
    -------
    float
        Padding [m].

    Raises
    ------
    ValueError
        If ``cfg.rope_contact`` is neither "centerline" nor "tube".

    Examples
    --------
    >>> round(collider_pad(SceneCfg()), 6), collider_pad(SceneCfg.tube_t0())
    (0.0045, 0.0)
    """
    if cfg.rope_contact not in ("centerline", "tube"):  # e.g. "centreline" would get neither padding nor tube contact
        raise ValueError(f"unknown rope_contact {cfg.rope_contact!r}")
    return cfg.rope_radius - DELTA if cfg.rope_contact == "centerline" else 0.0


def configure_solver(scene, cfg: SceneCfg):
    """Set the solver parameters of a scene: LDLT linear solver, Newton iterations, line search and the implicit
    normal force for dissipation.

    Parameters
    ----------
    scene : physics.Scene
        The SuperDex scene.
    cfg : SceneCfg
        Scene configuration.
    """
    params = scene.get_solver_params()
    params.linear_solver.solver_type = physics.LinearSolverType.LDLT
    params.non_linear_solver.max_iter = cfg.nonlinear_iters
    if cfg.line_search:
        params.non_linear_solver.line_search_type = getattr(physics.LineSearchType, cfg.line_search)
    params.experimental_eval.implicit_normal_force_for_dissipation = bool(cfg.implicit_normal_force)
    scene.set_solver_params(params)


def static_contact(cfg: SceneCfg, friction: float):
    """Contact parameters of the static colliders (the table and the pegs).

    Parameters
    ----------
    cfg : SceneCfg
        Scene configuration (padding and penalty).
    friction : float
        Coulomb friction coefficient.

    Returns
    -------
    physics.ContactParams
        The contact parameters.
    """
    return physics.ContactParams(coulomb_friction_coefficient=friction, penalty_threshold_default=DELTA,
                                 penalty_threshold_extra_padding=collider_pad(cfg),
                                 penalty_coefficient=cfg.contact_penalty, penalty_smoothing_half_distance=1e-3)


def rope_contact(cfg: SceneCfg):
    """Contact parameters of the rope itself.

    The rope's friction coefficient pairs with that of every collider it touches (geometric mean). Its padding,
    smoothing and penalty act only where the rope is the collider, i.e. in rope-rope contact.

    Parameters
    ----------
    cfg : SceneCfg
        Scene configuration.

    Returns
    -------
    physics.ContactParams
        The contact parameters.
    """
    pad = cfg.rope_radius - DELTA if cfg.rope_pad is None else cfg.rope_pad
    return physics.ContactParams(coulomb_friction_coefficient=cfg.rope_friction,
                                 penalty_coefficient=cfg.contact_penalty, penalty_threshold_default=DELTA,
                                 penalty_threshold_extra_padding=pad,
                                 penalty_smoothing_half_distance=cfg.rope_smoothing,
                                 friction_falloff_vel=cfg.rope_friction_falloff)


def create_rope(scene, cfg: SceneCfg, nodes, axes, name="rope"):
    """Create the rope: a rod actor with the configured material, contact and point-cloud collider.

    Parameters
    ----------
    scene : physics.Scene
        The SuperDex scene.
    cfg : SceneCfg
        Scene configuration.
    nodes : array_like, shape (num_nodes, 3)
        Initial node positions [m]; they are also the rod's rest shape.
    axes : array_like, shape (num_nodes - 1, 3)
        Element frame axis of each segment.
    name : str
        Actor name.

    Returns
    -------
    rod : rod actor
        The rope.
    rest_nodes : np.ndarray, shape (num_nodes, 3)
        Rest node positions as stored by the simulator [m].
    """
    radius = cfg.rope_radius
    model = physics.experimental.generate_tubular_rod_model_data(
        nodes=np.asarray(nodes, np.float32), element_frame_axes=np.asarray(axes, np.float32), radius=radius,
        num_cross_section_segments=8, is_closed_loop=False)
    material = physics.experimental.RodMaterialParams(
        linear_density=cfg.rope_linear_density, # Mass per unit length of the rope.
        linear_rotational_inertia=cfg.rope_linear_density * (radius * radius / 2), # Rotational inertia per unit length of the rope.
        axial_stiffness=cfg.rope_axial_stiffness, torsional_stiffness=cfg.rope_torsional_stiffness, # Axial and torsional stiffness of the rope.
        flexural_stiffness=[cfg.rope_bending_stiffness, cfg.rope_bending_stiffness], # Bending stiffness of the rope.
        mass_damping_coefficient=cfg.rope_mass_damping, stiffness_damping_coefficient=cfg.rope_stiffness_damping) # Mass and stiffness damping of the rope.
    
    # Point cloud instantiation
    collider = physics.experimental.PointCloudColliderParams()
    collider.radius = cfg.rope_pc_radius if cfg.rope_pc_radius > 0 else radius
    collider.self_contact = bool(cfg.rope_self_contact)
    collider.self_contact_exclusion_ratio = cfg.rope_pc_exclusion  # uses REST positions: keep the rest shape straight

    # Point cloud collider configuration.
    if cfg.rope_pc_quad:
        collider.collider_segment_element_type = getattr(physics.ActorSegmentElementType, cfg.rope_pc_quad)
    params = physics.experimental.RodActorParams(name=name, shape=physics.create_model_shape(model),
                                                 world_from_local=physics.TransformRT(), material=material,
                                                 contact=rope_contact(cfg),
                                                 use_visual_mesh_contact=cfg.rope_contact == "tube")
    params.collider_type = physics.ColliderType.POINT_CLOUD
    params.point_cloud_collider = collider
    rod = physics.experimental.create_rod_actor(scene, params)
    return rod, np.array(rod.get_mesh().coordinates, dtype=np.float64).reshape(-1, 3)


def homotopy(xy_from, xy_to, fraction):
    """Inextensible planar interpolation between two polylines with the same segment lengths.

    The first node, the heading of the first segment and every turning angle are interpolated linearly, and the
    polyline is rebuilt from them with the segment lengths of ``xy_from``. Every intermediate shape therefore keeps the
    rope's segment lengths.

    Parameters
    ----------
    xy_from, xy_to : np.ndarray, shape (num_nodes, 2)
        Start and end polylines [m].
    fraction : float
        Interpolation parameter in [0, 1]; 0 gives ``xy_from`` and 1 gives ``xy_to``.

    Returns
    -------
    np.ndarray, shape (num_nodes, 2)
        The interpolated polyline [m].

    Examples
    --------
    Halfway between a straight polyline and one that turns by 90 degrees at its middle node:

    >>> straight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> bent = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    >>> homotopy(straight, bent, 0.5).round(3).tolist()
    [[0.0, 0.0], [1.0, 0.0], [1.707, 0.707]]
    >>> bool(np.allclose(homotopy(straight, bent, 1.0), bent))
    True
    """
    def angles(xy):
        """Unwrapped segment headings [rad] and segment lengths [m] of a polyline."""
        delta = np.diff(xy, axis=0)
        return np.unwrap(np.arctan2(delta[:, 1], delta[:, 0])), np.linalg.norm(delta, axis=1)
    heading_from, lengths = angles(xy_from)
    heading_to, _ = angles(xy_to)
    start_turn = (heading_to[0] - heading_from[0] + np.pi) % (2 * np.pi) - np.pi
    turns_from, turns_to = np.diff(heading_from), np.diff(heading_to)
    heading = (heading_from[0] + fraction * start_turn
               + np.r_[0.0, np.cumsum((1 - fraction) * turns_from + fraction * turns_to)])
    start = (1 - fraction) * xy_from[0] + fraction * xy_to[0]
    return start + np.r_[[[0.0, 0.0]], np.cumsum(lengths[:, None] * np.c_[np.cos(heading), np.sin(heading)], axis=0)]


def frame_consistent_place(scene, rope, rest, anchor_constraint, layout_xy, z_table, steps, z_air, step,
                           morph_stiffness=1e4):
    """Move a rod that lies straight (with valid element frames) onto a planar layout without staling its frames.

    Boundary-condition values are written straight into the solution and bypass the Lie update that
    parallel-transports each element's material frame, so a boundary-condition jump may only TRANSLATE the rope. The
    placement therefore runs in three phases:

    1. Translate the xyz DOFs up to ``z_air`` with one boundary-condition step (twist free).
    2. Morph in the air with one node position constraint per node. The targets follow the inextensible curvature
       homotopy from the straight shape to the layout (each substep turns an element by less than about 0.2 rad), so
       every rotation goes through the solver.
    3. Remove the constraints, translate the planar rope down to ``z_table`` with one boundary-condition step and
       zero the velocities.

    The anchor constraint (node 0) is lifted and lowered together with the rope.

    Parameters
    ----------
    scene : physics.Scene
        The SuperDex scene.
    rope : rod actor
        The rope, lying straight.
    rest : np.ndarray, shape (num_nodes, 3)
        Rest node positions [m].
    anchor_constraint : node position constraint or None
        Constraint that holds node 0, if the rope is anchored.
    layout_xy : np.ndarray, shape (num_nodes, 2)
        Target layout [m].
    z_table : float
        Final node height [m].
    steps : int
        Substeps of the morph; 10 more substeps follow to converge.
    z_air : float
        Height of the morph [m].
    step : callable
        Advances the scene by one substep.
    morph_stiffness : float
        Stiffness of the node position constraints [N/m].

    Returns
    -------
    float
        Largest horizontal distance of a node from the layout after the morph [m].
    """
    num_nodes = len(rest)
    nodes = rest + np.array(rope.get_displacements(), dtype=np.float64).reshape(-1, 4)[:, :3]
    anchor_start = nodes[0].copy()
    dof_index = np.arange(4 * num_nodes, dtype=np.int32)
    xyz_dofs = dof_index[dof_index % 4 != 3]  # each node has 4 DOFs: x, y, z and twist

    # Phase 1: lift the rope and the anchor by translation.
    lift = nodes.copy()
    lift[:, 2] = z_air
    if anchor_constraint is not None:
        anchor_constraint.set_target_position([float(nodes[0, 0]), float(nodes[0, 1]), z_air])
    rope.add_boundary_condition_dofs_world(xyz_dofs, lift.ravel().astype(np.float64))
    step()
    rope.clear_boundary_conditions()
    rope.set_node_velocities_local(np.zeros(4 * num_nodes))

    # Phase 2: morph in the air along the curvature homotopy. The anchored node follows its own constraint.
    xy_from = (rest + np.array(rope.get_displacements(), dtype=np.float64).reshape(-1, 4)[:, :3])[:, :2]
    first_free = 1 if anchor_constraint is not None else 0
    constraints = [scene.create_deformable_node_position_constraint(
        actor=rope.get_handle(), node_index=node, position=[float(lift[node, 0]), float(lift[node, 1]), z_air],
        stiffness=morph_stiffness) for node in range(first_free, num_nodes)]
    for morph_step in range(steps + 10):
        target = homotopy(xy_from, layout_xy, min(1.0, (morph_step + 1) / steps))
        for index, constraint in enumerate(constraints):
            constraint.set_target_position([float(target[index + first_free, 0]), float(target[index + first_free, 1]),
                                            z_air])
        step()
    morphed = rest + np.array(rope.get_displacements(), dtype=np.float64).reshape(-1, 4)[:, :3]
    error = float(np.linalg.norm(morphed[:, :2] - layout_xy, axis=1).max())

    # Phase 3: remove the morph constraints and lower the rope and the anchor by translation.
    for constraint in constraints:
        scene.destroy_constraint(constraint)
    down = morphed.copy()
    down[:, 2] = z_table
    if anchor_constraint is not None:
        anchor_constraint.set_target_position([float(anchor_start[0]), float(anchor_start[1]), float(anchor_start[2])])
    rope.add_boundary_condition_dofs_world(xyz_dofs, down.ravel().astype(np.float64))
    step()
    rope.clear_boundary_conditions()
    rope.set_node_velocities_local(np.zeros(4 * num_nodes))
    return error


class GraspScene:
    """The physics world: FR3 + 2F-85, the anchored rope, three pegs and the table, with the arm and gripper
    controllers.

    The constructor builds the scene, lays the rope straight along -x from the anchor, settles the arm at its home pose
    for 100 substeps and captures a snapshot. :meth:`reset` restores the snapshot, places the rope on a new layout and
    lets it settle. Each :meth:`substep` applies the controller torques for the current targets (``target_pos``,
    ``target_yaw``, ``theta_target``), advances the physics by ``dt`` and updates the grasp.

    Parameters
    ----------
    task : TriangleTask
        Pegs, anchor and route; the rope is ``task.route_length + cfg.rope_tail`` long.
    cfg : SceneCfg
        Physics configuration.

    Attributes
    ----------
    task : TriangleTask
        The routing task.
    cfg : SceneCfg
        The physics configuration.
    dt : float
        Physics substep [s].
    rope_radius : float
        Rope radius [m].
    scene : physics.Scene
        The SuperDex scene.
    table : rigid actor
        The table plane z = 0.
    pegs : list of rigid actors
        The pegs (static cylinders).
    ctx, bot, arm
        SuperDex robotics context, the FR3 + 2F-85 bot and its articulated actor.
    num_arm_dofs : int
        Number of DOFs of the articulated actor (7 arm joints and 6 gripper joints).
    link_names : list of str
        Link names of the articulated actor.
    li : dict of str to int
        Index of each link name.
    link_actor : dict of str to actor
        Physics actor of each link (None for links without one).
    q_min : np.ndarray, shape (num_arm_dofs,), or None
        Lower joint limits [rad], if the robot provides them.
    all_dofs : np.ndarray of int32, shape (num_arm_dofs,)
        Indices of all DOFs, for applying the torque vector.
    osc
        BASIC_OSC_PD controller of the TCP.
    jsc
        BASIC_JSC_PD controller of the knuckles; exists only if ``cfg.grip_controller == "jsc"``.
    root_from_world : physics.TransformRT
        Inverse of the robot root pose, which converts world targets for the OSC.
    tipL, tipR, grip_base
        Actors of the left and right finger-tip links and of the gripper base link.
    rope_length : float
        Rope length [m].
    rope
        The rope's rod actor.
    rest_nodes : np.ndarray, shape (num_nodes, 3)
        Rest node positions of the straight rope [m].
    num_nodes : int
        Number of rope nodes.
    place_err : float
        Layout error after the last morph [m]; 0 with "bc" placement.
    seg : float
        Rest segment length [m].
    anchor_c : node position constraint
        Constraint that holds the rope's node 0 at the anchor (stiffness 1e6 N/m).
    theta_cmd, theta_target : float
        Rate-limited knuckle closure command and its target [rad].
    i_corr : np.ndarray, shape (3,)
        Integral correction added to the TCP target [m].
    target_pos : np.ndarray, shape (3,)
        TCP position target [m].
    target_yaw : float
        Yaw target of the finger closing axis [rad].
    grasp_c : constraint or None
        The no-slip grasp constraint while the rope is held.
    grasp_node : int
        Rope node held by the grasp constraint; -1 if none.
    home_target : np.ndarray, shape (3,)
        TCP target of the home snapshot [m].
    home_yaw : float
        Yaw target of the home snapshot [rad].
    snapshot
        Scene state at home (straight rope, robot at home, fingers open).

    Examples
    --------
    >>> from rope_routing.env.grasp_route_env import make_task  # doctest: +SKIP
    >>> scene = GraspScene(make_task())  # doctest: +SKIP
    >>> scene.num_nodes, scene.num_arm_dofs, round(scene.rope_length, 3)  # doctest: +SKIP
    (61, 13, 0.815)
    >>> scene.tcp().round(3).tolist(), round(scene.tool_yaw(), 3)  # home pose  # doctest: +SKIP
    ([0.441, 0.0, 0.306], -0.785)
    """

    def __init__(self, task: TriangleTask, cfg: SceneCfg = SceneCfg()):
        _init()
        self.task, self.cfg, self.dt = task, cfg, cfg.dt
        self.rope_radius = cfg.rope_radius
        scene = self.scene = physics.create_scene("grasp_route")
        scene.set_gravity([0, 0, -9.81])
        configure_solver(scene, cfg)
        pad = collider_pad(cfg)
        self.table = scene.create_rigid_actor(name="table",
                                              shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0),
                                              is_static=True, contact=static_contact(cfg, cfg.table_friction))
        vertices, faces = cylinder_mesh(task.peg_radius, cfg.peg_height)
        peg_shape = physics.create_tri_mesh_shape(vertices.reshape(-1), faces.reshape(-1))
        self.pegs = [scene.create_rigid_actor(
            name=f"peg{index}", shape=peg_shape, is_static=True, contact=static_contact(cfg, cfg.peg_friction),
            world_from_local=physics.TransformRT(translation=[float(peg[0]), float(peg[1]), 0.0]))
            for index, peg in enumerate(task.pegs)]

        # Robot: no link gravity (the real FR3 compensates gravity internally), by default no joint Coulomb friction,
        # home pose with the fingers open.
        prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(BOT_PATH)))
        for link_index in range(len(prefab.links)):
            prefab.links[link_index].has_gravity = False
        if not cfg.joint_coulomb_friction:
            for joint in prefab.joints:
                friction = joint.friction
                if friction.coulomb > 0:
                    friction.coulomb = 0.0
                    joint.friction = friction
        pose = np.array(prefab.default_pose, dtype=np.float32)
        pose[:7] = HOME
        pose[7:13] = 0.0
        prefab.default_pose = pose
        self.ctx = robotics.create_context()
        self.bot = robotics.create_bot(scene, prefab, self.ctx)
        self.arm = self.bot.get_articulated_actor()
        self.num_arm_dofs = self.arm.get_num_dofs()
        info = self.arm.get_articulated_shape_info()
        self.link_names = list(info.link_names)
        self.li = {link_name: index for index, link_name in enumerate(self.link_names)}
        self.link_actor = {link_name: scene.get_actor(handle)
                           for link_name, handle in zip(self.link_names, self.arm.get_nested_link_actors())}
        self.q_min = (np.array(info.joint_min_limits)[: self.num_arm_dofs]
                      if len(info.joint_min_limits) >= self.num_arm_dofs else None)
        self._buf = physics.DynamicArrayTransformRT(len(self.link_names))
        self._pose = physics.DynamicArrayReal(self.num_arm_dofs)
        self.all_dofs = np.arange(self.num_arm_dofs, dtype=np.int32)

        # Contact parameters of the gripper links: the collider padding of centreline contact, high-friction pads.
        for link_name, actor in self.link_actor.items():
            if actor is None or not link_name.startswith("2f_85"):
                continue
            try:
                contact = actor.get_contact_params()
            except Exception:  # links without a collider
                continue
            contact.penalty_threshold_extra_padding = pad
            contact.penalty_threshold_default = DELTA
            contact.penalty_coefficient = cfg.contact_penalty
            contact.penalty_smoothing_half_distance = 1e-3
            contact.friction_falloff_vel = 1e-4
            contact.coulomb_friction_coefficient = cfg.pad_friction if "finger_tip" in link_name else 0.5
            actor.set_contact_params(contact)

        # Operational-space control of the TCP: fr3_link8 shifted by tcp_offset along its z axis.
        bot_name = self.bot.get_name()
        self.osc = self.bot.create_controller("BASIC_OSC_PD")
        self.osc.initialize(f"{bot_name}/fr3_link0", f"{bot_name}/fr3_link8")
        osc_params = self.osc.get_params()
        osc_params.kp_p, osc_params.kd_p, osc_params.kp_r, osc_params.kd_r = cfg.osc_gains
        osc_params.max_translation_error, osc_params.max_rotation_error = 0.05, 0.4
        osc_params.b_apply_max_osc_torque_normalization = True
        osc_params.ee_link_from_ee = tf([0, 0, cfg.tcp_offset])
        self.osc.set_params(osc_params)
        self.root_from_world = self.osc.get_current_observations_from_mochi().world_from_root.inverse()

        # Gripper: BASIC_JSC_PD torques on the two knuckles, or (default) an implicit PD on them inside the physics
        # step, which is stiff and stable.
        if cfg.grip_controller == "jsc":
            self.jsc = self.bot.create_controller("BASIC_JSC_PD")
            jsc_params = robotics.ControllerBasicJscPdParams()
            kp, kd = np.zeros(self.num_arm_dofs, np.float32), np.zeros(self.num_arm_dofs, np.float32)
            saturation = np.full(self.num_arm_dofs, -1.0, np.float32)
            kp[KNUCKLE_DOFS], kd[KNUCKLE_DOFS], saturation[KNUCKLE_DOFS] = cfg.grip_kp, cfg.grip_kd, cfg.grip_sat
            jsc_params.kp, jsc_params.kd, jsc_params.saturation, jsc_params.deadband = (
                kp, kd, saturation, np.zeros(self.num_arm_dofs, np.float32))
            self.jsc.set_params(jsc_params)
        else:
            pose_params = physics.PoseControllerParams(len(self.link_names))
            tracking = pose_params.joint_tracking
            for knuckle_link in ("2f_85_left_knuckle_link", "2f_85_right_knuckle_link"):
                tracking[self.li[knuckle_link]] = physics.PoseTrackingParams(stiffness=cfg.grip_kp, damping=cfg.grip_kd,
                                                                            saturation=cfg.grip_sat / cfg.grip_kp)
            pose_params.joint_tracking = tracking
            self.arm.add_articulated_pose_controller(pose_params)
        self.tipL = self.link_actor["2f_85_left_finger_tip_link"]
        self.tipR = self.link_actor["2f_85_right_finger_tip_link"]
        self.grip_base = self.link_actor["2f_85_base_link"]

        # Rope: straight along -x from the anchor, 0.5 mm above the table, with node 0 held at the anchor.
        self.rope_length = task.route_length + cfg.rope_tail
        num_nodes, radius = cfg.rope_nodes, cfg.rope_radius
        nodes = np.zeros((num_nodes, 3), np.float32)
        nodes[:, 0] = task.anchor[0] - np.linspace(0, self.rope_length, num_nodes)
        nodes[:, 1] = task.anchor[1]
        nodes[:, 2] = radius + 0.0005
        axes = np.tile([0.0, 1.0, 0.0], (num_nodes - 1, 1)).astype(np.float32)
        self.rope, self.rest_nodes = create_rope(scene, cfg, nodes, axes)
        self.num_nodes = num_nodes
        self.place_err = 0.0
        self.seg = self.rope_length / (num_nodes - 1)
        self._bc_idx = np.arange(4 * num_nodes, dtype=np.int32)  # xyz and twist of every node
        self.anchor_c = scene.create_deformable_node_position_constraint(actor=self.rope.get_handle(), node_index=0,
                                                                         position=nodes[0].tolist(), stiffness=1e6)
        for actor in (self.tipL, self.tipR):
            actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

        # Settle the arm at home with the fingers open, then snapshot the scene for every reset.
        self.theta_cmd = self.theta_target = THETA_OPEN
        self.i_corr = np.zeros(3)
        self.target_pos = self.tcp().copy()
        self.target_yaw = self.tool_yaw()
        self.grasp_c, self.grasp_node = None, -1
        for _ in range(100):
            self.substep()
        self.home_target = self.target_pos.copy()
        self.home_yaw = self.target_yaw
        self.snapshot = scene.capture_state()

    # ------------------------------------------------------------------------------------------------ state readout
    def obsv(self):
        """Current observation of the OSC controller from Mochi (DOF state, end-effector pose, Jacobian)."""
        return self.osc.get_current_observations_from_mochi()

    def q(self):
        """Current DOF positions of the articulated actor, shape (num_arm_dofs,) [rad]."""
        self.arm.get_articulated_pose(self._pose)
        return np.array(self._pose, dtype=np.float64)

    def joint_state(self):
        """DOF positions [rad] and velocities [rad/s] of the articulated actor, each of shape (num_arm_dofs,)."""
        obs = self.obsv()
        return np.array(obs.dof_positions, dtype=np.float64), np.array(obs.dof_velocities, dtype=np.float64)

    def link_pos_rot(self, name):
        """World position, shape (3,) [m], and rotation matrix, shape (3, 3), of the named link."""
        self.arm.get_articulated_link_transforms(self._buf)
        transform = self._buf[self.li[name]]
        return np.array(transform.translation, dtype=np.float64), rot(transform.rotation)

    def tcp_T(self):
        """World pose of the TCP (``physics.TransformRT``)."""
        return self.obsv().world_from_ee_link * tf([0, 0, self.cfg.tcp_offset])

    def tcp(self):
        """World position of the TCP, shape (3,) [m]."""
        return np.array(self.tcp_T().translation, dtype=np.float64)

    def tool_yaw(self):
        """Yaw of the finger closing axis (fr3_link8 x) about world z [rad]."""
        rotation = rot(self.obsv().world_from_ee_link.rotation)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def pads(self):
        """World positions of the left and right pad centres, each of shape (3,) [m]."""
        pos_left, rot_left = self.link_pos_rot("2f_85_left_finger_tip_link")
        pos_right, rot_right = self.link_pos_rot("2f_85_right_finger_tip_link")
        return pos_left + rot_left @ PAD_L, pos_right + rot_right @ PAD_R

    def pad_gap(self):
        """Distance between the two pad centres [m]."""
        pad_left, pad_right = self.pads()
        return float(np.linalg.norm(pad_left - pad_right))

    def pad_forces(self):
        """Magnitudes of the rope's contact forces on the left and right finger tips [N]; (0, 0) if unavailable."""
        try:
            force_left = float(np.linalg.norm(np.array(self.tipL.get_contact_force_from_actor_world(self.rope))))
            force_right = float(np.linalg.norm(np.array(self.tipR.get_contact_force_from_actor_world(self.rope))))
        except Exception:
            force_left = force_right = 0.0
        return force_left, force_right

    def rope_nodes(self):
        """Current rope node positions, shape (num_nodes, 3) [m]."""
        displacement = np.array(self.rope.get_displacements(), dtype=np.float64).reshape(-1, 4)[:, :3]
        return self.rest_nodes + displacement

    def link_positions(self):
        """World positions of all links of the articulated actor, shape (n_links, 3) [m]."""
        self.arm.get_articulated_link_transforms(self._buf)
        return np.array([np.array(self._buf[index].translation, dtype=np.float64)
                         for index in range(len(self.link_names))])

    @property
    def grasped(self) -> bool:
        """Whether the rope is held.

        In "attach" mode: whether the grasp constraint exists. In "friction" mode: whether both pads press on the rope
        with at least ``cfg.grasp_min_force`` and a rope node lies within 1.5 cm of the TCP.
        """
        if self.cfg.grasp_mode == "attach":
            return self.grasp_c is not None
        force_left, force_right = self.pad_forces()
        nodes = self.rope_nodes()
        near = np.linalg.norm(nodes - self.tcp(), axis=1).min() < 0.015
        return bool(min(force_left, force_right) >= self.cfg.grasp_min_force and near)

    # ------------------------------------------------------------------------------------------------ control
    def _arm_tau(self):
        """Joint torques of the arm controller for the current TCP target.

        The OSC tracks the TCP target shifted by an integral correction (``i_corr``), which cancels the steady-state
        error left by joint friction. A posture term pulls the arm joints toward HOME, with the base joint turned to
        face the target; it acts through the null-space projector of the TCP task, so it does not disturb the TCP.

        Returns
        -------
        np.ndarray, shape (num_arm_dofs,)
            Torques [N m].
        """
        obs = self.obsv()
        tcp = np.array((obs.world_from_ee_link * tf([0, 0, self.cfg.tcp_offset])).translation, dtype=np.float64)
        self.i_corr = np.clip(self.i_corr + self.cfg.tcp_ki * self.dt * (self.target_pos - tcp), -self.cfg.tcp_i_max,
                              self.cfg.tcp_i_max)
        target = tf(self.target_pos + self.i_corr, quat_down(self.target_yaw))
        tau = np.array(self.osc.compute_output(obs, robotics.ControllerBasicOscPdTarget(
            root_from_target_ee=self.root_from_world * target)), dtype=np.float64)

        # Null-space posture: torques toward HOME, projected with the damped pseudo-inverse of the TCP Jacobian.
        joint_pos = np.array(obs.dof_positions, dtype=np.float64)[:7]
        joint_vel = np.array(obs.dof_velocities, dtype=np.float64)[:7]
        posture = HOME.copy()
        posture[0] = np.clip(np.arctan2(self.target_pos[1], self.target_pos[0]), -2.6, 2.6)  # face the target azimuth
        posture_tau = self.cfg.posture_kp * (posture - joint_pos) - self.cfg.posture_kd * joint_vel
        jacobian = np.array(obs.ee_jacobian, dtype=np.float64).reshape(6, -1)[:, :7]
        jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + 1e-4 * np.eye(6))
        tau[:7] += (np.eye(7) - jacobian.T @ jacobian_pinv.T) @ posture_tau
        return tau

    def _grip_tau(self):
        """Knuckle torques of the BASIC_JSC_PD gripper controller; zeros with the implicit gripper controller.

        Returns
        -------
        np.ndarray, shape (num_arm_dofs,)
            Torques [N m], nonzero only on the two knuckle DOFs.
        """
        if self.cfg.grip_controller != "jsc":
            return np.zeros(self.num_arm_dofs)
        obs = self.jsc.get_current_observations_from_mochi()
        obs.dt = self.dt
        target = np.array(obs.dof_positions, dtype=np.float32)
        target[GRIP_DOFS] = GRIP_SIGN * self.theta_cmd
        tau = np.array(self.jsc.compute_output(obs, robotics.ControllerBasicJscPdTarget(target_pose=target)),
                       dtype=np.float64)
        knuckle_tau = np.zeros(self.num_arm_dofs)
        knuckle_tau[KNUCKLE_DOFS] = tau[KNUCKLE_DOFS]
        return knuckle_tau

    def substep(self):
        """Advance the scene by one physics substep of ``dt``.

        The knuckle command moves toward ``theta_target`` by at most ``cfg.grip_speed * dt``. The arm torques (and, with
        the "jsc" gripper controller, the knuckle torques) are applied; with the implicit gripper controller the
        knuckle command is set as the pose-controller target instead. The physics then steps and the grasp is updated.
        """
        remaining = self.theta_target - self.theta_cmd
        self.theta_cmd += float(np.clip(remaining, -self.cfg.grip_speed * self.dt, self.cfg.grip_speed * self.dt))
        tau = self._arm_tau() + self._grip_tau()
        self.arm.set_external_forces_on_dofs(dof_indices=self.all_dofs, force_values=tau.astype(np.float32))
        if self.cfg.grip_controller != "jsc":
            pose = self.q()
            pose[GRIP_DOFS] = GRIP_SIGN * self.theta_cmd
            self.arm.set_articulated_target_pose(physics.DynamicArrayReal(pose.astype(np.float32)))
        self.scene.step(self.dt)
        self._update_grasp()

    def _update_grasp(self):
        """Create or release the no-slip grasp constraint ("attach" mode only).

        While the fingers are commanded closed (``theta_target`` above half of THETA_CLOSED) and both pads press on the
        rope with at least ``cfg.grasp_min_force``, the rope node nearest the TCP, if within 1.5 cm, is attached to the
        gripper base at its current relative position. The constraint saturates at GRASP_MAX_FORCE, so an overloaded
        grasp slips instead of exploding. Opening the fingers releases it.
        """
        if self.cfg.grasp_mode != "attach":
            return
        closing = self.theta_target > 0.5 * THETA_CLOSED
        if self.grasp_c is None and closing:
            force_left, force_right = self.pad_forces()
            if min(force_left, force_right) >= self.cfg.grasp_min_force:
                pinch = self.tcp()  # the rope is pinched near the finger tips (the TCP), not at the pad centres
                nodes = self.rope_nodes()
                node = int(np.argmin(np.linalg.norm(nodes - pinch, axis=1)))
                if np.linalg.norm(nodes[node] - pinch) < 0.015:
                    self.grasp_c = self.scene.create_deformable_node_to_rigid_constraint(
                        deformable_actor=self.rope.get_handle(), rigid_actor=self.grip_base.get_handle(),
                        deformable_node_index=node, rigid_local_pos=[0.0, 0.0, 0.0], fix_to_deformable_pos=True,
                        find_closest=False, stiffness=1e5)
                    # Friction-limited hold: beyond about 60 N (2 pads x friction 2 x about 15 N) the rope slips
                    # instead of exploding.
                    self.grasp_c.set_saturation(GRASP_MAX_FORCE / 1e5)
                    self.grasp_node = node
        elif self.grasp_c is not None and not closing:
            self.release()

    def release(self):
        """Remove the grasp constraint, if any."""
        if self.grasp_c is not None:
            self.scene.destroy_constraint(self.grasp_c)
        self.grasp_c, self.grasp_node = None, -1

    # ------------------------------------------------------------------------------------------------ reset and layout
    def reset(self, layout_xy: np.ndarray, settle_steps: int | None = None) -> int:
        """Restore the home snapshot, place the rope at rest on a layout and let it settle (robot holding home).

        Parameters
        ----------
        layout_xy : np.ndarray, shape (num_nodes, 2)
            Rope layout [m], e.g. from ``random_layout``.
        settle_steps : int, optional
            Number of settle substeps; None uses the configured settle (``cfg.settle``).

        Returns
        -------
        int
            Number of settle substeps run.
        """
        self.place(layout_xy)
        return self.settle(settle_steps)

    def place(self, layout_xy: np.ndarray):
        """Restore the home snapshot (straight rope, robot at home, fingers open), then put the rope on the layout,
        free and at rest.

        With ``cfg.placement == "frames"`` the rope is placed by :func:`frame_consistent_place`, and ``place_err``
        records the layout error after the morph. ``"bc"`` (legacy) pins every rope DOF (xyz and zero twist) at the
        layout for one step, which leaves stale element frames.

        Parameters
        ----------
        layout_xy : np.ndarray, shape (num_nodes, 2)
            Rope layout [m].

        Raises
        ------
        ValueError
            If ``cfg.placement`` is unknown.
        """
        self.release()
        self.scene.restore_state(self.snapshot, False)
        self.theta_cmd = self.theta_target = THETA_OPEN
        self.i_corr = np.zeros(3)
        self.target_pos, self.target_yaw = self.home_target.copy(), self.home_yaw
        height = self.rope_radius + self.cfg.layout_z0
        if self.cfg.placement == "frames":
            self.place_err = frame_consistent_place(self.scene, self.rope, self.rest_nodes, self.anchor_c,
                                                    np.asarray(layout_xy, np.float64), height, self.cfg.place_steps,
                                                    self.cfg.place_z, self.substep, self.cfg.place_k)
            return
        if self.cfg.placement != "bc":
            raise ValueError(f"unknown placement {self.cfg.placement!r}")
        target = np.c_[layout_xy, np.full(len(layout_xy), height), np.zeros(len(layout_xy))]  # positions, zero twist
        self.rope.add_boundary_condition_dofs_world(self._bc_idx, target.ravel().astype(np.float64))
        self.scene.step(self.dt)
        self.rope.clear_boundary_conditions()
        self.rope.set_node_velocities_local(np.zeros(4 * self.num_nodes))
        self.place_err = 0.0

    def settle(self, settle_steps: int | None = None) -> int:
        """Run idle substeps after the placement.

        The "adaptive" settle runs until the largest node speed stays below ``cfg.settle_v`` for ``cfg.settle_calm``
        consecutive substeps (at least ``cfg.settle_min`` and at most ``cfg.settle_max`` substeps). The placed layout
        keeps elastic energy that a short fixed settle would hand to the first actions as motion, and a long fixed
        settle wastes time.

        Parameters
        ----------
        settle_steps : int, optional
            If given, run exactly this many substeps instead.

        Returns
        -------
        int
            Number of substeps run.

        Raises
        ------
        ValueError
            If ``cfg.settle`` is unknown.
        """
        cfg = self.cfg
        if cfg.settle not in ("adaptive", "fixed"):  # a typo must not silently fall through to the adaptive settle
            raise ValueError(f"unknown settle {cfg.settle!r}")
        if settle_steps is not None or cfg.settle == "fixed":
            count = cfg.settle_steps if settle_steps is None else int(settle_steps)
            for _ in range(count):
                self.substep()
            return count
        previous, calm, count = self.rope_nodes(), 0, 0
        while count < cfg.settle_max:
            self.substep()
            count += 1
            nodes = self.rope_nodes()
            speed = float(np.linalg.norm(nodes - previous, axis=1).max()) / self.dt
            previous = nodes
            calm = calm + 1 if speed < cfg.settle_v else 0
            if calm >= cfg.settle_calm and count >= cfg.settle_min:
                break
        return count
