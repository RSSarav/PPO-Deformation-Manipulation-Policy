"""Tests of the GraspScene rope physics (sim/grasp_scene.py): SceneCfg() is the "C1-nc" configuration of notes C,
legacy_v3() reproduces the old scene, and the frame-consistent reset leaves valid rod frames
(notes/C_physics_and_infrastructure.md).

Real SuperDex physics, each test well under 60 s on one core. Run with -s to see the measured numbers:
    cd project_superdex && .venv/bin/python -m pytest ../rope_routing/tests/test_grasp_scene_physics.py -q -s
Full-scale measurements (50-100 layouts, capstan, pile, speed) are in scripts/check_scene_physics.py (run on d3).
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import superdex.physics as physics

from rope_routing.sim import grasp_scene as gs
from rope_routing.sim.grasp_scene import GraspScene, SceneCfg
from rope_routing.task.geometry import TriangleTask
from rope_routing.task.rope_init import random_layout

LAB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "rope_physics",
                                    "exp", "scripts"))
PEGS = np.array([[0.40, -0.09], [0.58, -0.09], [0.49, 0.066]])  # = env/grasp_route_env.py
SEEDS = (9000003, 9000014, 9000000)  # 9000003 / 9000014: tight curves that hopped and kinked under the old reset


def make_task():  # = grasp_route_env.make_task (not imported: the env is edited by another workflow)
    """The routing task of the environments."""
    return TriangleTask(pegs=PEGS.copy(), peg_height=0.07, rope_radius=0.005, waypoint_clearance=0.10, arc_radius=0.18)


def layout(sim, seed):
    """Random rope layout ``seed`` for the scene's task and rope."""
    return random_layout(np.random.default_rng(seed), sim.task.anchor, sim.task.pegs, sim.rope_length, sim.num_nodes,
                         sim.task.peg_radius, sim.rope_radius)


@pytest.fixture(scope="module")
def sim():
    """One default GraspScene shared by the tests of this module."""
    return GraspScene(make_task(), SceneCfg())


def seg_min_dist(points, min_sep=3):
    """Min distance between centreline segments i, j with |i - j| >= min_sep (clamped closest points).

    Closest points of two segments start + frac * dir, frac in [0, 1]: the unconstrained solution of the 2 x 2 normal
    equations, clamped to the segments.
    """
    starts, dirs = points[:-1], np.diff(points, axis=0)
    first, second = np.triu_indices(len(starts), k=min_sep)
    start1, dir1, start2, dir2 = starts[first], dirs[first], starts[second], dirs[second]
    offset = start1 - start2
    len1_sq, len2_sq, dot12, dot1r, dot2r = ((dir1 * dir1).sum(1), (dir2 * dir2).sum(1), (dir1 * dir2).sum(1),
                                             (dir1 * offset).sum(1), (dir2 * offset).sum(1))
    denom = len1_sq * len2_sq - dot12 * dot12
    frac1 = np.where(denom > 1e-14, np.clip((dot12 * dot2r - dot1r * len2_sq) / np.maximum(denom, 1e-30), 0, 1), 0.0)
    frac2 = (dot12 * frac1 + dot2r) / len2_sq
    frac1 = np.where(frac2 < 0, np.clip(-dot1r / len1_sq, 0, 1),
                     np.where(frac2 > 1, np.clip((dot12 - dot1r) / len1_sq, 0, 1), frac1))
    frac2 = np.clip(frac2, 0, 1)
    return float(np.linalg.norm(start1 + dir1 * frac1[:, None] - start2 - dir2 * frac2[:, None], axis=1).min())


def frame_dot_tangent(sim):
    """max |element frame axis . tangent| over all elements (0 for valid rod frames), from one recorded substep: the
    frames are internal solver state, exposed only by the HDF5 recording."""
    h5py = pytest.importorskip("h5py")
    path = os.path.join(tempfile.gettempdir(), f"gsp_frames_{os.getpid()}.h5")
    sim.scene.start_recording(path, physics.RecordingParams())
    sim.substep()
    sim.scene.stop_recording()
    out = []
    with h5py.File(path, "r") as handle:
        for event in sorted(handle["events"].keys(), key=int):
            group = handle[f"events/{event}"]
            for aid in group.get("actors", {}).keys():
                actor = group["actors"][aid]
                if "elementFrameAxes" in actor and actor["elementFrameAxes"].shape[0] == sim.num_nodes - 1:
                    out.append((actor["displacement"][:].astype(np.float64),
                                actor["elementFrameAxes"][:].astype(np.float64)))
    os.remove(path)
    disp, axes = out[-1]
    nodes = sim.rest_nodes + disp[:, :3]
    tangent = np.diff(nodes, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return float(np.abs((axes * tangent).sum(1)).max())


# ====================================================================================================== configuration
def test_defaults_are_c1_nc(sim):
    """SceneCfg() holds the C1-nc settings, and the built scene's solver and contact parameters match them."""
    cfg, radius = SceneCfg(), SceneCfg().rope_radius
    assert ((cfg.nonlinear_iters, cfg.line_search, cfg.implicit_normal_force, cfg.joint_coulomb_friction)
            == (8, "ARMIJO_OR_RESIDUAL_NORM", True, False))
    assert ((cfg.rope_self_contact, cfg.rope_pc_quad, cfg.rope_pc_exclusion, cfg.rope_contact)
            == (True, "P1Q3", 1.5, "centerline"))
    assert ((cfg.placement, cfg.settle, cfg.settle_v, cfg.settle_calm, cfg.settle_max)
            == ("frames", "adaptive", 0.002, 10, 400))
    assert (cfg.place_steps, cfg.place_z, cfg.place_k) == (30, 0.10, 1e4)
    # Read back from the built scene.
    solver_params = sim.scene.get_solver_params()
    assert solver_params.non_linear_solver.max_iter == 8
    assert solver_params.non_linear_solver.line_search_type == physics.LineSearchType.ARMIJO_OR_RESIDUAL_NORM
    assert solver_params.experimental_eval.implicit_normal_force_for_dissipation
    rope_params = sim.rope.get_contact_params()
    np.testing.assert_allclose([rope_params.coulomb_friction_coefficient, rope_params.penalty_coefficient,
                                rope_params.penalty_threshold_default, rope_params.penalty_threshold_extra_padding,
                                rope_params.penalty_smoothing_half_distance, rope_params.friction_falloff_vel],
                               [0.6, 1e10, 5e-4, radius - 5e-4, 1e-3, 1e-3], rtol=1e-6)
    for actor, mu in ((sim.table, 0.6), (sim.pegs[0], 0.5)):  # table / pegs unchanged from v3
        actor_params = actor.get_contact_params()
        np.testing.assert_allclose([actor_params.coulomb_friction_coefficient, actor_params.penalty_coefficient,
                                    actor_params.penalty_threshold_extra_padding,
                                    actor_params.penalty_smoothing_half_distance],
                                   [mu, 1e10, radius - 5e-4, 1e-3], rtol=1e-6)


def test_legacy_preset_is_v3():
    """legacy_v3() and tube_t0() hold the settings of the old scene and of the tube-contact scene."""
    cfg = SceneCfg.legacy_v3()
    assert ((cfg.nonlinear_iters, cfg.line_search, cfg.implicit_normal_force, cfg.joint_coulomb_friction)
            == (4, "", False, True))
    assert ((cfg.rope_self_contact, cfg.rope_pc_quad, cfg.rope_pad, cfg.rope_smoothing, cfg.rope_friction_falloff)
            == (False, "", 0.0, 5e-3, 1e-2))
    assert (cfg.placement, cfg.settle, cfg.settle_steps) == ("bc", "fixed", 40)
    defaults = physics.ContactParams()  # the v3 rope set only mu, penalty, threshold; the rest were defaults
    rope_params = gs.rope_contact(cfg)
    np.testing.assert_allclose([rope_params.penalty_threshold_extra_padding,
                                rope_params.penalty_smoothing_half_distance, rope_params.friction_falloff_vel],
                               [defaults.penalty_threshold_extra_padding, defaults.penalty_smoothing_half_distance,
                                defaults.friction_falloff_vel])
    tube = SceneCfg.tube_t0()
    assert ((tube.rope_contact, tube.contact_penalty, tube.rope_pad, tube.rope_friction_falloff, tube.nonlinear_iters)
            == ("tube", 1e11, 0.0, 1e-4, 4))
    assert gs.collider_pad(tube) == 0.0 and SceneCfg.legacy_v3(rope_nodes=81).rope_nodes == 81


@pytest.mark.skipif(not os.path.exists(os.path.join(LAB, "vscene.py")), reason="rope-physics lab not present")
def test_legacy_matches_frozen_v3_bit_for_bit():
    """legacy_v3() against the lab's frozen copy of the old GraspScene (vscene V0, validated bit-for-bit against it):
    same layout, reset and arm motion -> identical rope and TCP."""
    sys.path.insert(0, LAB)
    import vscene
    gs._init()
    vscene._initialized = True
    task = make_task()
    live, frozen = GraspScene(task, SceneCfg.legacy_v3()), vscene.VScene(vscene.SPECS["V0"], task)
    xy = layout(live, 9000014)
    live.reset(xy)
    frozen.reset(xy)
    for scene in (live, frozen):
        scene.target_pos = scene.target_pos + np.array([0.05, 0.0, -0.1])
    for _ in range(60):
        live.substep()
        frozen.substep()
    assert np.abs(live.rope_nodes() - frozen.rope_nodes()).max() == 0.0
    assert np.abs(live.tcp() - frozen.tcp()).max() == 0.0


# ====================================================================================================== reset
def test_placement_frames_consistent_and_at_layout(sim):
    """Frame-consistent placement: valid frames (|a.t| ~ 0) and the rope on the layout; the old BC jump leaves the
    straight rope's frames (|a.t| -> 1 on tight layouts)."""
    for seed in SEEDS:
        xy = layout(sim, seed)
        sim.place(xy)
        nodes = sim.rope_nodes()
        err_xy = float(np.linalg.norm(nodes[:, :2] - xy, axis=1).max())
        height = nodes[:, 2] - sim.rope_radius
        frame_dot = frame_dot_tangent(sim)
        count = sim.settle()
        settled = sim.rope_nodes()
        print(f"\n  seed {seed}: |a.t| {frame_dot:.1e}, layout error after the morph {1000 * sim.place_err:.3f} mm "
              f"(after lowering {1000 * err_xy:.3f} mm, "
              f"z-r in [{1000 * height.min():.2f}, {1000 * height.max():.2f}] mm); "
              f"after the {count}-substep settle the max node offset from the layout is "
              f"{1000 * np.linalg.norm(settled[:, :2] - xy, axis=1).max():.1f} mm "
              f"(elastic tail relaxation, notes/C C1: median 33 mm)")
        assert frame_dot < 1e-5
        assert sim.place_err < 2e-4 and err_xy < 2e-4
        assert np.all(np.abs(height - sim.cfg.layout_z0) < 1e-3)
        assert np.linalg.norm(settled[0] - nodes[0]) < 1e-3  # the anchor holds


def test_settle_terminates(sim):
    """Adaptive settle stops when every node is calm (< 2 mm/s for 10 substeps), well before the 400-substep cap."""
    for seed in SEEDS:
        sim.place(layout(sim, seed))
        count = sim.settle()
        before = sim.rope_nodes()
        sim.substep()
        speed = float(np.linalg.norm(sim.rope_nodes() - before, axis=1).max()) / sim.dt
        print(f"\n  seed {seed}: settled in {count} substeps, next-substep max node speed {1000 * speed:.2f} mm/s")
        assert sim.cfg.settle_min <= count < sim.cfg.settle_max
        assert speed < 2 * sim.cfg.settle_v
    assert sim.settle(5) == 5  # an explicit length overrides the adaptive settle


# ====================================================================================================== self-contact
def test_fold_does_not_pass_through_itself():
    """Pile test "fold" (notes C): the free end is lifted, carried back over the middle of the rope, laid onto it and
    dragged across the lower strand. The centreline distance between non-adjacent segments must stay near 2r (v3:
    5.3 mm, passes through in piles)."""
    gs._init()
    cfg = SceneCfg()
    radius, length, num_nodes, dt = cfg.rope_radius, 0.81, cfg.rope_nodes, 0.01
    scene = physics.create_scene("fold")
    scene.set_gravity([0, 0, -9.81])
    gs.configure_solver(scene, cfg)
    scene.create_rigid_actor(name="table", shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0),
                             is_static=True, contact=gs.static_contact(cfg, cfg.table_friction))
    nodes = np.c_[np.linspace(0, length, num_nodes), np.zeros(num_nodes), np.full(num_nodes, radius + 3e-4)]
    rope, rest = gs.create_rope(scene, cfg, nodes, np.tile([0.0, 1.0, 0.0], (num_nodes - 1, 1)))
    scene.create_deformable_node_position_constraint(actor=rope.get_handle(), node_index=0,
                                                     position=nodes[0].tolist(), stiffness=1e6)
    hand = scene.create_deformable_node_position_constraint(actor=rope.get_handle(), node_index=num_nodes - 1,
                                                            position=nodes[-1].tolist(), stiffness=1e5)
    # Key frames (time [s], hand position [m]) of the free end, interpolated linearly.
    end0 = nodes[-1]
    keys = [(0.0, end0), (0.5, end0 + [0, 0, 0.04]), (2.5, np.array([0.45 * length, -0.03, 0.045])),
            (3.5, np.array([0.45 * length, -0.03, 3 * radius])), (5.5, np.array([0.45 * length, 0.05, 3 * radius]))]
    dmin, zmax = 1.0, 0.0
    for step in range(700):
        now = (step + 1) * dt
        for (t_start, pos_start), (t_end, pos_end) in zip(keys[:-1], keys[1:]):
            if now <= t_end:
                target = pos_start + (pos_end - pos_start) * (now - t_start) / (t_end - t_start)
                break
        else:
            target = keys[-1][1]
        hand.set_target_position(target.tolist())
        scene.step(dt)
        current = rest + np.array(rope.get_displacements(), dtype=np.float64).reshape(-1, 4)[:, :3]
        assert np.isfinite(current).all()
        if step >= 250:  # strands overlap in top view from ~3 s on
            dmin = min(dmin, seg_min_dist(current))
            zmax = max(zmax, float(current[:, 2].max()))
    print(f"\n  fold: min centreline distance {1000 * dmin:.2f} mm (2r = {2000 * radius:.0f} mm), "
          f"max z {1000 * zmax:.1f} mm")
    assert dmin > 2 * radius - 1.5e-3
    assert zmax > 2.5 * radius  # the upper strand rests ON the lower one
