"""Tests of GenericSurfaceEnv (env/generic_surface_env.py): the object-agnostic goal (a target surface),
correspondence-free coverage, generic wraps, the generic grasp rule and roam starts (notes/B_rl_environment.md)."""

import numpy as np
import pytest

from rope_routing.env.generic_surface_env import (COVER_SUCCESS, N_TARGET, GRASP_MIN_FRAC, farthest_point_sample,
                                                  make_env)

# Observation blocks of SurfaceRouteEnv that name task structure; GenericSurfaceEnv must have none of them.
HAND_CODED = ("target_pos", "target_mask", "target_curvature", "grasp_pos", "grasp_yaw", "waypoint",
              "waypoint_progress", "wind_frac", "pegs")


@pytest.fixture(scope="module")
def env():
    """One GenericSurfaceEnv shared by the tests of this module."""
    return make_env(seed=0)


def test_observation_has_no_hand_coded_task_structure(env):
    """The observation has no hand-coded task blocks, and its blocks tile the observation vector."""
    assert not set(HAND_CODED) & set(env.obs_index)
    assert env.obs_index["target_surface"][1] - env.obs_index["target_surface"][0] == 4 * N_TARGET
    end = max(stop for _, stop in env.obs_index.values())
    assert end == env.observation_space.shape[0]
    obs, _ = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))


def test_target_surface_and_coverage(env):
    """The target covers itself exactly, coverage grows with a rigid shift, and the target wraps every obstacle."""
    assert env.coverage(env._surface(env.target_nodes)) == (0.0, 0.0)
    # moving the object rigidly away from the target raises coverage monotonically, at most by the shift (surface to
    # surface: a shift comparable to the tube radius still overlaps part of the target)
    covs = [env.coverage(env._surface(env.target_nodes + [0.0, 0.0, dz]))[0] for dz in (0.005, 0.01, 0.03, 0.06)]
    assert np.all(np.diff(covs) > 0) and all(cover <= dz for cover, dz in zip(covs, (0.005, 0.01, 0.03, 0.06)))
    assert covs[-1] > 0.8 * 0.06
    # every obstacle is wrapped by the target, and the target wraps itself fully
    assert env._wrap_mask.all()
    np.testing.assert_allclose(env.wrap_frac(env.target_nodes), 1.0, atol=1e-6)


def test_coverage_is_correspondence_free(env):
    """Coverage needs no correspondence: relabelling (reversing) the material does not change it."""
    # reversing the node order (relabelling the material) leaves the surface and therefore the coverage unchanged
    shifted = env.coverage(env._surface(env.target_nodes + [0.002, 0.0, 0.0]))
    nodes = env.sim.rope_nodes()
    forward_cov = env.coverage(env._surface(nodes))
    reversed_cov = env.coverage(env._surface(nodes[::-1].copy()))
    assert shifted[0] < 0.003
    assert abs(forward_cov[0] - reversed_cov[0]) < 2e-3


def test_potential_rises_toward_the_target(env):
    """The potential increases as the rope is blended from its current shape into the target configuration."""
    env.reset(seed=5)
    start = env.sim.rope_nodes()
    tcp = env.sim.tcp()
    phis = []
    for blend in np.linspace(0.0, 1.0, 6):
        nodes = (1 - blend) * start + blend * env.target_nodes
        phis.append(env._potential(tcp, nodes, None, env.wrap_frac(nodes), True)[0])
    assert phis[-1] - phis[0] > 3.0
    assert phis[-1] == max(phis)


def test_random_generic_grasp_point(env):
    """Start-state grasp nodes are random and at least GRASP_MIN_FRAC of the length from the pinned end."""
    grasp_nodes = set()
    lowest = GRASP_MIN_FRAC * (env.sim.num_nodes - 1)
    for index in range(8):
        env.reset(seed=100 + index, options={"start_mode": "hover"})
        assert env._gi is not None and env._gi >= lowest
        grasp_nodes.add(env._gi)
    assert len(grasp_nodes) > 2


def test_fps_is_even():
    """Farthest-point sampling of evenly spaced points on a line picks evenly spaced points."""
    points = np.linspace(0, 1, 101)[:, None] * [1.0, 0.0, 0.0]
    chosen = farthest_point_sample(points, 5)
    np.testing.assert_allclose(np.sort(points[chosen, 0]), [0, 0.25, 0.5, 0.75, 1.0])


def test_step_info(env):
    """step() reports coverage, curvature mismatch, wraps and the re-judged success."""
    env.reset(seed=7, options={"start_mode": "grasped"})
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert {"coverage", "curv_mismatch", "routed", "is_success"} <= set(info)
    assert info["is_success"] == (info["routed"] and info["coverage"] < COVER_SUCCESS and info["grasped"])
    assert np.isfinite(reward)


def test_roam_start_explores_grasp_locations():
    """roam starts: training 'full' starts begin from a random gripper position near the object (fingers open, nothing
    grasped, bookkeeping fresh); the nearest object point then varies along the whole object. Never in eval mode."""
    env = make_env(seed=1, roam_prob=1.0)
    home = env.sim.home_target.copy()
    fracs = []
    for index in range(6):
        obs, info = env.reset(seed=200 + index)
        assert info["start_mode"] == "full" and "roam" in info
        assert env.elapsed_steps == 0 and not env._grasp_state()[0] and np.all(np.isfinite(obs))
        if info["roam"]:
            tcp, nodes = env.sim.tcp(), env.sim.rope_nodes()
            assert np.linalg.norm(tcp[:2] - home[:2]) > 1e-3
            assert np.linalg.norm(nodes[:, :2] - tcp[:2], axis=1).min() < 0.06 * np.sqrt(2) + 0.03
            fracs.append(int(np.argmin(np.linalg.norm(nodes - tcp, axis=1))) / (env.sim.num_nodes - 1))
    assert len(fracs) >= 4 and max(fracs) - min(fracs) > 0.3  # starts spread along the object
    eval_env = make_env(seed=1, eval=True, roam_prob=1.0)
    _, info = eval_env.reset(seed=300)
    assert "roam" not in info and info["start_mode"] == "full"


def test_coverage_memo_matches_fresh_query(env):
    """The memoised coverage equals a fresh nearest-neighbour query and is returned from the cache."""
    env.reset(seed=9)
    surf = env._surface(env.sim.rope_nodes())
    cover = env.coverage(surf)
    radius_sq = env.sim.rope_radius ** 2
    distance, nearest = surf["tree"].query(env.target_V)
    assert cover == (float(distance.mean()),
                     float(np.clip(np.abs(radius_sq * env.target_K - radius_sq * surf["K_v"][nearest]), 0, 10).mean()))
    assert env.coverage(surf) is cover


def test_generic_grasp_rule_prefers_material_far_from_the_fixed_end(env):
    """Far region = top GRASP_FAR of rest geodesic distance from the pinned material: for the anchored rope that is the
    free end. Hovering above it scores a higher pre-grasp potential than hovering above the anchored part."""
    from rope_routing.env.generic_surface_env import GRASP_FAR
    ring = env.desc.ring_of_vertex[env._far_idx]
    rings = ring[ring >= 0]
    assert len(rings) and rings.min() >= (1.0 - GRASP_FAR - 0.05) * (env.sim.num_nodes - 1)
    env.reset(seed=11)
    nodes = env.sim.rope_nodes()
    phi = lambda point: env._potential(point, nodes, None, env.wrap_frac(nodes), False)[0]
    above = lambda node: nodes[node] + [0.0, 0.0, 0.02]
    assert phi(above(env.sim.num_nodes - 3)) > phi(above(int(0.3 * env.sim.num_nodes))) + 0.5
