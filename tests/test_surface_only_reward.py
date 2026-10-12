"""The surface-only reward ablation of GenericSurfaceEnv (reward_terms="surface")."""

import numpy as np
import pytest

from rope_routing.env.generic_surface_env import W_COVER, W_CURV, make_env


@pytest.fixture(scope="module")
def env():
    return make_env(seed=0, reward_terms="surface")


def test_potential_uses_only_surface_terms(env):
    """phi depends only on coverage and curvature mismatch: not on holding, gripper position or wraps."""
    env.reset(seed=3)
    nodes = env.sim.rope_nodes()
    cover, curv_err = env.coverage(env._surface(nodes))
    expected = -W_COVER * cover - W_CURV * curv_err
    wraps = env.wrap_frac(nodes)
    for tcp in (env.sim.tcp(), nodes[-1] + [0.0, 0.0, 0.02], nodes[10] + [0.0, 0.0, 0.05]):
        for grasped in (False, True):
            for wrap in (wraps, np.ones_like(wraps), np.zeros_like(wraps)):
                assert env._potential(tcp, nodes, None, wrap, grasped)[0] == expected


def test_observation_shows_nearest_surface_point(env):
    obs, _ = env.reset(seed=4)
    surf = env._surface(env.sim.rope_nodes())
    tcp = env.sim.tcp()
    start, end = env.obs_index["nearest_surface"]
    nearest = surf["V"][surf["tree"].query(tcp)[1]]
    np.testing.assert_allclose(obs[start:end], (nearest - tcp).astype(np.float32), atol=1e-6)


def test_success_and_info(env):
    env.reset(seed=5, options={"start_mode": "grasped"})
    for _ in range(3):
        _, reward, _, _, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert info["is_success"] == (info["coverage"] < 0.025 and info["r_terms"]["bad"] == 0.0)
    assert "success_full" in info


def test_default_is_the_full_reward():
    with pytest.raises(ValueError):
        make_env(seed=0, reward_terms="nope")
    assert make_env(seed=0).reward_terms == "full"
