"""Tests of SurfaceRouteEnv (env/surface_route_env.py): the pure-RL environment whose deformation observation and
reward terms are the first and second fundamental forms of the rope's tube surface (notes/A_deformation_math.md), with
an environment-generated start-state curriculum (notes/B_rl_environment.md).

Real SuperDex physics: about 2 min single-threaded (run on d3). Run with -s to see the measured numbers:
    cd project_superdex && .venv/bin/python -m pytest ../rope_routing/tests/test_surface_route_env.py -q -s
"""

import ast
import os
import time
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rope_routing.deformation import measures
from rope_routing.deformation.material_graph import from_polyline
from rope_routing.deformation.surface_forms import TubeSurfaceDescriptor, polyline_window_curvature
from rope_routing.env import grasp_route_env, surface_route_env as sre
from rope_routing.env.start_states import setup_start
from rope_routing.env.surface_route_env import DEF_FEATURES, L_KAPPA, MODES, SurfaceRouteEnv, make_env
from rope_routing.task.geometry import TriangleTask
from rope_routing.task.rope_init import random_layout

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GAMMA = 0.99  # the RL discount the environment declares (env.gamma)
INFO_KEYS = ("is_success", "start_mode", "max_stage", "grasped", "shape", "wind_frac", "routed", "area_strain_max",
             "curvature_match", "stage")
# (reset seed, layout seed) pairs tried in order until a "grasped" start succeeds (it failed on about 20% of layouts
# with the legacy scene, whose FR3 joint friction froze the arm; see start_states.py)
GRASP_CANDIDATES = [(5, 9000015), (1, 600001), (2, 600002), (3, 600003), (4, 600004), (6, 600006)]


@pytest.fixture(scope="module")
def env():
    """One SurfaceRouteEnv shared by the tests of this module."""
    return make_env(rank=0, seed=0)


def grasped_start(env):
    """Reset into a "grasped" start on the first candidate layout where the setup succeeds.

    Returns (obs, info, layout seed).
    """
    for seed, layout in GRASP_CANDIDATES:
        obs, info = env.reset(seed=seed, options={"layout_seed": layout, "no_prefix": True, "start_mode": "grasped"})
        if info["start_mode"] == "grasped":
            return obs, info, layout
    pytest.fail("no 'grasped' start on any candidate layout")


def potential_now(env):
    """Potential of the CURRENT physical state, from a fresh measurement (not the env's stored self.phi)."""
    nodes, samples, surf, wind, tcp = env._measure()
    return env._potential(tcp, nodes, samples, wind, env._grasp_state()[0])[0]


def check_obs(env, obs):
    """The observation is float32, of the declared shape, finite and inside the observation space."""
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs)) and obs in env.observation_space


class Diverged:
    """What Mochi's explosion control leaves after a diverged Newton solve: all rod displacements zero."""

    def __init__(self, rope):
        self._rope = rope

    def get_displacements(self):
        """All-zero displacements, as after the explosion control fired."""
        return np.zeros_like(np.asarray(self._rope.get_displacements(), np.float32))

    def __getattr__(self, name):
        """Everything else from the real rope."""
        return getattr(self._rope, name)


# ======================================================================================================== API
def test_check_env():
    """gymnasium's checker. Expected warning (printed): no registered spec, so the alternative-render-mode check is
    skipped. Reset/step determinism and the observation/action spaces must pass."""
    checked = make_env(rank=1, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(checked)
    messages = sorted({str(warning.message) for warning in caught})
    print("\ncheck_env warnings:", messages)
    assert all("spec" in message for message in messages), messages


def test_layout_and_spaces(env):
    """Layouts match GraspRouteEnv's, and the observation blocks have the documented sizes and order."""
    obs, info = env.reset(seed=0, options={"layout_seed": 9000015, "no_prefix": True})
    expected = random_layout(np.random.default_rng(9000015), env.task.anchor, env.task.pegs, env.sim.rope_length,
                             env.sim.num_nodes, env.task.peg_radius, env.sim.rope_radius)
    assert np.array_equal(env._layout, expected)  # same layout as GraspRouteEnv.reset for this layout_seed
    assert info["start_mode"] == "full" and info["start_tried"] == []
    check_obs(env, obs)
    assert env.observation_space.shape == (516,) and env.action_space.shape == (5,)
    index = env.obs_index
    assert len(DEF_FEATURES) == 17 and DEF_FEATURES[-1] == "valid" and "K_abs" in DEF_FEATURES
    assert index["q"][0] == 0 and index["pegs"][1] == 516 and index["deform"] == (33, 390)
    assert index["deform_global"] == (390, 392) and index["potential"] == (392, 393) and index["target_pos"][0] == 393
    # Contiguous and in order.
    assert all(index[name][1] == index[following][0] for name, following in zip(list(index), list(index)[1:]))
    assert env.horizon == 500 and env.expert is None and env.prefix_prob == 0.0
    # the potential the critic sees, and the discount the shaping assumes (the trainer checks env.gamma)
    assert obs[392] == pytest.approx(np.clip(info["phi"] / 10.0, -5.0, 5.0), abs=1e-6)
    from rope_routing.rl.async_ppo import env_gamma
    assert env.gamma == GAMMA and env_gamma(env) == GAMMA
    print("\nobs blocks:", index)


def test_start_probs_api(env):
    """set_start_probs normalises and validates its input, and the start modes are drawn with those probabilities."""
    env.set_start_probs((2.0, 1.0, 1.0))
    assert np.allclose(env.start_probs, [0.5, 0.25, 0.25])
    env.set_start_probs({"grasped": 3.0, "full": 1.0})
    assert np.allclose(env.start_probs, [0.25, 0.0, 0.75])
    for bad in ({"warp": 1.0}, (1.0, 1.0), (-1.0, 1.0, 1.0), {"full": 0.0}):
        with pytest.raises(ValueError):
            env.set_start_probs(bad)
    env.set_start_probs({"full": 0.2, "hover": 0.3, "grasped": 0.5})
    env.np_random = np.random.default_rng(0)
    draws = [env._pick_mode({}) for _ in range(4000)]
    freq = [draws.count(mode) / len(draws) for mode in MODES]
    assert np.allclose(freq, [0.2, 0.3, 0.5], atol=0.03), freq
    assert env._pick_mode({"start_mode": "hover"}) == "hover"
    with pytest.raises(ValueError):
        env._pick_mode({"start_mode": "flying"})
    env.set_start_probs({"full": 1.0})


def test_eval_mode_is_always_full():
    """In eval mode every reset is a "full" start with the robot at home, whatever the start probabilities."""
    eval_env = make_env(rank=2, seed=2, eval=True)
    eval_env.set_start_probs({"grasped": 1.0})
    for index in range(3):
        _, info = eval_env.reset(seed=index, options={"layout_seed": 9000000 + index, "no_prefix": True})
        assert info["start_mode"] == "full" and info["start_requested"] == "full" and info["start_tried"] == []
        assert np.allclose(eval_env.sim.target_pos, eval_env.sim.home_target)  # robot untouched at home
        _, _, _, _, step_info = eval_env.step(np.zeros(5))
        assert step_info["start_mode"] == "full"
    eval_env.eval_mode = False  # the same env outside eval mode does run the requested setup
    _, info = eval_env.reset(seed=0, options={"layout_seed": 9000015})
    assert info["start_requested"] == "grasped" and info["start_tried"]


# ======================================================================================================== episodes
FALLBACK_OF = {"full": ("full",), "hover": ("hover", "full"), "grasped": ("grasped", "hover", "full")}


@pytest.mark.parametrize("mode", MODES)
def test_random_episodes(mode):
    """3 episodes per start mode with uniform random actions (horizon 150): finite obs of constant dimension, finite
    rewards that equal the sum of their parts, the required info keys."""
    episode_env = make_env(rank=3, seed=3, horizon=150)
    rng = np.random.default_rng(1)
    used, steps, start_time = [], 0, time.perf_counter()
    for episode in range(3):
        obs, info = episode_env.reset(seed=10 + episode, options={"start_mode": mode})
        check_obs(episode_env, obs)
        used.append(info["start_mode"])
        assert info["start_mode"] in FALLBACK_OF[mode]
        while True:
            obs, reward, terminated, truncated, info = episode_env.step(rng.uniform(-1.0, 1.0, 5))
            steps += 1
            check_obs(episode_env, obs)
            assert np.isfinite(reward) and abs(reward - sum(info["r_terms"].values())) < 1e-9
            assert all(key in info for key in INFO_KEYS) and info["start_mode"] == used[-1]
            if terminated or truncated:
                break
        assert truncated or terminated
    print(f"\n{mode}: start modes used {used}, {steps} steps, "
          f"{(time.perf_counter() - start_time) / steps * 1e3:.1f} ms/step incl. setups")


def run_potential_episode(env, obs, info, rng, max_steps):
    """Returns per-step shaping terms and the independently recomputed potentials phi_0..phi_T."""
    phis, shaping = [potential_now(env)], []
    assert phis[0] == pytest.approx(info["phi"], rel=1e-12, abs=1e-12)  # reset/setup stored the current potential
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1.0, 1.0, 5))
        terms = info["r_terms"]
        assert abs(reward - sum(terms.values())) < 1e-9
        shaping.append(terms["shaping"])
        phis.append(potential_now(env))
        assert phis[-1] == pytest.approx(info["phi"], rel=1e-12, abs=1e-12)
        assert shaping[-1] == pytest.approx(10.0 * (env.shaping_gamma * phis[-1] - phis[-2]), rel=1e-9, abs=1e-9)
        # the observed potential is that of the returned state (the stored one: a bad step returns the last valid obs)
        assert obs[env.obs_index["potential"][0]] == pytest.approx(np.clip(env.phi / 10.0, -5.0, 5.0), abs=1e-6)
        if terminated or truncated:
            break
    return np.array(shaping), np.array(phis)


@pytest.mark.parametrize("mode", ["full", "grasped"])
def test_potential_consistency(env, mode):
    """sum_t gamma^t r_shaping_t = 10 (gamma^T phi_T - phi_0) over an episode (full start: the whole 500-step episode;
    grasped start: 200 steps, crossing the grasped branch and, with random grips, usually a release). phi includes the
    curvature-matching term, which is also checked against an independent descriptor."""
    rng = np.random.default_rng(7)
    if mode == "full":
        obs, info = env.reset(seed=21, options={"layout_seed": 9000021, "start_mode": "full"})
        max_steps = env.horizon
    else:
        obs, info, _ = grasped_start(env)
        max_steps = 200
    shaping, phis = run_potential_episode(env, obs, info, rng, max_steps)
    length = len(shaping)
    gamma = env.shaping_gamma
    lhs = float(np.sum(gamma ** np.arange(length) * shaping))
    rhs = 10.0 * (gamma**length * phis[-1] - phis[0])
    print(f"\n{mode}: T={length}  sum gamma^t shaping = {lhs:.9f}  10(gamma^T phi_T - phi_0) = {rhs:.9f}  "
          f"phi {phis[0]:.3f} -> {phis[-1]:.3f}")
    assert abs(lhs - rhs) <= 1e-6 * max(1.0, abs(rhs))
    # phi_kappa from scratch: fresh descriptor + target profile on the same windows
    desc = TubeSurfaceDescriptor(env.sim.rest_nodes, env.sim.rope_radius, env.sample_s)
    _, kappa_route = polyline_window_curvature(env.task.route, desc.sample_window)
    mask = env.goal_mask
    kappa = desc(env.sim.rope_nodes())["signed_planar_curvature"]
    curv_match = float(np.mean(np.abs(kappa - kappa_route)[mask]))
    assert env.curvature_match(env.sim.rope_nodes()) == pytest.approx(curv_match, rel=1e-12)
    _, _, _, _, info = env.step(np.zeros(5))
    kappa = desc(env.sim.rope_nodes())["signed_planar_curvature"]
    assert info["phi_kappa"] == pytest.approx(-L_KAPPA * float(np.mean(np.abs(kappa - kappa_route)[mask])), rel=1e-12)


def test_grasped_start_holds_under_zero_actions(env):
    """After a "grasped" start, zero actions keep the rope held, with small rewards and no hand-over spike."""
    obs, info, layout = grasped_start(env)
    check_obs(env, obs)
    assert obs[env.obs_index["gripper"][0] + 2] == 1.0
    rewards = []
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        rewards.append(reward)
        assert info["grasped"] and not truncated and info["r_terms"]["bad"] == 0.0
    rewards = np.array(rewards)
    print(f"\nlayout {layout}: rewards over 20 zero-action steps after a 'grasped' start: min {rewards.min():.3f} "
          f"max {rewards.max():.3f}; constant-potential value -(1 - gamma) 10 phi - 0.01 = "
          f"{-0.1 * info['phi'] - 0.01:.3f}")
    # holding still: reward ~ -(1 - gamma) 10 phi - 0.01 (constant potential), no hand-over spike
    assert np.all(np.abs(rewards) < 1.0)


def test_release_debounce(env):
    """While holding, the fingers open only on the RELEASE_STEPS-th consecutive opening command."""
    grasped_start(env)
    opening = np.r_[0.0, 0.0, 0.0, 0.0, -1.0]
    # Twice: fewer than RELEASE_STEPS opening commands keep holding, and an interruption resets the count.
    for _ in range(2):
        for step in range(sre.RELEASE_STEPS - 1):
            _, _, _, _, info = env.step(opening)
            assert info["grasped"], step
        _, _, _, _, info = env.step(np.zeros(5))
        assert info["grasped"]
    for step in range(sre.RELEASE_STEPS - 1):
        _, _, _, _, info = env.step(opening)
        assert info["grasped"], step
    _, _, _, _, info = env.step(opening)  # the RELEASE_STEPS-th consecutive one opens the fingers
    assert not info["grasped"] and not env.sim.grasped and env.sim.theta_target == 0.0


def test_start_states_are_deterministic(env):
    """Same (seed, layout seed, start mode) -> the same start state bit for bit, whatever ran before."""
    opts = {"layout_seed": 600505, "no_prefix": True, "start_mode": "grasped"}
    obs1, info1 = env.reset(seed=600505, options=opts)
    env.step(np.ones(5))
    env.reset(seed=1, options={"layout_seed": 9000001, "start_mode": "hover"})
    obs2, info2 = env.reset(seed=600505, options=opts)
    print(f"\nstart {info1['start_mode']} "
          f"(tried {[(attempt['mode'], attempt['ok']) for attempt in info1['start_tried']]}), "
          f"repeat identical: {np.array_equal(obs1, obs2)}")
    assert info1["start_mode"] == info2["start_mode"] and info1["start_tried"] == info2["start_tried"]
    assert np.array_equal(obs1, obs2)


def test_setup_divergence_is_never_handed_over(env):
    """The start-state driver bypasses env.step: a solver divergence during the setup must fail it (not hand a rope
    reset to its rest shape to the policy), and reset() then falls back to the untouched layout."""
    env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
    rope = env.sim.rope
    env.sim.rope = Diverged(rope)
    try:
        ok, start_info = setup_start(env, "hover", np.random.default_rng(0))
        assert not ok and start_info["diverged"] and not start_info["finite"] and start_info["steps"] == 1
        _, info = env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "grasped"})
    finally:
        env.sim.rope = rope
    assert info["start_mode"] == "full" and [attempt["ok"] for attempt in info["start_tried"]] == [False, False]
    _, info = env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
    *_, info = env.step(np.zeros(5))
    assert not info["sim_diverged"] and info["r_terms"]["bad"] == 0.0


# ======================================================================================================== deformation
def test_deformation_block_matches_descriptor(env):
    """The observation's deformation block IS the fundamental-form descriptor (fresh instance, evaluated directly), and
    none of the old point-graph features is computed while the env runs."""
    old = ("node_features", "deformation_gradients", "bending", "tautness_to", "overstretch", "rigidity")
    saved = {name: getattr(measures, name) for name in old}

    def forbidden(*_, **__):
        raise AssertionError("old point-graph deformation feature used")

    try:
        for name in old:
            setattr(measures, name, forbidden)
        obs, _ = env.reset(seed=4, options={"layout_seed": 9000004, "start_mode": "hover"})
        rng = np.random.default_rng(3)
        for _ in range(15):
            before = env.sim.rope_nodes()
            obs, *_ = env.step(rng.uniform(-1.0, 1.0, 5))
    finally:
        for name, func in saved.items():
            setattr(measures, name, func)
    desc = TubeSurfaceDescriptor(env.sim.rest_nodes, env.sim.rope_radius, env.sample_s)
    nodes, tcp, radius = env.sim.rope_nodes(), env.sim.tcp(), env.sim.rope_radius
    out = desc(nodes)
    graph = from_polyline(env.sim.rest_nodes, env.num_samples)
    samples = graph.samples(nodes)
    velocity = (samples - graph.samples(before)) / (env.substeps * env.sim.dt)
    expected = np.column_stack([samples - tcp, 0.1 * velocity, 10 * (out["area_ratio"] - 1),
                                10 * radius * (out["H"] - out["H_rest"]), radius * radius * out["K_mean"],
                                radius * radius * out["K_abs"], np.maximum(radius * radius * out["K_min"], -9.0),
                                radius * radius * out["K_max"], 0.03 * out["curvature_vector"],
                                0.03 * out["signed_planar_curvature"], out["valid"].astype(float)])
    deform, deform_global = env.deformation_block(obs)
    assert deform.shape == (21, 17) and len(DEF_FEATURES) == 17
    err = np.abs(deform - expected).max(0)
    print("\nmax |obs - direct descriptor| per feature:", dict(zip(DEF_FEATURES, np.round(err, 8))))
    assert np.allclose(deform, expected, rtol=1e-5, atol=1e-6)
    # willmore global: the smooth tube's closed form, continuous through folds (equal to willmore_rest at rest)
    w_smooth = desc.smooth_tube_forms(nodes)["willmore"]
    assert desc.smooth_tube_forms(env.sim.rest_nodes)["willmore"] == pytest.approx(out["willmore_rest"], rel=1e-9)
    assert np.allclose(deform_global, [10 * (out["area_ratio_total"] - 1), w_smooth / out["willmore_rest"] - 1],
                       rtol=1e-5, atol=1e-6)
    start, stop = env.obs_index["potential"]
    assert obs[start:stop] == pytest.approx([np.clip(potential_now(env) / 10.0, -5.0, 5.0)], abs=1e-6)
    _, kappa_route = polyline_window_curvature(env.task.route, desc.sample_window)
    start, stop = env.obs_index["target_curvature"]
    assert np.allclose(obs[start:stop], 0.03 * np.where(env.goal_mask, kappa_route, 0.0), atol=1e-6)
    assert np.all(obs[start:stop][~env.goal_mask] == 0.0)


def test_target_profile_matches_a_rope_on_the_route(env):
    """A rope laid exactly along the target route reads (nearly) the target curvature profile, on EVERY masked sample:
    the target is computed on the same footing as the measurement (residual from the rope's 13.5 mm chords cutting the
    2 cm arcs). Signs: the route turns left (counter-clockwise) around B, C and A, so the targets are >= 0, ~0 on the
    straight edges. The rope follows the route without its 5 cm end gap, i.e. it keeps wrapping peg A as the full route
    does: the last masked sample's window runs ~1.7 cm past the route end and its target (the window mean over the part
    on the route) is what such a rope reads; a rope cut straight at the route end would read far less there."""
    full = TriangleTask(pegs=env.task.pegs.copy(), peg_height=env.task.peg_height, rope_radius=env.task.rope_radius,
                        waypoint_clearance=env.task.waypoint_clearance, arc_radius=env.task.arc_radius, end_gap=0.0)
    assert np.allclose(full.route[: len(env.task.route)], env.task.route)  # same route, only longer
    node_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(env.sim.rest_nodes, axis=0), axis=1))]
    on_route = node_arc <= full.route_length
    nodes = np.zeros((env.sim.num_nodes, 3))
    nodes[on_route, 0] = np.interp(node_arc[on_route], full.route_s, full.route[:, 0])
    nodes[on_route, 1] = np.interp(node_arc[on_route], full.route_s, full.route[:, 1])
    end, direction = nodes[on_route][-1], nodes[on_route][-1] - nodes[on_route][-2]
    nodes[~on_route, :2] = (end[:2] + (node_arc[~on_route] - node_arc[on_route][-1])[:, None]
                            * (direction[:2] / np.linalg.norm(direction[:2])))
    nodes[:, 2] = env.sim.rope_radius
    kappa = env.desc(nodes)["signed_planar_curvature"]
    targets, mask = env.kappa_target, env.goal_mask
    err = np.abs(kappa - targets)
    print(f"\nrope on the route: max |kappa - kappa*| = {err[mask].max():.2f} 1/m on a {targets.max():.1f} 1/m peak; "
          f"targets {np.round(targets[mask], 1)}; masked mean {err[mask].mean():.2f} 1/m "
          f"-> phi_kappa {-L_KAPPA * err[mask].mean():.4f}")
    assert np.all(targets[mask] > -1e-9) and np.all(targets[~mask] == 0.0) and targets.max() > 30.0
    assert err[mask].max() < 0.05 * targets.max()
    assert env.curvature_match(nodes) == pytest.approx(err[mask].mean(), rel=1e-12)


def test_bad_states_are_truncated_with_finite_outputs(env):
    """Non-finite rope, an areal strain > 30% and a SuperDex divergence (rod reset to rest) are simulator failures: each
    gives reward -10 + the action terms only (no shaping into, no area penalty on the invalid state), truncation and the
    LAST VALID observation (the truncation bootstraps from it)."""
    obs0, _ = env.reset(seed=8, options={"layout_seed": 9000008})
    env.sim.rope_nodes = lambda: np.full((env.sim.num_nodes, 3), np.nan)  # instance attribute shadows the method
    try:
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
    finally:
        del env.sim.rope_nodes
    check_obs(env, obs)
    assert (truncated and not terminated and np.isfinite(reward) and info["r_terms"]["bad"] == -10.0
            and info["r_terms"]["shaping"] == 0.0)
    np.testing.assert_array_equal(obs, obs0)

    real_desc = env.desc

    def strained_step(ratio):
        """Reset, then step once with sample 3's area ratio replaced by ``ratio``."""
        def strained(nodes):
            out = dict(real_desc(nodes))
            out["area_ratio"] = out["area_ratio"].copy()
            out["area_ratio"][3] = ratio
            return out

        obs0, _ = env.reset(seed=8, options={"layout_seed": 9000008})
        env.desc = strained
        try:
            return (obs0, *env.step(np.zeros(5)))
        finally:
            env.desc = real_desc
            env._surf_nodes = None

    obs0, obs, reward, terminated, truncated, info = strained_step(1.35)
    check_obs(env, obs)
    assert truncated and info["r_terms"]["bad"] == -10.0 and info["area_strain_max"] == pytest.approx(0.35)
    assert info["r_terms"]["area"] == 0.0 and info["r_terms"]["shaping"] == 0.0 and reward == pytest.approx(-10.01)
    np.testing.assert_array_equal(obs, obs0)
    # real stretch below the 'bad' threshold is penalised: 10% -> -50 (0.10 - 0.02)^2, not truncated
    _, obs, reward, terminated, truncated, info = strained_step(1.10)
    check_obs(env, obs)
    assert not truncated and info["r_terms"]["bad"] == 0.0 and info["r_terms"]["area"] == pytest.approx(-50 * 0.08**2)
    # compression (-12%): a rope cannot carry it, so the one-sided penalty ignores it; not 'bad' either
    _, obs, reward, terminated, truncated, info = strained_step(0.88)
    check_obs(env, obs)
    assert not truncated and info["r_terms"]["bad"] == 0.0 and info["r_terms"]["area"] == 0.0
    assert info["area_strain_max"] == pytest.approx(0.12)

    obs0, _ = env.reset(seed=8, options={"layout_seed": 9000008})
    rope = env.sim.rope
    env.sim.rope = Diverged(rope)
    try:
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
    finally:
        env.sim.rope = rope
    check_obs(env, obs)
    assert truncated and info["sim_diverged"] and info["r_terms"]["bad"] == -10.0 and info["r_terms"]["shaping"] == 0.0
    np.testing.assert_array_equal(obs, obs0)
    env.reset(seed=8, options={"layout_seed": 9000008})
    *_, info = env.step(np.zeros(5))
    assert not info["sim_diverged"]


def hairpin(env, node=45, turn_deg=170.0):
    """The current rope folded back on itself at node k (a turn of `turn_deg` there, then straight on), segment lengths
    kept: a sharper fold than the 12-gon tube can follow (> ~106 degrees per node)."""
    nodes = env.sim.rope_nodes().copy()
    seg_len = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    tangent = (nodes[node] - nodes[node - 1]) / seg_len[node - 1]
    cos_t, sin_t = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    direction = np.array([cos_t * tangent[0] - sin_t * tangent[1], sin_t * tangent[0] + cos_t * tangent[1], 0.0])
    direction /= np.linalg.norm(direction)
    for index in range(node + 1, env.sim.num_nodes):
        nodes[index] = nodes[index - 1] + seg_len[index - 1] * direction
    return nodes


def test_forced_hairpin_reads_finite_and_flags_folds(env):
    """A forced hairpin: the descriptor flags the folded samples (valid = 0 in the observation), every value stays
    finite, and the area term reads the TRUE axial stretch (segment lengths kept -> no penalty, not "bad"), where the
    raw 12-gon mesh misreads it."""
    env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
    nodes = hairpin(env)
    env.sim.rope_nodes = lambda: nodes.copy()  # instance attribute shadows the method: the env measures the hairpin
    try:
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
    finally:
        del env.sim.rope_nodes
    check_obs(env, obs)
    deform, deform_global = env.deformation_block(obs)
    valid = deform[:, DEF_FEATURES.index("valid")]
    surf = env.desc(nodes)
    raw = env.desc.forms.group_stats(env.desc.last_out, env.desc.groups, env.desc.num_samples)["area_ratio"]
    print(f"\nhairpin: {info['n_invalid']} folded samples {np.flatnonzero(valid == 0)}, fold max "
          f"{info['fold_max_deg']:.0f} deg; area_ratio - 1 robust max {np.abs(surf['area_ratio'] - 1).max():.2e}, raw "
          f"mesh on folded samples {np.round(raw[valid == 0] - 1, 4)}")
    assert set(np.unique(valid)) == {0.0, 1.0} and info["n_invalid"] == int(np.sum(valid == 0)) > 0
    assert np.array_equal(valid, surf["valid"].astype(float)) and info["fold_max_deg"] > 150.0
    assert np.all(np.isfinite(deform)) and np.all(np.isfinite(deform_global))
    kmin = deform[:, DEF_FEATURES.index("K_min")]
    assert np.all(kmin >= -9.0 - 1e-6)
    assert not truncated and info["r_terms"]["bad"] == 0.0 and info["r_terms"]["area"] == 0.0
    assert np.all(np.isfinite(list(info["r_terms"].values()))) and np.isfinite(info["phi_kappa"])
    assert info["area_strain_max"] < 0.01  # true stretch of the (uncarried) rope


def test_willmore_global_is_continuous_through_folds(env):
    """Sweep one node's turn through the fold onset (~106-130 degrees): the descriptor's whole-rope Willmore switches
    from the mesh value to the closed form at the first fold (a jump of ~1.4% of the rest value); the Willmore global
    the environment observes is always the closed form, so it moves smoothly."""
    env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
    turns = np.arange(90.0, 171.0, 1.0)
    w_obs, w_desc, n_inv = [], [], []
    for turn in turns:
        nodes = hairpin(env, node=30, turn_deg=turn)
        surf = env._surface(nodes)
        w_obs.append(surf["willmore_smooth"] / env._willmore_rest - 1.0)
        w_desc.append(surf["willmore"] / surf["willmore_rest"] - 1.0)
        n_inv.append(surf["n_invalid"])
    w_obs, w_desc, n_inv = map(np.asarray, (w_obs, w_desc, n_inv))
    onset = turns[np.argmax(n_inv > 0)]
    print(f"\nfold onset at {onset:.0f} deg: max step per degree, observed willmore global "
          f"{np.abs(np.diff(w_obs)).max():.2e}, descriptor's {np.abs(np.diff(w_desc)).max():.2e}")
    assert n_inv[0] == 0 and n_inv[-1] > 0
    assert np.all(np.isfinite(w_obs)) and np.abs(np.diff(w_obs)).max() < 2e-3
    assert np.abs(np.diff(w_desc)).max() > 5e-3  # what the closed form avoids (if this fails the descriptor changed)


def test_shaping_is_progress_reward(env):
    """Shaping is 10 (shaping_gamma phi' - phi) with shaping_gamma = 1 by default (a progress reward: notes/B, run
    surf_ppo1), independent of env.gamma (the RL discount the trainer checks); shaping_gamma is honoured if changed."""
    assert env.shaping_gamma == 1.0 and env.gamma == GAMMA
    for gamma in (1.0, 0.95):
        env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
        phi0 = env.phi
        env.shaping_gamma = gamma
        try:
            _, _, _, _, info = env.step(np.zeros(5))
        finally:
            env.shaping_gamma = 1.0
        assert info["r_terms"]["shaping"] == pytest.approx(10.0 * (gamma * info["phi"] - phi0), rel=1e-12, abs=1e-12)


def test_reach_term_is_continuous(env):
    """The reach goal blends continuously (smoothstep over d_xy in [1, 3] cm) instead of the parent's jump at 1.5 cm:
    sweep the TCP away from the grasp point in 0.05 mm steps at several heights. Outside the blend band the potential is
    exactly GraspRouteEnv's + phi_kappa, and the grasped branch is untouched."""
    env.reset(seed=8, options={"layout_seed": 9000008, "start_mode": "full"})
    nodes, samples, surf, wind, tcp = env._measure()
    grasp_pos = nodes[env._grasp_point(nodes)]
    direction = np.array([1.0, 0.0, 0.0])
    offsets = np.arange(0.0, 0.05, 5e-5)
    kappa = sre.W_KAPPA * L_KAPPA * env.curvature_match(nodes)
    worst, parent_jump = 0.0, 0.0
    for height in (0.005, 0.03, 0.06, 0.10, 0.15):
        pts = [np.r_[grasp_pos[:2] + d_xy * direction[:2], height] for d_xy in offsets]
        phi = np.array([env._potential(point, nodes, samples, wind, False)[0] for point in pts])
        parent = np.array([grasp_route_env.GraspRouteEnv._potential(env, point, nodes, samples, wind, False)[0]
                           for point in pts])
        worst = max(worst, np.abs(np.diff(phi)).max() / 0.05)  # per mm
        parent_jump = max(parent_jump, np.abs(np.diff(parent)).max())
        out = (offsets < 0.01) | (offsets > 0.03)
        np.testing.assert_allclose(phi[out], parent[out] - kappa, rtol=0, atol=1e-12)
    print(f"\nreach sweep: max |dphi| per mm {worst:.4f} (parent: a single 0.05 mm step jumps by {parent_jump:.3f})")
    assert worst < 0.05 and parent_jump > 0.2
    for point in (tcp, tcp + 0.01, tcp - [0.0, 0.02, 0.0]):  # grasped branch: identical to the parent's
        assert env._potential(point, nodes, samples, wind, True)[0] == pytest.approx(
            grasp_route_env.GraspRouteEnv._potential(env, point, nodes, samples, wind, True)[0] - kappa, abs=1e-12)
    blend = [env.reach_blend(d_xy) for d_xy in (0.0, 0.01, 0.02, 0.03, 0.2)]
    assert blend == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


# ======================================================================================================== no expert
TRAIN_EVAL_PATHS = ["rope_routing/env/surface_route_env.py", "rope_routing/env/start_states.py",
                    "rope_routing/rl/async_ppo.py", "scripts/train_rl.py", "scripts/verify_rl.py"]


def _dotted(node):
    """Dotted name of a call target, e.g. "env.expert.act" or "make()"; "" for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func) + "()"
    return ""


def test_no_expert_in_training_or_eval_paths():
    """Static proof: in the training/evaluation code paths no call goes to anything named *expert* (other than the
    async_ppo blocker `_forbid_expert`) and ScriptedExpert is never referenced."""
    bad = []
    for rel in TRAIN_EVAL_PATHS:
        path = os.path.join(ROOT, rel)
        tree = ast.parse(open(path).read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if "expert" in name.lower() and not name.endswith("_forbid_expert"):
                    bad.append(f"{rel}:{node.lineno} calls {name}")
            elif isinstance(node, ast.Name) and node.id == "ScriptedExpert":
                bad.append(f"{rel}:{node.lineno} references ScriptedExpert")
    assert not bad, bad


def test_no_expert_at_runtime(env, monkeypatch):
    """Runtime proof: with every expert entry point booby-trapped, resets in all start modes and steps run clean."""
    from rope_routing.rl.async_ppo import _forbid_expert

    def trap(*_, **__):
        raise AssertionError("expert called")

    monkeypatch.setattr(grasp_route_env.ScriptedExpert, "act", trap)
    monkeypatch.setattr(grasp_route_env.ScriptedExpert, "reset", trap)
    monkeypatch.setattr(grasp_route_env.ScriptedExpert, "__init__", trap)
    guarded = _forbid_expert(make_env(rank=4, seed=4))
    assert guarded.expert is None
    with pytest.raises(RuntimeError):
        guarded.expert_action()
    with pytest.raises(ValueError):
        guarded.set_prefix_prob(0.5)
    guarded.set_start_probs({"full": 1, "hover": 1, "grasped": 1})
    for index, mode in enumerate(MODES):
        guarded.reset(seed=index, options={"layout_seed": 9000030 + index, "no_prefix": False, "start_mode": mode})
        for _ in range(3):
            guarded.step(guarded.action_space.sample())
    guarded.reset(seed=9, options={"no_prefix": False})  # curriculum draw, prefix flag ignored
    assert SurfaceRouteEnv.reset is not grasp_route_env.GraspRouteEnv.reset


# ======================================================================================================== speed
def test_step_time_breakdown(env):
    """Wall time per env.step and its parts (physics substeps / descriptor / everything else). Informative; the bound
    only catches gross regressions on a loaded machine (requirement: <= 40 ms on d3)."""
    env.reset(seed=6, options={"layout_seed": 9000006, "start_mode": "full"})
    sub, dsc = [0.0], [0.0]
    real_sub, real_desc = env.sim.substep, env.desc

    def timed_sub():
        """The physics substep, timed."""
        start = time.perf_counter()
        real_sub()
        sub[0] += time.perf_counter() - start

    def timed_desc(nodes):
        """The descriptor call, timed."""
        start = time.perf_counter()
        out = real_desc(nodes)
        dsc[0] += time.perf_counter() - start
        return out

    env.sim.substep, env.desc = timed_sub, timed_desc
    rng = np.random.default_rng(0)
    count, start_time = 60, time.perf_counter()
    try:
        for _ in range(count):
            env.step(rng.uniform(-1.0, 1.0, 5))
    finally:
        del env.sim.substep
        env.desc = real_desc
    total_ms = (time.perf_counter() - start_time) / count * 1e3
    print(f"\nstep {total_ms:.1f} ms = physics {sub[0] / count * 1e3:.1f} + descriptor {dsc[0] / count * 1e3:.2f} + "
          f"rest (readouts, potential, obs) {total_ms - (sub[0] + dsc[0]) / count * 1e3:.2f} ms")
    assert total_ms < 100.0
