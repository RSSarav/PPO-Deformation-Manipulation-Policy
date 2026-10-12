"""Correctness checks for the pure-RL asynchronous PPO trainer (rope_routing.rl.async_ppo).

Light tests run anywhere: GAE (including ragged fragments), running statistics, worker/learner log-prob agreement,
env-error robustness, straggler preemption (every transition used exactly once; preempt_frac 1 reproduces the old
fixed-T batch bit for bit), the KL-adaptive learning rate, critic training after the policy's KL stop, the gamma
lock, per-episode reward-term sums, 2-worker multi-process smoke runs, and fault injection (crashed, hung and
unrestartable worker, NaN gradient, dead evaluation-pool worker, hung evaluation):
    cd project_superdex && .venv/bin/python -m pytest ../rope_routing/tests/test_async_ppo.py -q
Heavy tests (Pendulum learning with 8 workers, 32-worker throughput including the heavy-tailed straggler benchmark,
old fixed-T vs preemption) need ASYNC_PPO_HEAVY=1; run them on d3:
    ASYNC_PPO_HEAVY=1 CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest ../rope_routing/tests/test_async_ppo.py -s \
        -k "pendulum or throughput"

Array conventions follow the trainer: (T, W) = (steps of a fragment, workers).
"""

import collections
import copy
import json
import math
import multiprocessing
import os
import sys
import time

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn.utils import parameters_to_vector

from rope_routing.rl import async_ppo as ap

HEAVY = pytest.mark.skipif(os.environ.get("ASYNC_PPO_HEAVY") != "1", reason="heavy: set ASYNC_PPO_HEAVY=1 (run on d3)")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GYM = "rope_routing.rl.async_ppo:make_gym_env"


def _iters(run_dir):
    """Iteration and evaluation records of a run's metrics.jsonl.

    Parameters
    ----------
    run_dir : str
        Run directory.

    Returns
    -------
    iterations, evaluations : list of dict
        Records of type "iter" and of type "eval", in file order.
    """
    with open(os.path.join(run_dir, "metrics.jsonl")) as handle:
        records = [json.loads(line) for line in handle]
    return ([record for record in records if record["type"] == "iter"],
            [record for record in records if record["type"] == "eval"])


# ============================================================================================ GAE
def gae_reference(rewards, values, v_last, term, trunc, v_final, gamma, lam):
    """Straightforward definition: A_t = sum_l (gamma lam)^l delta_{t+l}, summed forward until the episode ends.

    Same arguments as :func:`rope_routing.rl.async_ppo.gae` (without ragged lengths).

    Returns
    -------
    advantages, returns : np.ndarray, shape (T, W)
    """
    num_steps, num_workers = rewards.shape
    adv = np.zeros((num_steps, num_workers))
    for worker in range(num_workers):
        for step in range(num_steps):
            total, coef = 0.0, 1.0
            for later in range(step, num_steps):
                # Bootstrap value after step `later`: 0 at a termination, V(final obs) at a truncation, V(next obs)
                # after the fragment's last step, else the next step's value.
                if term[later, worker]:
                    next_value = 0.0
                elif trunc[later, worker]:
                    next_value = v_final[later, worker]
                elif later == num_steps - 1:
                    next_value = v_last[worker]
                else:
                    next_value = values[later + 1, worker]
                total += coef * (rewards[later, worker] + gamma * next_value - values[later, worker])
                if term[later, worker] or trunc[later, worker]:
                    break
                coef *= gamma * lam
            adv[step, worker] = total
    return adv, adv + values


def bootstrapped_returns(rewards, term, trunc, v_final, v_last, gamma):
    """lambda = 1 closed form: discounted reward sum to the episode/fragment end + discounted bootstrap value.

    Returns
    -------
    np.ndarray, shape (T, W)
        Return of every step.
    """
    num_steps, num_workers = rewards.shape
    returns = np.zeros((num_steps, num_workers))
    for worker in range(num_workers):
        for step in range(num_steps):
            ret, disc, later = 0.0, 1.0, step
            while True:
                ret += disc * rewards[later, worker]
                disc *= gamma
                if term[later, worker]:
                    break
                if trunc[later, worker]:
                    ret += disc * v_final[later, worker]
                    break
                if later == num_steps - 1:
                    ret += disc * v_last[worker]
                    break
                later += 1
            returns[step, worker] = ret
    return returns


def _random_fragment(seed, num_steps=60, num_workers=9):
    """Random rollout fragment with the episode-end edge cases GAE must handle.

    Parameters
    ----------
    seed : int
        Seed of all arrays.
    num_steps, num_workers : int, optional
        Fragment shape (T, W).

    Returns
    -------
    rewards, values, v_last, term, trunc, v_final
        The arguments of :func:`rope_routing.rl.async_ppo.gae` in order, without gamma and lambda.
    """
    rng = np.random.default_rng(seed)
    rewards, values = rng.normal(size=(num_steps, num_workers)), rng.normal(size=(num_steps, num_workers))
    v_last, v_final = rng.normal(size=num_workers), rng.normal(size=(num_steps, num_workers))
    term = rng.random((num_steps, num_workers)) < 0.06
    trunc = (rng.random((num_steps, num_workers)) < 0.06) & ~term
    # Edge cases: episode end on the first / last step, back-to-back ends, a column without any end.
    term[0, 0] = True
    term[-1, 1], trunc[-1, 1] = True, False
    trunc[-1, 2], term[-1, 2] = True, False
    term[5:8, 3], trunc[5:8, 3] = [True, False, True], [False, True, False]
    term[:, 4] = trunc[:, 4] = False
    return rewards, values, v_last, term, trunc, v_final


@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (0.9, 0.0), (0.97, 1.0), (1.0, 0.8)])
def test_gae_matches_reference(gamma, lam):
    """ap.gae equals the forward-sum definition of GAE on a fragment with every episode-end edge case."""
    rewards, values, v_last, term, trunc, v_final = _random_fragment(0)
    adv, ret = ap.gae(rewards, values, v_last, term, trunc, v_final, gamma, lam)
    adv_ref, ret_ref = gae_reference(rewards, values, v_last, term, trunc, v_final, gamma, lam)
    np.testing.assert_allclose(adv, adv_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ret, ret_ref, rtol=1e-12, atol=1e-12)


def test_gae_lambda1_is_bootstrapped_return():
    """With lambda = 1 the GAE return is the bootstrapped discounted return; a truncation bootstraps from
    V(final obs), a termination does not."""
    rewards, values, v_last, term, trunc, v_final = _random_fragment(1)
    _, ret = ap.gae(rewards, values, v_last, term, trunc, v_final, 0.95, 1.0)
    np.testing.assert_allclose(ret, bootstrapped_returns(rewards, term, trunc, v_final, v_last, 0.95), rtol=1e-12,
                               atol=1e-12)
    # Truncation bootstraps (changing V(final obs) changes the return), termination does not.
    v_final2 = v_final + 10.0
    _, ret2 = ap.gae(rewards, values, v_last, term, trunc, v_final2, 0.95, 1.0)
    step, worker = np.argwhere(trunc)[0]
    assert ret2[step, worker] == pytest.approx(ret[step, worker] + 0.95 * 10.0)
    step, worker = np.argwhere(term)[0]
    assert ret2[step, worker] == pytest.approx(rewards[step, worker])


# ============================================================================================ ragged GAE / preemption
@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (0.9, 0.0), (0.97, 1.0)])
def test_ragged_gae_matches_reference_per_worker(gamma, lam):
    """Preempted fragments have different lengths: padded (T_max, W) arrays with garbage in the padding must give, on
    each worker's rows 0 .. L_w-1, exactly the reference GAE of that worker's fragment alone (bootstrapping its last
    step from V(its next obs)), and zeros in the padding."""
    rewards, values, v_last, term, trunc, v_final = _random_fragment(3, num_steps=50, num_workers=7)
    lengths = np.array([50, 1, 17, 33, 2, 49, 25])
    rng = np.random.default_rng(4)
    # Fill the padding with large random rewards/values and random episode ends.
    pad = np.arange(50)[:, None] >= lengths[None]
    rewards, values, v_final = (np.where(pad, rng.normal(size=rewards.shape) * 1e3, array)
                                for array in (rewards, values, v_final))
    term, trunc = term | (pad & (rng.random(rewards.shape) < 0.5)), trunc | (pad & (rng.random(rewards.shape) < 0.5))
    term[16, 2], trunc[16, 2] = False, False  # worker 2's fragment ends mid-episode: bootstraps from v_last
    adv, ret = ap.gae(rewards, values, v_last, term, trunc, v_final, gamma, lam, lengths=lengths)
    for worker, length in enumerate(lengths):
        adv_ref, ret_ref = gae_reference(rewards[:length, worker:worker + 1], values[:length, worker:worker + 1],
                                         v_last[worker:worker + 1], term[:length, worker:worker + 1],
                                         trunc[:length, worker:worker + 1], v_final[:length, worker:worker + 1],
                                         gamma, lam)
        np.testing.assert_allclose(adv[:length, worker], adv_ref[:, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ret[:length, worker], ret_ref[:, 0], rtol=1e-12, atol=1e-12)
        assert not adv[length:, worker].any() and not ret[length:, worker].any()


class TokenEnv(gym.Env):
    """obs = (rank, number of env steps this env has taken so far): the obs of a worker's j-th transition is (rank, j),
    across episodes AND fragments, so the batches show directly whether a transition was lost or used twice. Rank 0
    steps slowly (the straggler). Episodes: terminated every 11th step, truncated at 7 steps.

    Parameters
    ----------
    rank : int
        Worker rank, the first observation entry.
    slow, fast : float, optional
        Sleep per step [s] of rank 0 and of the other ranks.

    Attributes
    ----------
    step_sleep : float
        Sleep per step [s].
    total_steps : int
        Steps taken since construction (the observation token).
    episode_steps : int
        Steps taken in the current episode.
    """

    def __init__(self, rank, slow=0.02, fast=0.001):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.rank, self.step_sleep, self.total_steps = rank, (slow if rank == 0 else fast), 0

    def _obs(self):
        """Observation (rank, total_steps)."""
        return np.array([self.rank, self.total_steps], np.float32)

    def reset(self, seed=None, options=None):
        """Start an episode; the step token keeps counting."""
        super().reset(seed=seed)
        self.episode_steps = 0
        return self._obs(), {}

    def step(self, action):
        """Sleep, advance both counters; reward -|action|^2."""
        time.sleep(self.step_sleep)
        self.total_steps += 1
        self.episode_steps += 1
        return self._obs(), float(-np.square(action).sum()), self.total_steps % 11 == 0, self.episode_steps >= 7, {}


def make_token_env(rank=0, seed=0, eval=False, **kwargs):
    """Factory of TokenEnv with the trainer's make_env signature (seed and eval are unused)."""
    return TokenEnv(max(rank, 0), **kwargs)


TOKEN = f"{__name__}:make_token_env"


def test_preemption_uses_every_transition_once(tmp_path):
    """Real spawn workers with preemption: the straggler (rank 0) is stopped before T, the others run past T; over
    several iterations every worker's transitions 0 .. N-1 appear in the batches exactly once, in order, and the
    learner reproduces every behaviour log-prob."""
    num_workers, num_steps = 4, 32
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "tok"), env=TOKEN, workers=num_workers, rollout=num_steps,
                       preempt_frac=0.75, epochs=2, minibatch=64, hidden=(32, 32), eval_every=0, ckpt_every=0,
                       device="cpu", curriculum="none", obs_clip=1e6)
    algo = ap.AsyncPPO(cfg)
    seen, lengths = {worker: [] for worker in range(num_workers)}, []
    try:
        algo.setup()
        for _ in range(5):  # obs statistics stay at the identity, so the batch obs are the raw (rank, j) tokens
            algo.shared_np[:] = parameters_to_vector(algo.actor.parameters()).detach().cpu().numpy()
            results = algo._collect(None)
            assert [result["rank"] for result in results] == list(range(num_workers))
            batch = algo._batch(results)
            lengths.append(batch["lengths"].tolist())
            assert batch["n"] == sum(batch["lengths"]) == len(batch["obs"])
            tok = batch["obs"].cpu().numpy().astype(np.int64)
            for worker in range(num_workers):
                seen[worker].extend(tok[tok[:, 0] == worker, 1].tolist())
            ppo = algo._update(batch, 3e-4)  # the weights change between iterations
            assert ppo["logp_err"] < 1e-5
    finally:
        algo.close()
    print("fragment lengths per iteration:", lengths)
    for worker in range(num_workers):
        assert seen[worker] == list(range(len(seen[worker]))), worker  # exactly once, in order, no gaps
    frag_lengths = np.array(lengths)
    assert ((frag_lengths[:, 0] < num_steps).all() and (frag_lengths[:, 1:] >= num_steps).all()
            and (frag_lengths <= 2 * num_steps).all())
    assert not multiprocessing.active_children()


def _old_fixed_T_batch(algo, results):
    """The trainer's batch construction before preemption (fixed T, stacked (T, W) arrays), kept as the reference for
    the preempt_frac = 1 path: the statements of the old trainer, with descriptive variable names.

    Parameters
    ----------
    algo : AsyncPPO
        Trainer whose critic, return accumulators and reward statistics are used (and updated).
    results : list of dict
        One fixed-length fragment per worker, as returned by RolloutWorker.rollout.

    Returns
    -------
    dict
        Batch in the format of AsyncPPO._batch.
    """
    cfg, num_steps, num_workers, obs_dim = algo.cfg, algo.cfg.rollout, len(results), algo.obs_dim
    stack = lambda key: np.stack([result[key] for result in results], 1)  # noqa: E731
    obs, act, logp, rew, term, trunc = (stack(key) for key in ("obs", "act", "logp", "rew", "term", "trunc"))
    values = algo._value(obs.reshape(num_steps * num_workers, obs_dim)).reshape(num_steps, num_workers)
    last_v = algo._value(np.stack([result["next_obs"] for result in results]))
    # Values of the final observations of truncated episodes, scattered to their (step, worker) cells.
    final_v = np.zeros((num_steps, num_workers))
    fv = algo._value(np.concatenate([result["final_obs"] for result in results]))
    offset = 0
    for worker, result in enumerate(results):
        count = len(result["final_t"])
        final_v[result["final_t"], worker] = fv[offset:offset + count]
        offset += count
    # Reward scaling by the running standard deviation of the discounted return.
    ranks = np.array([result["rank"] for result in results])
    acc, rets = algo.ret_acc[ranks].copy(), np.zeros((num_steps, num_workers))
    for step in range(num_steps):
        acc = acc * cfg.gamma + rew[step]
        rets[step] = acc
        acc[term[step] | trunc[step]] = 0.0
    algo.ret_acc[ranks] = acc
    algo.ret_rms.update(rets.ravel())
    rew_s = np.clip(rew / np.sqrt(algo.ret_rms.var + 1e-8), -cfg.reward_clip, cfg.reward_clip)
    adv, ret = ap.gae(rew_s, values, last_v, term, trunc, final_v, cfg.gamma, cfg.lam)
    dev = lambda array, *shape: torch.as_tensor(  # noqa: E731
        array, dtype=torch.float32, device=algo.device).reshape(num_steps * num_workers, *shape)
    return dict(obs=dev(obs, obs_dim), act=dev(act, algo.act_dim), logp=dev(logp), adv=dev(adv), ret=dev(ret),
                values=values.ravel(), returns=ret.ravel(), n=num_steps * num_workers,
                rew_scale=float(np.sqrt(algo.ret_rms.var)))


@pytest.mark.parametrize("device", DEVICES)
def test_preempt_one_reproduces_fixed_T_bit_for_bit(tmp_path, device):
    """preempt_frac = 1: fragments are exactly T steps (a preemptible worker that is never stopped runs to T_max) and
    the batch (obs, actions, log-probs, advantages, returns, reward-scaling state) equals the old fixed-T code's bit
    for bit, over consecutive iterations (accumulators carried across fragments, episodes crossing them)."""
    num_workers, num_steps = 3, 90
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "bits"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"},
                       workers=num_workers, rollout=num_steps, preempt_frac=1.0, hidden=(32, 32), device=device,
                       curriculum="none")
    new, old = ap.AsyncPPO(cfg), ap.AsyncPPO(cfg)  # same seed -> same critic
    net = dict(new.net_cfg)
    flat = parameters_to_vector(new.actor.parameters()).detach().cpu().numpy()
    mean, var = np.array([0.1, -0.2, 0.5]), np.array([0.4, 0.6, 2.0])
    fixed_worker, twin = (ap.RolloutWorker(0, GYM, {"env_id": "Pendulum-v1"}, 0, net) for _ in range(2))
    fixed, pre = (fixed_worker.rollout(flat, mean, var, 5.0, None, num_steps),
                  twin.rollout(flat, mean, var, 5.0, None, num_steps, 2 * num_steps, lambda: False))
    assert len(fixed["rew"]) == num_steps and len(pre["rew"]) == 2 * num_steps
    for key in ("obs", "act", "logp", "rew", "term", "trunc"):  # an unstopped preemptible fragment starts identically
        assert np.array_equal(fixed[key], pre[key][:num_steps]), key
    fixed_worker.close()
    twin.close()
    workers = [ap.RolloutWorker(rank, GYM, {"env_id": "Pendulum-v1"}, 0, net) for rank in range(num_workers)]
    for it in range(3):
        res = [worker.rollout(flat, mean, var, 5.0, None, num_steps) for worker in workers]
        assert all(len(result["rew"]) == num_steps for result in res)
        batch, ref = new._batch(res), _old_fixed_T_batch(old, res)
        for key in ("obs", "act", "logp", "adv", "ret"):
            assert torch.equal(batch[key], ref[key]), (it, key)
        for key in ("values", "returns"):
            assert np.array_equal(batch[key], ref[key]), (it, key)
        assert batch["n"] == ref["n"] and batch["rew_scale"] == ref["rew_scale"]
        assert np.array_equal(new.ret_acc, old.ret_acc) and new.ret_rms.var == old.ret_rms.var
    for worker in workers:
        worker.close()


def test_fixed_T_runs_are_reproducible(tmp_path):
    """Two fixed-T (preempt_frac = 1) multi-process runs on a deterministic env end with identical weights."""
    finals = []
    for name in ("a", "b"):
        cfg = ap.PPOConfig(run_dir=str(tmp_path / name), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=2,
                           rollout=64, steps=256, preempt_frac=1.0, epochs=2, minibatch=64, hidden=(32, 32),
                           eval_every=0, ckpt_every=0, device="cpu", curriculum="none")
        ap.AsyncPPO(cfg).train()
        finals.append(torch.load(os.path.join(cfg.run_dir, "last.pt"), map_location="cpu", weights_only=False))
    for net_name in ("actor", "critic"):
        for name, tensor in finals[0][net_name].items():
            assert torch.equal(tensor, finals[1][net_name][name]), (net_name, name)


def test_preempt_smoke_multiprocess(tmp_path):
    """Default preemption + KL-adaptive lr + stochastic evaluation through the whole pipeline; resume keeps the lr."""
    run = str(tmp_path / "pre")
    cfg = ap.PPOConfig(run_dir=run, env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=3, rollout=96, steps=1500,
                       epochs=3, minibatch=96, hidden=(64, 64), eval_every=700, n_eval=3, n_eval_workers=2,
                       eval_stochastic=True, ckpt_every=0, device=DEVICES[-1], curriculum="none")
    ap.AsyncPPO(cfg).train()
    iters, evals = _iters(run)
    steps = [record["step"] for record in iters]
    assert (steps == sorted(steps) and steps[-1] >= 1500
            and all(96 * 3 <= record["n"] <= 2 * 96 * 3 for record in iters))
    assert max(record["ppo"]["logp_err"] for record in iters) < 1e-5
    assert all(record["frag_max"] <= 192 and record["barrier_eff"] > 0 for record in iters)
    lrs = [record["ppo"]["lr"] for record in iters]
    for prev_lr, next_lr in zip(lrs, lrs[1:]):  # x1.5, /1.5 or unchanged, within the bounds
        assert (any(np.isclose(next_lr, candidate)
                    for candidate in (prev_lr, prev_lr * 1.5, prev_lr / 1.5, cfg.lr_min, cfg.lr_max))
                and 1e-5 <= next_lr <= 1e-3)
    with open(os.path.join(run, "metrics.jsonl")) as handle:
        kinds = [json.loads(line)["type"] for line in handle]
    assert kinds.count("eval") >= 2 and kinds.count("eval_stochastic") == kinds.count("eval")
    assert os.path.exists(os.path.join(run, "eval_stoch_history.json")) and os.path.exists(os.path.join(run, "best.pt"))
    ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
    assert ck["obs_count"] == ck["global_step"] == steps[-1]
    algo = ap.AsyncPPO(ap.PPOConfig(**{**cfg.__dict__, "resume": os.path.join(run, "last.pt")}))
    assert algo.lr == pytest.approx(ck["lr"]) and algo.global_step == steps[-1]
    assert not multiprocessing.active_children()


# ============================================================================================ optimisation rules
def test_kl_adaptive_lr_rule():
    """adapt_lr: x1.5 below target / 2, /1.5 above 2 target (an infinite KL counts as large), unchanged in band, for
    a NaN KL or without a target, and always clipped to [lr_min, lr_max]."""
    target = 0.01
    assert ap.adapt_lr(3e-4, 0.001, target, 1e-5, 1e-3) == pytest.approx(4.5e-4)  # KL < target / 2: x1.5
    assert ap.adapt_lr(3e-4, 0.03, target, 1e-5, 1e-3) == pytest.approx(2e-4)  # KL > 2 target: /1.5
    assert ap.adapt_lr(3e-4, 0.012, target, 1e-5, 1e-3) == 3e-4  # in band: unchanged
    assert (ap.adapt_lr(9e-4, 0.0, target, 1e-5, 1e-3) == 1e-3
            and ap.adapt_lr(1.2e-5, 1.0, target, 1e-5, 1e-3) == 1e-5)
    assert ap.adapt_lr(3e-4, math.inf, target, 1e-5, 1e-3) == pytest.approx(2e-4)  # an overflowing ratio: slow down
    assert (ap.adapt_lr(3e-4, float("nan"), target, 1e-5, 1e-3) == 3e-4
            and ap.adapt_lr(3e-4, 0.0, None, 1e-5, 1e-3) == 3e-4)
    lr = 3e-4  # a run of large-KL iterations walks down to the floor and stays there
    for _ in range(30):
        lr = ap.adapt_lr(lr, 0.5, target, 1e-5, 1e-3)
    assert lr == 1e-5


def _synthetic_batch(algo, batch_size=512, seed=0, logp_shift=0.0):
    """Random PPO batch in the format of AsyncPPO._batch, with log-probs from the current policy.

    Parameters
    ----------
    algo : AsyncPPO
        Trainer that provides the dimensions and the actor.
    batch_size : int, optional
        Number of transitions.
    seed : int, optional
        Seed of the torch generator.
    logp_shift : float, optional
        Added to the stored log-probs [nat]; negative values make the current policy look far from the behaviour one.

    Returns
    -------
    dict
    """
    gen = torch.Generator().manual_seed(seed)
    obs, act = (torch.randn(batch_size, algo.obs_dim, generator=gen),
                torch.randn(batch_size, algo.act_dim, generator=gen))
    with torch.no_grad():
        logp = algo.actor.dist(obs).log_prob(act).sum(-1) + logp_shift
    return dict(obs=obs, act=act, logp=logp, adv=torch.randn(batch_size, generator=gen),
                ret=torch.randn(batch_size, generator=gen), values=np.zeros(batch_size),
                returns=np.linspace(-1, 1, batch_size), n=batch_size, rew_scale=1.0)


def test_critic_keeps_training_after_policy_kl_stop(tmp_path):
    """Stored log-probs 1 nat below the current policy's: KL(ratio) = e - 2 = 0.72 >> kl_stop * target_kl on the
    first minibatch, so the policy must not move at all, while the critic still takes every one of its
    epochs x minibatches steps."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "ks"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=1,
                       epochs=4, minibatch=128, hidden=(32, 32), device="cpu", curriculum="none")
    algo = ap.AsyncPPO(cfg)
    batch = _synthetic_batch(algo, logp_shift=-1.0)
    actor_before = parameters_to_vector(algo.actor_params).detach().clone()
    critic_before = parameters_to_vector(algo.critic_params).detach().clone()
    out = algo._update(batch, 3e-4)
    assert out["early_stop"] and out["updates"] == 0 and out["critic_updates"] == 4 * 4
    assert out["approx_kl"] == pytest.approx(math.e - 2, rel=1e-4)
    assert torch.equal(parameters_to_vector(algo.actor_params), actor_before)
    assert not torch.equal(parameters_to_vector(algo.critic_params), critic_before)
    # Without the stop both networks train, and each gradient norm is clipped on its own.
    algo2 = ap.AsyncPPO(cfg)
    out2 = algo2._update(_synthetic_batch(algo2), 3e-4)
    assert not out2["early_stop"] and out2["updates"] == out2["critic_updates"] == 16
    assert "grad_norm_actor" in out2 and "grad_norm_critic" in out2
    assert len(out2["act_out_dims"]) == algo2.act_dim and 0.0 <= out2["act_out"] <= 1.0


def test_early_stop_lowers_adaptive_lr(tmp_path):
    """A policy early stop must lower the adaptive lr. The mean minibatch KL averages the path from 0 up to the stop
    (about half the KL reached), so on its own it stays inside [target/2, 2 target] and the lr never came down while
    the policy stopped after ~1 of 8 epochs every iteration."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "lr"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=1,
                       epochs=8, minibatch=256, hidden=(64, 64), device="cpu", curriculum="none")
    algo = ap.AsyncPPO(cfg)
    batch = _synthetic_batch(algo, batch_size=4096)
    batch["adv"] = batch["adv"].abs() * torch.sign(batch["act"][:, 0])  # a consistent signal: the policy moves fast
    algo.lr = 1e-3
    out = algo._update(batch, algo.lr)
    assert out["early_stop"] and out["epochs_done"] < cfg.epochs
    assert 0.5 * cfg.target_kl <= out["approx_kl"] <= 2.0 * cfg.target_kl  # the mean alone: "in band"
    assert ap.adapt_lr(algo.lr, out["approx_kl"], cfg.target_kl, cfg.lr_min, cfg.lr_max) == algo.lr
    assert algo._adapted_lr(out) == pytest.approx(1e-3 / 1.5)
    assert algo._adapted_lr(dict(early_stop=False, approx_kl=0.001)) == pytest.approx(1e-3)  # x1.5, capped at lr_max
    algo.cfg.lr_schedule = "linear"
    assert algo._adapted_lr(out) == algo.lr


def test_nonpositive_target_kl_disables_kl_control(tmp_path):
    """target_kl <= 0 (PPOConfig used directly, not through train_rl's flag) means off: a negative target used to be
    truthy, so kl_stop * target_kl < 0 stopped the policy on every first minibatch and the lr fell every iteration."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "off"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=1,
                       epochs=2, minibatch=128, hidden=(32, 32), device="cpu", curriculum="none", target_kl=-1.0)
    algo = ap.AsyncPPO(cfg)
    assert algo.cfg.target_kl is None
    out = algo._update(_synthetic_batch(algo, logp_shift=-1.0), 3e-4)  # KL 0.72: would stop with any target
    assert not out["early_stop"] and out["updates"] == 2 * 4
    assert algo._adapted_lr(out) == algo.lr


def test_min_std_floor(tmp_path):
    """The action std never drops below --min_std: in the distribution (workers, learner, Policy.sample) and in the
    parameter itself after optimiser steps that push it down."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "std"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=1,
                       epochs=2, minibatch=128, hidden=(32, 32), device="cpu", curriculum="none", min_std=0.2,
                       log_std_init=-3.0, target_kl=None)
    algo = ap.AsyncPPO(cfg)
    assert algo.actor.std().item() == pytest.approx(0.2)
    with torch.no_grad():
        algo.actor.log_std.fill_(-4.0)
    assert algo.actor.dist(torch.zeros(1, 3)).stddev.item() == pytest.approx(0.2)
    algo._update(_synthetic_batch(algo), 1e-2)
    assert algo.actor.log_std.item() >= math.log(0.2) - 1e-7
    pol = ap.Policy(algo._state())
    assert pol.actor.std().item() >= 0.2 - 1e-7


# ============================================================================================ gamma lock, metrics
class GammaEnv(ap.SleepEnv):
    """SleepEnv that declares its discount as a class attribute (gamma = 0.95)."""

    gamma = 0.95


def make_gamma_env(rank=0, seed=0, eval=False, **kwargs):
    """GammaEnv inside a TimeLimit wrapper, so the trainer must find gamma through the wrapper chain."""
    env = GammaEnv(obs_dim=3, act_dim=1, base=0.0, jitter=0.0)
    env.reset(seed=seed)
    return gym.wrappers.TimeLimit(env, 1000)  # found through the wrapper chain


def test_gamma_lock(tmp_path):
    """The trainer refuses a gamma that differs from the env's declared one (class attribute or module-level GAMMA);
    envs that declare none are not checked."""
    env = f"{__name__}:make_gamma_env"
    with pytest.raises(ValueError, match="gamma mismatch"):
        ap.AsyncPPO(ap.PPOConfig(run_dir=str(tmp_path / "g"), env=env, gamma=0.99, device="cpu"))
    assert ap.AsyncPPO(ap.PPOConfig(run_dir=str(tmp_path / "g"), env=env, gamma=0.95, device="cpu")).env_gamma == 0.95
    # A module-level GAMMA (how SurfaceRouteEnv's shaping gets it) is found too; envs without one are unchecked.
    mod = type(sys)("fake_env_module")
    mod.GAMMA = 0.99
    sys.modules["fake_env_module"] = mod
    try:
        Fake = type("Fake", (ap.SleepEnv,), {"__module__": "fake_env_module"})
        assert ap.env_gamma(Fake()) == 0.99 and ap.env_gamma(ap.SleepEnv()) is None
    finally:
        del sys.modules["fake_env_module"]
    assert ap.env_gamma(ap.make_gym_env()) is None


class TermsEnv(gym.Env):
    """Reward = sum of random parts reported in info['r_terms']; potential info['phi'] at reset and every step; every
    third episode ends in a 'bad' step (simulator failure: -10 'bad' term, exploded phi, truncated).

    Attributes
    ----------
    episodes : list of dict
        Ground truth per episode: summed reward terms ("terms"), "phi0", "phi_end" and the step rewards ("rews").
    n_ep : int
        Episodes started.
    episode_steps : int
        Steps taken in the current episode.
    """

    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.episodes, self.n_ep = [], 0

    def reset(self, seed=None, options=None):
        """Start an episode with a random potential and open its ground-truth record."""
        super().reset(seed=seed)
        self.episode_steps, self.n_ep = 0, self.n_ep + 1
        self.phi = float(self.np_random.normal())
        self.episodes.append(dict(terms=collections.defaultdict(float), phi0=self.phi, rews=[]))
        return np.zeros(2, np.float32), dict(phi=self.phi, start_mode="full")

    def step(self, action):
        """Random reward terms; the 9th step of every third episode is a simulator failure."""
        self.episode_steps += 1
        bad = self.n_ep % 3 == 0 and self.episode_steps == 9
        terms = dict(shaping=float(self.np_random.normal()), smooth=-float(self.np_random.random()), time=-0.01,
                     bad=-10.0 if bad else 0.0)
        if not bad:
            self.phi = float(self.np_random.normal())
            self.episodes[-1]["phi_end"] = self.phi
        for name, value in terms.items():
            self.episodes[-1]["terms"][name] += value
        reward = sum(terms.values())
        self.episodes[-1]["rews"].append(reward)
        info = dict(r_terms=terms, phi=1e6 if bad else self.phi, max_stage=0, start_mode="full", sim_diverged=bad)
        return np.zeros(2, np.float32), reward, False, bad or self.episode_steps >= 13, info


def test_episode_reward_terms_and_progress():
    """Per-episode sums of info['r_terms'] equal the sums of the env's step terms (and add up to the return); phi at
    start / last valid step, the discounted return and the simulator-failure flag are recorded."""
    net = dict(obs_dim=2, act_dim=1, hidden=[16], log_std_init=0.0)
    worker = ap.RolloutWorker(0, lambda **_: TermsEnv(), {}, 0, net, gamma=0.9)
    out = worker.rollout(parameters_to_vector(ap.Actor(**net).parameters()).detach().numpy(), np.zeros(2),
                         np.ones(2), 10.0, None, 200)
    truth = worker.env.episodes
    assert len(out["episodes"]) >= 10
    for episode, ref in zip(out["episodes"], truth):
        assert episode["r_terms"].keys() == ref["terms"].keys()
        for name, value in ref["terms"].items():
            assert episode["r_terms"][name] == pytest.approx(value, abs=1e-12)
        assert sum(episode["r_terms"].values()) == pytest.approx(episode["ret"], abs=1e-9)
        assert episode["phi0"] == ref["phi0"] and episode["phi_end"] == ref["phi_end"]  # never the exploded 1e6
        assert episode["disc_ret"] == pytest.approx(sum(0.9**step * reward for step, reward in enumerate(ref["rews"])),
                                                    abs=1e-12)
        assert episode["sim_fail"] == (ref["terms"]["bad"] < 0) and episode["len"] == len(ref["rews"])
    assert (any(episode["sim_fail"] for episode in out["episodes"])
            and not all(episode["sim_fail"] for episode in out["episodes"]))
    worker.close()


def test_episode_metrics_headline(tmp_path):
    """Per-mode success / grasp (full and hover starts only) / stage histogram / sim failures from episode records."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "m"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, device="cpu")
    algo = ap.AsyncPPO(cfg)
    algo.has_modes = True
    record = lambda mode, stage, success=False, sim_fail=False: dict(  # noqa
        ret=1.0, len=10, success=success, mode=mode, max_stage=stage, error=False, sim_fail=sim_fail, disc_ret=0.5,
        phi0=-2.0, phi_end=-1.0, r_terms=dict(shaping=1.0, bad=-10.0 if sim_fail else 0.0))
    episodes = [record("full", 0), record("full", 1), record("full", 4, True), record("hover", 2, sim_fail=True),
                record("grasped", 3), record("grasped", 4, True)]
    metrics = algo._episode_metrics(episodes)
    assert metrics["succ100"] == pytest.approx(2 / 6) and metrics["succ_full"] == pytest.approx(1 / 3)
    assert metrics["grasp100"] == pytest.approx(3 / 4)  # full + hover only
    assert metrics["stage100"] == pytest.approx(14 / 6) and metrics["dphi100"] == pytest.approx(1.0)
    assert metrics["simfail100"] == pytest.approx(1 / 6) and metrics["r_terms100"]["bad"] == pytest.approx(-10 / 6)
    assert (metrics["modes"]["full"]["stage_hist"] == {0: 1, 1: 1, 4: 1}
            and metrics["modes"]["grasped"]["grasp"] is None)
    assert metrics["modes"]["hover"]["simfail"] == 1.0


def test_best_checkpoint_order():
    """best.pt: deterministic success first, then the lower mean final shape error."""
    assert ap._eval_key(0.5, 0.2) > ap._eval_key(0.4, 0.01)
    assert ap._eval_key(0.5, 0.03) > ap._eval_key(0.5, 0.04) > ap._eval_key(0.5, None)


# ============================================================================================ running statistics
def test_running_stats_parallel_merge():
    """Merging per-chunk moments (what the learner does with the workers' statistics), batch updates and the scalar
    reward-scaling version all equal the moments of the concatenated data."""
    rng = np.random.default_rng(2)
    dim = 7
    sizes = (1, 9, 256, 3, 1000, 2)
    chunks = [rng.normal(5 * rng.normal(size=dim), rng.uniform(0.1, 3.0, dim), size=(size, dim)) for size in sizes]
    rms = ap.RunningMeanStd((dim,))
    for chunk in chunks:  # what the learner does with the per-worker moments of an iteration
        rms.merge(chunk.mean(0), chunk.var(0), len(chunk))
    allx = np.concatenate(chunks)
    np.testing.assert_allclose(rms.mean, allx.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rms.var, allx.var(0), rtol=1e-10)
    assert rms.count == len(allx)
    rms2 = ap.RunningMeanStd((dim,))  # batch updates from raw data agree as well
    for chunk in chunks:
        rms2.update(chunk)
    np.testing.assert_allclose(rms2.var, allx.var(0), rtol=1e-10)
    ret = ap.RunningMeanStd(())  # scalar version (reward scaling)
    for chunk in chunks:
        ret.update(chunk[:, 0])
    assert ret.mean == pytest.approx(allx[:, 0].mean()) and ret.var == pytest.approx(allx[:, 0].var())


# ============================================================================================ worker vs learner
def _learner_logp(actor, out, device, dtype=torch.float32):
    """Log-probs of a fragment's stored actions under a copy of ``actor`` on ``device`` in ``dtype``.

    Parameters
    ----------
    actor : Actor
        Policy network (not modified).
    out : dict
        Fragment from RolloutWorker.rollout ("obs" normalised, "act").
    device : str or torch.device
        Evaluation device.
    dtype : torch.dtype, optional
        Evaluation precision.

    Returns
    -------
    np.ndarray of float64, shape (T,)
    """
    local_actor = copy.deepcopy(actor).to(device=device, dtype=dtype)
    with torch.no_grad():
        return local_actor.dist(torch.as_tensor(out["obs"], device=device, dtype=dtype)).log_prob(
            torch.as_tensor(out["act"], device=device, dtype=dtype)).sum(-1).cpu().double().numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_worker_logprob_matches_learner(device):
    """The learner (any device) must reproduce the worker's stored log-probs from the stored normalised obs."""
    net = dict(obs_dim=3, act_dim=1, hidden=[512, 512, 256], log_std_init=-0.3)
    torch.manual_seed(0)
    learner = ap.Actor(**net)
    with torch.no_grad():  # non-trivial mean head and log-std so the check is not vacuous
        for param in learner.parameters():
            param.add_(0.05 * torch.randn_like(param))
    learner.to(device)
    flat = parameters_to_vector(learner.parameters()).detach().cpu().numpy()
    worker = ap.RolloutWorker(0, GYM, {"env_id": "Pendulum-v1"}, 0, net, keep_raw=True)
    mean, var, clip = np.array([0.1, -0.2, 0.5]), np.array([0.4, 0.6, 2.0]), 3.0
    out = worker.rollout(flat, mean, var, clip, None, 300)  # crosses the 200-step time limit
    lp, lp64 = _learner_logp(learner, out, device), _learner_logp(learner, out, "cpu", torch.float64)
    err = np.abs(lp - out["logp"]).max()
    print(f"[{device}] max |logp_learner - logp_worker| = {err:.2e}; vs float64: worker "
          f"{np.abs(out['logp'] - lp64).max():.1e}, learner {np.abs(lp - lp64).max():.1e}")
    assert err < 1e-5
    assert np.abs(out["logp"] - lp64).max() < 1e-5 and np.abs(lp - lp64).max() < 1e-5
    assert np.abs(out["act"]).max() > 0.1 and np.std(out["act"]) > 0.3  # actually stochastic
    np.testing.assert_array_equal(out["obs"], ap.normalize_obs(out["raw"], mean, var, clip))
    assert (np.abs(out["obs"]) == clip).any()  # clipping was exercised
    # The 200-step time limit is a TRUNCATION with its final obs kept for bootstrapping (not a termination).
    assert out["trunc"][199] and out["trunc"].sum() == 1 and not out["term"].any()
    assert list(out["final_t"]) == [199] and out["final_obs"].shape == (1, 3)
    assert len(out["episodes"]) == 1 and out["episodes"][0]["len"] == 200
    np.testing.assert_allclose(out["episodes"][0]["ret"], out["rew"][:200].sum())
    np.testing.assert_allclose(out["raw_mean"], out["raw"].mean(0))
    np.testing.assert_allclose(out["raw_var"], out["raw"].var(0))
    # Env, current obs and episode persist across fragments.
    out2 = worker.rollout(flat, mean, var, clip, None, 150)
    assert list(out2["final_t"]) == [99] and out2["episodes"][0]["len"] == 200
    worker.close()


@pytest.mark.parametrize("device", DEVICES)
def test_sharp_policy_logprob_is_fp32_roundoff(device):
    """A trained policy is sharp (small std, large |mu|), so float32 round-off in mu, ~1e-6 * |mu| (a few ulp), is
    amplified by d logp / d mu = (a - mu) / std^2. Worker and learner must both equal the exact (float64) log-density
    within that round-off model; a weights/normalisation bug would miss it by orders of magnitude."""
    net = dict(obs_dim=3, act_dim=1, hidden=[512, 512, 256], log_std_init=-1.6)
    torch.manual_seed(1)
    actor = ap.Actor(**net)
    with torch.no_grad():
        actor.net[-1].weight.mul_(1500.0)  # |mu| up to ~12, like the trained Pendulum policy (7.3)
    worker = ap.RolloutWorker(0, GYM, {"env_id": "Pendulum-v1"}, 5, net)
    flat = parameters_to_vector(actor.parameters()).detach().numpy()
    out = worker.rollout(flat, np.zeros(3), np.array([0.5, 0.5, 4.0]), 5.0, None, 1000)
    lp, lp64 = _learner_logp(actor, out, device), _learner_logp(actor, out, "cpu", torch.float64)
    with torch.no_grad():
        mu = actor.double()(torch.as_tensor(out["obs"], dtype=torch.float64)).numpy()[:, 0]
    std = float(np.exp(net["log_std_init"]))
    slope = (np.abs(out["act"][:, 0] - mu) / std**2).max()
    tol = 1e-6 * np.abs(mu).max() * slope + 1e-6
    err_worker, err_learner, gap = (np.abs(out["logp"] - lp64).max(), np.abs(lp - lp64).max(),
                                    np.abs(lp - out["logp"]).max())
    print(f"[{device}] sharp policy (std {std:.2f}, max|mu| {np.abs(mu).max():.1f}, max dlogp/dmu {slope:.0f}): "
          f"vs float64 worker {err_worker:.1e}, learner {err_learner:.1e}; worker-learner gap {gap:.1e}; "
          f"round-off bound {tol:.1e}")
    assert np.abs(mu).max() > 5.0 and err_worker < tol and err_learner < tol and gap < 2 * tol
    worker.close()


# ============================================================================================ robustness
class FlakyEnv(gym.Env):
    """Raises or returns NaNs at random, like a diverging physics solver.

    Parameters
    ----------
    p_raise : float, optional
        Probability that a step raises.
    p_nan_obs : float, optional
        Probability of a NaN observation with the finite -10 penalty (a blow-up).
    p_nan_rew : float, optional
        Probability of a NaN reward.
    p_reset_raise : float, optional
        Probability that a reset raises.

    Attributes
    ----------
    probs : tuple of float
        (p_raise, p_nan_obs, p_nan_rew, p_reset_raise).
    episode_steps : int
        Steps taken in the current episode (truncation at 50).
    """

    def __init__(self, p_raise=0.03, p_nan_obs=0.03, p_nan_rew=0.02, p_reset_raise=0.1):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.probs = (p_raise, p_nan_obs, p_nan_rew, p_reset_raise)

    def reset(self, seed=None, options=None):
        """Random observation, or RuntimeError with probability p_reset_raise."""
        super().reset(seed=seed)
        if self.np_random.random() < self.probs[3]:
            raise RuntimeError("reset diverged")
        self.episode_steps = 0
        return self.np_random.normal(size=4).astype(np.float32), {}

    def step(self, action):
        """One uniform draw selects the outcome: raise, NaN observation, NaN reward, or a normal step (reward 1)."""
        draw = self.np_random.random()
        if draw < self.probs[0]:
            raise RuntimeError("solver diverged")
        self.episode_steps += 1
        obs, reward = self.np_random.normal(size=4).astype(np.float32), 1.0
        if draw < self.probs[0] + self.probs[1]:  # blow-up: NaN state with a finite penalty, as in GraspRouteEnv
            obs[1], reward = np.nan, -10.0
        elif draw < self.probs[0] + self.probs[1] + self.probs[2]:
            reward = float("nan")
        return obs, reward, False, self.episode_steps >= 50, {"is_success": False}


def make_flaky_env(rank=0, seed=0, eval=False, **kwargs):
    """Factory of FlakyEnv with the trainer's make_env signature (rank, seed and eval are unused)."""
    return FlakyEnv(**kwargs)


def test_worker_survives_env_errors():
    """A worker on FlakyEnv keeps collecting: every exception or non-finite value ends the episode as a truncation
    with finite stored data, and the rewards of errored steps follow the documented rule."""
    net = dict(obs_dim=4, act_dim=2, hidden=[32, 32], log_std_init=0.0)
    worker = ap.RolloutWorker(3, make_flaky_env, {}, 0, net)
    flat = parameters_to_vector(ap.Actor(**net).parameters()).detach().numpy()
    out = worker.rollout(flat, np.zeros(4), np.ones(4), 10.0, None, 3000)
    bad = [episode for episode in out["episodes"] if episode["error"]]
    print(f"env errors {out['errors']}, errored episodes {len(bad)} of {len(out['episodes'])}")
    assert out["errors"] > 50 and len(bad) > 50
    assert np.isfinite(out["obs"]).all() and np.isfinite(out["rew"]).all() and np.isfinite(out["final_obs"]).all()
    # Every episode end is a truncation (no termination in this env) with a stored final obs.
    assert not out["term"].any() and out["trunc"].sum() == len(out["final_t"]) == len(out["episodes"])
    # An errored step keeps a finite env reward (the -10 blow-up penalty of a NaN state) and counts a missing or
    # non-finite one (exception, NaN reward) as 0; good steps have reward 1.
    err_steps = out["final_t"][[episode["error"] for episode in out["episodes"]]]
    assert (set(np.unique(out["rew"][err_steps])) == {0.0, -10.0}
            and set(np.unique(out["rew"])) == {0.0, 1.0, -10.0})
    worker.close()


def test_resumed_workers_draw_fresh_streams():
    """Workers (re)started at iteration k > 0 (a resume) must not replay the episode seeds and action noise of the
    original run's first iterations; a fresh run (stream 0) stays reproducible."""
    net = dict(obs_dim=3, act_dim=1, hidden=[16], log_std_init=0.0)
    flat = parameters_to_vector(ap.Actor(**net).parameters()).detach().numpy()
    acts = []
    for stream in (0, 0, 7):
        worker = ap.RolloutWorker(0, GYM, {"env_id": "Pendulum-v1"}, 0, net, stream=stream)
        acts.append(worker.rollout(flat, np.zeros(3), np.ones(3), 10.0, None, 20)["act"])
        worker.close()
    assert np.array_equal(acts[0], acts[1]) and not np.array_equal(acts[0], acts[2])


def test_expert_is_blocked():
    """_forbid_expert makes both the env's expert object and its expert_action method raise (RL-only training)."""

    class ExpertEnv(gym.Env):
        """Env that exposes a scripted expert, the way the legacy environments did."""

        class _Expert:
            """Scripted expert stand-in."""

            def act(self):
                """A fixed action."""
                return np.zeros(5)

        def __init__(self):
            self.expert = self._Expert()

        def expert_action(self):
            """The expert's action."""
            return self.expert.act()

    env = ap._forbid_expert(ExpertEnv())
    with pytest.raises(RuntimeError, match="forbidden"):
        env.expert.act()
    with pytest.raises(RuntimeError, match="forbidden"):
        env.expert_action()


# ============================================================================================ multi-process smoke run
def test_trainer_smoke_multiprocess(tmp_path):
    """2 spawn workers, 3 iterations, non-blocking evaluation, checkpoints, resume, clean shutdown."""
    run = str(tmp_path / "smoke")
    cfg = ap.PPOConfig(run_dir=run, env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=2, rollout=128, steps=768,
                       epochs=2, minibatch=128, hidden=(64, 64), eval_every=512, n_eval=3, n_eval_workers=2,
                       ckpt_every=512, device=DEVICES[-1], curriculum="none", info_keys=("is_success",),
                       preempt_frac=1.0)  # fixed T-step fragments: exact step counts
    ap.AsyncPPO(cfg).train()
    iters, evals = _iters(run)
    assert [record["step"] for record in iters] == [256, 512, 768]
    assert max(record["ppo"]["logp_err"] for record in iters) < 1e-5
    assert all(record["workers"] == 2 and record["errors"] == 0 for record in iters)
    assert sorted(record["step"] for record in evals) == [512, 768] and all(record["n"] == 3 for record in evals)
    with open(os.path.join(run, "eval_history.json")) as handle:
        hist = json.load(handle)
    assert [entry["episodes"][0]["seed"] for entry in hist] == [ap.EVAL_SEED0] * 2
    assert all(entry["episodes"][0]["steps"] == 200 for entry in hist)
    for name in ("last.pt", "best.pt", "ckpt_512.pt", "ckpt_768.pt"):
        assert os.path.exists(os.path.join(run, name)), name
    ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
    assert ck["obs_count"] == 768 and ck["global_step"] == 768 and ck["env"]["factory"] == GYM
    pol = ap.load_policy(os.path.join(run, "best.pt"))
    action = pol(np.array([1.0, 0.0, 0.5]))
    assert action.shape == (1,) and action.dtype == np.float32 and np.abs(action).max() <= 1.0
    assert pol(np.zeros((5, 3))).shape == (5, 1) and np.abs(pol.sample(np.zeros((5, 3)))).max() <= 1.0
    assert not multiprocessing.active_children()  # every worker / eval process was reaped

    # Resume continues the step count, the optimiser and the statistics.
    cfg2 = ap.PPOConfig(**{**cfg.__dict__, "steps": 1024, "resume": os.path.join(run, "last.pt"), "eval_every": 0})
    ap.AsyncPPO(cfg2).train()
    iters, _ = _iters(run)
    assert [record["step"] for record in iters] == [256, 512, 768, 1024]
    ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
    assert ck["obs_count"] == 1024 and ck["iteration"] == 4
    assert not multiprocessing.active_children()


# ============================================================================================ fault injection
class FaultEnv(ap.SleepEnv):
    """Misbehaves once at step 40: 'crash' exits the process (like a simulator segfault), 'hang' blocks forever.

    The factory sets the attributes ``fault`` ("crash", "hang" or None) and ``marker`` (path of the file left behind
    when the fault fires).
    """

    def step(self, action):
        """SleepEnv.step, except at step 40 of a faulty instance."""
        if self.fault and self.elapsed_steps == 40:
            open(self.marker, "w").close()
            if self.fault == "crash":
                os._exit(3)
            time.sleep(3600)
        return super().step(action)


def make_fault_env(rank=0, seed=0, eval=False, fault="crash", marker="", fail_restart=False):
    """Rank 1 faults once; the marker file it leaves makes its restarted incarnation behave (or, with fail_restart,
    makes that incarnation's factory raise)."""
    restarted = os.path.exists(marker)
    if rank == 1 and restarted and fail_restart:
        raise RuntimeError("factory exploded after a restart")
    env = FaultEnv(obs_dim=3, act_dim=1, horizon=100, base=0.0, jitter=0.0)
    env.fault, env.marker = (fault if rank == 1 and not restarted else None), marker
    env.reset(seed=seed)
    return env


FAULT = f"{__name__}:make_fault_env"  # importable by the spawned workers (they inherit sys.path)


def _fault_cfg(tmp_path, name, **kwargs):
    """Small 2-worker PPOConfig on the fault-injection env; keyword arguments override the defaults."""
    return ap.PPOConfig(**{**dict(run_dir=str(tmp_path / name), env=FAULT, workers=2, rollout=64, epochs=1,
                                  minibatch=128, hidden=(32, 32), eval_every=0, ckpt_every=0, device="cpu",
                                  curriculum="none"), **kwargs})


@pytest.mark.parametrize("preempt", [1.0, 0.5])
@pytest.mark.parametrize("fault", ["crash", "hang"])
def test_dead_or_hung_worker_is_restarted(tmp_path, fault, preempt):
    """A crashed or hung simulator costs one fragment: the learner restarts that worker (a hung one after
    worker_timeout) and keeps training with every worker; no process is left behind. With preemption the dead worker
    drops out of the stop count (no deadlock); a hung one, which never reports, is restarted at the timeout."""
    cfg = _fault_cfg(tmp_path, fault, env_kwargs=dict(fault=fault, marker=str(tmp_path / "m")), steps=4 * 128,
                     worker_timeout=10.0, preempt_frac=preempt)
    ap.AsyncPPO(cfg).train()
    iters, _ = _iters(cfg.run_dir)
    if preempt == 1.0:
        assert [record["workers"] for record in iters] == [1, 2, 2, 2, 2]
    else:
        assert iters[0]["workers"] == 1 and all(record["workers"] == 2 for record in iters[1:]) and len(iters) >= 3
    assert iters[-1]["restarts"] == 1
    assert (iters[0]["sample_s"] > 9.0) == (fault == "hang")
    assert not multiprocessing.active_children()


def test_restart_failure_reports_root_cause(tmp_path):
    """A restarted worker that dies in its env factory before the next iteration: the abort message must carry the
    factory's traceback, not only the resulting 'send failed: BrokenPipeError'."""
    cfg = _fault_cfg(tmp_path, "rc", env_kwargs=dict(fault="crash", marker=str(tmp_path / "m"), fail_restart=True),
                     max_worker_fails=1)
    algo = ap.AsyncPPO(cfg)
    try:
        algo.setup()
        algo.iterate()  # worker 1 crashes and is restarted; its new incarnation dies in the factory
        deadline = time.time() + 120
        while algo.workers[1].proc.is_alive() and time.time() < deadline:
            time.sleep(0.2)
        with pytest.raises(RuntimeError, match="factory exploded"):
            algo.iterate()
    finally:
        algo.close()
    assert not multiprocessing.active_children()


def test_update_survives_nonfinite_gradient(tmp_path):
    """A finite loss can have a NaN gradient: exp(log_ratio) overflows to inf on a positive-advantage sample, the min
    picks the clipped branch, and its zero gradient times exp' = inf is NaN. That step must be skipped; the update
    must neither crash (Normal's argument check on a NaN mean) nor leave NaN weights."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "nan"), env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=1,
                       epochs=3, minibatch=64, hidden=(32, 32), device="cpu", target_kl=None, curriculum="none")
    algo = ap.AsyncPPO(cfg)
    gen = torch.Generator().manual_seed(0)
    batch_size = 256
    obs, act = torch.randn(batch_size, 3, generator=gen), torch.randn(batch_size, 1, generator=gen)
    with torch.no_grad():
        logp = algo.actor.dist(obs).log_prob(act).sum(-1)
    logp[0] = -1000.0  # ratio exp(~1000) = inf on sample 0
    adv = torch.randn(batch_size, generator=gen)
    adv[0] = 50.0
    batch = dict(obs=obs, act=act, logp=logp, adv=adv, ret=torch.randn(batch_size, generator=gen),
                 values=np.zeros(batch_size), returns=np.linspace(-1, 1, batch_size), n=batch_size, rew_scale=1.0)
    before = parameters_to_vector(algo.params).detach().clone()
    out = algo._update(batch, 1e-3)
    after = parameters_to_vector(algo.params).detach()
    assert torch.isfinite(after).all() and not out["rolled_back"]
    assert out["skipped"] == cfg.epochs and out["updates"] == 3 * cfg.epochs  # only the poisoned minibatches skipped
    assert not torch.equal(before, after)


def _exit_on_seed_3(seed):
    """Pool task that kills its worker process on seed 3 and succeeds otherwise."""
    if seed == 3:
        os._exit(1)  # like a simulator segfault inside an evaluation/verification pool worker
    return dict(seed=seed, success=True)


def test_pool_results_survives_dead_pool_worker():
    """multiprocessing.Pool never returns the task of a worker that died mid-episode (a plain imap_unordered waits
    forever): evaluation and verification must still finish, with that layout reported as a failed episode."""
    with ap._single_thread_env():
        pool = multiprocessing.get_context("spawn").Pool(2)
    with pool:
        res = sorted(ap.pool_results(pool, _exit_on_seed_3, list(range(6)), timeout=10),
                     key=lambda result: result["seed"])
    assert [result["seed"] for result in res] == list(range(6))
    assert ([result["success"] for result in res] == [True, True, True, False, True, True]
            and "no result" in res[3]["error"])


def test_hung_evaluation_is_killed(tmp_path):
    """An evaluation that never finishes (a hung simulator, or a Pool whose worker died mid-episode and so never
    returns that result) is killed after eval_timeout and logged as failed, and the queued evaluations still run."""
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "hung"), env="rope_routing.rl.async_ppo:make_sleep_env",
                       env_kwargs=dict(obs_dim=3, act_dim=1, base=60.0, jitter=0.0), workers=1, hidden=(32, 32),
                       n_eval=2, n_eval_workers=1, device="cpu", curriculum="none", eval_timeout=5.0)
    algo = ap.AsyncPPO(cfg)
    t0 = time.time()
    try:
        algo._queue_eval(algo._save_ckpt(), 1)
        algo._queue_eval(algo._save_ckpt(), 2)
        while (algo.eval_proc is not None or algo.eval_queue) and time.time() - t0 < 60:
            time.sleep(0.2)
            algo._poll_eval()
        assert algo.eval_proc is None and not algo.eval_queue and not algo.eval_history
        assert time.time() - t0 < 45
    finally:
        algo.close()
    assert not multiprocessing.active_children()


# ============================================================================================ heavy tests (d3)
@HEAVY
def test_pendulum_learns(tmp_path):
    """End-to-end learning through the full multi-process pipeline. Pendulum episodes are TRUNCATED at 200 steps,
    so this also exercises truncation bootstrapping."""
    run = os.environ.get("ASYNC_PPO_RUN_DIR") or str(tmp_path / "pendulum")
    num_workers, num_steps = 8, 256
    cfg = ap.PPOConfig(run_dir=run, env=GYM, env_kwargs={"env_id": "Pendulum-v1"}, workers=num_workers,
                       rollout=num_steps, steps=600_000 - 2 * num_workers * num_steps, gamma=0.95, lam=0.95, lr=1e-3,
                       epochs=10, minibatch=512, log_std_init=0.0, eval_every=200_000, n_eval=8, n_eval_workers=2,
                       ckpt_every=0, device=DEVICES[-1], curriculum="none")
    ap.AsyncPPO(cfg).train()
    iters, evals = _iters(run)
    print("learning curve (env steps, mean return of the last 100 training episodes):")
    for record in iters[::15] + [iters[-1]]:
        print(f"  {record['step']:>8,d}  {record['ret100']:8.1f}   std {record['ppo']['std']:.3f}  "
              f"lr {record['ppo']['lr']:.1e}  "
              f"kl {record['ppo'].get('approx_kl', float('nan')):.4f}  ep {record['ppo']['epochs_done']:.1f}  "
              f"frag {record['frag_min']}-{record['frag_max']}")
    for evaluation in evals:
        print(f"  eval @ {evaluation['step']:,}: deterministic mean return {evaluation['mean_return']:.1f} "
              f"on {evaluation['n']} held-out seeds")
    within = [record for record in iters if record["step"] <= 600_000]  # preempted iterations have ragged sizes
    print(f"  last iteration within 600k: step {within[-1]['step']:,}, ret100 {within[-1]['ret100']:.1f}, "
          f"lr {within[-1]['ppo']['lr']:.1e}")
    assert within[-1]["ret100"] > -300
    # Learner (GPU) vs worker (CPU) log-probs: float32 round-off only (it grows as the policy sharpens: ~4e-5 at
    # std 0.32 here; see test_sharp_policy_logprob_is_fp32_roundoff); a real mismatch would be >= 1e-2.
    assert max(record["ppo"]["logp_err"] for record in iters) < 1e-4


@HEAVY
def test_throughput_sleep_env(tmp_path):
    """32 workers on an env whose step sleeps 30 ms + U(0, 30) ms: ideal = 32 / 0.045 s = 711 steps/s. Lock-stepped
    envs would wait for the slowest of 32 every step (E[max] = 59.1 ms -> 76% of ideal)."""
    num_workers, num_steps, n_iter = 32, 128, 4
    cfg = ap.PPOConfig(run_dir=str(tmp_path / "sleep"), env="rope_routing.rl.async_ppo:make_sleep_env",
                       workers=num_workers, rollout=num_steps, steps=num_workers * num_steps * n_iter, epochs=1,
                       minibatch=num_workers * num_steps, eval_every=0, ckpt_every=0, device=DEVICES[-1],
                       curriculum="none")
    ap.AsyncPPO(cfg).train()
    iters, _ = _iters(cfg.run_dir)
    ideal = num_workers / 0.045
    # Preempted fragments are ragged: count the transitions actually collected.
    steps = sum(record["n"] for record in iters)
    sample_sps = steps / sum(record["sample_s"] for record in iters)
    wall_sps = steps / sum(record["sample_s"] + record["update_s"] for record in iters)
    print(f"ideal {ideal:.0f} steps/s | sampling {sample_sps:.0f} steps/s ({sample_sps / ideal:.1%}) | "
          f"wall incl. update {wall_sps:.0f} steps/s ({wall_sps / ideal:.1%}) | per iteration: "
          + ", ".join(f"{record['sample_s']:.2f}+{record['update_s']:.2f}s" for record in iters))
    assert sample_sps >= 0.85 * ideal


@HEAVY
@pytest.mark.parametrize("preempt", [1.0, 0.8])
def test_throughput_heavy_tailed(tmp_path, preempt):
    """The straggler benchmark: 32 workers, step time 70% 20 ms / 25% 60 ms / 5% U(1, 3) s (free step, contact step,
    start-state reset; mean 129 ms -> ideal 248 steps/s). Fixed T-step fragments wait for the slowest worker every
    iteration; preemption (0.8) must keep the workers >= 85% busy."""
    num_workers, num_steps, n_iter = 32, 128, 5
    cfg = ap.PPOConfig(run_dir=str(tmp_path / f"ht{preempt}"), env="rope_routing.rl.async_ppo:make_sleep_env",
                       env_kwargs=dict(mix=ap.HEAVY_TAIL, horizon=500), workers=num_workers, rollout=num_steps,
                       preempt_frac=preempt, steps=num_workers * num_steps * n_iter, epochs=1,
                       minibatch=num_workers * num_steps, eval_every=0, ckpt_every=0, device=DEVICES[-1],
                       curriculum="none")
    ap.AsyncPPO(cfg).train()
    iters, _ = _iters(cfg.run_dir)
    prob, lo, hi = np.asarray(ap.HEAVY_TAIL).T
    ideal = num_workers / float(np.sum(prob * (lo + hi) / 2))
    steps = sum(record["n"] for record in iters)
    sample_sps = steps / sum(record["sample_s"] for record in iters)
    wall_sps = steps / sum(record["sample_s"] + record["update_s"] for record in iters)
    eff = float(np.mean([record["barrier_eff"] for record in iters]))
    out = dict(preempt_frac=preempt, ideal_sps=ideal, sample_sps=sample_sps, wall_sps=wall_sps, barrier_eff=eff,
               barrier_effs=[record["barrier_eff"] for record in iters],
               frag=[(record["frag_min"], record["frag_max"]) for record in iters], steps=steps)
    print(json.dumps(out))
    if os.environ.get("ASYNC_PPO_BENCH_OUT"):
        with open(os.environ["ASYNC_PPO_BENCH_OUT"], "a") as handle:
            handle.write(json.dumps(out) + "\n")
    if preempt < 1.0:
        assert eff >= 0.85
