"""Asynchronous-rollout PPO: pure reinforcement learning for slow CPU simulators with a GPU learner.

Why not Stable-Baselines3's SubprocVecEnv: it steps all environments in LOCKSTEP, so every step waits for the slowest
environment. The contact-rich rope environment has a heavy-tailed step time, and 40 lock-stepped environments ran at
about 6 steps/s each, against 30-35 standalone. Here every worker process owns ONE environment and a CPU copy of the
actor and collects a rollout fragment on its own; the learner synchronises once per iteration:

    learner : actor weights -> shared memory; observation statistics and start-state probabilities -> pipes
    workers : environment steps, independently -> (normalised obs, action, log-prob, reward, terminated, truncated,
              final obs of truncated steps, next obs, episode statistics, moments of the raw obs)
    learner : V(s) on the GPU, return-based reward scaling, GAE(lambda), clipped PPO update, observation-statistics
              update, start-state curriculum, non-blocking evaluation, checkpoints, logging

Straggler preemption (DD-PPO, Wijmans et al. 2020). A contact step costs about 2.6 times a free one and start-state
resets take 1-6 s, so with fixed T-step fragments the workers idled about 43% of every sampling phase waiting for the
slowest one. Instead, every worker keeps stepping (up to rollout_max = 2T steps) until the learner tells it to stop,
which the learner does once at least preempt_frac of the workers have collected T steps. Fragments are therefore
ragged: GAE and the reward-scaling accumulators run per worker (on padded arrays), and the batch is the time-major
concatenation. The algorithm stays exactly on-policy: one set of weights and observation statistics per iteration,
every transition used once, behaviour log-probabilities stored. preempt_frac = 1 is the fixed-T scheme, bit for bit.

The ONLY learning signal is the environment reward: no demonstrations, no behaviour cloning, no imitation loss, and
every environment built here has its privileged scripted expert disabled (:func:`_forbid_expert`: any call raises).

Environment factory contract ("module:function"):
``factory(rank: int, seed: int, eval: bool = False, **env_kwargs) -> gym.Env``, with a Box action space in [-1, 1]
(the environment clips) and a flat Box observation. Optional:

    env.set_start_probs({"full": p, "hover": p, "grasped": p})   start-state curriculum, applied at resets (any
                                                                  wrapper level)
    env.gamma, or a module-level GAMMA in the env's module        the discount its potential-based shaping assumes; it
                                                                  must equal the trainer's gamma (checked at start-up)
    info["is_success"], info["max_stage"], info["start_mode"], info["shape"], info["dropped"]    logging, evaluation
    info["r_terms"] (reward parts, summed per episode), info["phi"] (shaping potential, at reset and every step),
    info["sim_diverged"]                                                                          honest metrics

Role in the system: ``scripts/train_rl.py`` builds a :class:`PPOConfig` and runs :class:`AsyncPPO`; the verification,
evaluation and rendering scripts load its checkpoints with :func:`load_policy` (``best.pt`` is the checkpoint with the
best deterministic held-out success).
"""

import collections
import ctypes
import importlib
import json
import math
import multiprocessing
import os
import shutil
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing.connection import wait as mp_wait

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as tmp
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

MODES = ("full", "hover", "grasped")  # start modes of the curriculum
EVAL_SEED0 = 700_000  # held-out evaluation layouts 700000 .. 700000 + n_eval - 1
TRAIN_SEED_LO = 10_000_000  # fresh training seeds are drawn above every held-out seed range (700k.., 9M..9M+99)
LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0  # bounds of the actor's log standard deviation
EMA_ALPHA = 0.05  # smoothing factor of the per-mode success averages that drive the curriculum


# ============================================================================================ small utilities
def resolve(spec):
    """Resolve a "package.module:function" string to the callable; callables pass through (for in-process use).

    Examples
    --------
    >>> resolve("math:sqrt")(16.0)
    4.0
    """
    if callable(spec):
        return spec
    module, attr = spec.split(":")
    return getattr(importlib.import_module(module), attr)


def _wrapper_chain(env):
    """The wrapper chain env, env.env, ..., env.unwrapped (gymnasium >= 1.0 wrappers no longer forward attribute
    lookups)."""
    chain, level = [], env
    while level is not None and all(level is not seen for seen in chain):
        chain.append(level)
        level = getattr(level, "env", None)
    return chain


def env_attr(env, name):
    """Attribute ``name`` from the outermost level of the wrapper chain that defines it, else None."""
    for level in _wrapper_chain(env):
        try:
            return object.__getattribute__(level, name)
        except AttributeError:
            pass
    return None


def _forbid_expert(env):
    """Pure-RL guarantee: make any use of a privileged scripted expert inside the environment fail loudly.

    Every level of the wrapper chain gets ``expert_action`` and ``expert.act`` replaced by a function that raises.
    Returns the environment.
    """

    def _blocked(*_, **__):
        raise RuntimeError("scripted-expert use is forbidden in pure-RL training/evaluation")

    for level in _wrapper_chain(env):
        if callable(getattr(level, "expert_action", None)):
            level.expert_action = _blocked
        expert = getattr(level, "expert", None)
        if expert is not None and hasattr(expert, "act"):
            expert.act = _blocked
    return env


def _set_start_probs(env, probs):
    """Pass start-state probabilities to the environment, if it has a curriculum and ``probs`` is given."""
    setter = env_attr(env, "set_start_probs")
    if setter is not None and probs is not None:
        setter(dict(probs))


def _die_with_parent(parent_pid=None):
    """Linux: receive SIGTERM when the parent dies (even by SIGKILL), so no simulator or pool is left orphaned.

    Exits at once if the parent ``parent_pid`` is already gone.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        return
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(0)


@contextmanager
def _single_thread_env():
    """Child processes inherit the environment variables at start(): one BLAS/OpenMP thread per simulator process."""
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    old = {key: os.environ.get(key) for key in keys}
    os.environ.update({key: "1" for key in keys})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _np_default(value):
    """JSON fallback for NumPy scalars and arrays; anything else is written as its string.

    Examples
    --------
    >>> json.dumps(dict(n=np.int64(3), x=np.float32(0.5), a=np.arange(2)), default=_np_default)
    '{"n": 3, "x": 0.5, "a": [0, 1]}'
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _atomic_json(obj, path):
    """Write JSON through a temporary file and an atomic rename, so readers never see a partial file."""
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(obj, handle, indent=1, default=_np_default)
    os.replace(tmp_path, path)


def _atomic_save(obj, path):
    """``torch.save`` through a temporary file and an atomic rename."""
    tmp_path = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _num(value):
    """A finite numeric scalar as float; anything else as None.

    Examples
    --------
    >>> _num(np.float32(0.5)), _num(True), _num(float("nan")), _num("1")
    (0.5, 1.0, None, None)
    """
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)) and math.isfinite(float(value)):
        return float(value)
    return None


def env_gamma(env):
    """The discount the environment's reward assumes, or None.

    Potential-based shaping 10 (gamma phi' - phi) is policy-invariant only for the trainer's own gamma. The value is
    ``env.gamma`` at any wrapper level, else a ``GAMMA`` constant of the environment class's module.
    """
    gamma = _num(env_attr(env, "gamma"))
    if gamma is None:
        gamma = _num(getattr(sys.modules.get(type(getattr(env, "unwrapped", env)).__module__), "GAMMA", None))
    return gamma


def _sim_failed(info):
    """Whether a step was a simulator failure: the environment flags a solver divergence or charges its "bad"
    penalty."""
    terms = info.get("r_terms") or {}
    return bool(info.get("sim_diverged", False)) or (_num(terms.get("bad")) or 0.0) < 0.0


# ============================================================================================ test / benchmark envs
class _UnitBox(gym.ActionWrapper):
    """The policy acts in [-1, 1]^d as on the rope environments: clip, then rescale to the wrapped env's Box."""

    def __init__(self, env):
        super().__init__(env)
        self._low = env.action_space.low.astype(np.float64)
        self._high = env.action_space.high.astype(np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, env.action_space.shape, np.float32)

    def action(self, act):
        """Map an action in [-1, 1]^d (clipped) linearly onto the wrapped action space."""
        act = np.clip(np.asarray(act, np.float64), -1.0, 1.0)
        return (self._low + 0.5 * (act + 1.0) * (self._high - self._low)).astype(self.env.action_space.dtype)


def make_gym_env(rank=0, seed=0, eval=False, env_id="Pendulum-v1", **kwargs):
    """Factory for standard gymnasium tasks (end-to-end learning test): actions in [-1, 1], time limit kept."""
    env = _UnitBox(gym.make(env_id, **kwargs))
    env.reset(seed=seed)
    return env


HEAVY_TAIL = ((0.70, 0.02, 0.02), (0.25, 0.06, 0.06), (0.05, 1.0, 3.0))  # free step, contact step, start-state reset


class SleepEnv(gym.Env):
    """Throughput benchmark: a stand-in for a contact-heavy simulator whose step time varies.

    ``step()`` sleeps base + U(0, jitter) s; episodes last ``horizon`` steps (truncated) and the reward is -|a|^2. With
    ``mix`` = ((p, lo, hi), ...) the sleep is drawn from that mixture of uniform distributions instead (HEAVY_TAIL
    mimics the rope environment: free and contact steps and 1-3 s resets).
    """

    def __init__(self, obs_dim=8, act_dim=5, horizon=200, base=0.03, jitter=0.03, mix=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self.horizon, self.base, self.jitter = horizon, base, jitter
        self.mix = None if mix is None else np.asarray(mix, np.float64)

    def reset(self, seed=None, options=None):
        """Start an episode with a random observation."""
        super().reset(seed=seed)
        self.elapsed_steps = 0
        return self.np_random.normal(size=self.observation_space.shape).astype(np.float32), {}

    def _sleep_s(self):
        """Duration of the next step's sleep [s]."""
        if self.mix is None:
            return self.base + self.jitter * self.np_random.random()
        prob, low, high = self.mix.T
        component = self.np_random.choice(len(prob), p=prob / prob.sum())
        return low[component] + (high[component] - low[component]) * self.np_random.random()

    def step(self, action):
        """Sleep, then return a random observation and the reward -|a|^2."""
        time.sleep(self._sleep_s())
        self.elapsed_steps += 1
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        obs = self.np_random.normal(size=self.observation_space.shape).astype(np.float32)
        return obs, float(-np.sum(action**2)), False, self.elapsed_steps >= self.horizon, {}


def make_sleep_env(rank=0, seed=0, eval=False, **kwargs):
    """Factory of :class:`SleepEnv` (``kwargs`` are its arguments)."""
    env = SleepEnv(**kwargs)
    env.reset(seed=seed)
    return env


# ============================================================================================ networks
def mlp(inp, hidden, out, out_gain):
    """Multilayer perceptron with ELU activations and orthogonal initialisation.

    Parameters
    ----------
    inp : int
        Input size.
    hidden : sequence of int
        Hidden layer sizes.
    out : int
        Output size.
    out_gain : float
        Orthogonal-initialisation gain of the output layer (hidden layers use sqrt(2)).

    Returns
    -------
    nn.Sequential
        The network.
    """
    layers, width = [], inp
    for size in hidden:
        linear = nn.Linear(width, size)
        nn.init.orthogonal_(linear.weight, math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ELU()]
        width = size
    last = nn.Linear(width, out)
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class Actor(nn.Module):
    """Gaussian policy: an MLP mean and a state-independent log standard deviation (a learned parameter).

    ``log_std_min`` is the exploration floor (``--min_std``): the smoothness and action penalties charge the action
    noise itself, a constant push toward a small sigma on top of PPO's own drift, so sigma is never allowed below it.

    Parameters
    ----------
    obs_dim, act_dim : int
        Observation and action sizes.
    hidden : sequence of int
        Hidden layer sizes.
    log_std_init : float
        Initial log standard deviation (raised to ``log_std_min`` if below it).
    log_std_min : float
        Lower bound of the log standard deviation.

    Examples
    --------
    >>> actor = Actor(obs_dim=4, act_dim=2, hidden=(8,))
    >>> tuple(actor(torch.zeros(3, 4)).shape), [round(float(std), 4) for std in actor.std()]
    ((3, 2), [0.6065, 0.6065])
    """

    def __init__(self, obs_dim, act_dim, hidden=(512, 512, 256), log_std_init=-0.5, log_std_min=LOG_STD_MIN):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.log_std_min = float(log_std_min)
        self.net = mlp(self.obs_dim, tuple(hidden), self.act_dim, 0.01)  # small output gain: near-zero initial mean
        self.log_std = nn.Parameter(torch.full((self.act_dim,), max(float(log_std_init), self.log_std_min)))

    def forward(self, obs):
        """Mean action for (normalised) observations."""
        return self.net(obs)

    def std(self):
        """Action standard deviation, with the log standard deviation clamped to [log_std_min, LOG_STD_MAX]."""
        return self.log_std.clamp(self.log_std_min, LOG_STD_MAX).exp()

    def dist(self, obs):
        """Action distribution (independent Normal per dimension) for (normalised) observations."""
        mean = self.net(obs)
        # No argument validation: a non-finite mean must reach the learner's skip and roll-back logic, not raise.
        return Normal(mean, self.std().expand_as(mean), validate_args=False)


class Critic(nn.Module):
    """State-value function V(s): an MLP on (normalised) observations."""

    def __init__(self, obs_dim, hidden=(512, 512, 256)):
        super().__init__()
        self.net = mlp(int(obs_dim), tuple(hidden), 1, 1.0)

    def forward(self, obs):
        """Values, with the trailing singleton dimension removed."""
        return self.net(obs).squeeze(-1)


# ============================================================================================ statistics, GAE
class RunningMeanStd:
    """Running mean and variance merged from batch moments (the parallel Welford update of Chan et al.).

    Merging the moments of any partition of the data gives exactly the mean and variance of the concatenated data.
    Starts empty (identity normalisation).

    Examples
    --------
    >>> stats = RunningMeanStd()
    >>> stats.update(np.array([1.0, 2.0])); stats.update(np.array([3.0, 4.0]))
    >>> float(stats.mean), float(stats.var), stats.count
    (2.5, 1.25, 4.0)
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 0.0

    def update(self, batch):
        """Merge the moments of a batch of samples, shape (n, *shape)."""
        batch = np.asarray(batch, np.float64).reshape(-1, *self.mean.shape)
        self.merge(batch.mean(0), batch.var(0), len(batch))

    def merge(self, mean, var, count):
        """Merge precomputed moments (mean, variance, count) of a batch."""
        if count <= 0:
            return
        mean, var = np.asarray(mean, np.float64), np.asarray(var, np.float64)
        if self.count == 0:
            self.mean, self.var, self.count = mean.copy(), var.copy(), float(count)
            return
        total = self.count + count
        delta = mean - self.mean
        sum_sq = self.var * self.count + var * count + delta**2 * self.count * count / total
        self.mean, self.var, self.count = self.mean + delta * count / total, sum_sq / total, total

    def state(self):
        """Checkpoint state: dict with "mean", "var" (tensors) and "count"."""
        return dict(mean=torch.as_tensor(self.mean), var=torch.as_tensor(self.var), count=float(self.count))

    def load(self, state):
        """Restore from a :meth:`state` dict."""
        self.mean = np.asarray(state["mean"], np.float64).copy()
        self.var = np.asarray(state["var"], np.float64).copy()
        self.count = float(state["count"])


def normalize_obs(obs, mean, var, clip):
    """Normalise observations with the running statistics and clip them to [-clip, clip] (float32 result).

    Examples
    --------
    >>> normalize_obs(np.array([3.0, -30.0]), np.zeros(2), np.full(2, 4.0), 10.0).tolist()
    [1.5, -10.0]
    """
    return np.clip((np.asarray(obs, np.float64) - mean) / np.sqrt(var + 1e-8), -clip, clip).astype(np.float32)


def gae(rewards, values, last_values, terminated, truncated, final_values, gamma, lam, lengths=None):
    """Generalised advantage estimation, GAE(lambda), on (T, W) arrays (time x worker).

    The bootstrap value after step t is:

    * terminated: 0 (the true end of the MDP episode);
    * truncated (time limit or error): V(final observation of the episode), normalised with the SAME statistics as
      the fragment;
    * otherwise: V(s_{t+1}), and V(next observation) after the last step of the fragment.

    The recursion is cut at every episode end. Ragged fragments (preemption): worker w's fragment is rows
    0 .. lengths[w] - 1 of column w; its last step bootstraps from last_values[w], and the padding rows (any content)
    come out as zeros.

    Parameters
    ----------
    rewards, values : np.ndarray, shape (T, W)
        Rewards and value estimates.
    last_values : np.ndarray, shape (W,)
        Value of the observation after each fragment.
    terminated, truncated : np.ndarray of bool, shape (T, W)
        Episode ends.
    final_values : np.ndarray, shape (T, W)
        Value of the final observation at truncated steps.
    gamma, lam : float
        Discount and GAE lambda.
    lengths : np.ndarray of int, shape (W,), optional
        Fragment lengths of ragged fragments.

    Returns
    -------
    advantages, returns : np.ndarray, shape (T, W)
        Advantages, and returns = advantages + values.

    Examples
    --------
    One worker, two steps of reward 1, the episode terminating at the second step:

    >>> advantages, returns = gae(np.ones((2, 1)), np.zeros((2, 1)), np.zeros(1), np.array([[False], [True]]),
    ...                           np.zeros((2, 1), bool), np.zeros((2, 1)), 0.5, 1.0)
    >>> returns.ravel().tolist()
    [1.5, 1.0]
    """
    if lengths is None:
        next_values = np.concatenate([values[1:], np.asarray(last_values)[None]], 0)
    else:
        lengths = np.asarray(lengths, np.int64)
        pad = np.arange(len(rewards))[:, None] >= lengths[None]
        rewards, values = np.where(pad, 0.0, rewards), np.where(pad, 0.0, values)
        terminated, truncated = terminated | pad, truncated & ~pad  # padding: zero delta, zero bootstrap, cut
        next_values = np.concatenate([values[1:], np.zeros_like(values[:1])], 0)
        next_values[lengths - 1, np.arange(len(lengths))] = last_values
    next_values = np.where(truncated, final_values, next_values)
    next_values = np.where(terminated, 0.0, next_values)
    delta = rewards + gamma * next_values - values
    keep = gamma * lam * (1.0 - (terminated | truncated))
    advantages = np.zeros_like(delta, dtype=np.float64)
    last = np.zeros(delta.shape[1:], np.float64)
    for step in range(len(delta) - 1, -1, -1):
        last = delta[step] + keep[step] * last
        advantages[step] = last
    return advantages, advantages + values


def explained_variance(pred, target):
    """Explained variance 1 - Var(target - pred) / Var(target); NaN if the target is constant.

    Examples
    --------
    >>> round(explained_variance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])), 6)
    0.857143
    """
    variance = np.var(target)
    return float("nan") if variance == 0 else float(1.0 - np.var(target - pred) / variance)


# ============================================================================================ rollout worker
class RolloutWorker:
    """One environment and a CPU copy of the actor; collects rollout fragments.

    Each fragment uses whatever weights and normalisation statistics it is sent; the environment, its current
    observation and the running episode persist across fragments. Environment failures (exceptions, non-finite
    observations or rewards) never crash the worker: the step is recorded as truncated with reward 0, counted, and the
    environment is reset (and rebuilt if the reset fails too).

    Per episode it also keeps the honest progress measures (notes B: the undiscounted return of the shaped reward
    FALLS as the policy improves): per-term sums of ``info["r_terms"]``, the shaping potential at the start and at the
    end (last valid step), the discounted return sum_t gamma^t r_t, and whether the simulator failed.

    Parameters
    ----------
    rank : int
        Worker index.
    factory : str or callable
        Environment factory ("module:function").
    env_kwargs : dict
        Extra factory arguments.
    base_seed : int
        Run seed; the environment seed is ``base_seed + 1000 rank + 100000 restarts``.
    net_cfg : dict
        Actor arguments.
    info_keys : sequence of str
        Info entries to record per episode.
    restarts : int
        How often this worker has been restarted (changes the seeds).
    keep_raw : bool
        Also return the raw (unnormalised) observations.
    stream : int
        Iteration this worker started at (0 in a fresh run): a resumed run does not replay the layouts and the noise
        of the original run's first iterations.
    gamma : float
        Discount of the per-episode discounted return.
    """

    def __init__(self, rank, factory, env_kwargs, base_seed, net_cfg, info_keys=(), restarts=0, keep_raw=False,
                 stream=0, gamma=0.99):
        self.rank, self.factory, self.env_kwargs = rank, factory, dict(env_kwargs)
        self.env_seed = base_seed + 1000 * rank + 100_000 * restarts
        self.info_keys, self.keep_raw, self.gamma = tuple(info_keys), keep_raw, float(gamma)
        # Episode seeds and action noise; see `stream` in the class docstring.
        self.rng = np.random.default_rng([base_seed, rank, restarts] + ([int(stream)] if stream else []))
        self.actor = Actor(**net_cfg)
        self.actor.requires_grad_(False)
        self.start_probs = None
        self.errors, self.error_msgs = 0, []
        self.env = self._build()
        self.obs = self._reset()

    def _build(self):
        """Create the environment (with the scripted expert disabled) and apply the current start probabilities."""
        env = _forbid_expert(resolve(self.factory)(rank=self.rank, seed=self.env_seed, **self.env_kwargs))
        _set_start_probs(env, self.start_probs)
        return env

    def _error(self, msg):
        """Count an environment error and keep the first 10 messages of the fragment."""
        self.errors += 1
        if len(self.error_msgs) < 10:
            self.error_msgs.append(f"worker {self.rank}: {msg}"[:400])

    def close(self):
        """Close the environment, ignoring errors."""
        try:
            self.env.close()
        except Exception:
            pass

    def _reset(self):
        """Start a fresh random episode (no expert prefix); rebuild the environment if the reset raises or returns a
        non-finite observation. Returns the raw observation."""
        for attempt in range(4):
            try:
                if attempt:
                    self.close()
                    self.env = self._build()
                seed = int(self.rng.integers(TRAIN_SEED_LO, 2**31 - 1))
                obs, info = self.env.reset(seed=seed, options={"no_prefix": True})
                obs = np.asarray(obs, np.float64).ravel()
                if np.all(np.isfinite(obs)):
                    info = info or {}
                    self.ep_ret, self.ep_len, self.ep_disc, self.ep_coef = 0.0, 0, 0.0, 1.0
                    self.ep_mode = str(info.get("start_mode", "full"))
                    self.ep_terms, self.ep_simfail = collections.defaultdict(float), False
                    self.ep_phi0 = self.ep_phi = _num(info.get("phi"))
                    return obs
                self._error("reset returned a non-finite observation")
            except Exception as exc:
                self._error(f"reset: {exc!r}")
        raise RuntimeError(f"worker {self.rank}: env reset failed 4 times in a row: {self.error_msgs[-1:]}")

    def _track(self, reward, info, ok):
        """Per-step episode accounting: return, reward parts, potential, discounted return, simulator failure."""
        self.ep_ret += reward
        self.ep_len += 1
        self.ep_disc += self.ep_coef * reward
        self.ep_coef *= self.gamma
        terms = info.get("r_terms")
        if isinstance(terms, dict):
            for key, value in terms.items():
                value = _num(value)
                if value is not None:
                    self.ep_terms[key] += value
        failed = (not ok) or _sim_failed(info)
        self.ep_simfail |= failed
        phi = _num(info.get("phi"))
        if phi is not None and not failed:  # the potential of an exploded rope is not progress
            self.ep_phi = phi

    def _episode_stats(self, info, error):
        """Statistics of the episode that just ended (return, length, success, mode, stage, potentials, ...)."""
        max_stage = info.get("max_stage")
        stats = dict(ret=float(self.ep_ret), len=int(self.ep_len), success=bool(info.get("is_success", False)),
                     mode=str(info.get("start_mode", self.ep_mode)),
                     max_stage=None if max_stage is None else int(max_stage), error=bool(error),
                     disc_ret=float(self.ep_disc), sim_fail=bool(self.ep_simfail))
        if self.ep_terms:
            stats["r_terms"] = dict(self.ep_terms)
        if self.ep_phi0 is not None and self.ep_phi is not None:
            stats.update(phi0=self.ep_phi0, phi_end=self.ep_phi)
        for key in self.info_keys:
            value = _num(info.get(key))
            if value is not None:
                stats[key] = value
        return stats

    def rollout(self, weights, obs_mean, obs_var, clip, start_probs, num_steps, max_steps=None, should_stop=None,
                on_reached=None):
        """Collect a fragment: a fixed one of ``num_steps`` steps (``max_steps`` None), or a preemptible one of at
        least 1 and at most ``max_steps`` steps, ending when ``should_stop()`` turns true (checked between steps);
        ``on_reached()`` is called once ``num_steps`` steps are done.

        Parameters
        ----------
        weights : np.ndarray of float32 or None
            Flat actor parameters; None keeps the current ones.
        obs_mean, obs_var : np.ndarray, shape (obs_dim,)
            Observation-normalisation statistics.
        clip : float
            Observation clip after normalisation.
        start_probs : dict or None
            Start-state probabilities to apply; None keeps the current ones.
        num_steps, max_steps : int
            Fragment length and the cap of a preemptible fragment.
        should_stop, on_reached : callable, optional
            Preemption hooks.

        Returns
        -------
        dict
            The fragment: "obs", "act", "logp", "rew", "term", "trunc", "final_t", "final_obs", "next_obs",
            "episodes", the raw-observation moments, error counts and timings.
        """
        if weights is not None:
            with torch.no_grad():
                vector_to_parameters(torch.tensor(np.asarray(weights, np.float32)), self.actor.parameters())
        if start_probs is not None:
            self.start_probs = dict(start_probs)
            _set_start_probs(self.env, self.start_probs)
        max_steps = num_steps if max_steps is None else max(int(num_steps), int(max_steps))
        with torch.inference_mode():
            return self._steps(np.asarray(obs_mean, np.float64), np.asarray(obs_var, np.float64), clip, int(num_steps),
                               max_steps, should_stop, on_reached)

    def _steps(self, mean, var, clip, num_steps, max_steps, should_stop=None, on_reached=None):
        """The sampling loop of :meth:`rollout`."""
        obs_dim, act_dim = len(mean), self.actor.act_dim
        obs, raw = np.zeros((max_steps, obs_dim), np.float32), np.zeros((max_steps, obs_dim), np.float64)
        act, logp = np.zeros((max_steps, act_dim), np.float32), np.zeros(max_steps, np.float32)
        rew, term, trunc = np.zeros(max_steps), np.zeros(max_steps, bool), np.zeros(max_steps, bool)
        final_t, final_obs, episodes = [], [], []
        errors0, start_time, env_time, count = self.errors, time.perf_counter(), 0.0, 0
        while count < max_steps:
            if count and should_stop is not None and should_stop():
                break
            step = count
            # Sample an action from the current policy (the noise comes from the worker's own generator).
            norm_obs = normalize_obs(self.obs, mean, var, clip)
            obs[step], raw[step] = norm_obs, self.obs
            dist = self.actor.dist(torch.from_numpy(norm_obs)[None])
            noise = torch.from_numpy(self.rng.standard_normal((1, act_dim)).astype(np.float32))
            action = dist.mean + dist.stddev * noise
            logp[step] = dist.log_prob(action).sum(-1).item()
            act[step] = action[0].numpy()
            # Step the environment; any failure becomes a truncation.
            step_start = time.perf_counter()
            try:
                next_obs, reward, terminated, truncated, info = self.env.step(act[step].copy())
                next_obs, reward = np.asarray(next_obs, np.float64).ravel(), float(reward)
                ok = bool(np.all(np.isfinite(next_obs)) and math.isfinite(reward))
                if not ok:
                    self._error("step returned a non-finite observation/reward")
            except Exception as exc:
                ok, reward, info = False, float("nan"), {}
                self._error(f"step: {exc!r}")
            env_time += time.perf_counter() - step_start
            info = info if isinstance(info, dict) else {}
            if not ok:  # cut the episode as a truncation, bootstrapping from the last valid observation; a finite
                # reward is kept (e.g. the environment's blow-up penalty that comes with non-finite rope nodes), else 0
                next_obs, reward, terminated, truncated = (self.obs, (reward if math.isfinite(reward) else 0.0), False,
                                                           True)
            self._track(reward, info, ok)
            rew[step], term[step], trunc[step] = reward, bool(terminated), bool(truncated) and not bool(terminated)
            if terminated or truncated:
                if trunc[step]:
                    final_t.append(step)
                    final_obs.append(normalize_obs(next_obs, mean, var, clip))
                episodes.append(self._episode_stats(info if ok else {**info, "is_success": False}, error=not ok))
                self.obs = self._reset()
            else:
                self.obs = next_obs
            count += 1
            if count == num_steps and on_reached is not None:
                on_reached()
        messages, self.error_msgs = self.error_msgs, []
        fragment = dict(rank=self.rank, obs=obs[:count], act=act[:count], logp=logp[:count], rew=rew[:count],
                        term=term[:count], trunc=trunc[:count], final_t=np.asarray(final_t, np.int64),
                        final_obs=np.asarray(final_obs, np.float32).reshape(-1, obs_dim),
                        next_obs=normalize_obs(self.obs, mean, var, clip), episodes=episodes,
                        raw_mean=raw[:count].mean(0), raw_var=raw[:count].var(0), raw_count=count,
                        errors=self.errors - errors0, error_msgs=messages, sample_s=time.perf_counter() - start_time,
                        env_s=env_time)
        if self.keep_raw:
            fragment["raw"] = raw[:count]
        return fragment


def _worker_main(rank, conn, shared_w, factory, env_kwargs, base_seed, net_cfg, info_keys, stream, gamma, restarts,
                 parent_pid):
    """Command loop of a worker process.

    ("rollout", cid, mean, var, clip, probs, T, T_max, preempt) -> ("ok", fragment). While a preemptible fragment
    runs, the worker sends ("reached", cid) at T steps and polls its pipe between steps for ("stop", cid). A "stop" of
    an older collection (it crossed the fragment of a worker that hit T_max) is ignored. The actor weights are read
    from the shared memory ``shared_w``.
    """
    _die_with_parent(parent_pid)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C goes to the learner, which then closes the workers
    torch.set_num_threads(1)
    worker, closing = None, [False]
    try:
        worker = RolloutWorker(rank, factory, env_kwargs, base_seed, net_cfg, info_keys, restarts, stream=stream,
                               gamma=gamma)
        flat = np.frombuffer(shared_w, dtype=np.float32)
        conn.send(("ready", rank))
        while not closing[0]:
            msg = conn.recv()
            if msg[0] == "close":
                break
            if msg[0] != "rollout":  # a stale "stop"
                continue
            _, cid, mean, var, clip, probs, num_steps, max_steps, preempt = msg

            def should_stop(cid=cid):
                while conn.poll():
                    message = conn.recv()
                    if message[0] == "close":
                        closing[0] = True
                        return True
                    if message[0] == "stop" and message[1] == cid:
                        return True
                return False

            fragment = worker.rollout(flat.copy(), mean, var, clip, probs, num_steps, max_steps,
                                      should_stop if preempt else None,
                                      (lambda cid=cid: conn.send(("reached", cid))) if preempt else None)
            conn.send(("ok", fragment))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass  # the learner went away
    except Exception:
        try:
            conn.send(("fatal", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if worker is not None:
            worker.close()


class _WorkerHandle:
    """Learner-side handle of one worker process: the process, its pipe, and its restart and failure counts."""

    def __init__(self, ctx, rank, args, restarts=0, fails=0):
        self.rank, self.restarts, self.fails = rank, restarts, fails
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(rank, child, *args, restarts, os.getpid()), daemon=True,
                                name=f"rollout-{rank}")
        self.proc.start()
        child.close()  # so that a dead worker shows up as EOF on this end

    def stop(self, timeout=2.0):
        """Terminate (then kill) the process and close the pipe."""
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(timeout)
        self.conn.close()

    def death_reason(self, what):
        """Why the worker is gone: the traceback it sent before dying (e.g. its environment factory failed after a
        restart while the learner was busy), else ``what`` with the exit code (-11 = SIGSEGV: a simulator crash)."""
        try:
            while self.conn.poll():
                kind, payload = self.conn.recv()
                if kind == "fatal":
                    return payload
        except (EOFError, OSError):
            pass
        self.proc.join(1.0)  # reap it, so that its exit code is known
        return f"{what} (exit code {self.proc.exitcode})"


def _stop_process(proc, grace=15.0):
    """SIGTERM (an evaluation process then terminates its own pool), SIGKILL if still alive after ``grace`` s; reap."""
    if proc.is_alive():
        proc.terminate()
        proc.join(grace)
    if proc.is_alive():
        proc.kill()
    proc.join(5.0)


# ============================================================================================ policy loading / eval
class Policy:
    """Checkpoint policy on RAW observations.

    ``__call__`` returns the deterministic action (the mean, clipped to [-1, 1]) and :meth:`sample` a stochastic one
    (clipped). Both accept one observation (D,) or a batch (N, D).

    Parameters
    ----------
    ckpt : dict
        Loaded checkpoint (see ``AsyncPPO._state``).
    device : str
        Torch device.

    Attributes
    ----------
    actor : Actor
        The policy network.
    obs_mean, obs_var : np.ndarray
        Observation-normalisation statistics.
    clip : float
        Observation clip after normalisation.
    env : dict
        Environment factory and keyword arguments the policy was trained with ("factory", "kwargs").
    global_step : int
        Training steps of the checkpoint.

    Examples
    --------
    >>> policy = load_policy("data/runs/generic_ppo4/best.pt")  # doctest: +SKIP
    >>> policy.env["factory"], policy.global_step  # doctest: +SKIP
    ('rope_routing.env.generic_surface_env:make_env', 10007055)
    """

    def __init__(self, ckpt, device="cpu"):
        net = ckpt["net"]
        self.actor = Actor(**{**net, "hidden": tuple(net["hidden"])})  # (older checkpoints: no log_std_min)
        self.actor.load_state_dict(ckpt["actor"])
        self.device = torch.device(device)
        self.actor.to(self.device).eval()
        self.obs_mean = np.asarray(ckpt["obs_mean"], np.float64)
        self.obs_var = np.asarray(ckpt["obs_var"], np.float64)
        self.clip = float(ckpt["clip"])
        self.env = ckpt.get("env", {})
        self.global_step = int(ckpt.get("global_step", 0))

    def _input(self, obs):
        """Normalised network input and whether a single observation was given."""
        batch = np.asarray(obs, np.float64)
        single = batch.ndim == 1
        inputs = torch.as_tensor(normalize_obs(batch.reshape(-1, len(self.obs_mean)), self.obs_mean, self.obs_var,
                                               self.clip), device=self.device)
        return inputs, single

    @torch.inference_mode()
    def __call__(self, obs):
        """Deterministic action(s), float32 in [-1, 1]."""
        inputs, single = self._input(obs)
        action = self.actor(inputs).clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return action[0] if single else action

    @torch.inference_mode()
    def sample(self, obs):
        """Sampled action(s), float32 clipped to [-1, 1] (uses the torch random generator)."""
        inputs, single = self._input(obs)
        action = self.actor.dist(inputs).sample().clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return action[0] if single else action


def load_policy(path, device="cpu"):
    """Load a checkpoint file as a :class:`Policy`."""
    return Policy(torch.load(path, map_location=device, weights_only=False), device)


def run_episode(env, policy, seed, deterministic=True, max_steps=100_000, on_reset=None):
    """Run one held-out episode on layout ``seed`` (no expert prefix).

    Parameters
    ----------
    env : gym.Env
        The environment.
    policy : Policy
        The policy.
    seed : int
        Reset seed and layout seed.
    deterministic : bool
        Act with the policy mean; otherwise sample, with the torch random generator seeded by ``seed``
        (reproducible).
    max_steps : int
        Step limit.
    on_reset : callable, optional
        Called with the environment right after the reset (e.g. to record the initial rope layout).

    Returns
    -------
    record : dict
        "seed", "success", "steps", "ret", "max_stage", "shape", "drops", "terminated", "truncated", "error",
        "sim_fail" and "dphi" (potential progress).
    info : dict
        The final info dict.
    """
    obs, info = env.reset(seed=int(seed), options={"layout_seed": int(seed), "no_prefix": True})
    if not deterministic:
        torch.manual_seed(int(seed))
    if on_reset is not None:
        on_reset(env)
    phi0 = phi = _num((info or {}).get("phi"))
    ret, steps, drops, max_stage, sim_fail = 0.0, 0, 0, None, False
    terminated = truncated = False
    while steps < max_steps:
        action = policy(obs) if deterministic else policy.sample(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        steps += 1
        drops += int(_num(info.get("dropped")) or 0)
        stage = info.get("max_stage", info.get("stage"))
        if _num(stage) is not None:
            max_stage = int(stage) if max_stage is None else max(max_stage, int(stage))
        failed = _sim_failed(info)
        sim_fail |= failed
        if _num(info.get("phi")) is not None and not failed:
            phi = _num(info.get("phi"))
        if terminated or truncated:
            break
    record = dict(seed=int(seed), success=bool(info.get("is_success", False)), steps=steps, ret=ret,
                  max_stage=max_stage, shape=_num(info.get("shape")), drops=drops, terminated=bool(terminated),
                  truncated=bool(truncated), error=None, sim_fail=bool(sim_fail),
                  dphi=None if phi0 is None or phi is None else phi - phi0)
    return record, info


def failed_episode(seed, error):
    """Record of an episode that crashed or never returned: counted as a failure, with the reason."""
    return dict(seed=int(seed), success=False, steps=0, ret=float("nan"), max_stage=None, shape=None, drops=0,
                terminated=False, truncated=False, error=str(error)[:300], sim_fail=True, dphi=None)


def pool_results(pool, fn, seeds, timeout):
    """``pool.imap_unordered(fn, seeds)`` that cannot hang.

    ``multiprocessing.Pool`` never returns the task of a worker that died mid-episode (e.g. a simulator segfault), so
    after ``timeout`` s without any new result the missing seeds are reported as failed episodes. Yields the results in
    completion order.
    """
    results, done = pool.imap_unordered(fn, seeds), set()
    try:
        for _ in seeds:
            result = results.next(timeout=timeout)
            done.add(result["seed"])
            yield result
    except multiprocessing.TimeoutError:
        for seed in seeds:
            if seed not in done:
                yield failed_episode(seed, f"no result within {timeout:.0f} s (worker died or hung)")


_EVAL = {}  # per-process state of an evaluation pool worker: environment factory, environment, policy, mode


def _eval_init(ckpt_path, factory, env_kwargs, parent_pid, deterministic=True):
    """Initialiser of an evaluation pool worker: build its environment and load the policy."""
    _die_with_parent(parent_pid)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    ident = tmp.current_process()._identity
    rank = int(ident[-1]) if ident else 0
    _EVAL["make"] = lambda: _forbid_expert(resolve(factory)(rank=rank, seed=EVAL_SEED0 + rank, eval=True,
                                                            **env_kwargs))
    _EVAL["env"] = _EVAL["make"]()
    _EVAL["policy"] = load_policy(ckpt_path)
    _EVAL["det"] = bool(deterministic)


def _eval_episode(seed):
    """One evaluation episode in a pool worker; a crash yields a failed-episode record."""
    try:
        return run_episode(_EVAL["env"], _EVAL["policy"], seed, deterministic=_EVAL["det"])[0]
    except Exception as exc:  # rebuild the env: a simulator left broken must not fail this worker's remaining layouts
        try:
            _EVAL["env"].close()
        except Exception:
            pass
        try:
            _EVAL["env"] = _EVAL["make"]()
        except Exception:
            pass
        return failed_episode(seed, repr(exc))


def summarize_eval(episodes):
    """Summary statistics of evaluation episodes.

    Parameters
    ----------
    episodes : list of dict
        Episode records (:func:`run_episode`, :func:`failed_episode`).

    Returns
    -------
    dict
        "n", "success_rate", "n_success", "grasp_rate", "mean_stage", "sim_fail_rate", "mean_dphi", "mean_return",
        "mean_steps", "mean_shape", "stage_hist" and "errors" (None where undefined).

    Examples
    --------
    >>> summary = summarize_eval([failed_episode(1, "crash"), failed_episode(2, "crash")])
    >>> summary["n"], summary["success_rate"], summary["errors"], summary["mean_return"]
    (2, 0.0, 2, None)
    """
    success = [episode["success"] for episode in episodes]
    stages = [episode["max_stage"] for episode in episodes if episode["max_stage"] is not None]
    shapes = [episode["shape"] for episode in episodes if episode["shape"] is not None]
    returns = [episode["ret"] for episode in episodes if math.isfinite(episode["ret"])]
    dphi = [episode["dphi"] for episode in episodes if episode.get("dphi") is not None]
    return dict(n=len(episodes), success_rate=float(np.mean(success)) if episodes else 0.0,
                n_success=int(np.sum(success)),
                grasp_rate=float(np.mean([stage >= 1 for stage in stages])) if stages else None,
                mean_stage=float(np.mean(stages)) if stages else None,
                sim_fail_rate=(float(np.mean([bool(episode.get("sim_fail")) for episode in episodes]))
                               if episodes else None),
                mean_dphi=float(np.mean(dphi)) if dphi else None,
                mean_return=float(np.mean(returns)) if returns else None,
                mean_steps=float(np.mean([episode["steps"] for episode in episodes])) if episodes else None,
                mean_shape=float(np.mean(shapes)) if shapes else None,
                stage_hist=({int(stage): int(count) for stage, count in zip(*np.unique(stages, return_counts=True))}
                            if stages else {}),
                errors=int(sum(episode.get("error") is not None for episode in episodes)))


def _eval_main(ckpt_path, factory, env_kwargs, seeds, n_workers, out_path, parent_pid, episode_timeout,
               deterministic=True):
    """Evaluation process: evaluate a checkpoint on held-out layouts with its own small pool, write a JSON, exit."""
    _die_with_parent(parent_pid)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # -> Pool.__exit__ terminates the pool workers
    torch.set_num_threads(1)
    start_time = time.time()
    ctx = tmp.get_context("spawn")
    with _single_thread_env():
        pool = ctx.Pool(max(1, min(n_workers, len(seeds))), initializer=_eval_init,
                        initargs=(ckpt_path, factory, env_kwargs, os.getpid(), deterministic))
    with pool:
        episodes = sorted(pool_results(pool, _eval_episode, seeds, episode_timeout),
                          key=lambda episode: episode["seed"])
    _atomic_json(dict(ckpt=ckpt_path, eval_s=time.time() - start_time, deterministic=bool(deterministic),
                      **summarize_eval(episodes), episodes=episodes), out_path)


# ============================================================================================ learner
@dataclass
class PPOConfig:
    """Configuration of an :class:`AsyncPPO` run. The field names appear on the command line (``train_rl.py``), in
    ``config.json`` and in checkpoints, so they are part of the interface.

    Attributes
    ----------
    run_dir : str
        Output directory: checkpoints, metrics, evaluations.
    env : str
        Environment factory, "module:function".
    env_kwargs : dict or None
        Extra factory arguments.
    workers : int
        Rollout worker processes.
    rollout : int
        T: steps per worker per iteration (the preemption threshold).
    rollout_max : int
        Cap of a preemptible fragment; 0 means 2 * rollout.
    preempt_frac : float
        Stop every worker once this share has at least T steps; 1.0 gives fixed T-step fragments.
    stop_timeout : float
        A worker that has not returned its fragment this long after "stop" is restarted [s].
    steps : int
        Total environment steps.
    gamma, lam : float
        Discount and GAE lambda.
    epochs, minibatch : int
        PPO epochs per iteration and minibatch size [transitions].
    lr : float
        Initial learning rate.
    lr_schedule : str
        "adaptive" (KL-driven, the rsl_rl rule) or "linear" (lr to lr_final over ``steps``).
    lr_min, lr_max : float
        Bounds of the adaptive learning rate.
    lr_final : float
        Final learning rate of the linear schedule.
    clip : float
        PPO ratio clip.
    ent, vf : float
        Entropy-bonus and value-loss coefficients.
    max_grad : float
        Gradient-norm clip, applied to the actor and to the critic SEPARATELY.
    target_kl : float or None
        Adaptive learning rate: x1.5 if the mean KL < target / 2, /1.5 if > 2 target or on an early stop. None or
        <= 0 turns it off (and disables the early stop).
    kl_stop : float
        The policy stops for the iteration when a minibatch KL exceeds kl_stop * target_kl (<= 0: never); the critic
        keeps training for the remaining epochs.
    min_std : float
        Exploration floor: the action standard deviation is clamped to at least min_std (0: the old floor e^-5).
    log_std_init : float
        Initial log standard deviation of the actor.
    hidden : tuple of int
        Hidden layer sizes of the actor and the critic.
    obs_clip, reward_clip : float
        Clips of the normalised observations and of the scaled rewards.
    eval_every : int
        Evaluation interval [environment steps]; 0 disables evaluation.
    n_eval, n_eval_workers : int
        Held-out layouts per evaluation and evaluation processes.
    eval_stochastic : bool
        Also evaluate sampled actions (logged separately; never selects best.pt).
    ckpt_every : int
        Checkpoint interval [environment steps]; 0 disables periodic checkpoints (last.pt is still written every
        iteration).
    device : str
        Learner device.
    seed : int
        Run seed.
    curriculum : str
        "adaptive" or "none" (start-state curriculum).
    resume : str or None
        Checkpoint to resume from.
    info_keys : tuple of str
        Info entries averaged per episode in the logs.
    worker_timeout : float
        A worker that returns no fragment within this time is killed and restarted [s].
    max_worker_fails : int
        Consecutive failures of one worker before the run is aborted.
    final_eval : bool
        Evaluate the final weights at the end of the run.
    eval_timeout : float
        An evaluation process still running after this is killed and logged as failed [s].
    eval_episode_timeout : float
        Held-out layouts without a result for this long count as crashed episodes [s].
    learner_threads : int
        Torch CPU threads of the learner.
    """

    run_dir: str
    env: str = "rope_routing.env.surface_route_env:make_env"
    env_kwargs: dict | None = None
    workers: int = 48
    rollout: int = 256
    rollout_max: int = 0
    preempt_frac: float = 0.8
    stop_timeout: float = 60.0
    steps: int = 30_000_000
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 8
    minibatch: int = 2048
    lr: float = 3e-4
    lr_schedule: str = "adaptive"
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    lr_final: float = 3e-5
    clip: float = 0.2
    ent: float = 0.0
    vf: float = 0.5
    max_grad: float = 0.5
    target_kl: float | None = 0.01
    kl_stop: float = 3.0
    min_std: float = 0.1
    log_std_init: float = -0.5
    hidden: tuple = (512, 512, 256)
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    eval_every: int = 1_000_000
    n_eval: int = 64
    n_eval_workers: int = 8
    eval_stochastic: bool = False
    ckpt_every: int = 1_000_000
    device: str = "cuda"
    seed: int = 0
    curriculum: str = "adaptive"
    resume: str | None = None
    info_keys: tuple = ()
    worker_timeout: float = 300.0
    max_worker_fails: int = 5
    final_eval: bool = True
    eval_timeout: float = 3600.0
    eval_episode_timeout: float = 900.0
    learner_threads: int = 4


def adapt_lr(lr, kl, target, lo, hi):
    """The rsl_rl KL-adaptive learning-rate rule, applied once per iteration to the mean minibatch KL.

    A fixed learning rate gets more aggressive as the policy sharpens (the same parameter step moves a narrow Gaussian
    further); this rule keeps the step at about the target KL.

    Examples
    --------
    >>> adapt_lr(1e-3, 0.05, 0.01, 1e-5, 1e-2) < 1e-3, adapt_lr(1e-3, 0.001, 0.01, 1e-5, 1e-2) > 1e-3
    (True, True)
    """
    if not target or kl is None or math.isnan(kl):
        return lr
    if kl > 2.0 * target:
        lr /= 1.5
    elif kl < 0.5 * target:
        lr *= 1.5
    return float(min(hi, max(lo, lr)))


def _eval_key(success, shape):
    """Ordering key of best.pt: deterministic held-out success, ties broken by the lower mean final shape error.

    Examples
    --------
    >>> _eval_key(0.8, 0.02) > _eval_key(0.8, 0.03) > _eval_key(0.7, 0.01)
    True
    """
    return (float(success), -(math.inf if shape is None else float(shape)))


class AsyncPPO:
    """Asynchronous-rollout PPO trainer (see the module docstring).

    The constructor probes the environment (dimensions, curriculum, shaping discount), builds the networks and the
    optimiser and restores a checkpoint if ``cfg.resume`` is set; :meth:`train` starts the workers and runs
    iterations until ``cfg.steps`` environment steps, then the final evaluation.

    Parameters
    ----------
    cfg : PPOConfig
        Run configuration.

    Attributes
    ----------
    actor, critic : Actor, Critic
        The networks, on ``device``.
    opt : torch.optim.Adam
        Optimiser with one parameter group for the actor and one for the critic.
    lr : float
        Current learning rate.
    obs_rms, ret_rms : RunningMeanStd
        Observation statistics and statistics of the discounted returns (reward scaling).
    ret_acc : np.ndarray, shape (workers,)
        Per-worker discounted-return accumulators of the reward scaling.
    global_step, iteration : int
        Environment steps and iterations done.
    succ_ema, mode_total : dict
        Smoothed success rate and episode count per start mode (curriculum).
    eval_history, eval_stoch_history : list of dict
        Deterministic and stochastic evaluation records.
    best_key : tuple
        :func:`_eval_key` of best.pt.
    workers : list of _WorkerHandle
        The rollout workers.
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        cfg.env_kwargs = dict(cfg.env_kwargs or {})
        cfg.hidden, cfg.info_keys = tuple(cfg.hidden), tuple(cfg.info_keys)
        assert cfg.curriculum in ("adaptive", "none"), cfg.curriculum
        assert cfg.lr_schedule in ("adaptive", "linear"), cfg.lr_schedule
        assert 0.0 < cfg.preempt_frac <= 1.0, cfg.preempt_frac
        if cfg.target_kl is not None and not cfg.target_kl > 0:  # <= 0 means off (a negative target was truthy:
            cfg.target_kl = None  # kl_stop < 0 froze the policy, and the learning rate fell every iteration)
        os.makedirs(cfg.run_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.learner_threads)
        torch.backends.cuda.matmul.allow_tf32 = False  # full fp32: the learner's log-probs must match the CPU workers'
        torch.backends.cudnn.allow_tf32 = False
        device = cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[async_ppo] CUDA not available -> CPU learner", flush=True)
            device = "cpu"
        self.device = torch.device(device)
        self.ctx = tmp.get_context("spawn")  # never fork a process that holds a CUDA context

        # Probe environment: dimensions, the start-state curriculum, and the discount its shaping assumes.
        env = resolve(cfg.env)(rank=-1, seed=cfg.seed, **cfg.env_kwargs)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))
        self.has_modes = env_attr(env, "set_start_probs") is not None
        self.env_gamma = env_gamma(env)
        try:
            env.close()
        except Exception:
            pass
        del env
        if self.env_gamma is not None and abs(self.env_gamma - cfg.gamma) > 1e-12:
            raise ValueError(
                f"gamma mismatch: trainer gamma = {cfg.gamma} but the env ({cfg.env}) shapes its reward with gamma = "
                f"{self.env_gamma}. Potential-based shaping 10 (gamma_env phi' - phi) is policy-invariant only when "
                f"the two agree; otherwise it adds a real reward -10 (gamma - gamma_env) phi(s') per step that pushes "
                f"away from the goal (notes/B R1). Pass --gamma {self.env_gamma} or change the env's GAMMA as well.")

        # Networks and optimiser; a resumed run keeps the checkpoint's architecture.
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False) if cfg.resume else None
        self.net_cfg = dict(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden=list(cfg.hidden),
                            log_std_init=float(cfg.log_std_init))
        if ckpt is not None:
            assert (ckpt["net"]["obs_dim"], ckpt["net"]["act_dim"]) == (self.obs_dim, self.act_dim), "env/ckpt mismatch"
            self.net_cfg = dict(ckpt["net"])
        # The std floor is a training setting, not architecture: a resumed run uses the current --min_std.
        self.net_cfg["log_std_min"] = math.log(cfg.min_std) if cfg.min_std > 0 else LOG_STD_MIN
        self.actor = Actor(**self.net_cfg).to(self.device)
        self.critic = Critic(self.obs_dim, self.net_cfg["hidden"]).to(self.device)
        self.actor_params, self.critic_params = list(self.actor.parameters()), list(self.critic.parameters())
        self.params = self.actor_params + self.critic_params
        self.opt = torch.optim.Adam([dict(params=self.actor_params), dict(params=self.critic_params)], lr=cfg.lr,
                                    eps=1e-5)
        self.lr = float(cfg.lr)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        # Statistics, curriculum and logging state.
        self.obs_rms, self.ret_rms = RunningMeanStd((self.obs_dim,)), RunningMeanStd(())
        self.ret_acc = np.zeros(cfg.workers)  # per-worker discounted-return accumulators (reward scaling)
        self.global_step = self.iteration = 0
        self._cid = 0  # collection id: tags "reached" / "stop" messages so that stale ones are ignored
        self.succ_ema = {mode: 0.0 for mode in MODES}
        self.mode_total = {mode: 0 for mode in MODES}
        self.episodes = collections.deque(maxlen=100)
        self.mode_eps = collections.defaultdict(lambda: collections.deque(maxlen=100))
        self.total_episodes = self.total_errors = self.total_restarts = 0
        self.best_key, self.last_eval, self.last_eval_stoch = (-math.inf, -math.inf), None, None
        self.eval_history, self.eval_queue, self.eval_proc = [], collections.deque(), None
        self.eval_stoch_history = []
        self.workers = []
        if ckpt is not None:
            self._load_state(ckpt)
            for name, attr in (("eval_history.json", "eval_history"),
                               ("eval_stoch_history.json", "eval_stoch_history")):
                path = os.path.join(cfg.run_dir, name)
                if os.path.exists(path):
                    with open(path) as handle:
                        setattr(self, attr, json.load(handle))
            self.best_key = max([_eval_key(entry["success"], entry.get("mean_shape")) for entry in self.eval_history],
                                default=self.best_key)
        self.next_ckpt = self._next(cfg.ckpt_every)
        self.next_eval = self._next(cfg.eval_every)
        self.metrics_path = os.path.join(cfg.run_dir, "metrics.jsonl")

    @property
    def best_eval(self):
        """Deterministic held-out success of best.pt (-inf before the first evaluation)."""
        return self.best_key[0]

    # ---------------------------------------------------------------------------------------- state / checkpoints
    def _next(self, every):
        """The next multiple of ``every`` above the current step (inf if ``every`` <= 0)."""
        return (self.global_step // every + 1) * every if every > 0 else math.inf

    def _state(self):
        """Checkpoint dict: networks, observation statistics, network and environment configuration, optimiser,
        reward scaling, learning rate, curriculum, best evaluation and the run configuration."""
        cpu = lambda state_dict: {key: value.detach().cpu().clone() for key, value in state_dict.items()}  # noqa: E731
        cfg = asdict(self.cfg)
        return dict(
            algo="async_ppo (pure RL: no demonstrations, no behaviour cloning)",
            actor=cpu(self.actor.state_dict()), critic=cpu(self.critic.state_dict()),
            log_std=self.actor.log_std.detach().cpu().clone(),
            obs_mean=torch.as_tensor(self.obs_rms.mean.copy()), obs_var=torch.as_tensor(self.obs_rms.var.copy()),
            obs_count=float(self.obs_rms.count), clip=float(self.cfg.obs_clip), net=dict(self.net_cfg),
            env=dict(factory=self.cfg.env, kwargs=dict(self.cfg.env_kwargs)),
            global_step=int(self.global_step), iteration=int(self.iteration),
            optimizer=self.opt.state_dict(), ret_rms=self.ret_rms.state(), lr=float(self.lr),
            curriculum=dict(succ_ema=dict(self.succ_ema), totals=dict(self.mode_total)),
            best_eval=None if not math.isfinite(self.best_eval) else float(self.best_eval), config=cfg,
        )

    def _load_state(self, ckpt):
        """Restore the training state from a checkpoint dict (see :meth:`_state`)."""
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        try:
            self.opt.load_state_dict(ckpt["optimizer"])
        except Exception as exc:  # e.g. a different network or an older single-group optimiser: fresh Adam moments
            print(f"[async_ppo] optimizer state not restored: {exc!r}", flush=True)
        self.obs_rms.load(dict(mean=ckpt["obs_mean"], var=ckpt["obs_var"], count=ckpt["obs_count"]))
        self.ret_rms.load(ckpt["ret_rms"])
        self.global_step, self.iteration = int(ckpt["global_step"]), int(ckpt["iteration"])
        if self.cfg.lr_schedule == "adaptive" and ckpt.get("lr") is not None:
            self.lr = float(ckpt["lr"])
        with torch.no_grad():
            self.actor.log_std.clamp_(self.actor.log_std_min, LOG_STD_MAX)
        self.succ_ema.update(ckpt["curriculum"]["succ_ema"])
        self.mode_total.update(ckpt["curriculum"]["totals"])
        print(f"[async_ppo] resumed from {self.cfg.resume} at step {self.global_step:,} (iteration {self.iteration})",
              flush=True)

    # ---------------------------------------------------------------------------------------- workers
    def _worker_args(self):
        """Arguments of :func:`_worker_main` shared by all workers."""
        return (self.shared_w, self.cfg.env, self.cfg.env_kwargs, self.cfg.seed, self.net_cfg, self.cfg.info_keys,
                self.iteration, self.cfg.gamma)

    def _start_workers(self):
        """Allocate the shared weight buffer, start every worker and wait until all have built their environment."""
        num_params = sum(param.numel() for param in self.actor.parameters())
        self.shared_w = self.ctx.RawArray("f", num_params)  # actor weights, rewritten in place every iteration
        self.shared_np = np.frombuffer(self.shared_w, dtype=np.float32)
        with _single_thread_env():
            self.workers = [_WorkerHandle(self.ctx, rank, self._worker_args()) for rank in range(self.cfg.workers)]
        pending = {handle.conn: handle for handle in self.workers}
        deadline = time.time() + self.cfg.worker_timeout
        while pending:
            ready = mp_wait(list(pending), timeout=max(0.0, deadline - time.time()))
            if not ready:
                raise RuntimeError(f"workers {sorted(handle.rank for handle in pending.values())} not ready in time")
            for conn in ready:
                handle = pending.pop(conn)
                try:
                    kind, payload = conn.recv()
                except (EOFError, OSError):
                    kind, payload = "dead", handle.death_reason("process died")
                if kind != "ready":
                    raise RuntimeError(f"worker {handle.rank} failed to start:\n{payload}")

    def _restart(self, handle, reason):
        """Replace a failed worker with a fresh process (aborting the run after ``max_worker_fails`` in a row)."""
        self.total_restarts += 1
        print(f"[async_ppo] restarting worker {handle.rank} ({handle.fails + 1} consecutive failures): {reason}",
              flush=True)
        if handle.fails + 1 > self.cfg.max_worker_fails:
            raise RuntimeError(f"worker {handle.rank} failed {handle.fails + 1} times in a row; last error:\n{reason}")
        handle.stop()
        self.ret_acc[handle.rank] = 0.0
        with _single_thread_env():
            self.workers[handle.rank] = _WorkerHandle(self.ctx, handle.rank, self._worker_args(), handle.restarts + 1,
                                                      handle.fails + 1)

    def _rollout_max(self):
        """Cap of a preemptible fragment [steps]."""
        return self.cfg.rollout_max if self.cfg.rollout_max > 0 else 2 * self.cfg.rollout

    def _collect(self, probs):
        """Collect one fragment from every worker.

        Sends (statistics, start probabilities, T, T_max) to every worker (the weights are in shared memory) and
        gathers the fragments as they finish. Preemption: once at least preempt_frac of the workers still taking part
        (dead ones drop out, so that a death cannot deadlock the count) have reported T steps, every worker still
        sampling is told to stop after its current step. Bounds: worker_timeout from the start, and stop_timeout after
        the stop.

        Returns
        -------
        list of dict
            The fragments, sorted by worker rank.
        """
        cfg = self.cfg
        self._cid += 1
        cid, preempt, num_steps = self._cid, cfg.preempt_frac < 1.0, cfg.rollout
        max_steps = self._rollout_max() if preempt else num_steps
        message = ("rollout", cid, self.obs_rms.mean, self.obs_rms.var, cfg.obs_clip, probs, num_steps, max_steps,
                   preempt)
        live = {}
        for handle in list(self.workers):
            try:
                handle.conn.send(message)
                live[handle.conn] = handle
            except (BrokenPipeError, OSError) as exc:
                self._restart(handle, handle.death_reason(f"send failed: {exc!r}"))
        results, reached, stopped = [], set(), False
        start_time = time.time()
        deadline = start_time + cfg.worker_timeout
        self._stop_s = None
        while live:
            # Preemption: stop the stragglers once enough workers have reached T steps.
            if preempt and not stopped:
                taking_part = {handle.rank for handle in live.values()} | {result["rank"] for result in results}
                if len(reached & taking_part) >= math.ceil(cfg.preempt_frac * len(taking_part) - 1e-9):
                    for handle in live.values():
                        try:
                            handle.conn.send(("stop", cid))
                        except (BrokenPipeError, OSError):
                            pass  # its death shows up as EOF below
                    stopped, self._stop_s = True, time.time() - start_time
                    deadline = min(deadline, time.time() + cfg.stop_timeout)
            ready = mp_wait(list(live), timeout=max(0.0, deadline - time.time()))
            if not ready:
                reason = (f"no fragment within {cfg.stop_timeout:.0f} s of the stop signal"
                          if stopped and deadline < start_time + cfg.worker_timeout
                          else f"no fragment within {cfg.worker_timeout:.0f} s")
                for handle in list(live.values()):
                    self._restart(handle, reason)
                break
            for conn in ready:
                handle = live[conn]
                try:
                    kind, payload = conn.recv()
                except (EOFError, OSError):
                    kind, payload = "dead", handle.death_reason("process died")
                if kind == "ready":  # a restarted worker reporting in; its fragment follows
                    continue
                if kind == "reached":
                    if payload == cid:
                        reached.add(handle.rank)
                    continue
                del live[conn]
                if kind == "ok":
                    handle.fails = 0
                    results.append(payload)
                    if len(payload["rew"]) >= num_steps:
                        reached.add(handle.rank)
                else:
                    self._restart(handle, payload)
        return sorted(results, key=lambda result: result["rank"])

    # ---------------------------------------------------------------------------------------- learning
    @torch.no_grad()
    def _value(self, obs):
        """Critic values of a batch of normalised observations, as a float64 array (chunks of 65536)."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs) == 0:
            return np.zeros(0)
        return torch.cat([self.critic(chunk) for chunk in obs.split(65536)]).double().cpu().numpy()

    def _batch(self, results):
        """Turn the ragged fragments into one training batch.

        The arrays are padded to (T_max, W) (time x worker); the batch holds the valid entries in time-major order,
        which for equal lengths is exactly the old (T, W).reshape(T * W) layout. Rewards are divided by the running
        standard deviation of the per-worker discounted return, then GAE gives the advantages and returns.
        """
        cfg, num_workers, obs_dim = self.cfg, len(results), self.obs_dim
        lengths = np.array([len(result["rew"]) for result in results], np.int64)
        max_len = int(lengths.max())
        valid = np.arange(max_len)[:, None] < lengths[None]  # (max_len, W)
        ragged = bool((lengths != max_len).any())

        def pad(key):
            if not ragged:
                return np.stack([result[key] for result in results], 1)
            first = results[0][key]
            padded = np.zeros((max_len, num_workers) + first.shape[1:], first.dtype)
            for column, result in enumerate(results):
                padded[:lengths[column], column] = result[key]
            return padded

        obs, act, logp, rew, term, trunc = (pad(key) for key in ("obs", "act", "logp", "rew", "term", "trunc"))
        values = np.zeros((max_len, num_workers))
        values[valid] = self._value(obs[valid])
        last_values = self._value(np.stack([result["next_obs"] for result in results]))
        final_values = np.zeros((max_len, num_workers))
        final_flat = self._value(np.concatenate([result["final_obs"] for result in results]))
        offset = 0
        for column, result in enumerate(results):
            count = len(result["final_t"])
            final_values[result["final_t"], column] = final_flat[offset:offset + count]
            offset += count
        # Reward scaling: divide by the running std of the per-worker discounted return (reset at episode ends).
        ranks = np.array([result["rank"] for result in results])
        acc, disc_returns = self.ret_acc[ranks].copy(), np.zeros((max_len, num_workers))
        for step in range(max_len):
            new = acc * cfg.gamma + rew[step]
            disc_returns[step] = new
            acc = np.where(valid[step], new, acc)  # a worker's accumulator freezes after the end of its fragment
            acc[(term[step] | trunc[step]) & valid[step]] = 0.0
        self.ret_acc[ranks] = acc
        self.ret_rms.update(disc_returns[valid])
        rew_scaled = np.clip(rew / np.sqrt(self.ret_rms.var + 1e-8), -cfg.reward_clip, cfg.reward_clip)
        advantages, returns = gae(rew_scaled, values, last_values, term, trunc, final_values, cfg.gamma, cfg.lam,
                                  lengths if ragged else None)
        to_device = lambda array: torch.as_tensor(array[valid], dtype=torch.float32, device=self.device)  # noqa: E731
        return dict(obs=to_device(obs), act=to_device(act), logp=to_device(logp), adv=to_device(advantages),
                    ret=to_device(returns), values=values[valid], returns=returns[valid], n=int(lengths.sum()),
                    lengths=lengths, rew_scale=float(np.sqrt(self.ret_rms.var)))

    def _lr(self):
        """Learning rate of this iteration (adaptive: the current one; linear: interpolated over the run)."""
        if self.cfg.lr_schedule == "adaptive":
            return self.lr
        frac = min(1.0, self.global_step / max(1, self.cfg.steps))
        return self.cfg.lr + (self.cfg.lr_final - self.cfg.lr) * frac

    def _adapted_lr(self, ppo):
        """The next iteration's learning rate (adaptive schedule).

        An early stop means the policy passed kl_stop * target_kl (> 2 target) within this iteration, so it always
        counts as "too large": the mean minibatch KL alone averages the path from 0 up to the stop (about half the KL
        reached) and stays in band, which left the learning rate unchanged while the policy stopped after about 1 of 8
        epochs in every iteration.
        """
        cfg = self.cfg
        if cfg.lr_schedule != "adaptive":
            return self.lr
        kl = math.inf if ppo.get("early_stop") else ppo.get("approx_kl")
        return adapt_lr(self.lr, kl, cfg.target_kl, cfg.lr_min, cfg.lr_max)

    def _update(self, batch, lr):
        """Clipped PPO epochs over random minibatches.

        If a minibatch's KL exceeds kl_stop * target_kl, the POLICY stops for this iteration, but the critic (a
        separate network) keeps fitting the returns for the remaining epochs: its steps cannot move the policy, and
        stopping both left the critic half-trained in early-stopped iterations. Actor and critic gradients are clipped
        separately (one joint norm is dominated by the critic and then clips the actor at every step). A non-finite
        loss or gradient skips the minibatch, and non-finite parameters after the update roll the networks back.

        Returns
        -------
        dict
            Loss, KL, gradient, clip-fraction and standard-deviation statistics of the update.
        """
        cfg = self.cfg
        for group in self.opt.param_groups:
            group["lr"] = lr
        with torch.no_grad():  # the learner must reproduce the workers' log-probs (same weights, same inputs)
            logp_learner = torch.cat([self.actor.dist(obs).log_prob(act).sum(-1)
                                      for obs, act in zip(batch["obs"].split(65536), batch["act"].split(65536))])
            logp_err = float((logp_learner - batch["logp"]).abs().max())
        if logp_err > 1e-2:  # float32 CPU-vs-GPU round-off is ~1e-6..1e-4; this is a sync bug, never noise
            print(f"[async_ppo] WARNING: worker/learner log-prob mismatch {logp_err:.3g}", flush=True)
        snapshot = parameters_to_vector(self.params).detach().clone()
        num_minibatches = max(1, int(round(batch["n"] / cfg.minibatch)))
        kl_stop = cfg.kl_stop * cfg.target_kl if cfg.target_kl and cfg.kl_stop > 0 else None
        logs, stopped, skipped, n_upd, n_vupd = collections.defaultdict(list), False, 0, 0, 0
        for _ in range(cfg.epochs):
            for index in torch.randperm(batch["n"], device=self.device, generator=self.gen).chunk(num_minibatches):
                v_loss = (self.critic(batch["obs"][index]) - batch["ret"][index]).pow(2).mean()
                if not stopped:
                    dist = self.actor.dist(batch["obs"][index])
                    log_ratio = dist.log_prob(batch["act"][index]).sum(-1) - batch["logp"][index]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl = float(((ratio - 1.0) - log_ratio).mean())
                    logs["approx_kl"].append(torch.tensor(kl))
                    stopped = kl_stop is not None and not kl <= kl_stop  # a NaN KL stops too
                if stopped:  # critic only
                    pg_loss = entropy = None
                    loss = cfg.vf * v_loss
                else:
                    advantages = batch["adv"][index]
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    pg_loss = -torch.min(advantages * ratio,
                                         advantages * ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = pg_loss + cfg.vf * v_loss - cfg.ent * entropy
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                self.opt.zero_grad(set_to_none=True)  # actor gradients stay None when critic-only: Adam skips them
                loss.backward()
                grad_critic = nn.utils.clip_grad_norm_(self.critic_params, cfg.max_grad)
                grad_actor = None if stopped else nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad)
                # A finite loss with a NaN gradient (e.g. an overflowing ratio): never step on it.
                if not torch.isfinite(grad_critic) or (grad_actor is not None and not torch.isfinite(grad_actor)):
                    skipped += 1
                    continue
                self.opt.step()
                with torch.no_grad():  # keep the parameter itself in range (no gradient reaches it below the floor)
                    self.actor.log_std.clamp_(self.actor.log_std_min, LOG_STD_MAX)
                logs["v_loss"].append(v_loss.detach())
                logs["grad_norm_critic"].append(grad_critic.detach())
                if stopped:
                    n_vupd += 1
                    continue
                n_upd += 1
                for key, value in (("pg_loss", pg_loss), ("entropy", entropy), ("grad_norm_actor", grad_actor),
                                   ("clip_frac", ((ratio - 1.0).abs() > cfg.clip).float().mean())):
                    logs[key].append(value.detach())
        rolled_back = not bool(torch.isfinite(parameters_to_vector(self.params)).all())
        if rolled_back:  # never let one bad batch destroy the run
            vector_to_parameters(snapshot, self.params)
            self.opt.state = collections.defaultdict(dict)
        stats = {key: float(torch.stack([value.float().cpu() for value in values]).mean())
                 for key, values in logs.items()}
        std = self.actor.std().detach().cpu().numpy()
        stats.update(updates=n_upd, critic_updates=n_upd + n_vupd, epochs_done=n_upd / num_minibatches,
                     minibatches=num_minibatches, early_stop=stopped, skipped=skipped, rolled_back=rolled_back,
                     logp_err=logp_err, lr=lr, explained_var=explained_variance(batch["values"], batch["returns"]),
                     std=float(std.mean()), std_dims=[float(scale) for scale in std], rew_scale=batch["rew_scale"],
                     # share of sampled actions outside [-1, 1] (the environment clips them): exploration wasted on
                     # the clip (notes B)
                     act_out=float((batch["act"].abs() > 1.0).float().mean()),
                     act_out_dims=[float(share) for share in (batch["act"].abs() > 1.0).float().mean(0).cpu()])
        return stats

    # ---------------------------------------------------------------------------------------- curriculum
    def _start_probs(self):
        """Start-state probabilities of the next iteration.

        Adaptive curriculum: "grasped" starts get 0.4 while their smoothed success is below 0.6 (else 0.15), "hover"
        starts 0.3 (else 0.15), and "full" starts the rest. None if the environment has no start-state curriculum.
        """
        if not self.has_modes:
            return None
        if self.cfg.curriculum == "none":
            return {"full": 1.0, "hover": 0.0, "grasped": 0.0}
        prob_grasped = 0.4 if self.succ_ema["grasped"] < 0.6 else 0.15
        prob_hover = 0.3 if self.succ_ema["hover"] < 0.6 else 0.15
        return {"full": 1.0 - prob_grasped - prob_hover, "hover": prob_hover, "grasped": prob_grasped}

    def _episode_metrics(self, new_episodes):
        """Training-episode metrics over the last 100 episodes, overall and per start mode.

        The shaped reward's undiscounted return runs OPPOSITE to task progress (notes B), so the headline metrics are
        the success rate, the grasp rate, the maximum stage, the potential progress and the simulator failures;
        ret100 stays in metrics.jsonl only.
        """
        for episode in new_episodes:
            self.episodes.append(episode)
            mode = episode["mode"]
            self.mode_eps[mode].append(episode)
            self.succ_ema[mode] = ((1.0 - EMA_ALPHA) * self.succ_ema.get(mode, 0.0)
                                   + EMA_ALPHA * float(episode["success"]))
            self.mode_total[mode] = self.mode_total.get(mode, 0) + 1
        self.total_episodes += len(new_episodes)
        mean = lambda values: float(np.mean(values)) if len(values) else None  # noqa: E731
        stages = lambda episodes: [episode["max_stage"] for episode in episodes  # noqa: E731
                                   if episode["max_stage"] is not None]
        hist = lambda values: ({int(stage): int(count) for stage, count in zip(*np.unique(values, return_counts=True))}
                               if values else {})  # noqa: E731
        dphi = lambda episodes: [episode["phi_end"] - episode["phi0"] for episode in episodes  # noqa: E731
                                 if "phi0" in episode]
        window = list(self.episodes)
        full_hover = [episode for episode in window if episode["mode"] in ("full", "hover")]
        grasp_src = full_hover if self.has_modes else window  # a grasped start is grasped by construction
        metrics = dict(episodes=len(new_episodes), succ100=mean([episode["success"] for episode in window]),
                       succ_full=(mean([episode["success"] for episode in self.mode_eps["full"]])
                                  if self.has_modes else None),
                       grasp100=mean([stage >= 1 for stage in stages(grasp_src)]) if stages(grasp_src) else None,
                       stage100=mean(stages(window)), dphi100=mean(dphi(window)),
                       simfail100=mean([episode.get("sim_fail", False) for episode in window]),
                       disc_ret100=mean([episode.get("disc_ret", episode["ret"]) for episode in window]),
                       ret100=mean([episode["ret"] for episode in window]),
                       len100=mean([episode["len"] for episode in window]))
        terms = collections.defaultdict(list)
        for episode in window:
            for key, value in episode.get("r_terms", {}).items():
                terms[key].append(value)
        metrics["r_terms100"] = {key: mean(value) for key, value in sorted(terms.items())}
        modes = {}
        for mode in sorted(set(MODES) | {episode["mode"] for episode in new_episodes}):
            new_mode, window_mode = ([episode for episode in new_episodes if episode["mode"] == mode],
                                     list(self.mode_eps[mode]))
            if mode not in MODES and not new_mode:
                continue
            modes[mode] = dict(ema=self.succ_ema.get(mode, 0.0), n=len(new_mode), total=self.mode_total.get(mode, 0),
                               succ=mean([episode["success"] for episode in window_mode]),
                               grasp=(mean([stage >= 1 for stage in stages(window_mode)])
                                      if mode != "grasped" and stages(window_mode) else None),
                               stage_hist=hist(stages(window_mode)),
                               simfail=mean([episode.get("sim_fail", False) for episode in window_mode]),
                               dphi=mean(dphi(window_mode)), ret=mean([episode["ret"] for episode in window_mode]),
                               len=mean([episode["len"] for episode in window_mode]))
        metrics["modes"] = modes
        metrics["info"] = {key: mean([episode[key] for episode in window if key in episode])
                           for key in self.cfg.info_keys}
        return metrics

    # ---------------------------------------------------------------------------------------- evaluation
    def _queue_eval(self, ckpt_path, step):
        """Queue the deterministic (and, if configured, the stochastic) evaluation of a checkpoint."""
        self.eval_queue.append((ckpt_path, step, True))
        if self.cfg.eval_stochastic:
            self.eval_queue.append((ckpt_path, step, False))
        self._poll_eval()

    def _poll_eval(self, block=False):
        """Reap a finished evaluation process (logging its result) and start the next queued one.

        Non-blocking unless ``block`` (then it waits for every queued evaluation). An evaluation still running after
        eval_timeout is killed and logged as failed, so one stuck evaluation never stalls all later ones (and best.pt)
        for the rest of the run.
        """
        while True:
            entry = self.eval_proc
            if entry is not None:
                process = entry["proc"]
                if block:
                    process.join(max(0.0, entry["t0"] + self.cfg.eval_timeout - time.time()))
                if process.is_alive():
                    if time.time() - entry["t0"] < self.cfg.eval_timeout:
                        return
                    print(f"[eval] step {entry['step']:,}: still running after {self.cfg.eval_timeout:.0f} s -> killed",
                          flush=True)
                    _stop_process(process)
                process.join(5.0)
                self._finish_eval(entry, process.exitcode)
                if not process.is_alive():
                    process.close()
                self.eval_proc = None
            if not self.eval_queue:
                return
            path, step, deterministic = self.eval_queue.popleft()
            os.makedirs(os.path.join(self.cfg.run_dir, "eval"), exist_ok=True)
            out_path = os.path.join(self.cfg.run_dir, "eval", f"eval_{step}{'' if deterministic else '_stoch'}.json")
            if os.path.exists(out_path):  # e.g. from the run before a resume: never report it for a failed evaluation
                os.remove(out_path)
            seeds = list(range(EVAL_SEED0, EVAL_SEED0 + self.cfg.n_eval))
            proc = self.ctx.Process(target=_eval_main, name=f"eval-{step}", daemon=False,  # it runs its own pool
                                    args=(path, self.cfg.env, self.cfg.env_kwargs, seeds, self.cfg.n_eval_workers,
                                          out_path, os.getpid(), self.cfg.eval_episode_timeout, deterministic))
            proc.start()
            self.eval_proc = dict(proc=proc, out=out_path, ckpt=path, step=step, t0=time.time(), det=deterministic)
            if not block:
                return

    def _finish_eval(self, entry, exitcode):
        """Log a finished evaluation, update the evaluation histories and copy the checkpoint to best.pt if it is the
        best so far (deterministic success, ties broken by the lower mean final shape error, then the later
        checkpoint)."""
        deterministic = entry.get("det", True)
        tag = "eval" if deterministic else "eval-stoch"
        try:
            with open(entry["out"]) as handle:
                result = json.load(handle)
        except Exception as exc:
            print(f"[{tag}] step {entry['step']:,}: evaluation failed (exit code {exitcode}): {exc!r}", flush=True)
            return
        record = dict(step=entry["step"], ckpt=os.path.basename(entry["ckpt"]), deterministic=deterministic,
                      success=result["success_rate"], n_success=result["n_success"], n=result["n"],
                      mean_return=result["mean_return"], mean_steps=result["mean_steps"],
                      mean_shape=result["mean_shape"], grasp_rate=result.get("grasp_rate"),
                      mean_stage=result.get("mean_stage"), mean_dphi=result.get("mean_dphi"),
                      sim_fail_rate=result.get("sim_fail_rate"), stage_hist=result["stage_hist"],
                      errors=result["errors"], eval_s=round(time.time() - entry["t0"], 1),
                      episodes=result["episodes"])
        if deterministic:
            key = _eval_key(record["success"], record["mean_shape"])
            record["is_best"] = key >= self.best_key
            if record["is_best"]:
                self.best_key = key
                best = os.path.join(self.cfg.run_dir, "best.pt")
                shutil.copyfile(entry["ckpt"], best + ".tmp")
                os.replace(best + ".tmp", best)
            record["best"] = self.best_eval
            self.eval_history.append(record)
            _atomic_json(self.eval_history, os.path.join(self.cfg.run_dir, "eval_history.json"))
            self.last_eval = record
        else:
            self.eval_stoch_history.append(record)
            _atomic_json(self.eval_stoch_history, os.path.join(self.cfg.run_dir, "eval_stoch_history.json"))
            self.last_eval_stoch = record
        self._write_metrics(dict(type="eval" if deterministic else "eval_stochastic",
                                 **{key: value for key, value in record.items() if key != "episodes"}))
        show = lambda value, spec: "-" if value is None else format(value, spec)  # noqa: E731
        best = f" | best {self.best_eval:.3f}{' -> best.pt' if record['is_best'] else ''}" if deterministic else ""
        print(f"[{tag}] step {record['step']:,}: held-out success {record['success']:.3f} ({record['n_success']}/"
              f"{record['n']}){best} | grasp {show(record['grasp_rate'], '.2f')} | stage "
              f"{show(record['mean_stage'], '.2f')} | dphi {show(record['mean_dphi'], '+.2f')} | shape "
              f"{show(record['mean_shape'], '.3f')} | sim fail {show(record['sim_fail_rate'], '.2f')} | stages "
              f"{record['stage_hist']} | steps {show(record['mean_steps'], '.0f')} | errors {record['errors']} | "
              f"{record['eval_s']:.0f} s", flush=True)

    # ---------------------------------------------------------------------------------------- main loop
    def _write_metrics(self, record):
        """Append one record to metrics.jsonl."""
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(record, default=_np_default) + "\n")

    def setup(self):
        """Write config.json and start the rollout workers."""
        _atomic_json(asdict(self.cfg), os.path.join(self.cfg.run_dir, "config.json"))
        start_time = time.time()
        self._start_workers()
        sampling = (f"preempt {self.cfg.preempt_frac:.2f} of T={self.cfg.rollout} (max {self._rollout_max()})"
                    if self.cfg.preempt_frac < 1.0 else f"fixed T={self.cfg.rollout}")
        print(f"[async_ppo] {self.cfg.workers} workers ready in {time.time() - start_time:.1f} s | obs {self.obs_dim} "
              f"act {self.act_dim} | learner {self.device} | {sampling} | gamma {self.cfg.gamma} (env "
              f"{'n/a' if self.env_gamma is None else self.env_gamma}) | lr {self.cfg.lr_schedule} | curriculum "
              f"{self.cfg.curriculum if self.has_modes else 'n/a (env has no set_start_probs)'}", flush=True)

    def iterate(self):
        """One training iteration: collect fragments, update the networks, merge the observation statistics, log,
        checkpoint and queue evaluations.

        Returns
        -------
        dict or None
            The iteration's metrics record, or None if no worker returned a fragment.
        """
        cfg = self.cfg
        start_time = time.perf_counter()
        probs = self._start_probs()
        self.shared_np[:] = parameters_to_vector(self.actor.parameters()).detach().cpu().numpy()
        results = self._collect(probs)
        t_sample = time.perf_counter() - start_time
        if not results:
            print("[async_ppo] no fragments this iteration (all workers failed); retrying", flush=True)
            return None
        batch = self._batch(results)
        lr = self._lr()
        ppo = self._update(batch, lr)
        self.lr = self._adapted_lr(ppo)
        # The observation statistics change only AFTER the iteration: workers and stored observations stay consistent.
        for result in results:
            self.obs_rms.merge(result["raw_mean"], result["raw_var"], result["raw_count"])
        self.global_step += batch["n"]
        self.iteration += 1
        errors = int(sum(result["errors"] for result in results))
        self.total_errors += errors
        for result in results:
            for message in result["error_msgs"][:2]:
                print(f"[env error] {message}", flush=True)
        episode_metrics = self._episode_metrics([episode for result in results for episode in result["episodes"]])
        t_learn = time.perf_counter() - start_time

        # Checkpoints: last.pt every iteration, ckpt_<step>.pt periodically and before every evaluation.
        state = self._state()
        _atomic_save(state, os.path.join(cfg.run_dir, "last.pt"))
        ckpt = None
        if self.global_step >= self.next_ckpt:
            ckpt = self._save_ckpt(state)
            self.next_ckpt = self._next(cfg.ckpt_every)
        do_eval = self.global_step >= self.next_eval
        if do_eval:
            ckpt = ckpt or self._save_ckpt(state)
        t_iter = time.perf_counter() - start_time  # checkpoint writes included: sps is the true wall-clock rate
        lengths = batch["lengths"]
        record = dict(type="iter", iter=self.iteration, step=self.global_step, time=time.time(),
                      sps=batch["n"] / t_iter, sample_s=t_sample, update_s=t_iter - t_sample,
                      save_s=t_iter - t_learn, sample_sps=batch["n"] / t_sample, workers=len(results), n=batch["n"],
                      frag_min=int(lengths.min()), frag_mean=float(lengths.mean()), frag_max=int(lengths.max()),
                      preempted=int((lengths < cfg.rollout).sum()), stop_s=self._stop_s,
                      worker_env_frac=float(np.mean([result["env_s"] / max(result["sample_s"], 1e-9)
                                                     for result in results])),
                      # mean worker busy time / sampling wall time: 1 - the share of time the workers idle at the
                      # per-iteration barrier, waiting for the slowest fragment
                      barrier_eff=float(np.mean([result["sample_s"] for result in results]) / max(t_sample, 1e-9)),
                      errors=errors, total_errors=self.total_errors, restarts=self.total_restarts,
                      start_probs=probs, **episode_metrics, ppo=ppo)
        self._write_metrics(record)
        self._print(record)
        if do_eval:
            self._queue_eval(ckpt, self.global_step)
            self.next_eval = self._next(cfg.eval_every)
        self._poll_eval()
        return record

    def _save_ckpt(self, state=None):
        """Write ckpt_<global_step>.pt and return its path."""
        path = os.path.join(self.cfg.run_dir, f"ckpt_{self.global_step}.pt")
        _atomic_save(state or self._state(), path)
        return path

    def _print(self, record):
        """Print the one-line summary of an iteration."""
        show = lambda value, spec: "-" if value is None else format(value, spec)  # noqa: E731
        ppo = record["ppo"]
        line = f"[it {record['iter']}] step {record['step']:,} | succ {show(record['succ100'], '.2f')}"
        if self.has_modes:
            line += f" (full {show(record['succ_full'], '.2f')})"
        line += (f" | grasp {show(record['grasp100'], '.2f')} | stage {show(record['stage100'], '.2f')} | dphi "
                 f"{show(record['dphi100'], '+.2f')} | simfail {show(record['simfail100'], '.3f')}")
        if self.last_eval is not None:
            line += f" | eval@{self.last_eval['step']:,} {self.last_eval['success']:.2f} (best {self.best_eval:.2f})"
        if self.last_eval_stoch is not None:
            line += f" stoch {self.last_eval_stoch['success']:.2f}"
        line += f" | len {show(record['len100'], '.0f')}"
        if self.has_modes:
            line += " | ema " + " ".join(f"{mode} {record['modes'][mode]['ema']:.2f}" for mode in MODES)
            line += " | p " + "/".join(f"{record['start_probs'][mode]:.2f}" for mode in MODES)
        for key, value in record["info"].items():
            line += f" | {key} {show(value, '.3g')}"
        line += (f" | {record['sps']:,.0f} sps (sample {record['sample_s']:.1f}s, barrier {record['barrier_eff']:.2f}, "
                 f"frag {record['frag_min']}-{record['frag_max']}, update {record['update_s']:.1f}s)")
        line += (f" | kl {show(ppo.get('approx_kl'), '.4f')} clip {show(ppo.get('clip_frac'), '.3f')} ev "
                 f"{ppo['explained_var']:.2f} std {ppo['std']:.3f} lr {ppo['lr']:.1e} ep {ppo['epochs_done']:.1f} | "
                 f"logp_err {ppo['logp_err']:.1e}")
        if record["errors"] or record["restarts"]:
            line += f" | env errors {record['errors']} restarts {record['restarts']}"
        print(line, flush=True)

    def finish(self):
        """Evaluate the final weights too (unless just done) and wait for every pending evaluation."""
        cfg = self.cfg
        if cfg.eval_every > 0 and cfg.final_eval:
            done = {entry["step"] for entry in self.eval_history} | {queued[1] for queued in self.eval_queue}
            if self.eval_proc is not None:
                done.add(self.eval_proc["step"])
            if self.global_step not in done and self.global_step > 0:
                self._queue_eval(self._save_ckpt(), self.global_step)
        self._poll_eval(block=True)

    def train(self):
        """Run the whole training: set up, iterate until ``cfg.steps``, final evaluation, close."""
        try:
            self.setup()
            while self.global_step < self.cfg.steps:
                self.iterate()
            self.finish()
        finally:
            self.close()

    def close(self):
        """Stop the workers (politely, then by force) and any running evaluation process."""
        for handle in self.workers:
            try:
                handle.conn.send(("close",))
            except Exception:
                pass
        deadline = time.time() + 10.0
        for handle in self.workers:
            handle.proc.join(max(0.1, deadline - time.time()))
        for handle in self.workers:
            handle.stop()
        self.workers = []
        if self.eval_proc is not None:
            _stop_process(self.eval_proc["proc"])
            self.eval_proc = None
