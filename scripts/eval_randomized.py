"""Held-out evaluation of a policy with RANDOMISED ROPE PHYSICS: n episodes, each on an unseen layout AND with a rope
whose physical parameters are drawn at random (the policy was trained on the nominal rope only).

Randomised per episode (multiplicative factors are log-uniform):
  linear density x[0.6, 1.6], axial stiffness EA x[0.5, 2], bending stiffness EI x[0.4, 2.5],
  torsional stiffness GJ x[0.5, 2], mass damping x[0.5, 2], rope friction U[0.4, 0.8],
  table friction U[0.4, 0.8], peg friction U[0.3, 0.7]
The radius is fixed: the goal is a target surface at that radius.

Each episode runs the deterministic policy from the robot's home pose until success or the 50 s time limit; after a
success the gripper releases and moves away, and the rope must stay routed (env.release_and_hold).

Writes <out>/results.json (every episode: parameters and outcome) and <out>/summary.md (success rates); <out> is
data/runs/<run>/eval_randomized_rope by default. Results of the final policy: notes/D_results.md, section 2.

  cd project_superdex && .venv/bin/python ../rope_routing/scripts/eval_randomized.py --run generic_ppo3 --model best
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED0 = 9_100_000  # layouts never used in training (own RNG streams there; held-out evaluations use 700000+, 9000000+)
# Randomisation ranges: log-uniform factors on the nominal SceneCfg value, and uniform absolute values.
LOGU = dict(rope_linear_density=(0.6, 1.6), rope_axial_stiffness=(0.5, 2.0), rope_bending_stiffness=(0.4, 2.5),
            rope_torsional_stiffness=(0.5, 2.0), rope_mass_damping=(0.5, 2.0))
UNIF = dict(rope_friction=(0.4, 0.8), table_friction=(0.4, 0.8), peg_friction=(0.3, 0.7))


def rope_settings(seed):
    """Randomised rope parameters of episode ``seed`` (reproducible): SceneCfg field name -> value.

    Examples
    --------
    >>> settings = rope_settings(9_100_000)
    >>> len(settings), round(settings["rope_friction"], 3), round(settings["rope_bending_stiffness"] / 3e-4, 3)
    (8, 0.518, 0.426)
    """
    from rope_routing.sim.grasp_scene import SceneCfg

    rng = np.random.default_rng(seed)
    nominal = SceneCfg()
    settings = {name: float(getattr(nominal, name) * np.exp(rng.uniform(np.log(low), np.log(high))))
                for name, (low, high) in LOGU.items()}
    settings.update({name: float(rng.uniform(low, high)) for name, (low, high) in UNIF.items()})
    return settings


def episode(args):
    """Run one evaluation episode in a pool worker.

    Parameters
    ----------
    args : tuple of (str, int)
        Checkpoint path and seed (layout and rope parameters).

    Returns
    -------
    dict
        "seed", "rope" (the parameters), "success", "held_after_release", "steps", "coverage", "wrap", "grasped",
        "sim_fail", "error" and "wall_s". A crash is recorded as a failure, never dropped.
    """
    model_path, seed = args
    import torch

    from rope_routing.rl.async_ppo import load_policy, resolve

    torch.set_num_threads(1)
    policy = load_policy(model_path)
    rope = rope_settings(seed)
    start_time = time.time()
    try:
        env = resolve(policy.env["factory"])(rank=0, seed=seed, eval=True,
                                             **{**policy.env.get("kwargs", {}), "scene_cfg": rope})
        obs, _ = env.reset(seed=seed, options={"layout_seed": seed})
        info, steps = {}, 0
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            steps += 1
            if terminated or truncated:
                break
        record = dict(seed=seed, rope=rope, success=bool(info["is_success"]), steps=steps,
                      coverage=float(info.get("coverage", info.get("shape", np.nan))),
                      wrap=np.round(np.asarray(info["wind_frac"], float), 3).tolist(), grasped=bool(info["grasped"]),
                      sim_fail=bool(info["r_terms"].get("bad", 0) != 0), error=None)
        record["held_after_release"] = bool(env.release_and_hold()["held"]) if record["success"] else False
    except Exception as exc:  # recorded as a failure, never dropped
        record = dict(seed=seed, rope=rope, success=False, held_after_release=False, steps=0, coverage=None,
                      wrap=None, grasped=False, sim_fail=True, error=repr(exc))
    record["wall_s"] = round(time.time() - start_time, 1)
    print(f"[{seed}] success={record['success']} held={record['held_after_release']} steps={record['steps']} "
          f"cov={record['coverage']} err={record['error']}", flush=True)
    return record


def wilson(successes, trials, z_score=1.96):
    """Wilson score interval for a binomial proportion; (0, 0) without trials.

    Examples
    --------
    >>> [round(float(bound), 3) for bound in wilson(88, 100)]
    [0.802, 0.93]
    """
    if trials == 0:
        return (0.0, 0.0)
    rate = successes / trials
    denom = 1 + z_score * z_score / trials
    centre = (rate + z_score * z_score / (2 * trials)) / denom
    half_width = z_score * np.sqrt(rate * (1 - rate) / trials + z_score * z_score / (4 * trials * trials)) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def main():
    """Evaluate the policy on --n randomised episodes in a process pool and write results.json and summary.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="best")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    model_path = os.path.join(ROOT, "data", "runs", args.run, args.model + ".pt")
    out_dir = args.out or os.path.join(ROOT, "data", "runs", args.run, "eval_randomized_rope")
    os.makedirs(out_dir, exist_ok=True)
    seeds = [SEED0 + index for index in range(args.n)]
    start_time = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(episode, [(model_path, seed) for seed in seeds], chunksize=1)

    # Rates and results.json.
    count = len(results)
    n_success = sum(result["success"] for result in results)
    n_held = sum(result["held_after_release"] for result in results)
    fails = sum(result["sim_fail"] for result in results)
    json.dump(dict(run=args.run, model=args.model, model_path=model_path, n=count, seeds=[seeds[0], seeds[-1]],
                   randomization=dict(log_uniform_factors=LOGU, uniform=UNIF), episodes=results),
              open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    succ_lo, succ_hi = wilson(n_success, count)
    held_lo, held_hi = wilson(n_held, count)

    # summary.md: rates, then one row per episode with the rope's main parameters.
    lines = [f"# Randomized-rope evaluation: `{args.run}/{args.model}` (pure RL, deterministic policy)", "",
             f"- episodes: {count} unseen layouts (seeds {seeds[0]}..{seeds[-1]}), each with randomized rope physics",
             f"- **success: {n_success}/{count} = {n_success / count:.0%}** "
             f"(95% Wilson CI {succ_lo:.0%}-{succ_hi:.0%})",
             f"- **held after release: {n_held}/{count} = {n_held / count:.0%}** (95% CI {held_lo:.0%}-{held_hi:.0%})",
             f"- simulator failures / crashes: {fails}",
             f"- mean coverage of successes: "
             f"{np.mean([result['coverage'] for result in results if result['success']] or [np.nan]):.4f} m",
             f"- wall time: {time.time() - start_time:.0f} s", "",
             "Randomized per episode (log-uniform factors on the nominal value): "
             + ", ".join(f"{name} x[{low}, {high}]" for name, (low, high) in LOGU.items()),
             "; uniform: " + ", ".join(f"{name} [{low}, {high}]" for name, (low, high) in UNIF.items()) + ".", "",
             "| seed | success | held | steps | coverage [m] | EI x | density x | rope mu |",
             "|---|---|---|---|---|---|---|---|"]
    from rope_routing.sim.grasp_scene import SceneCfg
    nominal = SceneCfg()
    for result in results:
        coverage = "-" if result["coverage"] is None else f"{result['coverage']:.3f}"
        lines.append(f"| {result['seed']} | {result['success']} | {result['held_after_release']} | {result['steps']} | "
                     f"{coverage} | {result['rope']['rope_bending_stiffness'] / nominal.rope_bending_stiffness:.2f} | "
                     f"{result['rope']['rope_linear_density'] / nominal.rope_linear_density:.2f} | "
                     f"{result['rope']['rope_friction']:.2f} |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]), flush=True)


if __name__ == "__main__":
    main()
