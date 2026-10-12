"""Verify a pure-RL (async PPO) policy on never-seen rope layouts (seeds 9,000,000 .. 9,000,000 + n - 1).

The policy acts deterministically (action mean, clipped; --stochastic 1 samples actions, with the torch random
generator seeded per layout) in eval=True environments, reset with reset(seed=s, options={'layout_seed': s,
'no_prefix': True}); the environment's scripted expert is disabled. After a success, if the environment has
release_and_hold(), the gripper opens and moves away and the rope must STAY wrapped (the release-and-hold test).

Outputs in data/runs/<run>/verify100_<model>[_stoch]/:
  results.json          summary and per-layout metrics
  summary.md            success rate with 95% Wilson CI, held-after-release rate, grasp rate, simulator-failure
                        rate, potential progress, stage histogram of the failures
  montage_initial.png   initial layouts (rope environments only)
  montage_final.png     final rope states: green = success and held after release, red = failure

  cd project_superdex && .venv/bin/python ../rope_routing/scripts/verify_rl.py --run rl1 --model best --workers 20

The environment factory and its arguments come from the checkpoint unless --env / --env_kwargs override them. Results
of the final policy: notes/D_results.md.
"""

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as tmp

from rope_routing.rl import async_ppo as ap

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEED0 = 9_000_000  # first verification layout
_W = {}  # per-process state of a pool worker: environment factory, environment, policy, mode, task geometry


def wilson(successes, trials, z_score=1.96):
    """Wilson score interval for a binomial proportion (sound for small samples and rates near 0 or 1).

    Parameters
    ----------
    successes, trials : int
        Number of successes and of trials.
    z_score : float
        Normal quantile of the confidence level (1.96 for 95%).

    Returns
    -------
    tuple of float
        Lower and upper bound; (0, 1) without trials.

    Examples
    --------
    >>> [round(bound, 3) for bound in wilson(87, 100)]
    [0.79, 0.922]
    """
    if trials == 0:
        return 0.0, 1.0
    rate, denom = successes / trials, 1.0 + z_score * z_score / trials
    centre = (rate + z_score * z_score / (2 * trials)) / denom
    half_width = z_score * math.sqrt(rate * (1 - rate) / trials + z_score * z_score / (4 * trials * trials)) / denom
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _rope_xy(env):
    """Current rope node positions (x, y) as a list, or None if the environment has no rope simulator."""
    sim = getattr(env.unwrapped, "sim", None)
    return np.asarray(sim.rope_nodes())[:, :2].tolist() if hasattr(sim, "rope_nodes") else None


def _floats(value):
    """A scalar or array as a list of floats (None stays None)."""
    return None if value is None else [float(item) for item in np.atleast_1d(value)]


def _init(model_path, factory, env_kwargs, parent_pid, deterministic=True):
    """Initialiser of a pool worker: build the environment, load the policy, record the task geometry."""
    ap._die_with_parent(parent_pid)
    torch.set_num_threads(1)
    ident = tmp.current_process()._identity
    rank = int(ident[-1]) if ident else 0
    _W["make"] = lambda: ap._forbid_expert(ap.resolve(factory)(rank=rank, seed=SEED0 + rank, eval=True, **env_kwargs))
    env = _W["make"]()
    task = getattr(env.unwrapped, "task", None)
    _W.update(env=env, policy=ap.load_policy(model_path), det=bool(deterministic), task=None if task is None else dict(
        route=np.asarray(task.route).tolist(), pegs=np.asarray(task.pegs).tolist(), peg_radius=float(task.peg_radius)))


def _failed(seed, error):
    """Record of a crashed episode (a failure, not held after release)."""
    record = ap.failed_episode(seed, error)
    record.update(held_after_release=False, routed=False, wind_frac=None, max_strain=None, _task=None)
    return record


def _episode(seed):
    """One verification episode, followed by the release-and-hold test after a success.

    A crash never stops the verification: an episode that crashed is a failure; a crash after the episode ended
    (in the release test or the read-out) keeps the episode's outcome, marked as not held. The environment is then
    rebuilt so that a broken simulator does not fail the worker's remaining layouts.
    """
    env, start_time, initial, record = _W["env"], time.time(), {}, None
    try:
        record, info = ap.run_episode(env, _W["policy"], seed, deterministic=_W["det"],
                                      on_reset=lambda reset_env: initial.update(xy=_rope_xy(reset_env)))
        record.update(routed=bool(info.get("routed", False)), wind_frac=_floats(info.get("wind_frac")),
                      max_strain=ap._num(info.get("max_strain")), held_after_release=False)
        release = ap.env_attr(env, "release_and_hold")
        if record["success"] and release is not None:
            hold = release()
            record.update(held_after_release=bool(hold["held"]), shape_after_release=float(hold["shape"]),
                          wind_frac_after_release=_floats(hold["wind_frac"]))
        record.update(x0=initial.get("xy"), x_final=_rope_xy(env))
    except Exception as exc:  # a crash never stops the verification
        if record is None:  # the episode itself crashed: a failure
            record = _failed(seed, repr(exc))
        else:  # the release test or read-out crashed after the episode ended: its outcome stands, not held
            for key, value in dict(routed=False, wind_frac=None, max_strain=None).items():
                record.setdefault(key, value)
            record.update(held_after_release=False, error=repr(exc)[:300])
        record.setdefault("x0", initial.get("xy"))
        try:  # do not let a broken simulator fail the remaining layouts of this worker
            env.close()
        except Exception:
            pass
        try:
            _W["env"] = _W["make"]()
        except Exception:
            pass
    record["wall_s"] = round(time.time() - start_time, 1)
    record["_task"] = _W["task"]
    return record


def montage(results, key, path, task, title):
    """Save a grid of small plots of the rope states ``results[i][key]`` over the target route and the pegs.

    Initial layouts ("x0") are drawn in blue; final states in green (success and held after release) or red.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 10
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows), squeeze=False)
    for cell in axes.flat:
        cell.axis("off")
    route = np.asarray(task["route"])
    for cell, result in zip(axes.flat, results):
        if result.get(key) is None:
            continue
        xy = np.asarray(result[key])
        ok = result["success"] and (key == "x0" or result["held_after_release"])
        cell.plot(route[:, 0], route[:, 1], "--", color="0.7", lw=0.6)
        color = "tab:blue" if key == "x0" else ("tab:green" if ok else "tab:red")
        cell.plot(xy[:, 0], xy[:, 1], "-", lw=1.6, color=color)
        for peg in task["pegs"]:
            cell.add_patch(plt.Circle(peg, task["peg_radius"], color="k"))
        cell.set_aspect("equal")
        cell.set_xlim(-0.05, 0.75)
        cell.set_ylim(-0.7, 0.25)
        cell.set_title(str(result["seed"] % 1000), fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    """Run the verification in a process pool and write results.json, summary.md and the montages."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="best", help="best | last | ckpt_<steps> | path to a .pt file")
    parser.add_argument("--n", type=int, default=100, help="layouts 9000000 .. 9000000+n-1")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--env", default=None,
                        help="env factory 'module:function' (default: the one in the checkpoint)")
    parser.add_argument("--env_kwargs", default=None, help="JSON dict (default: the kwargs in the checkpoint)")
    parser.add_argument("--stochastic", type=int, choices=[0, 1], default=0,
                        help="1: sample actions from the policy (seeded per layout) instead of the mean")
    parser.add_argument("--episode_timeout", type=float, default=900.0,
                        help="layouts with no result for this long (a worker died or hung) count as crashed failures")
    args = parser.parse_args()

    # Model, environment and output directory.
    run_dir = os.path.join(ROOT, "data", "runs", args.run)
    model_path = args.model if args.model.endswith(".pt") else os.path.join(run_dir, args.model + ".pt")
    name = os.path.splitext(os.path.basename(model_path))[0]
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    factory = args.env or ckpt["env"]["factory"]
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else dict(ckpt["env"]["kwargs"])
    step = int(ckpt.get("global_step", 0))
    del ckpt
    out_dir = os.path.join(run_dir, f"verify100_{name}{'_stoch' if args.stochastic else ''}")
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + args.n))
    print(f"[verify_rl] {model_path} ({step:,} env steps) | env {factory} {env_kwargs} | {args.n} layouts, "
          f"{args.workers} workers | {'stochastic' if args.stochastic else 'deterministic'} policy", flush=True)

    # Episodes, in a pool of single-threaded worker processes.
    start_time, results = time.time(), []
    with ap._single_thread_env():
        pool = tmp.get_context("spawn").Pool(min(args.workers, args.n), initializer=_init,
                                             initargs=(model_path, factory, env_kwargs, os.getpid(),
                                                       not args.stochastic))
    with pool:
        for result in ap.pool_results(pool, _episode, seeds, args.episode_timeout):
            if "_task" not in result:  # no result: the pool worker died (e.g. a simulator segfault) or hung
                result = {**_failed(result["seed"], result["error"]), "wall_s": None}
            results.append(result)
            print(f"[{len(results):3d}/{args.n}] seed {result['seed']}: success={result['success']} "
                  f"held={result['held_after_release']} "
                  f"shape={result['shape'] if result['shape'] is None else round(result['shape'], 3)} "
                  f"stage={result['max_stage']} "
                  f"steps={result['steps']}{' ERROR ' + result['error'] if result.get('error') else ''}", flush=True)
    results.sort(key=lambda result: result["seed"])
    task = next((result["_task"] for result in results if result["_task"] is not None), None)
    for result in results:
        result.pop("_task")

    # Summary statistics and results.json.
    num_layouts = len(results)
    success = np.array([result["success"] for result in results])
    held = np.array([result["held_after_release"] for result in results])
    fail_stages = [result["max_stage"] for result in results if not result["success"]]
    hist = lambda values: dict(sorted(collections.Counter("none" if value is None else str(value)  # noqa: E731
                                                          for value in values).items()))
    mean = lambda values: float(np.mean(values)) if len(values) else None  # noqa: E731
    strains = [result["max_strain"] for result in results if result.get("max_strain") is not None]
    summary = dict(
        algo="async PPO, pure RL (no demonstrations / behaviour cloning / expert)", model=name, model_path=model_path,
        global_step=step, env=factory, env_kwargs=env_kwargs, n=num_layouts, seeds=[SEED0, SEED0 + args.n - 1],
        n_success=int(success.sum()), success_rate=float(success.mean()),
        success_ci95=wilson(int(success.sum()), num_layouts),
        n_held=int(held.sum()), held_after_release_rate=float(held.mean()),
        held_ci95=wilson(int(held.sum()), num_layouts),
        routed_rate=float(np.mean([result["routed"] for result in results])),
        deterministic=not args.stochastic,
        grasp_rate=mean([result["max_stage"] >= 1 for result in results if result["max_stage"] is not None]),
        sim_fail_rate=float(np.mean([bool(result.get("sim_fail")) for result in results])),
        mean_dphi=mean([result["dphi"] for result in results if result.get("dphi") is not None]),
        mean_shape_success=mean([result["shape"] for result in results if result["success"]]),
        mean_steps_success=mean([result["steps"] for result in results if result["success"]]),
        mean_drops=mean([result["drops"] for result in results]), max_strain=max(strains) if strains else None,
        stage_histogram=hist([result["max_stage"] for result in results]),
        failure_stage_histogram=hist(fail_stages),
        errors=int(sum(bool(result.get("error")) for result in results)), wall_s=round(time.time() - start_time, 1),
    )
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(dict(summary=summary, episodes=results), handle, default=ap._np_default)

    # summary.md and the montages.
    pct = lambda interval: f"{interval[0]:.0%}-{interval[1]:.0%}"  # noqa: E731
    show = lambda value, spec: "-" if value is None else format(value, spec)  # noqa: E731
    lines = [f"# Verification on {num_layouts} random layouts (async PPO, pure RL; model `{name}` at {step:,} env "
             f"steps, {'stochastic' if args.stochastic else 'deterministic'} policy)", "",
             f"- success (grasped + routed + shape): **{success.mean():.0%}** ({success.sum()}/{num_layouts}; 95% "
             f"Wilson CI {pct(summary['success_ci95'])})",
             f"- held after release: **{held.mean():.0%}** ({held.sum()}/{num_layouts}; 95% CI "
             f"{pct(summary['held_ci95'])})",
             f"- routed (all windings >= 85%): {summary['routed_rate']:.0%}",
             f"- grasp rate (episodes reaching stage >= 1): {show(summary['grasp_rate'], '.0%')}",
             f"- simulator failures (divergence / 'bad' step / crash): {summary['sim_fail_rate']:.0%}",
             f"- mean potential progress phi_end - phi_start: {show(summary['mean_dphi'], '+.2f')}",
             f"- mean shape error (successes): {show(summary['mean_shape_success'], '.4f')} m",
             f"- mean steps (successes): {show(summary['mean_steps_success'], '.1f')}",
             f"- mean drops per episode: {show(summary['mean_drops'], '.2f')}",
             f"- max strain: {show(summary['max_strain'], '.4f')}",
             f"- furthest stage reached by the {len(fail_stages)} FAILURES: {summary['failure_stage_histogram']}",
             f"- furthest stage histogram (all episodes): {summary['stage_histogram']}",
             f"- crashed episodes (exceptions): {summary['errors']}",
             f"- env: `{factory}` {env_kwargs}; layouts {SEED0}..{SEED0 + args.n - 1} (never used in training)", "",
             "| seed | success | held | shape [m] | steps | max stage | drops |", "|---|---|---|---|---|---|---|"]
    for result in results:
        lines.append(f"| {result['seed']} | {result['success']} | {result['held_after_release']} | "
                     f"{show(result['shape'], '.3f')} | {result['steps']} | {result['max_stage']} | {result['drops']} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    # Cells without a rope state stay empty.
    if task is not None and any(result.get("x0") is not None for result in results):
        montage(results, "x0", os.path.join(out_dir, "montage_initial.png"), task,
                f"{num_layouts} random initial layouts")
        montage(results, "x_final", os.path.join(out_dir, "montage_final.png"), task,
                f"final states: success {success.mean():.0%}, held after release {held.mean():.0%}")
    print(json.dumps(summary, indent=1, default=ap._np_default))
    print(f"[verify_rl] wrote {out_dir}")


if __name__ == "__main__":
    main()
