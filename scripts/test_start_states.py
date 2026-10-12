"""Test of the environment-generated start states (rope_routing/env/start_states.py): no expert, no demonstrations.

For each layout (seeds 600000 and up, never the evaluation seeds 700000+ or the verification seeds 9000000+): reset and
set up "hover", reset and set up "grasped" (and a "full" identity check). Reports the weld success, the setup wall and
simulated time, the TCP error against the target, the largest rope strain during the setup, the hover geometry, the
stationarity under zero actions after the hand-over, and that the first env.step(zero action) returns a finite
observation and reward without a stale-potential spike (the counterfactual spike with the reset's potential is
reported too). The rope's activity after the reset (robot static, 3 s) is measured per layout, because a hopping rope
stalls the arm through the shared solver.

  ssh d3 'cd .../project_superdex && .venv/bin/python ../rope_routing/scripts/test_start_states.py --n 30'

The environment is SurfaceRouteEnv (make_env), the environment the policy is trained in: its action map (release
debounce) is the one the setup drives, and no ScriptedExpert object exists in it (GraspRouteEnv would build one and
reset it every episode).

--scene picks the GraspScene physics: "default" (SceneCfg(), the "C1-nc" scene of notes C that the environment trains
on) or "legacy_v3" (SceneCfg.legacy_v3(), the old scene the driver was first tuned on, for before/after comparisons).
Results are written to data/runs/start_states/ by default (--out).
"""

import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def rope_activity(env, seed, seconds=3.0):
    """Largest rope node speed (after the first 0.5 s) [m/s] and largest node height [m], with the robot static at
    home after a reset."""
    env.reset(seed=seed, options={"layout_seed": seed, "no_prefix": True})
    sim = env.sim
    prev, max_speed, max_height = sim.rope_nodes(), 0.0, 0.0
    for step in range(int(seconds * 10)):
        for _ in range(env.substeps):
            sim.substep()
        nodes = sim.rope_nodes()
        if step >= 5:
            max_speed = max(max_speed, float(np.linalg.norm(nodes - prev, axis=1).max()) / (env.substeps * sim.dt))
            max_height = max(max_height, float(nodes[:, 2].max()))
        prev = nodes
    return max_speed, max_height


def run_mode(env, seed, mode):
    """Reset on layout ``seed``, set up the start ``mode`` and record the setup statistics and the first steps."""
    from rope_routing.env.grasp_route_env import wrap_half
    from rope_routing.env.start_states import setup_start

    obs0, _ = env.reset(seed=seed, options={"layout_seed": seed, "no_prefix": True})
    sim = env.sim
    phi_reset = float(env.phi)
    rng = np.random.default_rng([seed, {"full": 0, "hover": 1, "grasped": 2}[mode]])
    ok, info = setup_start(env, mode, rng)
    obs = info["obs"]
    row = dict(seed=seed, mode=mode, ok=bool(ok), reached=info["reached"], wall=info["wall"], sim_time=info["sim_time"],
               obs_ok=bool(obs is env._last_obs and obs.shape == env.observation_space.shape
                           and np.all(np.isfinite(obs))))
    if mode == "full":
        row["full_identity"] = bool(np.array_equal(obs, obs0))
        return row
    nodes = sim.rope_nodes()
    grasp_node = env._gi
    tcp = sim.tcp()
    row.update(steps=info["steps"], tcp_err=info["tcp_err"], rest_err=info["rest_err"], tcp_speed=info["tcp_speed"],
               yaw_err=float(abs(wrap_half(sim.tool_yaw() - env._rope_dir_yaw(nodes, grasp_node)))),
               max_strain=info["max_strain"], max_seg_strain=info["max_seg_strain"], finite=info["finite"],
               diverged=info["diverged"],
               attempts=info["attempts"], attempt_log=info["attempt_log"], weld_node=info["weld_node"],
               gi=int(grasp_node), height=float(tcp[2] - nodes[grasp_node, 2]), tcp_z=float(tcp[2]),
               dxy=float(np.linalg.norm(tcp[:2] - nodes[grasp_node, :2])), theta_cmd=float(sim.theta_cmd),
               grasped=bool(env._grasp_state()[0]), phi=float(env.phi), phi_reset=phi_reset)
    # First step after the hand-over: a zero action (as a fresh policy might output); then 4 more for stationarity.
    obs1, reward, terminated, truncated, step_info = env.step(np.zeros(5))
    row.update(step_reward=float(reward), step_obs_finite=bool(np.all(np.isfinite(obs1))), step_te=bool(terminated),
               step_tr=bool(truncated), stale_spike=float(10.0 * (0.99 * env.phi - phi_reset)),
               grasped_after_step=bool(step_info["grasped"]))
    for _ in range(4):
        env.step(np.zeros(5))
    row["drift_5steps"] = float(np.linalg.norm(sim.tcp() - tcp))
    row["grasped_after_5"] = bool(env._grasp_state()[0])
    return row


def fail_mode(row):
    """Short classification of a setup's outcome ("ok" or the failure mode)."""
    if row["ok"]:
        return "ok"
    if not row.get("finite", True):
        return "solver diverged" if row.get("diverged") else "non-finite"
    if row["reached"] == "none":
        return "at hover, not settled (rope moving)" if row["tcp_err"] < 0.02 else "travel stalled (arm frozen)"
    logs = row.get("attempt_log", [])
    if any(log["weld"] for log in logs):
        return "welded, lift not reached"
    if logs and not any(log["reached_rope"] for log in logs):
        return "descent stalled above the rope"
    return "closed, no weld"


def summarize(rows, activity):
    """Per-mode summary of the setup rows; ``activity`` maps seed -> (max rope speed, max height) after the reset."""
    summary = {}
    for mode in ("hover", "grasped"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        ok = np.array([row["ok"] for row in mode_rows])
        wall = np.array([row["wall"] for row in mode_rows])
        sim_time = np.array([row["sim_time"] for row in mode_rows])
        ok_rows = [row for row in mode_rows if row["ok"]]
        active = (np.array([activity[row["seed"]][0] > 0.03 for row in mode_rows]) if activity
                  else np.zeros(len(mode_rows), bool))
        modes = {}
        for row in mode_rows:
            failure = fail_mode(row)
            modes[failure] = modes.get(failure, 0) + 1
        summary[mode] = dict(
            n=len(mode_rows), success=float(ok.mean()), n_ok=int(ok.sum()),
            success_calm_rope=float(ok[~active].mean()) if (~active).any() else None, n_calm=int((~active).sum()),
            success_active_rope=float(ok[active].mean()) if active.any() else None, n_active=int(active.sum()),
            wall_mean=float(wall.mean()), wall_max=float(wall.max()),
            wall_mean_ok=float(wall[ok].mean()) if ok.any() else None,
            sim_time_mean=float(sim_time.mean()), sim_time_max=float(sim_time.max()),
            ms_per_substep=float(1e3 * wall.sum() / (10 * sum(row["steps"] for row in mode_rows))),
            tcp_err_mm_mean_ok=float(1e3 * np.mean([row["tcp_err"] for row in ok_rows])) if ok_rows else None,
            tcp_err_mm_max_ok=float(1e3 * np.max([row["tcp_err"] for row in ok_rows])) if ok_rows else None,
            rest_err_mm_max_ok=float(1e3 * np.max([row["rest_err"] for row in ok_rows])) if ok_rows else None,
            yaw_err_mean_ok=float(np.mean([row["yaw_err"] for row in ok_rows])) if ok_rows else None,
            max_strain=float(max(row["max_strain"] for row in mode_rows)),
            max_seg_strain=float(max(row["max_seg_strain"] for row in mode_rows)),
            step_reward_absmax=float(max(abs(row["step_reward"]) for row in mode_rows)),
            step_reward_absmax_ok=float(max(abs(row["step_reward"]) for row in ok_rows)) if ok_rows else None,
            stale_spike_absmean=float(np.mean([abs(row["stale_spike"]) for row in mode_rows])),
            step_finite=bool(all(row["step_obs_finite"] and np.isfinite(row["step_reward"]) for row in mode_rows)),
            obs_ok=bool(all(row["obs_ok"] for row in mode_rows)),
            drift_mm_max_ok=float(1e3 * max(row["drift_5steps"] for row in ok_rows)) if ok_rows else None,
            failure_modes=modes)
        if mode == "hover" and ok_rows:
            summary[mode].update(height_cm=[float(100 * min(row["height"] for row in ok_rows)),
                                            float(100 * max(row["height"] for row in ok_rows))],
                                 dxy_mm_max=float(1e3 * max(row["dxy"] for row in ok_rows)),
                                 fingers_open=bool(all(row["theta_cmd"] < 0.05 for row in ok_rows)))
        if mode == "grasped" and ok_rows:
            summary[mode].update(held_after_step=float(np.mean([row["grasped_after_step"] for row in ok_rows])),
                                 held_after_5=float(np.mean([row["grasped_after_5"] for row in ok_rows])),
                                 one_attempt=float(np.mean([row["attempts"] == 1 for row in ok_rows])),
                                 lift_cm=[float(100 * min(row["tcp_z"] for row in ok_rows)),
                                          float(100 * max(row["tcp_z"] for row in ok_rows))])
    return summary


def main():
    """Run the start-state test on --n layouts and write the JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=600000)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--scene", default="default", choices=("default", "legacy_v3"),
                        help="GraspScene physics preset")
    parser.add_argument("--no_activity", action="store_true", help="skip the static rope-activity measurement")
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "runs", "start_states"))
    args = parser.parse_args()
    from rope_routing.env.surface_route_env import make_env
    from rope_routing.sim.grasp_scene import SceneCfg

    seeds = list(range(args.start, args.start + args.n))
    assert not any(700000 <= seed < 800000 or seed >= 9_000_000 for seed in seeds), "never use eval/verification seeds"
    cfg = SceneCfg.legacy_v3() if args.scene == "legacy_v3" else SceneCfg()
    # The start probabilities default to "full": every reset() below leaves the robot at home.
    env = make_env(rank=0, seed=0, scene_cfg=cfg)
    rows, activity = [], {}
    start_time = time.time()
    for seed in seeds:
        if not args.no_activity:
            activity[seed] = rope_activity(env, seed)
        for mode in ("full", "hover", "grasped"):
            row = run_mode(env, seed, mode)
            if mode == "full":
                assert row["ok"] and row["full_identity"] and row["obs_ok"], row
                continue
            rows.append(row)
            rope_act = activity.get(seed, (np.nan, np.nan))
            print(f"{seed} {mode:7s} ok={int(row['ok'])} {fail_mode(row):38s} wall={row['wall']:.2f}s "
                  f"sim={row['sim_time']:.1f}s tcp_err={1e3 * row['tcp_err']:5.1f}mm z={100 * row['tcp_z']:4.1f}cm "
                  f"yaw={row['yaw_err']:.2f} strain={row['max_strain']:.4f} r0={row['step_reward']:+.2f} "
                  f"(stale {row['stale_spike']:+.1f}) drift={1e3 * row['drift_5steps']:.1f}mm "
                  f"rope_v={1e3 * rope_act[0]:.0f}mm/s", flush=True)
    summary = summarize(rows, activity)
    summary["meta"] = dict(seeds=[seeds[0], seeds[-1]], scene=args.scene, total_wall_s=time.time() - start_time,
                           host=os.uname().nodename, active_rope_threshold_mps=0.03,
                           rope_activity={str(key): [float(value[0]), float(value[1])]
                                          for key, value in activity.items()})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"test_{args.scene}_{seeds[0]}_{len(seeds)}.json")
    with open(path, "w") as handle:
        json.dump(dict(summary=summary, rows=rows), handle, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    print("saved", path)


if __name__ == "__main__":
    main()
