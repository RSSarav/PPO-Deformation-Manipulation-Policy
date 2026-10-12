"""Videos of a randomized-rope evaluation (eval_randomized.py): the first 12 successes and the first 12 failures (by
seed, not hand-picked; selection.json), rendered from the SAME layout and rope settings as in the evaluation.

Output folder <eval_dir>/:
  successes_grid_12.mp4      4 x 3 grid of the 12 successes (third-person, real time)
  failures_grid_12.mp4       4 x 3 grid of the 12 failures
  successes/<seed>.mp4       one video per success: third-person, top and wrist cameras and the live deformation charts
  successes/<seed>.json      its outcome and rope settings
The per-camera files and the grid cells are temporary and deleted at the end.

Each episode is rendered by a scripts/render_hd.py subprocess, alternating between the two GPUs of DEV; the grids are
assembled by scripts/compose_grid.py.

Run on d3 (EGL 7 = GPU 4, EGL 5 = GPU 6), from project_superdex/:
  .venv/bin/python ../rope_routing/scripts/render_eval_set.py --run generic_ppo4 --model best
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV = [(7, 4), (5, 6)]  # (EGL device, nvidia-smi GPU) on d3


def main():
    """Render the selected episodes in parallel, assemble the grids and keep one video per success."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="best")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--success_videos_only", action="store_true", help="re-render only successes/<seed>.mp4")
    args = parser.parse_args()
    eval_dir = os.path.join(ROOT, "data", "runs", args.run, "eval_randomized_rope")
    selection = json.load(open(os.path.join(eval_dir, "selection.json")))
    cells, success_dir, logs = (os.path.join(eval_dir, "_cells"), os.path.join(eval_dir, "successes"),
                                os.path.join(eval_dir, "_logs"))
    for folder in (cells, success_dir, logs):
        os.makedirs(folder, exist_ok=True)
    render_script = os.path.join(ROOT, "scripts", "render_hd.py")
    label = "pure-RL policy (PPO)"

    # One render job per selected episode.
    jobs = []
    for kind in (("successes",) if args.success_videos_only else ("successes", "failures")):
        for entry in selection[kind]:
            seed, rope = entry["seed"], json.dumps(entry["rope"])
            cell_label = (f"seed {seed} · EI ×{entry['rope']['rope_bending_stiffness'] / 3e-4:.1f} · "
                          f"mass ×{entry['rope']['rope_linear_density'] / 0.06:.1f} · "
                          f"μ {entry['rope']['rope_friction']:.2f}")
            base = [sys.executable, render_script, "--run", args.run, "--model", args.model, "--seed", str(seed),
                    "--rope", rope, "--label", label, "--cell_label", cell_label]
            if kind == "successes" and args.success_videos_only:
                jobs.append((f"s_{seed}", base + ["--outputs", "multiview", "--out_dir", cells, "--tag", f"cell_{seed}",
                                                  "--subtitle", f"random rope · seed {seed}"]))
            elif kind == "successes":  # 3-camera video (multiview) + a full-HD third-person file used only as the cell
                jobs.append((f"s_{seed}", base + ["--outputs", "multiview,third", "--clean", "--third_cam", "close",
                                                  "--out_dir", cells, "--tag", f"cell_{seed}",
                                                  "--subtitle", f"random rope · seed {seed}"]))
            else:
                jobs.append((f"f_{seed}", base + ["--outputs", "third", "--size", "960", "540", "--clean",
                                                  "--third_cam", "close", "--hold", "0", "--out_dir", cells, "--tag",
                                                  f"cell_{seed}"]))

    def run(indexed_job):
        """Run one render job on the GPU pair selected by its index; returns (name, exit code)."""
        index, (name, cmd) = indexed_job
        egl, gpu = DEV[index % 2]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2")
        start_time = time.time()
        with open(os.path.join(logs, name + ".log"), "w") as log:
            rc = subprocess.call(cmd + ["--egl_device", str(egl), "--nvenc_gpu", str(gpu)], stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
        print(f"[{name}] rc={rc} {time.time() - start_time:.0f}s", flush=True)
        return name, rc

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, enumerate(jobs)))
    bad = [name for name, rc in results if rc != 0]
    if bad:
        print("FAILED renders:", bad, flush=True)
    if args.success_videos_only:
        for entry in selection["successes"]:
            shutil.move(os.path.join(cells, f"cell_{entry['seed']}_multiview.mp4"),
                        os.path.join(success_dir, f"{entry['seed']}.mp4"))
        if not bad:
            shutil.rmtree(cells)
            shutil.rmtree(logs)
        return
    # Grids (3 s hold at the end).
    grid_script = os.path.join(ROOT, "scripts", "compose_grid.py")
    for kind, title in (("successes", "12 successes"), ("failures", "12 failures")):
        seeds = [str(entry["seed"]) for entry in selection[kind]]
        subprocess.check_call([sys.executable, grid_script, "--cells_dir", cells, "--seeds", *seeds, "--cols", "4",
                               "--rows", "3",
                               "--title", f"{title} · randomized rope physics · first 12 by seed, not selected",
                               "--out", os.path.join(eval_dir, f"{kind}_grid_12.mp4"), "--nvenc_gpu", "4"],
                              env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
    # Keep one 3-camera video and its JSON per success; delete the per-camera files, the cells and the logs.
    for entry in selection["successes"]:
        seed = entry["seed"]
        shutil.move(os.path.join(cells, f"cell_{seed}_multiview.mp4"), os.path.join(success_dir, f"{seed}.mp4"))
        shutil.copy(os.path.join(cells, f"cell_{seed}.json"), os.path.join(success_dir, f"{seed}.json"))
    # The rendered episodes must reproduce the evaluation's outcomes.
    outcomes = {entry["seed"]: json.load(open(os.path.join(cells, f"cell_{entry['seed']}.json")))["success"]
                for kind in ("successes", "failures") for entry in selection[kind]}
    mismatches = [seed for kind in ("successes", "failures") for entry in selection[kind]
                  if outcomes[(seed := entry["seed"])] != (kind == "successes")]
    print("render outcome matches evaluation:", not mismatches, mismatches, flush=True)
    if not bad:
        shutil.rmtree(cells)
        shutil.rmtree(logs)


if __name__ == "__main__":
    main()
