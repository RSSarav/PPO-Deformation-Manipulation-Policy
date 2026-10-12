# C. Rope physics, compute, and how to run everything

## 1. Simulator: SuperDex (Mochi) rope, configured to behave like the "Rope Braid" demo
**Code:** `rope_routing/sim/grasp_scene.py` (`SceneCfg`, `GraspScene`).

- **Rope:** a discrete elastic rod.

  | Parameter | Value |
  |---|---|
  | Nodes | 61 |
  | Length | 0.81 m (0.61 m on the route + a 0.20 m tail) |
  | Radius | 5 mm |
  | Linear density | 0.06 kg/m |
  | EA / EI / GJ | 5e3 N / 3e-4 N·m² / 2e-4 N·m² |
  | Rayleigh damping | (1.0, 0.01) |
  | Friction: rope / table / pegs / gripper pads | 0.6 / 0.6 / 0.5 / 2.0 |
  | Timestep | 10 ms |

  These match the demo clip: the rope there is 0.8–1 cm thick, and its drape implies EI ≈ 0.4–3 × 10⁻⁴.
- **Solver ("C1-nc"):**
  - LDLT linear solver
  - 8 Newton iterations; 4 makes wraps sticky
  - Armijo-or-residual line search
  - implicit normal force for dissipation
  - FR3 joint Coulomb friction off: its regularised stick could freeze the arm for seconds
- **Contact:**
  - Point-cloud collider (3 samples per segment) with rope self-contact; the rope padding `r − 0.5 mm` makes strands touch at 2r.
  - Penalty 1e10; smoothing 1 mm.
  - Friction fall-off 1e-3 m/s. The default 1e-2 creeps, and 1e-4 is sticky.
- **Reset ("frame-consistent placement"):** the rope is lifted 10 cm, morphed into the random layout in the air along a curvature homotopy, lowered, and settled until it moves under 2 mm/s.
  - This fixed the old artefacts (hops, loops, blow-ups). They were caused by stale rod frames after a one-step boundary-condition jump, not by twist.
- **Robot:**
  - FR3 under operational-space control of the pinch point, with null-space posture.
  - Robotiq 2F-85 under an implicit knuckle PD; contact-gated, no-slip "attach" grasp.
- **Remaining gaps vs the demo:**
  - Centreline contact can't roll or twist a rope under contact; tube contact can, at about 3× the cost.
  - About 0.3 mm/s creep at rest.
  - Up to about 2.7 mm interpenetration at taut crossings.
- **Tests:** `tests/test_grasp_scene_physics.py`; scene checks in `scripts/check_scene_physics.py`.

## 2. Compute: why training is CPU-bound and GPUs can't help (audit 2026-10-11)
- SuperDex steps **one scene at a time on one CPU core**; it has no batched GPU stepping.
  - SuperDex Lab docs: "Both run on CPU — `num_gpus_per_learner=0`" (https://projectsuperdex.com/lab/docs/latest/superdex_gym/rllib/).
  - Batching docs: workers step environments "sequentially in-process"; start `num_workers` at the physical core count.
- Its CUDA linear solvers are not in the installed build. They would be slower anyway: the linear solve is 13% of step time, and each GPU call copies the matrix (1.4–9× slower measured).
- **Measured step cost:** 14 ms (rope free) to 43 ms (rope held, where Newton runs all 8 iterations). Physics is 75–84% of a step.
- **Levers that work:**
  - one run per machine: stopping a second run gave 1.59×
  - single-threaded Mochi
  - more CPU cores
- **Biggest open lever:** while the rope is held, Newton never converges (7.98/8 iterations). Fixing that could save up to about 45% of a held step.
- The GPU runs only the PPO learner (<1% of wall time) and video rendering.

## 4. How to run (from `project_superdex/`, with `.venv/bin/python`)
```bash
# tests
.venv/bin/python -m pytest -q ../rope_routing/tests
# train (pure RL); see data/runs/generic_ppo3_launch.sh for the exact final command
.venv/bin/python ../scripts/train_rl.py --run NAME --env rope_routing.env.generic_surface_env:make_env \
    --env_kwargs '{"roam_prob": 0.5}' --workers 24 --n_eval_workers 6
# verify on 100 unseen layouts (+ release-and-hold)
.venv/bin/python ../scripts/verify_rl.py --run NAME --model best --n 100
# randomized rope physics, 100 episodes
.venv/bin/python ../scripts/eval_randomized.py --run NAME --model best
# HD video of one episode: 3 camera views + live deformation charts in one MP4
.venv/bin/python ../scripts/render_hd.py --run NAME --model best --seed S --outputs multiview \
    --egl_device 7 --nvenc_gpu 4
# grid video of rendered cells
python ../scripts/compose_grid.py --cells_dir DIR --seeds ... --title T --out OUT.mp4
```
