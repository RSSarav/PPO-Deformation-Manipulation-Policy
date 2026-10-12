# D. Results (final policy: `data/runs/generic_ppo4/best.pt`)

**What it is:** a pure-RL policy (async PPO from scratch; no demonstrations, BC or expert), trained 10.0M env steps in the object-agnostic environment of notes B. It sees the rope's surface-area and curvature deformation (notes A) and is rewarded by correspondence-free coverage of a target surface, generic obstacle wraps, the stretch penalty and the generic grasp rule.

## 1. Standard verification: 100 unseen layouts, nominal rope (`data/runs/generic_ppo4/verify100_best/summary.md`)
| Metric | Value |
|---|---|
| **Success** (held + all wraps ≥ 85% + coverage < 2.5 cm) | **87/100 = 87%** (95% CI 79–92%) |
| **Still wrapped after release** (gripper opens and leaves) | **82/100** |
| Grasp rate | 99% |
| Simulator failures | 0 |
| Mean coverage of successes | 1.6 cm |
| Mean time to success | 8.9 s (89 steps) |
| Max area strain | 0.9% |

## 2. Randomized rope physics: 100 unseen layouts, a different rope each episode (`data/runs/generic_ppo4/eval_randomized_rope/`)
The policy was trained on the nominal rope only. Each episode draws:

| Parameter | Range |
|---|---|
| Mass | ×0.6–1.6 |
| Stretch stiffness | ×0.5–2 |
| Bending stiffness | ×0.4–2.5 |
| Twist stiffness | ×0.5–2 |
| Damping | ×0.5–2 |
| Friction | rope 0.4–0.8, table 0.4–0.8, pegs 0.3–0.7 |

| Metric | Value |
|---|---|
| **Success** | **88/100 = 88%** (95% CI 80–93%) |
| **Still wrapped after release** | **76/100** (76 of the 88 successes) |
| Simulator failures | 0 |
| Mean coverage of successes | 1.5 cm |

- **Where it fails:** mostly ropes **softer** than the training rope.
  - Bending stiffness < 0.8× nominal: 31/41 succeed. ≥ 0.8×: 57/59.
  - 10 of the 12 failures have EI ≤ 0.77×.
- **Next step:** randomizing rope physics during training (domain randomization) would likely close this gap.

**Files:**
- `summary.md`: rates and a per-episode table with each rope's parameters
- `results.json`: every episode
- `selection.json`: the videoed episodes (the first 12 successes and first 12 failures by seed; not hand-picked)
- `successes_grid_12.mp4`: 4×3 grid of the 12 successes, real time
- `failures_grid_12.mp4`: 4×3 grid of the 12 failures
- `successes/<seed>.mp4`: each success as one video with third-person, top and wrist cameras plus live charts of wrap, coverage and surface-area stretch; `<seed>.json` holds the outcome and rope settings

## 3. Learning curve (held-out evaluation, 64 layouts, from the robot's home pose)
| Steps | Change | Success |
|---|---|---|
| 1M, 2M | GenericSurfaceEnv: always grasped next to the anchor | 0%, 2% |
| 3M | + roam starts (random gripper starts) | 0% |
| 4M, 5M | + generic grasp rule, far region 30% | 3%, 9% |
| 6M | Far region tightened to 15% | 31% |
| 7M | | 45% |
| 8M | | 69% |
| 9M | | 86% |
| 10M | Plateau; training stopped (stop rule) | 86% |

Wall time: about 6.5 h on a 64-thread machine (24 CPU workers, about 330–450 steps/s).

## 4. Best previous policy (kept for comparison): `data/runs/surf_ppo2/best_2M.pt`
- Pure RL in the older SurfaceRouteEnv: 97/100 verified, 97% held after release.
- It relied on rope- and task-specific structure: 15 gripper waypoints, a hand-chosen grasp node and per-point targets. It is not object-agnostic.
- Videos are in `data/runs/surf_ppo2/videos_best_2M/`.

## 5. Known limitations
- **Generic grasp rule:** the grasp rule is a designed, object-agnostic prior (farthest from the fixed material), not learned (notes B).
- **Rope-specific observation parts:** the deformation samples are spaced along the rope's length, and 2 of 17 features (curvature vector, signed planar curvature) are curve quantities. For cloth, the samples would be farthest-point samples of the surface and those 2 features dropped. The reward and the target are fully surface-based.
- **Release-and-hold:** 5 of 87 successes (nominal) and 12 of 88 (randomized) do not stay wrapped once released.
- **Training speed:** CPU-bound (notes C §2).
