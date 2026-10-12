# B. The RL environment and how the policy is trained

**Code:** `rope_routing/env/generic_surface_env.py` (`GenericSurfaceEnv`). It subclasses `surface_route_env.py` (`SurfaceRouteEnv`) for the robot interface, the stretch penalty, bad-step handling and the start-state curriculum. The trainer is `rope_routing/rl/async_ppo.py`.

## Task
A Franka FR3 with a Robotiq 2F-85 gripper picks up a soft rope and routes it around three pegs into a triangle. The rope is a SuperDex discrete elastic rod: 61 nodes, 0.81 m long, 5 mm radius, one end pinned near the first peg. The physics is in notes C.
- **Episode:** at most 500 control steps of 0.1 s (50 s); each step is 10 physics substeps.
- **Layout:** every episode draws a new random rope layout (`task/rope_init.py:random_layout`).

## Action (5-D, each in [−1, 1]); `grasp_route_env.py:_apply_action`
| Index | Meaning |
|---|---|
| 0–2 | Move the gripper target by up to 2 cm per axis per step. Clamped to the workspace and to within 4 cm of the gripper. |
| 3 | Rotate the gripper by up to 0.15 rad per step. |
| 4 | Finger closure. While holding, the fingers open only after 3 consecutive "release" commands (a debounce, so exploration noise can't drop the rope). |

## Observation (602 numbers); `_obs_blocks` / `_make_obs`
| Block | Size | Content |
|---|---|---|
| Robot | 33 | Joint angles and velocities, gripper position and yaw, target offset, finger state, pad forces, previous action, time |
| **Deformation** | 21×17 + 2 | Per material sample: position relative to the gripper, velocity, **area strain** (1st fundamental form), **mean/Gaussian/principal curvature**, curvature vector, validity (2nd form). Globals: total area strain and Willmore energy. See notes A. |
| Potential | 1 | The current score φ (below) |
| Grasp region | 3 | Nearest point of the object's *far region* (grasp rule) relative to the gripper |
| **Target surface** | 48×4 | 48 farthest-point samples of the target surface relative to the gripper, with their target `r²K` |
| Shape error | 2 | Coverage and curvature mismatch |
| Obstacles | 6 + 6 | Each obstacle's position relative to the gripper; current wrap fraction and "target wraps it" flag |

Nothing names a peg, a point of the rope or a waypoint. Obstacles are an unlabelled list.

## Goal: a target surface
The goal is the surface the object should end up with. Here it is the tube surface of the rope lying along the triangle route from the pinned end (`_build_target`); the remaining rope is free. For another object you would supply its target mesh instead.

## Reward
The score (potential) φ of a state:
```
holding:      φ = 2 + 2·Σ wrap_frac − 10·coverage − 0.3·curv_mismatch
not holding:  φ = −3·reach + 2·Σ wrap_frac − 10·coverage − 0.3·curv_mismatch
```
| Term | Definition | Object-agnostic because |
|---|---|---|
| `coverage` [m] | Mean distance from each target-surface vertex to the nearest current-surface vertex | No correspondence |
| `curv_mismatch` | Mean `|r²K* − r²K|` at those nearest points | Second fundamental form |
| `wrap_frac` | Per obstacle: `|winding of object| / |winding of target|`, clipped to [0, 1]; only obstacles the target wraps count | Computed from the target surface; obstacles unlabelled |
| `reach` | Distance from the gripper to the nearest point of the **far region**, with a hover/descend height blend | **Generic grasp rule**, see below |

**Per-step reward**, `r = 10(φ′ − φ)` (progress, which sums to `10(φ_end − φ_start)`) plus:
| Term | Value |
|---|---|
| Area-stretch penalty | `−50 Σ max(0, J − 1 − 0.02)²` |
| Action smoothness | `−0.02‖Δa‖²` |
| Effort | `−0.005‖a_xyz,yaw‖²` |
| Pressing on the table | `−0.5` |
| Time | `−0.01` per step |
| Success | `+20` (ends the episode) |
| Simulator failure | `−10` (truncates) |

**Lesson:** strictly policy-invariant shaping `10(γφ′ − φ)` taught nothing (run surf_ppo1). Rewarding progress itself (`γ_shaping = 1`) made RL learn.

**Success:** the object is held, every wrapped obstacle reaches `wrap_frac ≥ 0.85`, and coverage is under 2.5 cm.

**Verification also checks release-and-hold:** the gripper opens and moves away, and the rope must stay wrapped (coverage under 3.75 cm).

## Generic grasp rule
Before grasping, the reach term pulls toward the **far region**: the 15% (`GRASP_FAR`) of the object's surface that is farthest, measured as rest geodesic distance (Dijkstra on the rest mesh), from the material that is held fixed (the pinned end). It needs only the mesh and which material is fixed. For a rope that is the free end; for a towel pinned at a corner, the opposite corner.
- **Why it's needed:** without it, the policy always grabbed the rope point nearest its start pose (next to the anchor), and it can't finish from there: 0/16 from the start pose, versus 5/6 from far grasps.
- **What the policy still decides:** exactly where to grasp within the region, how to approach, and the whole routing.

## Start states and exploration (`env/start_states.py`)
The adaptive curriculum mixes episode starts. Every start is reached through legal actions, and nothing is used as a label.
| Start | Description |
|---|---|
| `full` | Robot at home |
| `full`, roam variant | With probability 0.5 in training, the gripper first goes to a random point above the object (`setup_roam`) |
| `hover` | Fingers open, 3–6 cm above a random point of the object |
| `grasped` | The rope already held at a random point |

Evaluation always uses `full` from home.

## Trainer: asynchronous PPO, pure RL (`rl/async_ppo.py`, `scripts/train_rl.py`)
- No demonstrations, no behaviour cloning, no expert.
- 24 CPU simulator workers (SuperDex physics is CPU-only, notes C); the learner runs on the GPU.
- DD-PPO-style straggler preemption.
- PPO settings:

| Setting | Value |
|---|---|
| γ / λ | 0.99 / 0.95 |
| Clip | 0.2 |
| Epochs / minibatch | 8 / 2048 |
| Learning rate | KL-adaptive (target KL 0.01) |
| Action std | min 0.1 |
| Actor and critic MLPs | 512-512-256 |
| Normalisation | Observations normalised; rewards scaled |

- **Held-out evaluation:** every 1M steps on 64 layouts. `best.pt` is chosen by deterministic success.
- **Verification:** `scripts/verify_rl.py`, 100 unseen layouts. `scripts/eval_randomized.py` adds randomized rope physics.

## Run history
| Run | What | Result |
|---|---|---|
| surf_ppo2 | Old env (waypoints, chosen grasp point, per-point targets) | 97/100. Kept as the best previous policy, but not object-agnostic. |
| generic_ppo1 | GenericSurfaceEnv | Learned routing; always grasped next to the anchor (0–1/64) |
| generic_ppo2 | + roam starts, resumed | Grasp choice improved slowly |
| generic_ppo3 | + generic grasp rule (far region = top 30%), resumed at 3.48M | Grasp moved to ~0.6 of the length; 6/64 at 5M |
| generic_ppo4 | Far region tightened to the top 15% (≈ last 13% of the rope), resumed at 5.17M | See notes D (results) |
