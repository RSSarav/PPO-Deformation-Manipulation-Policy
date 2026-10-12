# Notes: rope routing with an object-agnostic, pure-RL policy

A Franka FR3 with a Robotiq 2F-85 gripper picks up a soft rope in SuperDex and routes it around three pegs into a triangle. The policy is trained **only with reinforcement learning**. Its observation and reward describe the rope through the **surface area and curvature of its boundary surface** (the first and second fundamental forms), so the same formulation applies to any deformable object.

| Note | Contents |
|---|---|
| [A_deformation_math.md](A_deformation_math.md) | The deformation math (first and second fundamental forms, discrete estimators, tube identities, folds) and exactly where each piece is in the code |
| [B_rl_environment.md](B_rl_environment.md) | The RL environment: action, observation, target surface, reward, success, generic grasp rule, start states, the PPO trainer, run history |
| [C_physics_and_infrastructure.md](C_physics_and_infrastructure.md) | Rope physics (matched to SuperDex's Rope Braid demo), why GPUs can't speed up SuperDex sampling, machines, and how to run training, verification, evaluation and videos |
| [E_code_workflow.md](E_code_workflow.md) | **Code workflow diagram** and the order to read the code |
| [D_results.md](D_results.md) | Final results: 87% on 100 unseen layouts, 88% with randomized rope physics, learning curve, videos, limitations |

**Final policy:** `data/runs/generic_ppo4/best.pt`.
**Best previous policy (kept):** `data/runs/surf_ppo2/best_2M.pt`.

**Videos:**
- `data/runs/generic_ppo4/eval_randomized_rope/`: 12-success and 12-failure grids, plus 12 three-camera success videos
- `data/runs/surf_ppo2/videos_best_2M/`: videos of the previous policy
