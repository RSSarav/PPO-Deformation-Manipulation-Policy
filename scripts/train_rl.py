"""Train a policy with pure reinforcement learning (asynchronous-rollout PPO): no demonstrations, no behaviour
cloning, no expert.

Each of the --workers processes owns one environment and a CPU copy of the actor and samples on its own; the learner
(GPU) synchronises once per iteration. Straggler preemption: workers keep stepping (up to --rollout_max) until
--preempt_frac of them have --rollout steps, then all stop (--preempt_frac 1 gives fixed --rollout-step fragments).
The learning rate adapts to --target_kl (rsl_rl rule), and --min_std floors the exploration noise. The trainer's
--gamma must equal the environment's shaping GAMMA (checked at start-up). See rope_routing/rl/async_ppo.py and
notes/B_rl_environment.md.

Outputs in data/runs/<run>/: metrics.jsonl (one record per iteration and one per evaluation), last.pt (every iteration),
ckpt_<steps>.pt, best.pt (best deterministic held-out success; ties go to the lower shape error), eval_history.json,
eval_stoch_history.json (--eval_stochastic 1), eval/eval_<steps>[_stoch].json and config.json.

  cd project_superdex && CUDA_VISIBLE_DEVICES=4 .venv/bin/python ../rope_routing/scripts/train_rl.py --run rl1 \
      --workers 24 --info_keys shape,max_stage
  # resume:   ... train_rl.py --run rl1 --resume ../rope_routing/data/runs/rl1/last.pt
  # verify:   .venv/bin/python ../rope_routing/scripts/verify_rl.py --run rl1 --model best

The final policy was trained with --env rope_routing.env.generic_surface_env:make_env --env_kwargs
'{"roam_prob": 0.5}' (notes/C_physics_and_infrastructure.md, section 4).
"""

import argparse
import json
import os
import signal
import sys

from rope_routing.rl.async_ppo import AsyncPPO, PPOConfig

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    """Parse the command line into a ``PPOConfig`` and run ``AsyncPPO`` until the step budget is spent."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run name -> data/runs/<run>/")
    parser.add_argument("--env", default="rope_routing.env.surface_route_env:make_env",
                        help="env factory 'module:function'")
    parser.add_argument("--env_kwargs", default="{}", help="JSON dict passed to the env factory")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--rollout", type=int, default=256,
                        help="T: steps per worker per iteration (preemption threshold)")
    parser.add_argument("--rollout_max", type=int, default=0, help="cap of a preemptible fragment (0: 2 x --rollout)")
    parser.add_argument("--preempt_frac", type=float, default=0.8,
                        help="stop all workers once this share has >= --rollout steps; 1.0 = fixed-length fragments")
    parser.add_argument("--stop_timeout", type=float, default=60.0,
                        help="restart a worker that has not returned its fragment this long after the stop signal")
    parser.add_argument("--steps", type=int, default=30_000_000, help="total env steps")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--lr_schedule", choices=["adaptive", "linear"], default="adaptive",
                        help="adaptive: x1.5 / /1.5 per iteration toward --target_kl (/1.5 after a policy early "
                             "stop), within [--lr_min, --lr_max]; linear: --lr -> --lr_final over --steps")
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_final", type=float, default=3e-5, help="end of the linear schedule")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.0)
    parser.add_argument("--vf", type=float, default=0.5)
    parser.add_argument("--max_grad", type=float, default=0.5, help="gradient-norm clip, actor and critic separately")
    parser.add_argument("--target_kl", type=float, default=0.01,
                        help="KL target of the adaptive lr (<= 0 disables the adaptation and the policy early stop)")
    parser.add_argument("--kl_stop", type=float, default=3.0,
                        help="the policy stops for the iteration when a minibatch KL > kl_stop x target_kl; the critic "
                             "keeps training (<= 0: never stop)")
    parser.add_argument("--min_std", type=float, default=0.1, help="floor of the action std (0: e^-5)")
    parser.add_argument("--log_std_init", type=float, default=-0.5)
    parser.add_argument("--hidden", default="512,512,256", help="actor/critic hidden sizes")
    parser.add_argument("--eval_every", type=int, default=1_000_000,
                        help="env steps between held-out evaluations (0: off)")
    parser.add_argument("--n_eval", type=int, default=64, help="held-out layouts 700000 .. 700000+n-1")
    parser.add_argument("--n_eval_workers", type=int, default=8)
    parser.add_argument("--eval_stochastic", type=int, choices=[0, 1], default=0,
                        help="1: also evaluate sampled actions on the same layouts (logged separately, "
                             "not for best.pt)")
    parser.add_argument("--ckpt_every", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum", choices=["adaptive", "none"], default="adaptive",
                        help="start-state curriculum (full/hover/grasped), used only if the env has set_start_probs")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to resume from (weights, optimiser, statistics, step)")
    parser.add_argument("--info_keys", default="", help="comma list of numeric info keys averaged at episode ends")
    parser.add_argument("--worker_timeout", type=float, default=300.0,
                        help="restart a worker that returns no fragment within this many seconds")
    args = parser.parse_args()

    cfg = PPOConfig(
        run_dir=os.path.join(ROOT, "data", "runs", args.run), env=args.env, env_kwargs=json.loads(args.env_kwargs),
        workers=args.workers, rollout=args.rollout, rollout_max=args.rollout_max, preempt_frac=args.preempt_frac,
        stop_timeout=args.stop_timeout, steps=args.steps, gamma=args.gamma, lam=args.lam, epochs=args.epochs,
        minibatch=args.minibatch, lr=args.lr, lr_schedule=args.lr_schedule, lr_min=args.lr_min, lr_max=args.lr_max,
        lr_final=args.lr_final, clip=args.clip, ent=args.ent, vf=args.vf, max_grad=args.max_grad,
        target_kl=args.target_kl if args.target_kl > 0 else None, kl_stop=args.kl_stop, min_std=args.min_std,
        log_std_init=args.log_std_init, hidden=tuple(int(size) for size in args.hidden.split(",")),
        eval_every=args.eval_every, n_eval=args.n_eval, n_eval_workers=args.n_eval_workers,
        eval_stochastic=bool(args.eval_stochastic),
        ckpt_every=args.ckpt_every, device=args.device, seed=args.seed, curriculum=args.curriculum,
        resume=os.path.abspath(args.resume) if args.resume else None,
        info_keys=tuple(key for key in args.info_keys.split(",") if key), worker_timeout=args.worker_timeout,
    )
    print("[train_rl] pure RL (async PPO, no demonstrations / BC / expert):", json.dumps(cfg.__dict__, default=str),
          flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run AsyncPPO's cleanup (workers, eval) on kill
    AsyncPPO(cfg).train()


if __name__ == "__main__":
    main()
