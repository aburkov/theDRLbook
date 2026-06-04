"""REINFORCE with trajectory batches only.

Compared with ``vanilla_reinforce.py``, this keeps the same full-trajectory
return objective but averages several trajectory-gradient estimates before each
optimizer step. Here R(tau_i) is the globally discounted trajectory return:

    loss = mean_i -R(tau_i) * sum_t log pi(a_t^i | o_t^i)
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch

from vanilla_reinforce import (
    Policy,
    PlatformLander,
    add_output_args,
    animate,
    log,
    open_log,
    resolve_project_path,
    run_episode,
    save_policy,
    save_training_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch REINFORCE for PlatformLander.")
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-power", type=float, default=5.0)
    parser.add_argument("--no-animation", action="store_true")
    add_output_args(parser, "batch_reinforce")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Policy:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = PlatformLander(
        enable_wind=args.wind,
        wind_power=args.wind_power,
        wind_direction=(1.0, 0.0),
    )
    policy = Policy(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        hidden_dim=args.hidden_dim,
    )
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    optimizer = torch.optim.SGD(policy.parameters(), lr=args.learning_rate)
    recent_returns: deque[float] = deque(maxlen=args.target_window)
    recent_successes: deque[bool] = deque(maxlen=args.target_window)
    recent_jet_fires: deque[int] = deque(maxlen=args.target_window)
    training_rows: list[dict[str, object]] = []
    log_file = open_log(args.log_file)

    episode = 0
    try:
        log(
            f"training_start script=batch_reinforce "
            f"episodes={args.episodes} batch_size={args.batch_size} "
            f"max_steps={args.max_steps} seed={args.seed} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )
        while episode < args.episodes:
            loss_terms: list[torch.Tensor] = []

            for _ in range(args.batch_size):
                if episode >= args.episodes:
                    break
                episode += 1

                log_probs, rewards, episode_return, steps, info = run_episode(
                    env,
                    policy,
                    rng,
                    gamma=args.gamma,
                    max_steps=args.max_steps,
                    train=True,
                )
                sum_log_probs = torch.stack(log_probs).sum()
                loss_terms.append(-sum_log_probs * episode_return)

                recent_returns.append(episode_return)
                recent_successes.append(bool(info.get("success", False)))
                recent_jet_fires.append(int(info.get("jet_fires_used", 0)))
                average_return = float(np.mean(recent_returns))
                success_count = int(sum(recent_successes))
                average_jet_fires = float(np.mean(recent_jet_fires))
                training_rows.append(
                    {
                        "episode": episode,
                        "return": episode_return,
                        "average_return": average_return,
                        "success_count": success_count,
                        "success_rate": success_count / len(recent_successes),
                        "jet_fires": int(info.get("jet_fires_used", 0)),
                        "average_jet_fires": average_jet_fires,
                        "batch_index": len(loss_terms),
                        "batch_size": args.batch_size,
                        "steps": steps,
                        "success": bool(info.get("success", False)),
                        "failure_reason": info.get("failure_reason"),
                    }
                )

                if episode == 1 or episode % args.print_every == 0:
                    log(
                        f"episode={episode:5d} "
                        f"return={episode_return:8.2f} "
                        f"avg{len(recent_returns):02d}={average_return:8.2f} "
                        f"success{len(recent_successes):02d}={success_count:2d} "
                        f"fires={info.get('jet_fires_used', 0):3d} "
                        f"avgfires{len(recent_jet_fires):02d}={average_jet_fires:6.1f} "
                        f"batch={len(loss_terms):2d}/{args.batch_size:2d} "
                        f"steps={steps:4d} "
                        f"failure={info.get('failure_reason')}",
                        log_file,
                    )

            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_path = save_policy(policy, args, args.model_file, obs_dim=obs_dim, action_dim=action_dim)
        log(f"saved_model={model_path}", log_file)
        csv_path = save_training_csv(training_rows, args.csv_file)
        log(f"saved_csv={csv_path}", log_file)
    finally:
        env.close()
        log_file.close()

    return policy


def main() -> None:
    args = parse_args()
    policy = train(args)
    if not args.no_animation:
        animate(policy, args)


if __name__ == "__main__":
    main()
