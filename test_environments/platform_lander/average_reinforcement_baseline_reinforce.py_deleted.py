"""REINFORCE with reward-to-go and a running average-reinforcement baseline.

Compared with ``rtg_reinforce.py``, this keeps one trajectory per policy update
but subtracts a scalar baseline from each reward-to-go weight:

    A_t = G_t - running_rtg_mean
    loss = -sum_t gamma^t * A_t * log pi_theta(a_t | o_t)

The baseline is the mean of all reward-to-go samples seen in previous
trajectories. It is updated after the current policy-gradient step, so the
baseline used for an episode is independent of that episode's sampled actions.
This is a non-value-function analogue of reinforcement comparison.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch

from rtg_reinforce import Policy, discount_factors, rewards_to_go, trajectory_entropies
from vanilla_reinforce import (
    PlatformLander,
    add_output_args,
    animate,
    log,
    open_log,
    resolve_project_path,
    run_episode_data,
    save_policy,
    save_training_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reward-to-go REINFORCE with a running average-reinforcement baseline."
    )
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Coefficient for the policy entropy bonus. A value of 0 disables it.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-power", type=float, default=5.0)
    parser.add_argument("--no-animation", action="store_true")
    add_output_args(parser, "average_reinforcement_baseline_reinforce")
    return parser.parse_args()


def update_running_rtg_mean(running_mean: float, running_count: int, rtg: torch.Tensor) -> tuple[float, int]:
    """Update the all-past-samples mean with one trajectory's reward-to-go values."""
    batch_count = int(rtg.numel())
    if batch_count == 0:
        return running_mean, running_count

    total = running_mean * running_count + float(rtg.detach().sum().item())
    new_count = running_count + batch_count
    return total / new_count, new_count


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

    running_rtg_mean = 0.0
    running_rtg_count = 0

    recent_returns: deque[float] = deque(maxlen=args.target_window)
    recent_successes: deque[bool] = deque(maxlen=args.target_window)
    recent_jet_fires: deque[int] = deque(maxlen=args.target_window)

    training_rows: list[dict[str, object]] = []
    log_file = open_log(args.log_file)

    try:
        log(
            f"training_start script=average_reinforcement_baseline_reinforce "
            f"episodes={args.episodes} max_steps={args.max_steps} seed={args.seed} "
            f"entropy_coef={args.entropy_coef} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )

        for episode in range(1, args.episodes + 1):
            observations, log_probs, rewards, episode_return, steps, info = run_episode_data(
                env,
                policy,
                rng,
                gamma=args.gamma,
                max_steps=args.max_steps,
                train=True,
            )

            rtg = rewards_to_go(rewards, args.gamma)
            log_prob_tensor = torch.stack(log_probs).reshape(-1)
            discounts = discount_factors(len(rewards), args.gamma)
            entropy_tensor = trajectory_entropies(policy, observations)

            rtg = rtg.to(
                device=log_prob_tensor.device,
                dtype=log_prob_tensor.dtype,
            )
            discounts = discounts.to(
                device=log_prob_tensor.device,
                dtype=log_prob_tensor.dtype,
            )
            entropy_tensor = entropy_tensor.to(
                device=log_prob_tensor.device,
                dtype=log_prob_tensor.dtype,
            )

            assert log_prob_tensor.shape == rtg.shape == discounts.shape == entropy_tensor.shape

            baseline = running_rtg_mean
            advantages = rtg - baseline
            policy_loss = -(log_prob_tensor * discounts * advantages).sum()
            entropy_bonus = entropy_tensor.sum()
            loss = policy_loss - args.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_rtg_mean, running_rtg_count = update_running_rtg_mean(
                running_rtg_mean,
                running_rtg_count,
                rtg,
            )

            recent_returns.append(episode_return)
            recent_successes.append(bool(info.get("success", False)))
            recent_jet_fires.append(int(info.get("jet_fires_used", 0)))

            average_return = float(np.mean(recent_returns))
            success_count = int(sum(recent_successes))
            average_jet_fires = float(np.mean(recent_jet_fires))
            mean_advantage = float(advantages.detach().mean().item())
            average_entropy = float(entropy_tensor.detach().mean().item())

            training_rows.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "rtg_mean": float(rtg.detach().mean().item()),
                    "baseline": baseline,
                    "average_advantage": mean_advantage,
                    "running_rtg_mean": running_rtg_mean,
                    "running_rtg_count": running_rtg_count,
                    "average_return": average_return,
                    "policy_loss": float(policy_loss.detach().item()),
                    "entropy_bonus": float(entropy_bonus.detach().item()),
                    "average_entropy": average_entropy,
                    "loss": float(loss.detach().item()),
                    "success_count": success_count,
                    "success_rate": success_count / len(recent_successes),
                    "jet_fires": int(info.get("jet_fires_used", 0)),
                    "average_jet_fires": average_jet_fires,
                    "steps": steps,
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                }
            )

            if episode == 1 or episode % args.print_every == 0:
                log(
                    f"episode={episode:5d} "
                    f"return={episode_return:8.2f} "
                    f"baseline={baseline:8.2f} "
                    f"advantage={mean_advantage:8.2f} "
                    f"avg{len(recent_returns):02d}={average_return:8.2f} "
                    f"entropy={average_entropy:5.3f} "
                    f"success{len(recent_successes):02d}={success_count:2d} "
                    f"fires={info.get('jet_fires_used', 0):3d} "
                    f"avgfires{len(recent_jet_fires):02d}={average_jet_fires:6.1f} "
                    f"steps={steps:4d} "
                    f"failure={info.get('failure_reason')}",
                    log_file,
                )

        model_path = save_policy(
            policy,
            args,
            args.model_file,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
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
