"""Batch reward-to-go REINFORCE with selectable scalar advantage baselines.

This combines trajectory batches with reward-to-go advantages. The chapter's
convention uses gamma^t * (G_t^i - b_i) as the weight for action a_t^i:

    loss = mean_i -sum_t gamma^t * (G_t^i - b_i) * log pi(a_t^i | o_t^i)
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Any

import numpy as np
import torch

from rtg_reinforce import discount_factors, rewards_to_go
from vanilla_reinforce import (
    Policy,
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


BASELINE_CHOICES = ("batch-mean", "trajectory-mean", "moving-average")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch advantage reward-to-go REINFORCE for PlatformLander.")
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument(
        "--baseline",
        choices=BASELINE_CHOICES,
        default="batch-mean",
        help=(
            "Scalar baseline: leave-one-out batch mean reward-to-go, current "
            "trajectory mean reward-to-go practical centering heuristic, or "
            "moving average of recent trajectory mean reward-to-go values."
        ),
    )
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-power", type=float, default=5.0)
    parser.add_argument("--no-animation", action="store_true")
    add_output_args(parser, "full_reinforce")
    return parser.parse_args()


def collect_trajectory(
    env: PlatformLander,
    policy: Policy,
    rng: np.random.Generator,
    args: argparse.Namespace,
    episode: int,
) -> dict[str, Any]:
    _observations, log_probs, rewards, episode_return, steps, info = run_episode_data(
        env,
        policy,
        rng,
        gamma=args.gamma,
        max_steps=args.max_steps,
        train=True,
    )
    rtg = rewards_to_go(rewards, args.gamma)
    return {
        "episode": episode,
        "log_probs": log_probs,
        "rtg": rtg,
        "return": episode_return,
        "rtg_mean": float(rtg.mean().item()),
        "steps": steps,
        "info": info,
    }


def baseline_for_trajectory(
    trajectory: dict[str, Any],
    *,
    baseline_mode: str,
    batch_total: float,
    batch_count: int,
    moving_baseline: float,
) -> float:
    rtg = trajectory["rtg"]
    if baseline_mode == "trajectory-mean":
        return float(rtg.mean().item())
    if baseline_mode == "batch-mean":
        own_total = float(rtg.sum().item())
        own_count = int(rtg.numel())
        other_count = batch_count - own_count
        return (batch_total - own_total) / other_count if other_count > 0 else 0.0
    if baseline_mode == "moving-average":
        return moving_baseline
    raise ValueError(f"Unknown baseline mode: {baseline_mode}")


def update_from_batch(
    trajectories: list[dict[str, Any]],
    *,
    baseline_mode: str,
    moving_baseline: float,
    gamma: float,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, list[float], list[float]]:
    batch_total = sum(float(trajectory["rtg"].sum().item()) for trajectory in trajectories)
    batch_count = sum(int(trajectory["rtg"].numel()) for trajectory in trajectories)
    loss_terms: list[torch.Tensor] = []
    baselines: list[float] = []
    mean_advantages: list[float] = []

    for trajectory in trajectories:
        rtg = trajectory["rtg"]
        baseline = baseline_for_trajectory(
            trajectory,
            baseline_mode=baseline_mode,
            batch_total=batch_total,
            batch_count=batch_count,
            moving_baseline=moving_baseline,
        )
        advantages = rtg - baseline
        log_prob_tensor = torch.stack(trajectory["log_probs"])
        discounts = discount_factors(int(rtg.numel()), gamma).to(log_prob_tensor.device)
        loss_terms.append(-(log_prob_tensor * discounts * advantages).sum())
        baselines.append(baseline)
        mean_advantages.append(float(advantages.mean().item()))

    loss = torch.stack(loss_terms).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item()), baselines, mean_advantages


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
    recent_rtg_means: deque[float] = deque(maxlen=args.target_window)
    recent_successes: deque[bool] = deque(maxlen=args.target_window)
    recent_jet_fires: deque[int] = deque(maxlen=args.target_window)
    training_rows: list[dict[str, object]] = []
    log_file = open_log(args.log_file)

    episode = 0
    try:
        log(
            f"training_start script=full_reinforce "
            f"episodes={args.episodes} batch_size={args.batch_size} baseline={args.baseline} "
            f"max_steps={args.max_steps} seed={args.seed} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )
        while episode < args.episodes:
            moving_baseline = float(np.mean(recent_rtg_means)) if recent_rtg_means else 0.0
            trajectories: list[dict[str, Any]] = []
            for _ in range(args.batch_size):
                if episode >= args.episodes:
                    break
                episode += 1
                trajectories.append(collect_trajectory(env, policy, rng, args, episode))

            policy_loss, baselines, mean_advantages = update_from_batch(
                trajectories,
                baseline_mode=args.baseline,
                moving_baseline=moving_baseline,
                gamma=args.gamma,
                optimizer=optimizer,
            )

            for batch_index, (trajectory, baseline, mean_advantage) in enumerate(
                zip(trajectories, baselines, mean_advantages),
                start=1,
            ):
                info = trajectory["info"]
                episode_return = float(trajectory["return"])
                recent_returns.append(episode_return)
                recent_rtg_means.append(float(trajectory["rtg_mean"]))
                recent_successes.append(bool(info.get("success", False)))
                recent_jet_fires.append(int(info.get("jet_fires_used", 0)))
                average_return = float(np.mean(recent_returns))
                success_count = int(sum(recent_successes))
                average_jet_fires = float(np.mean(recent_jet_fires))
                current_episode = int(trajectory["episode"])
                training_rows.append(
                    {
                        "episode": current_episode,
                        "return": episode_return,
                        "baseline_mode": args.baseline,
                        "baseline": baseline,
                        "average_advantage": mean_advantage,
                        "policy_loss": policy_loss,
                        "average_return": average_return,
                        "success_count": success_count,
                        "success_rate": success_count / len(recent_successes),
                        "jet_fires": int(info.get("jet_fires_used", 0)),
                        "average_jet_fires": average_jet_fires,
                        "batch_index": batch_index,
                        "batch_size": args.batch_size,
                        "steps": int(trajectory["steps"]),
                        "success": bool(info.get("success", False)),
                        "failure_reason": info.get("failure_reason"),
                    }
                )

                if current_episode == 1 or current_episode % args.print_every == 0:
                    log(
                        f"episode={current_episode:5d} "
                        f"return={episode_return:8.2f} "
                        f"baseline={baseline:8.2f} "
                        f"advantage={mean_advantage:8.2f} "
                        f"loss={policy_loss:8.2f} "
                        f"avg{len(recent_returns):02d}={average_return:8.2f} "
                        f"success{len(recent_successes):02d}={success_count:2d} "
                        f"fires={info.get('jet_fires_used', 0):3d} "
                        f"avgfires{len(recent_jet_fires):02d}={average_jet_fires:6.1f} "
                        f"batch={batch_index:2d}/{args.batch_size:2d} "
                        f"steps={int(trajectory['steps']):4d} "
                        f"failure={info.get('failure_reason')}",
                        log_file,
                    )

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
