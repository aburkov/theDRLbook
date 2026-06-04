"""Batched REINFORCE with reward-to-go and optional entropy regularization.

Compared with ``rtg_reinforce.py``, this averages several complete
trajectory-gradient estimates before each optimizer step. This practical
variant drops the outer ``gamma^t`` factor from the policy-gradient term while
keeping discounted reward-to-go targets. An entropy bonus can be enabled with
``--entropy-coef`` to discourage premature policy collapse. The policy network
uses the common OpenAI-Baselines/SB3-style initialization for on-policy RL:
orthogonal hidden layers with gain sqrt(2), zero biases, and a small 0.01-gain
policy-logit head so the initial categorical policy is close to uniform.

    loss = mean_i [-sum_t G_t^i * log pi(a_t^i | o_t^i)
                   - entropy_coef * sum_t H(pi(. | o_t^i))]

where

    G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from platform_lander import PlatformLander
from vanilla_reinforce import (
    add_output_args,
    animate,
    log,
    open_log,
    resolve_project_path,
    run_episode_data,
    save_policy,
    save_training_csv,
)


def init_layer(
    layer: nn.Linear,
    *,
    gain: float = math.sqrt(2.0),
    bias_const: float = 0.0,
) -> nn.Linear:
    """Initialize a linear layer using the common on-policy RL scheme."""
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    """Categorical policy with OpenAI-Baselines/SB3-style initialization."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class TrajectoryLoss(NamedTuple):
    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy_bonus: torch.Tensor
    average_entropy: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch reward-to-go REINFORCE for PlatformLander.")
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=10)
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
    add_output_args(parser, "gamma_dropped_batch_rtg_reinforce")
    return parser.parse_args()


def rewards_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32)


def trajectory_entropies(policy: Policy, observations: list[torch.Tensor]) -> torch.Tensor:
    """Return H(pi(. | o_t)) for each observation in one trajectory."""
    observation_tensor = torch.stack(observations)
    logits = policy(observation_tensor)
    dist = Categorical(logits=logits)
    return dist.entropy().reshape(-1)


def trajectory_rtg_loss(
    policy: Policy,
    observations: list[torch.Tensor],
    log_probs: list[torch.Tensor],
    rewards: list[float],
    *,
    gamma: float,
    entropy_coef: float,
) -> TrajectoryLoss:
    """Return one complete-trajectory RTG loss term.

    The returned loss keeps the policy log-probability and entropy graphs intact
    so the caller can average several trajectory losses and backpropagate once.
    """
    rtg = rewards_to_go(rewards, gamma)
    log_prob_tensor = torch.stack(log_probs).reshape(-1)
    entropy_tensor = trajectory_entropies(policy, observations)

    rtg = rtg.to(
        device=log_prob_tensor.device,
        dtype=log_prob_tensor.dtype,
    )
    entropy_tensor = entropy_tensor.to(
        device=log_prob_tensor.device,
        dtype=log_prob_tensor.dtype,
    )

    assert log_prob_tensor.shape == rtg.shape == entropy_tensor.shape

    policy_loss = -(log_prob_tensor * rtg).sum()
    entropy_bonus = entropy_tensor.sum()
    loss = policy_loss - entropy_coef * entropy_bonus

    return TrajectoryLoss(
        loss=loss,
        policy_loss=policy_loss,
        entropy_bonus=entropy_bonus,
        average_entropy=entropy_tensor.mean(),
    )


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
            f"training_start script=gamma_dropped_batch_rtg_reinforce "
            f"episodes={args.episodes} batch_size={args.batch_size} "
            f"max_steps={args.max_steps} seed={args.seed} "
            f"entropy_coef={args.entropy_coef} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )

        while episode < args.episodes:
            loss_terms: list[torch.Tensor] = []
            policy_loss_values: list[float] = []
            entropy_bonus_values: list[float] = []
            average_entropy_values: list[float] = []

            for _ in range(args.batch_size):
                if episode >= args.episodes:
                    break
                episode += 1

                observations, log_probs, rewards, episode_return, steps, info = run_episode_data(
                    env,
                    policy,
                    rng,
                    gamma=args.gamma,
                    max_steps=args.max_steps,
                    train=True,
                )

                trajectory_loss = trajectory_rtg_loss(
                    policy,
                    observations,
                    log_probs,
                    rewards,
                    gamma=args.gamma,
                    entropy_coef=args.entropy_coef,
                )
                loss_terms.append(trajectory_loss.loss)
                policy_loss_values.append(float(trajectory_loss.policy_loss.detach().item()))
                entropy_bonus_values.append(float(trajectory_loss.entropy_bonus.detach().item()))
                average_entropy_values.append(float(trajectory_loss.average_entropy.detach().item()))

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
                        "policy_loss": policy_loss_values[-1],
                        "entropy_bonus": entropy_bonus_values[-1],
                        "average_entropy": average_entropy_values[-1],
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
                        f"entropy={average_entropy_values[-1]:5.3f} "
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

            if training_rows:
                training_rows[-1]["batch_policy_loss_mean"] = float(np.mean(policy_loss_values))
                training_rows[-1]["batch_entropy_bonus_mean"] = float(np.mean(entropy_bonus_values))
                training_rows[-1]["batch_average_entropy_mean"] = float(np.mean(average_entropy_values))
                training_rows[-1]["batch_loss_mean"] = float(loss.detach().item())

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
