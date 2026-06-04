"""REINFORCE with reward-to-go.

Compared with ``vanilla_reinforce.py``, this keeps one trajectory per update
and no baseline. This practical variant drops the outer ``gamma^t`` factor
from the policy-gradient term while keeping discounted reward-to-go targets.
The policy network uses the common OpenAI-Baselines/SB3-style initialization
for on-policy RL: orthogonal hidden layers with gain sqrt(2), zero biases, and
a small 0.01-gain policy-logit head so the initial categorical policy is close
to uniform.

    loss = -sum_t G_t * log pi(a_t | o_t)

where

    G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
"""

from __future__ import annotations

import argparse
import math
from collections import deque

import numpy as np
import torch
from torch import nn

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-to-go REINFORCE for PlatformLander.")
    parser.add_argument("--episodes", type=int, default=150_000)
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
    add_output_args(parser, "gamma_dropped_rtg_reinforce")
    return parser.parse_args()


def rewards_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32)


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

    try:
        log(
            f"training_start script=gamma_dropped_rtg_reinforce "
            f"episodes={args.episodes} max_steps={args.max_steps} seed={args.seed} "
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

            rtg = rtg.to(
                device=log_prob_tensor.device,
                dtype=log_prob_tensor.dtype,
            )

            assert log_prob_tensor.shape == rtg.shape

            policy_loss = -(log_prob_tensor * rtg).sum()
            loss = policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                    "policy_loss": float(policy_loss.detach().item()),
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
                    f"avg{len(recent_returns):02d}={average_return:8.2f} "
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
