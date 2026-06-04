"""REINFORCE with reward-to-go and a learned value-function baseline.

Compared with ``rtg_reinforce.py``, this script trains an additional value
network V_phi(o_t) and uses it as the baseline for the policy-gradient update:

    A_t = G_t - V_phi(o_t)
    policy_loss = -sum_t A_t * log pi_theta(a_t | o_t)
                  - entropy_coef * sum_t H(pi(. | o_t))
    value_loss = mean_t (V_phi(o_t) - G_t)^2

This practical variant drops the outer ``gamma^t`` factor from the
policy-gradient term while keeping discounted reward-to-go targets.

This is still Monte Carlo REINFORCE with baseline, not actor-critic/A2C: the
value network is trained to predict the sampled reward-to-go G_t, and the policy
update does not bootstrap from V_phi(o_{t+1}).
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gamma_dropped_rtg_reinforce import Policy, rewards_to_go
from platform_lander import PlatformLander
from vanilla_reinforce import (
    add_output_args,
    animate,
    log,
    open_log,
    resolve_project_path,
    run_episode_data,
    save_training_csv,
)


class ValueFunction(nn.Module):
    """Small state-value baseline network V_phi(o)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            self._init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            self._init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self._init_layer(nn.Linear(hidden_dim, 1)),
        )

    @staticmethod
    def _init_layer(layer: nn.Linear) -> nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        return layer

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


def trajectory_entropies(policy: Policy, observations: list[torch.Tensor]) -> torch.Tensor:
    """Return H(pi(. | o_t)) for each observation in one trajectory."""
    observation_tensor = torch.stack(observations)
    logits = policy(observation_tensor)
    dist = Categorical(logits=logits)
    return dist.entropy().reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE with reward-to-go and learned value-function baseline.")
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Coefficient for the policy entropy bonus. A value of 0 disables it.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--value-hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-power", type=float, default=5.0)
    parser.add_argument("--no-animation", action="store_true")
    add_output_args(parser, "gamma_dropped_value_function_baseline_reinforce")
    return parser.parse_args()


def collect_trajectory(
    env: PlatformLander,
    policy: Policy,
    rng: np.random.Generator,
    args: argparse.Namespace,
    episode: int,
) -> dict[str, Any]:
    observations, log_probs, rewards, episode_return, steps, info = run_episode_data(
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
        "observations": observations,
        "log_probs": log_probs,
        "rtg": rtg,
        "return": episode_return,
        "rtg_mean": float(rtg.mean().item()),
        "steps": steps,
        "info": info,
    }


def update_from_trajectory(
    trajectory: dict[str, Any],
    *,
    entropy_coef: float,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    policy: Policy,
    value_function: ValueFunction,
) -> tuple[float, float, float, float, float, float, float]:
    """Update policy and value baseline from one complete sampled trajectory.

    Returns:
        policy_loss, value_loss, mean_value_prediction, mean_advantage,
        entropy_bonus, average_entropy, loss
    """
    observation_tensor = torch.stack(trajectory["observations"])
    log_prob_tensor = torch.stack(trajectory["log_probs"]).reshape(-1)
    rtg = trajectory["rtg"].to(
        device=log_prob_tensor.device,
        dtype=log_prob_tensor.dtype,
    )
    entropy_tensor = trajectory_entropies(policy, trajectory["observations"])

    value_predictions = value_function(observation_tensor).to(log_prob_tensor.device)
    entropy_tensor = entropy_tensor.to(
        device=log_prob_tensor.device,
        dtype=log_prob_tensor.dtype,
    )

    assert log_prob_tensor.shape == rtg.shape == entropy_tensor.shape

    # The baseline is treated as a constant for the policy-gradient step.
    # This prevents the policy loss from updating the value network.
    advantages = rtg - value_predictions.detach()
    policy_loss = -(log_prob_tensor * advantages).sum()
    entropy_bonus = entropy_tensor.sum()
    loss = policy_loss - entropy_coef * entropy_bonus

    # The value network is trained by Monte Carlo regression toward G_t.
    # No bootstrapped target is used, so this remains REINFORCE with baseline.
    value_loss = F.mse_loss(value_predictions, rtg)

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return (
        float(policy_loss.detach().item()),
        float(value_loss.detach().item()),
        float(value_predictions.detach().mean().item()),
        float(advantages.detach().mean().item()),
        float(entropy_bonus.detach().item()),
        float(entropy_tensor.detach().mean().item()),
        float(loss.detach().item()),
    )


def save_policy_and_value(
    policy: Policy,
    value_function: ValueFunction,
    args: argparse.Namespace,
    model_file: Path,
    *,
    obs_dim: int,
    action_dim: int,
    value_hidden_dim: int,
) -> Path:
    path = resolve_project_path(model_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_function.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "value_hidden_dim": value_hidden_dim,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )
    return path


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
    value_hidden_dim = int(args.value_hidden_dim or args.hidden_dim)

    policy_rng_state = torch.get_rng_state()
    value_function = ValueFunction(obs_dim=obs_dim, hidden_dim=value_hidden_dim)
    torch.set_rng_state(policy_rng_state)

    policy_optimizer = torch.optim.SGD(policy.parameters(), lr=args.learning_rate)
    value_optimizer = torch.optim.Adam(value_function.parameters(), lr=args.value_learning_rate)

    recent_returns: deque[float] = deque(maxlen=args.target_window)
    recent_value_losses: deque[float] = deque(maxlen=args.target_window)
    recent_successes: deque[bool] = deque(maxlen=args.target_window)
    recent_jet_fires: deque[int] = deque(maxlen=args.target_window)
    training_rows: list[dict[str, object]] = []
    log_file = open_log(args.log_file)

    try:
        log(
            f"training_start script=gamma_dropped_value_function_baseline_reinforce "
            f"episodes={args.episodes} max_steps={args.max_steps} seed={args.seed} "
            f"entropy_coef={args.entropy_coef} "
            f"value_learning_rate={args.value_learning_rate} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )
        for episode in range(1, args.episodes + 1):
            trajectory = collect_trajectory(env, policy, rng, args, episode)
            (
                policy_loss,
                value_loss,
                mean_value,
                mean_advantage,
                entropy_bonus,
                average_entropy,
                loss,
            ) = update_from_trajectory(
                trajectory,
                entropy_coef=args.entropy_coef,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                policy=policy,
                value_function=value_function,
            )

            info = trajectory["info"]
            episode_return = float(trajectory["return"])
            recent_returns.append(episode_return)
            recent_value_losses.append(value_loss)
            recent_successes.append(bool(info.get("success", False)))
            recent_jet_fires.append(int(info.get("jet_fires_used", 0)))
            average_return = float(np.mean(recent_returns))
            average_value_loss = float(np.mean(recent_value_losses))
            success_count = int(sum(recent_successes))
            average_jet_fires = float(np.mean(recent_jet_fires))

            training_rows.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "rtg_mean": float(trajectory["rtg_mean"]),
                    "value_prediction_mean": mean_value,
                    "average_advantage": mean_advantage,
                    "policy_loss": policy_loss,
                    "entropy_bonus": entropy_bonus,
                    "average_entropy": average_entropy,
                    "loss": loss,
                    "value_loss": value_loss,
                    "average_value_loss": average_value_loss,
                    "average_return": average_return,
                    "success_count": success_count,
                    "success_rate": success_count / len(recent_successes),
                    "jet_fires": int(info.get("jet_fires_used", 0)),
                    "average_jet_fires": average_jet_fires,
                    "steps": int(trajectory["steps"]),
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                }
            )

            if episode == 1 or episode % args.print_every == 0:
                log(
                    f"episode={episode:5d} "
                    f"return={episode_return:8.2f} "
                    f"value={mean_value:8.2f} "
                    f"advantage={mean_advantage:8.2f} "
                    f"policy_loss={policy_loss:8.2f} "
                    f"value_loss={value_loss:8.2f} "
                    f"avg_vloss{len(recent_value_losses):02d}={average_value_loss:8.2f} "
                    f"avg{len(recent_returns):02d}={average_return:8.2f} "
                    f"entropy={average_entropy:5.3f} "
                    f"success{len(recent_successes):02d}={success_count:2d} "
                    f"fires={info.get('jet_fires_used', 0):3d} "
                    f"avgfires{len(recent_jet_fires):02d}={average_jet_fires:6.1f} "
                    f"steps={int(trajectory['steps']):4d} "
                    f"failure={info.get('failure_reason')}",
                    log_file,
                )

        model_path = save_policy_and_value(
            policy,
            value_function,
            args,
            args.model_file,
            obs_dim=obs_dim,
            action_dim=action_dim,
            value_hidden_dim=value_hidden_dim,
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
