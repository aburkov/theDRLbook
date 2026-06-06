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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from platform_lander import DEFAULT_WIND_POWER, PlatformLander
from vanilla_reinforce import (
    add_output_args,
    add_reward_args,
    animate,
    log,
    open_log,
    platform_lander_reward_kwargs,
    resolve_project_path,
    run_episode_data,
    save_policy,
    save_training_csv,
)


@dataclass(frozen=True)
class TrainingResult:
    seed: int
    policy: "Policy"
    final_row: dict[str, object]
    model_path: Path
    best_model_path: Path | None
    csv_path: Path


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
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save the model and CSV every N episodes. Use 0 to save only at the end.",
    )
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-power", type=float, default=DEFAULT_WIND_POWER)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Keep training after the target window reaches all successes.",
    )
    add_reward_args(parser)
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


def with_seed_suffix(path: Path, seed: int) -> Path:
    return path.with_name(f"{path.stem}_seed{seed}{path.suffix}")


def args_for_seed(args: argparse.Namespace, seed: int, *, multi_seed: bool) -> argparse.Namespace:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = int(seed)
    if multi_seed:
        seed_args.log_file = with_seed_suffix(args.log_file, seed)
        seed_args.model_file = with_seed_suffix(args.model_file, seed)
        seed_args.best_model_file = with_seed_suffix(args.best_model_file, seed)
        seed_args.csv_file = with_seed_suffix(args.csv_file, seed)
    return seed_args


def save_checkpoint(
    policy: Policy,
    args: argparse.Namespace,
    training_rows: list[dict[str, object]],
    *,
    obs_dim: int,
    action_dim: int,
    log_file,
    episode: int,
) -> tuple[Path, Path]:
    model_path = save_policy(
        policy,
        args,
        args.model_file,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    csv_path = save_training_csv(training_rows, args.csv_file)
    log(
        f"saved_checkpoint episode={episode} "
        f"model_file={model_path} csv_file={csv_path}",
        log_file,
    )
    return model_path, csv_path


def save_best_checkpoint(
    policy: Policy,
    args: argparse.Namespace,
    *,
    obs_dim: int,
    action_dim: int,
    log_file,
    episode: int,
    success_count: int,
    average_return: float,
) -> Path:
    best_model_path = save_policy(
        policy,
        args,
        args.best_model_file,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    log(
        f"saved_best_checkpoint episode={episode} "
        f"success{args.target_window}={success_count} "
        f"avg{args.target_window}={average_return:.2f} "
        f"model_file={best_model_path}",
        log_file,
    )
    return best_model_path


def clear_existing_best_checkpoint(args: argparse.Namespace, log_file) -> None:
    best_model_path = resolve_project_path(args.best_model_file)
    if best_model_path.exists():
        best_model_path.unlink()
        log(f"cleared_best_checkpoint model_file={best_model_path}", log_file)


def train(args: argparse.Namespace) -> TrainingResult:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = PlatformLander(
        enable_wind=args.wind,
        wind_power=args.wind_power,
        wind_direction=(1.0, 0.0),
        **platform_lander_reward_kwargs(args),
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
    recent_entropies: deque[float] = deque(maxlen=args.target_window)

    training_rows: list[dict[str, object]] = []
    log_file = open_log(args.log_file)
    model_path: Path | None = None
    best_model_path: Path | None = None
    csv_path: Path | None = None
    completed_episode = 0
    last_saved_episode = 0
    best_success_count = 0

    try:
        log(
            f"training_start script=gamma_dropped_rtg_reinforce "
            f"episodes={args.episodes} max_steps={args.max_steps} seed={args.seed} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )
        clear_existing_best_checkpoint(args, log_file)

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

            recent_returns.append(episode_return)
            recent_successes.append(bool(info.get("success", False)))
            recent_jet_fires.append(int(info.get("jet_fires_used", 0)))
            policy_entropy = float(info.get("policy_entropy", 0.0))
            recent_entropies.append(policy_entropy)

            average_return = float(np.mean(recent_returns))
            success_count = int(sum(recent_successes))
            average_jet_fires = float(np.mean(recent_jet_fires))
            average_entropy = float(np.mean(recent_entropies))
            effective_actions = float(math.exp(policy_entropy))
            average_effective_actions = float(math.exp(average_entropy))

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
                    "policy_entropy": policy_entropy,
                    "average_policy_entropy": average_entropy,
                    "effective_actions": effective_actions,
                    "average_effective_actions": average_effective_actions,
                    "steps": steps,
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                }
            )
            completed_episode = episode

            target_window_boundary = (
                len(recent_successes) == args.target_window
                and episode % args.target_window == 0
            )
            improved_best = target_window_boundary and success_count > best_success_count
            if improved_best:
                best_success_count = success_count
                best_model_path = save_best_checkpoint(
                    policy,
                    args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    log_file=log_file,
                    episode=episode,
                    success_count=success_count,
                    average_return=average_return,
                )

            if target_window_boundary and success_count == args.target_window and not args.no_early_stop:
                log(
                    f"early_stop episode={episode} "
                    f"success{args.target_window}={success_count} "
                    f"avg{args.target_window}={average_return:.2f} "
                    f"best_model_file={best_model_path}",
                    log_file,
                )
                break

            if episode == 1 or episode % args.print_every == 0:
                log(
                    f"episode={episode:5d} "
                    f"return={episode_return:8.2f} "
                    f"avg{len(recent_returns):02d}={average_return:8.2f} "
                    f"success{len(recent_successes):02d}={success_count:2d} "
                    f"fires={info.get('jet_fires_used', 0):3d} "
                    f"avgfires{len(recent_jet_fires):02d}={average_jet_fires:6.1f} "
                    f"entropy={policy_entropy:5.3f} "
                    f"effact={effective_actions:4.2f} "
                    f"avgeff{len(recent_entropies):02d}={average_effective_actions:4.2f} "
                    f"steps={steps:4d} "
                    f"failure={info.get('failure_reason')}",
                    log_file,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.save_every and episode % args.save_every == 0:
                model_path, csv_path = save_checkpoint(
                    policy,
                    args,
                    training_rows,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    log_file=log_file,
                    episode=episode,
                )
                last_saved_episode = episode

    except KeyboardInterrupt:
        log(f"training_interrupted episode={completed_episode}", log_file)

    finally:
        if training_rows and completed_episode != last_saved_episode:
            model_path, csv_path = save_checkpoint(
                policy,
                args,
                training_rows,
                obs_dim=obs_dim,
                action_dim=action_dim,
                log_file=log_file,
                episode=completed_episode,
            )
        env.close()
        log_file.close()

    if not training_rows or model_path is None or csv_path is None:
        raise RuntimeError("training produced no episodes to save")

    return TrainingResult(
        seed=int(args.seed),
        policy=policy,
        final_row=training_rows[-1],
        model_path=model_path,
        best_model_path=best_model_path,
        csv_path=csv_path,
    )


def report_multi_seed_results(results: list[TrainingResult], target_window: int) -> None:
    final_avg_returns = np.asarray(
        [float(result.final_row["average_return"]) for result in results],
        dtype=np.float64,
    )
    final_success_rates = np.asarray(
        [float(result.final_row["success_rate"]) for result in results],
        dtype=np.float64,
    )
    final_returns = np.asarray(
        [float(result.final_row["return"]) for result in results],
        dtype=np.float64,
    )
    final_avg_jet_fires = np.asarray(
        [float(result.final_row["average_jet_fires"]) for result in results],
        dtype=np.float64,
    )
    final_effective_actions = np.asarray(
        [float(result.final_row["average_effective_actions"]) for result in results],
        dtype=np.float64,
    )

    print("")
    print("multi_seed_report script=gamma_dropped_rtg_reinforce")
    print(f"seeds={','.join(str(result.seed) for result in results)}")
    print(f"runs={len(results)}")
    print(
        f"final_avg{target_window}_mean={final_avg_returns.mean():.2f} "
        f"std={final_avg_returns.std(ddof=0):.2f} "
        f"min={final_avg_returns.min():.2f} "
        f"max={final_avg_returns.max():.2f}"
    )
    print(
        f"final_success_rate_mean={final_success_rates.mean():.3f} "
        f"std={final_success_rates.std(ddof=0):.3f}"
    )
    print(
        f"final_return_mean={final_returns.mean():.2f} "
        f"std={final_returns.std(ddof=0):.2f}"
    )
    print(
        f"final_avg_jet_fires_mean={final_avg_jet_fires.mean():.1f} "
        f"std={final_avg_jet_fires.std(ddof=0):.1f}"
    )
    print(
        f"final_effective_actions_mean={final_effective_actions.mean():.2f} "
        f"std={final_effective_actions.std(ddof=0):.2f}"
    )
    for result in results:
        row = result.final_row
        print(
            f"seed={result.seed} "
            f"final_avg{target_window}={float(row['average_return']):.2f} "
            f"final_success_rate={float(row['success_rate']):.3f} "
            f"final_return={float(row['return']):.2f} "
            f"final_avgfires{target_window}={float(row['average_jet_fires']):.1f} "
            f"final_effact={float(row['average_effective_actions']):.2f} "
            f"model_file={result.model_path} "
            f"best_model_file={result.best_model_path} "
            f"csv_file={result.csv_path}"
        )


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seed]
    multi_seed = len(seeds) > 1
    results: list[TrainingResult] = []

    for seed in seeds:
        result = train(args_for_seed(args, seed, multi_seed=multi_seed))
        results.append(result)

    if multi_seed:
        report_multi_seed_results(results, args.target_window)

    if not args.no_animation:
        animate(results[-1].policy, args_for_seed(args, results[-1].seed, multi_seed=multi_seed))


if __name__ == "__main__":
    main()
