"""Train PlatformLander with the single-trajectory vanilla REINFORCE algorithm.

This script intentionally follows the textbook version of REINFORCE:

    loss = -R(tau) * sum_t log pi_theta(a_t | o_t)

Here R(tau) is the globally discounted trajectory return computed by
discounted_return(rewards, gamma). Minimizing that loss with SGD is gradient
ascent on R(tau) * sum_t grad log pi_theta(a_t | o_t).

Run from the project root:

    python vanilla_reinforce.py

After training stops, the script opens a Pygame window and shows three policy
rollouts from randomized booster and platform starts.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from platform_lander import PlatformLander
from platform_lander.platform_lander import BOOSTER_BOTTOM, BOOSTER_START_CLEARANCE, SCALE

PROJECT_ROOT = Path(__file__).resolve().parent


RUNS_DIR = PROJECT_ROOT / "runs"


def init_layer(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    """Small categorical policy for the four discrete booster actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, action_dim)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla REINFORCE for PlatformLander.")
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
    add_output_args(parser, "vanilla_reinforce")
    return parser.parse_args()


def add_output_args(parser: argparse.ArgumentParser, run_name: str) -> None:
    parser.add_argument(
        "--log-file",
        type=Path,
        default=RUNS_DIR / f"{run_name}.log",
        help="Path to write the training log.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=RUNS_DIR / f"{run_name}.pt",
        help="Path to write the trained policy checkpoint.",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=RUNS_DIR / f"{run_name}.csv",
        help="Path to write per-episode training data as CSV.",
    )


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def open_log(log_file: Path) -> TextIO:
    path = resolve_project_path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def log(message: str, log_file: TextIO | None = None) -> None:
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def save_policy(
    policy: Policy,
    args: argparse.Namespace,
    model_file: Path,
    *,
    obs_dim: int,
    action_dim: int,
) -> Path:
    path = resolve_project_path(model_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )
    return path


def save_training_csv(rows: list[dict[str, object]], csv_file: Path) -> Path:
    path = resolve_project_path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def load_policy(model_file: Path) -> tuple[Policy, dict]:
    checkpoint = torch.load(resolve_project_path(model_file), map_location="cpu")
    policy = Policy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
    )
    policy_state = policy.state_dict()
    checkpoint_state = checkpoint["policy_state_dict"]
    compatible_state = {
        key: value for key, value in checkpoint_state.items() if key in policy_state
    }
    policy.load_state_dict(compatible_state)
    policy.eval()
    return policy, checkpoint


def randomize_start(env: PlatformLander, rng: np.random.Generator) -> np.ndarray:
    """Randomize platform and booster start, then return the fresh observation."""

    assert env.booster is not None
    assert env.platform is not None

    w = 600 / 30.0
    h = 400 / 30.0

    env.platform.position = (
        float(rng.uniform(env.platform_min_x, env.platform_max_x)),
        env.platform_y,
    )
    env.platform_direction = int(rng.choice([-1, 1]))
    env.platform.linearVelocity = (
        env.platform_speed * env.platform_direction,
        0.0,
    )

    env.booster.position = (
        float(rng.uniform(0.18 * w, 0.82 * w)),
        float(h + BOOSTER_BOTTOM / SCALE + rng.uniform(BOOSTER_START_CLEARANCE, 0.18)),
    )
    env.booster.angle = float(rng.uniform(-0.45, 0.45))
    env.booster.linearVelocity = (
        float(rng.uniform(-1.0, 1.0)),
        float(rng.uniform(-0.6, 0.4)),
    )
    env.booster.angularVelocity = float(rng.uniform(-0.6, 0.6))
    env.booster.awake = True

    env.ocean_contact = False
    env.platform_contact = False
    env.body_platform_contact = False
    env.left_foot_contact = False
    env.right_foot_contact = False
    env.failure_reason = None
    env.prev_shaping = None
    env.bottom_flame_power = 0.0
    env.top_flame_power = 0.0
    env.top_flame_direction = 0
    env.jet_fires_used = 0
    return env._get_state()


def discounted_return(rewards: list[float], gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * reward
        discount *= gamma
    return total


def run_episode(
    env: PlatformLander,
    policy: Policy,
    rng: np.random.Generator,
    *,
    gamma: float,
    max_steps: int,
    train: bool,
) -> tuple[list[torch.Tensor], list[float], float, int, dict]:
    _observations, log_probs, rewards, episode_return, steps, info = run_episode_data(
        env,
        policy,
        rng,
        gamma=gamma,
        max_steps=max_steps,
        train=train,
    )
    return log_probs, rewards, episode_return, steps, info


def run_episode_data(
    env: PlatformLander,
    policy: Policy,
    rng: np.random.Generator,
    *,
    gamma: float,
    max_steps: int,
    train: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], float, int, dict]:
    observation, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    observation = randomize_start(env, rng)

    observations: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    info: dict = {}

    for step in range(max_steps):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        observations.append(obs_tensor)
        logits = policy(obs_tensor)
        dist = Categorical(logits=logits)

        if train:
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(logits)

        action = int(action_tensor.item())
        log_probs.append(dist.log_prob(action_tensor))

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))

        if terminated or truncated:
            break

    episode_return = discounted_return(rewards, gamma)
    return observations, log_probs, rewards, episode_return, step + 1, info


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
            f"training_start script=vanilla_reinforce "
            f"episodes={args.episodes} max_steps={args.max_steps} seed={args.seed} "
            f"model_file={resolve_project_path(args.model_file)}",
            log_file,
        )
        for episode in range(1, args.episodes + 1):
            log_probs, rewards, episode_return, steps, info = run_episode(
                env,
                policy,
                rng,
                gamma=args.gamma,
                max_steps=args.max_steps,
                train=True,
            )

            sum_log_probs = torch.stack(log_probs).sum()
            loss = -sum_log_probs * episode_return

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


def animate(policy: Policy, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed + 10_000)
    env = PlatformLander(
        render_mode="human",
        enable_wind=args.wind,
        wind_power=args.wind_power,
        wind_direction=(1.0, 0.0),
    )

    try:
        animation_runs = int(getattr(args, "animation_runs", 3))
        for run in range(animation_runs):
            _, rewards, episode_return, steps, info = run_episode(
                env,
                policy,
                rng,
                gamma=args.gamma,
                max_steps=args.max_steps,
                train=False,
            )
            print(
                f"animation_run={run + 1} "
                f"return={episode_return:.2f} "
                f"undiscounted={sum(rewards):.2f} "
                f"steps={steps} "
                f"fires={info.get('jet_fires_used', 0)} "
                f"info={info}"
            )
            time.sleep(0.75)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    policy = train(args)
    if not args.no_animation:
        animate(policy, args)


if __name__ == "__main__":
    main()
