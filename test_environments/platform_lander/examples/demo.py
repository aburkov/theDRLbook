"""Render PlatformLander in a local Pygame window.

Run from the project root with:

    python examples/demo.py

The tests are intentionally headless; this script is for visual inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from platform_lander import PlatformLander, heuristic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the PlatformLander environment.")
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--wind", action="store_true", help="Enable wind during the demo.")
    parser.add_argument("--wind-power", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PlatformLander(
        render_mode="human",
        enable_wind=args.wind,
        wind_power=args.wind_power,
        wind_direction=(1.0, 0.0),
    )

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0

            for step in range(1000):
                if args.policy == "random":
                    action = env.action_space.sample()
                else:
                    action = heuristic(env, obs)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    print(
                        f"episode={episode} step={step} "
                        f"reward={total_reward:.1f} info={info}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()

