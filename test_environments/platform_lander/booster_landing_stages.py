"""Capture three real PlatformLander render frames.

Run:

    python booster_landing_stages.py

The script saves three individual environment screenshots and one combined
three-panel image in the same directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
from platform_lander import PlatformLander
from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE


OUT_DIR = PROJECT_ROOT


def set_scene(
    env: PlatformLander,
    *,
    platform_x: float,
    booster_x: float,
    booster_y: float,
    booster_angle: float,
    bottom_flame_power: float,
    top_flame_power: float,
    top_flame_direction: int,
):
    assert env.platform is not None
    assert env.booster is not None

    env.platform.position = (platform_x, env.platform_y)
    env.platform_direction = 1
    env.platform.linearVelocity = (env.platform_speed, 0.0)

    env.booster.position = (booster_x, booster_y)
    env.booster.angle = booster_angle
    env.booster.linearVelocity = (0.0, 0.0)
    env.booster.angularVelocity = 0.0
    env.booster.awake = True

    env.bottom_flame_power = bottom_flame_power
    env.top_flame_power = top_flame_power
    env.top_flame_direction = top_flame_direction

    return env.render()


def save_frame(image, filename: str):
    path = OUT_DIR / filename
    plt.imsave(path, image)
    return path


def main():
    env = PlatformLander(render_mode="rgb_array")
    env.reset(seed=7)

    target_y = env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE

    scenes = [
        (
            "High and correcting",
            set_scene(
                env,
                platform_x=3.0,
                booster_x=4.1,
                booster_y=10.7,
                booster_angle=-0.42,
                bottom_flame_power=0.95,
                top_flame_power=1.0,
                top_flame_direction=-1,
            ),
            "booster_landing_stage_1_high.png",
        ),
        (
            "Close to landing",
            set_scene(
                env,
                platform_x=5.0,
                booster_x=5.15,
                booster_y=target_y + 2.2,
                booster_angle=-0.08,
                bottom_flame_power=1.0,
                top_flame_power=0.85,
                top_flame_direction=1,
            ),
            "booster_landing_stage_2_close.png",
        ),
        (
            "Landed",
            set_scene(
                env,
                platform_x=7.0,
                booster_x=7.0,
                booster_y=target_y,
                booster_angle=0.0,
                bottom_flame_power=0.0,
                top_flame_power=0.0,
                top_flame_direction=0,
            ),
            "booster_landing_stage_3_landed.png",
        ),
    ]

    for _, image, filename in scenes:
        save_frame(image, filename)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (title, image, _) in zip(axes, scenes):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    combined_path = OUT_DIR / "booster_landing_stages.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    env.close()

    print(f"saved {combined_path}")
    for _, _, filename in scenes:
        print(f"saved {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
