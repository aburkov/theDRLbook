"""Watch the saved gamma-dropped reward-to-go REINFORCE policy.

This script loads the checkpoint produced by:

    python gamma_dropped_rtg_reinforce.py --wind --wind-power 5.0 --gamma 0.99 --learning-rate 1e-6 --seed 42

It shows animated greedy rollouts and overlays separate fire counters for the
    upper-left attitude jet, bottom engine, and upper-right attitude jet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from platform_lander import (
    DEFAULT_FAILURE_REWARD,
    DEFAULT_SHAPING_FACTOR,
    DEFAULT_SUCCESS_REWARD,
    DEFAULT_WIND_POWER,
    PlatformLander,
)
from platform_lander.platform_lander import BOOSTER_BOTTOM, MAX_JET_FIRES, SCALE
from vanilla_reinforce import (
    RUNS_DIR,
    discounted_return,
    load_policy,
    nonnegative_float,
    positive_int,
    randomize_start,
    resolve_project_path,
)


DEFAULT_MODEL_FILE = RUNS_DIR / "gamma_dropped_rtg_reinforce.pt"
DEFAULT_BEST_MODEL_FILE = RUNS_DIR / "gamma_dropped_rtg_reinforce_best.pt"
DESCRIPTION = "Animate a saved gamma-dropped reward-to-go REINFORCE policy."
TRAIN_COMMAND = (
    "python gamma_dropped_rtg_reinforce.py --wind --wind-power 5.0 "
    "--gamma 0.99 --learning-rate 1e-6 --seed 42"
)
WINDOW_TITLE = "PlatformLander policy with per-engine fire counters"
DEFAULT_RUNS = 3
DEFAULT_MAX_STEPS = 400
DEFAULT_GAMMA = 0.99
DEFAULT_SEED = 42
CRASH_HOLD_FRAMES = 30

ACTION_NAMES = {
    0: "noop",
    1: "upper-left",
    2: "bottom",
    3: "upper-right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the best checkpoint when present, otherwise the latest checkpoint.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="Sample actions from the policy distribution instead of using greedy argmax.",
    )
    parser.add_argument("--wind", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wind-power", type=float, default=None)
    parser.add_argument("--success-reward", type=float, default=None)
    parser.add_argument("--failure-reward", type=float, default=None)
    parser.add_argument("--shaping-factor", type=nonnegative_float, default=None)
    parser.add_argument("--max-jet-fires", type=positive_int, default=None)
    return parser.parse_args()


def checkpoint_arg(checkpoint: dict, name: str, default):
    return checkpoint.get("args", {}).get(name, default)


def draw_overlay(
    screen,
    pygame,
    font,
    *,
    run: int,
    step: int,
    action: int,
    return_so_far: float,
    counts: dict[int, int],
    total_fires: int,
    max_fires: int,
    info: dict,
) -> None:
    rows = [
        f"run {run}   step {step}   action {action}: {ACTION_NAMES[action]}",
        f"upper-left:  {counts[1]:2d}",
        f"bottom:      {counts[2]:2d}",
        f"upper-right: {counts[3]:2d}",
        f"total fires: {total_fires:2d}/{max_fires}",
        f"return: {return_so_far:7.2f}",
        f"contact: L={int(info.get('left_foot_contact', False))} R={int(info.get('right_foot_contact', False))} B={int(info.get('body_platform_contact', False))}",
        f"anchored: {','.join(info.get('anchored_feet', ())) or 'none'}",
        f"stable: {int(info.get('stable_landing_steps', 0))}",
    ]

    if info.get("failure_reason") is not None:
        rows.append(f"failure: {info['failure_reason']}")
    elif info.get("success", False):
        rows.append("success")

    padding = 8
    line_h = font.get_height() + 4
    width = 235
    height = padding * 2 + line_h * len(rows)
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((10, 14, 20, 190))
    screen.blit(panel, (10, 10))

    for i, text in enumerate(rows):
        color = (245, 248, 252)
        if text.startswith("bottom"):
            color = (255, 228, 135)
        elif text.startswith("upper"):
            color = (176, 217, 255)
        elif text == "success":
            color = (122, 240, 160)
        elif text.startswith("failure"):
            color = (255, 145, 130)
        screen.blit(font.render(text, True, color), (10 + padding, 10 + padding + i * line_h))


def show_rollout(
    *,
    policy,
    run: int,
    rng: np.random.Generator,
    gamma: float,
    max_steps: int,
    wind: bool,
    wind_power: float,
    max_jet_fires: int,
    success_reward: float,
    failure_reward: float,
    shaping_factor: float,
    sample_actions: bool,
) -> bool:
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("pygame is required for animation. Install it with `pip install pygame`.") from exc

    env = PlatformLander(
        render_mode="rgb_array",
        enable_wind=wind,
        wind_power=wind_power,
        wind_direction=(1.0, 0.0),
        max_jet_fires=max_jet_fires,
        success_reward=success_reward,
        failure_reward=failure_reward,
        shaping_factor=shaping_factor,
    )

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 16) or pygame.font.Font(None, 18)

    counts = {1: 0, 2: 0, 3: 0}
    rewards: list[float] = []
    info: dict = {}
    step = 0
    ended = False

    try:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        observation = randomize_start(env, rng)

        for step in range(1, max_steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                logits = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                probs = dist.probs.detach().cpu().numpy()
                if sample_actions:
                    action = int(dist.sample().item())
                else:
                    action = int(torch.argmax(logits).item())

            fires_before = int(env.jet_fires_used)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            fires_after = int(info.get("jet_fires_used", env.jet_fires_used))
            booster = env.booster
            platform = env.platform
            assert booster is not None
            assert platform is not None
            target_y = env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE
            bottom_y = float(booster.position.y) - BOOSTER_BOTTOM / SCALE
            ocean_clearance = bottom_y - float(env.ocean_y)
            relative_x = float(observation[0])
            speed_error = float(np.sqrt(observation[2] * observation[2] + observation[3] * observation[3]))
            abs_angle = float(abs(observation[4]))
            impact_velocity = info.get("platform_impact_velocity")
            if impact_velocity is None:
                impact_vx = impact_vy = impact_av = float("nan")
            else:
                impact_vx, impact_vy, impact_av = (float(value) for value in impact_velocity)

            if action in counts and fires_after > fires_before:
                counts[action] += fires_after - fires_before

            print(
                f"run={run} "
                f"step={step:4d} "
                f"action={action} "
                f"action_name={ACTION_NAMES[action]} "
                f"p_noop={float(probs[0]):5.3f} "
                f"p_ul={float(probs[1]):5.3f} "
                f"p_bottom={float(probs[2]):5.3f} "
                f"p_ur={float(probs[3]):5.3f} "
                f"fired={int(fires_after > fires_before)} "
                f"upper_left={counts[1]} "
                f"bottom={counts[2]} "
                f"upper_right={counts[3]} "
                f"total_fires={fires_after} "
                f"remaining={info.get('jet_fires_remaining', 0)} "
                f"booster_x={float(booster.position.x):7.3f} "
                f"booster_y={float(booster.position.y):7.3f} "
                f"platform_x={float(platform.position.x):7.3f} "
                f"platform_y={float(platform.position.y):7.3f} "
                f"target_y={float(target_y):7.3f} "
                f"bottom_y={bottom_y:7.3f} "
                f"ocean_clearance={ocean_clearance:7.3f} "
                f"dx={float(booster.position.x - platform.position.x):7.3f} "
                f"dy={float(booster.position.y - platform.position.y):7.3f} "
                f"y_error={float(booster.position.y - target_y):7.3f} "
                f"relative_x={relative_x:7.3f} "
                f"speed_error={speed_error:7.3f} "
                f"abs_angle={abs_angle:7.3f} "
                f"vx={float(booster.linearVelocity.x):7.3f} "
                f"vy={float(booster.linearVelocity.y):7.3f} "
                f"rel_vx={float(booster.linearVelocity.x - platform.linearVelocity.x):7.3f} "
                f"impact_vx={impact_vx:7.3f} "
                f"impact_vy={impact_vy:7.3f} "
                f"impact_av={impact_av:7.3f} "
                f"platform_vx={float(platform.linearVelocity.x):6.3f} "
                f"angle={float(booster.angle):7.3f} "
                f"stable={int(info.get('stable_landing_steps', 0)):2d} "
                f"anchored={','.join(info.get('anchored_feet', ())) or 'none'} "
                f"reward={float(reward):8.2f} "
                f"failure={info.get('failure_reason')}",
                flush=True,
            )

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            draw_overlay(
                screen,
                pygame,
                font,
                run=run,
                step=step,
                action=action,
                return_so_far=discounted_return(rewards, gamma),
                counts=counts,
                total_fires=fires_after,
                max_fires=env.max_jet_fires,
                info=info,
            )
            pygame.display.flip()
            clock.tick(env.metadata["render_fps"])

            if terminated or truncated:
                if not info.get("success", False):
                    for _ in range(CRASH_HOLD_FRAMES):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return False
                        frame = env.render()
                        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                        screen.blit(surface, (0, 0))
                        draw_overlay(
                            screen,
                            pygame,
                            font,
                            run=run,
                            step=step,
                            action=action,
                            return_so_far=discounted_return(rewards, gamma),
                            counts=counts,
                            total_fires=fires_after,
                            max_fires=env.max_jet_fires,
                            info=info,
                        )
                        pygame.display.flip()
                        clock.tick(env.metadata["render_fps"])
                ended = True
                break

        if not ended and rewards:
            rewards[-1] = float(failure_reward)
            info = {
                **info,
                "success": False,
                "failure_reason": "timeout",
            }

        episode_return = discounted_return(rewards, gamma)
        print(
            f"animation_run={run} "
            f"return={episode_return:.2f} "
            f"undiscounted={sum(rewards):.2f} "
            f"steps={step} "
            f"upper_left={counts[1]} "
            f"bottom={counts[2]} "
            f"upper_right={counts[3]} "
            f"total_fires={sum(counts.values())} "
            f"info={info}"
        )
        time.sleep(0.75)
        return True
    finally:
        env.close()
        pygame.display.quit()
        pygame.quit()


def main() -> None:
    args = parse_args()
    model_file = args.model_file
    if model_file is None:
        best_model_path = resolve_project_path(DEFAULT_BEST_MODEL_FILE)
        latest_model_path = resolve_project_path(DEFAULT_MODEL_FILE)
        use_best = best_model_path.exists() and (
            not latest_model_path.exists()
            or best_model_path.stat().st_mtime >= latest_model_path.stat().st_mtime
        )
        model_file = DEFAULT_BEST_MODEL_FILE if use_best else DEFAULT_MODEL_FILE

    model_path = resolve_project_path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {model_path}. Train it first with:\n"
            f"{TRAIN_COMMAND}"
        )

    policy, checkpoint = load_policy(model_file)
    seed = int(args.seed if args.seed is not None else checkpoint_arg(checkpoint, "seed", DEFAULT_SEED))
    gamma = float(args.gamma if args.gamma is not None else checkpoint_arg(checkpoint, "gamma", DEFAULT_GAMMA))
    max_steps = int(
        args.max_steps if args.max_steps is not None else checkpoint_arg(checkpoint, "max_steps", DEFAULT_MAX_STEPS)
    )
    wind = bool(args.wind if args.wind is not None else checkpoint_arg(checkpoint, "wind", False))
    wind_power = float(
        args.wind_power if args.wind_power is not None else checkpoint_arg(checkpoint, "wind_power", DEFAULT_WIND_POWER)
    )
    success_reward = float(
        args.success_reward
        if args.success_reward is not None
        else checkpoint_arg(checkpoint, "success_reward", DEFAULT_SUCCESS_REWARD)
    )
    failure_reward = float(
        args.failure_reward
        if args.failure_reward is not None
        else checkpoint_arg(checkpoint, "failure_reward", DEFAULT_FAILURE_REWARD)
    )
    shaping_factor = float(
        args.shaping_factor
        if args.shaping_factor is not None
        else checkpoint_arg(checkpoint, "shaping_factor", DEFAULT_SHAPING_FACTOR)
    )
    max_jet_fires = int(
        args.max_jet_fires
        if args.max_jet_fires is not None
        else checkpoint_arg(checkpoint, "max_jet_fires", MAX_JET_FIRES)
    )

    print(f"loaded_model={model_path}")
    print(
        f"animation_settings seed={seed} gamma={gamma} max_steps={max_steps} "
        f"wind={wind} wind_power={wind_power} max_jet_fires={max_jet_fires} success_reward={success_reward} "
        f"failure_reward={failure_reward} shaping_factor={shaping_factor}"
    )

    rng = np.random.default_rng(seed + 10_000)
    for run in range(1, args.runs + 1):
        if not show_rollout(
            policy=policy,
            run=run,
            rng=rng,
            gamma=gamma,
            max_steps=max_steps,
            wind=wind,
            wind_power=wind_power,
            max_jet_fires=max_jet_fires,
            success_reward=success_reward,
            failure_reward=failure_reward,
            shaping_factor=shaping_factor,
            sample_actions=args.sample_actions,
        ):
            break


if __name__ == "__main__":
    main()
