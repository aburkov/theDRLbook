"""Export three saved-policy PlatformLander tests to a 1080p MP4 video."""

from __future__ import annotations

import argparse
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
from watch_gamma_dropped_rtg_policy import ACTION_NAMES, checkpoint_arg, draw_overlay


DEFAULT_RUNS = 3
DEFAULT_MAX_STEPS = 400
DEFAULT_GAMMA = 0.99
DEFAULT_SEED = 42
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 50
HOLD_FRAMES = 40

MODEL_DEFAULTS = {
    "batch": (
        RUNS_DIR / "gamma_dropped_batch_rtg_reinforce_best.pt",
        RUNS_DIR / "gamma_dropped_batch_rtg_reinforce.pt",
        RUNS_DIR / "gamma_dropped_batch_rtg_policy_tests_1080p.mp4",
    ),
    "rtg": (
        RUNS_DIR / "gamma_dropped_rtg_reinforce_best.pt",
        RUNS_DIR / "gamma_dropped_rtg_reinforce.pt",
        RUNS_DIR / "gamma_dropped_rtg_policy_tests_1080p.mp4",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 1080p video of saved PlatformLander policy tests.")
    parser.add_argument("--model-kind", choices=sorted(MODEL_DEFAULTS), default="batch")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the best checkpoint when present, otherwise the latest checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runs", type=positive_int, default=DEFAULT_RUNS)
    parser.add_argument("--max-steps", type=positive_int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--wind", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wind-power", type=float, default=None)
    parser.add_argument("--success-reward", type=float, default=None)
    parser.add_argument("--failure-reward", type=float, default=None)
    parser.add_argument("--shaping-factor", type=nonnegative_float, default=None)
    parser.add_argument("--max-jet-fires", type=positive_int, default=None)
    parser.add_argument("--width", type=positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=positive_int, default=DEFAULT_FPS)
    return parser.parse_args()


def default_model_file(model_kind: str) -> Path:
    best_model_file, latest_model_file, _output_file = MODEL_DEFAULTS[model_kind]
    best_model_path = resolve_project_path(best_model_file)
    if best_model_path.exists():
        return best_model_file
    return latest_model_file


def writer_for(output_path: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - optional export dependency
        raise RuntimeError(
            "imageio and imageio-ffmpeg are required for video export. "
            "Install them with `python -m pip install imageio imageio-ffmpeg`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def render_video_frame(
    *,
    pygame,
    font,
    frame: np.ndarray,
    width: int,
    height: int,
    run: int,
    step: int,
    action: int,
    return_so_far: float,
    counts: dict[int, int],
    total_fires: int,
    max_fires: int,
    info: dict,
) -> np.ndarray:
    source = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    draw_overlay(
        source,
        pygame,
        font,
        run=run,
        step=step,
        action=action,
        return_so_far=return_so_far,
        counts=counts,
        total_fires=total_fires,
        max_fires=max_fires,
        info=info,
    )

    scale = min(width / source.get_width(), height / source.get_height())
    scaled_size = (
        int(round(source.get_width() * scale)),
        int(round(source.get_height() * scale)),
    )
    scaled = pygame.transform.smoothscale(source, scaled_size)
    canvas = pygame.Surface((width, height))
    canvas.fill((10, 14, 20))
    canvas.blit(
        scaled,
        (
            (width - scaled_size[0]) // 2,
            (height - scaled_size[1]) // 2,
        ),
    )
    return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))


def run_test_to_video(
    *,
    writer,
    pygame,
    font,
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
    width: int,
    height: int,
) -> None:
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

    counts = {1: 0, 2: 0, 3: 0}
    rewards: list[float] = []
    info: dict = {}
    action = 0
    step = 0
    ended = False

    try:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        observation = randomize_start(env, rng)

        for step in range(1, max_steps + 1):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                logits = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item() if sample_actions else torch.argmax(logits).item())

            fires_before = int(env.jet_fires_used)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            fires_after = int(info.get("jet_fires_used", env.jet_fires_used))
            if action in counts and fires_after > fires_before:
                counts[action] += fires_after - fires_before

            frame = env.render()
            writer.append_data(
                render_video_frame(
                    pygame=pygame,
                    font=font,
                    frame=frame,
                    width=width,
                    height=height,
                    run=run,
                    step=step,
                    action=action,
                    return_so_far=discounted_return(rewards, gamma),
                    counts=counts,
                    total_fires=fires_after,
                    max_fires=env.max_jet_fires,
                    info=info,
                )
            )

            if terminated or truncated:
                ended = True
                break

        if not ended:
            rewards.append(float(failure_reward))
            info = {
                **info,
                "success": False,
                "failure_reason": "timeout",
            }

        for _ in range(HOLD_FRAMES):
            frame = env.render()
            writer.append_data(
                render_video_frame(
                    pygame=pygame,
                    font=font,
                    frame=frame,
                    width=width,
                    height=height,
                    run=run,
                    step=step,
                    action=action,
                    return_so_far=discounted_return(rewards, gamma),
                    counts=counts,
                    total_fires=int(info.get("jet_fires_used", env.jet_fires_used)),
                    max_fires=env.max_jet_fires,
                    info=info,
                )
            )

        print(
            f"video_run={run} "
            f"return={discounted_return(rewards, gamma):.2f} "
            f"undiscounted={sum(rewards):.2f} "
            f"steps={step} "
            f"upper_left={counts[1]} "
            f"bottom={counts[2]} "
            f"upper_right={counts[3]} "
            f"total_fires={sum(counts.values())} "
            f"success={info.get('success', False)} "
            f"failure={info.get('failure_reason')}",
            flush=True,
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    model_file = args.model_file or default_model_file(args.model_kind)
    model_path = resolve_project_path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {model_path}")

    output_file = args.output or MODEL_DEFAULTS[args.model_kind][2]
    output_path = resolve_project_path(output_file)

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
        f"video_settings output={output_path} runs={args.runs} size={args.width}x{args.height} fps={args.fps} "
        f"seed={seed} gamma={gamma} max_steps={max_steps} wind={wind} wind_power={wind_power} "
        f"max_jet_fires={max_jet_fires} success_reward={success_reward} failure_reward={failure_reward} "
        f"shaping_factor={shaping_factor}",
        flush=True,
    )

    import pygame

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Menlo", 16) or pygame.font.Font(None, 18)
    rng = np.random.default_rng(seed + 10_000)

    writer = writer_for(output_path, args.fps)
    try:
        for run in range(1, args.runs + 1):
            run_test_to_video(
                writer=writer,
                pygame=pygame,
                font=font,
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
                width=args.width,
                height=args.height,
            )
    finally:
        writer.close()
        pygame.quit()

    print(f"saved_video={output_path}")


if __name__ == "__main__":
    main()
