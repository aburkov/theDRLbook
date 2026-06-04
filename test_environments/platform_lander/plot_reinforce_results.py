"""Plot REINFORCE training results from the saved per-episode CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"

STYLE_ROOT = PROJECT_ROOT.parent
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))

try:
    from plot_style import configure_cambria_math, image_path
except ImportError:

    def configure_cambria_math(font_size: int) -> None:
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size - 1,
                "legend.fontsize": font_size - 3,
                "xtick.labelsize": font_size - 3,
                "ytick.labelsize": font_size - 3,
            }
        )

    def image_path(name: str) -> Path:
        return RUNS_DIR / "plots" / name


VARIANTS = {
    "vanilla_reinforce": "Vanilla REINFORCE",
    "batch_reinforce": "Batch REINFORCE",
    "rtg_reinforce": "Reward-to-Go REINFORCE",
    "average_reinforcement_baseline_reinforce": "Average-Reinforcement Baseline REINFORCE",
    "value_function_baseline_reinforce": "Value-Function Baseline REINFORCE",
    "full_reinforce": "Full REINFORCE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot REINFORCE results from CSV files.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_training_csv(csv_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes: list[int] = []
    average_returns: list[float] = []
    successes: list[bool] = []

    with csv_file.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"episode", "average_return", "success"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_file} is missing required column(s): {missing}")

        for row in reader:
            episodes.append(int(row["episode"]))
            average_returns.append(float(row["average_return"]))
            successes.append(parse_bool(row["success"]))

    if not episodes:
        raise ValueError(f"{csv_file} has no episode rows")

    return (
        np.asarray(episodes, dtype=int),
        np.asarray(average_returns, dtype=float),
        np.cumsum(np.asarray(successes, dtype=int)),
    )


def save_figure(fig: plt.Figure, base_path: Path, formats: list[str]) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for file_format in formats:
        output_path = base_path.with_suffix(f".{file_format}")
        fig.savefig(output_path, format=file_format, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def plot_variant(
    variant_name: str,
    display_name: str,
    csv_file: Path,
    output_dir: Path | None,
    formats: list[str],
    *,
    show: bool,
) -> list[Path]:
    episodes, average_returns, cumulative_successes = load_training_csv(csv_file)

    colors = {
        "blue": "#2B6CB0",
        "green": "#38A169",
        "gray": "#4A5568",
    }
    font_size = 16
    configure_cambria_math(font_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(display_name, fontsize=font_size + 2)

    axes[0].plot(episodes, average_returns, color=colors["blue"], linewidth=2.5)
    axes[0].set_title("Average Total Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average return")
    axes[0].grid(True, linestyle=":", alpha=0.7)

    axes[1].plot(episodes, cumulative_successes, color=colors["green"], linewidth=2.5)
    axes[1].set_title("Successful Landings")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total successes")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, linestyle=":", alpha=0.7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=colors["gray"])

    plt.tight_layout()

    if output_dir is None:
        output_base = image_path(f"{variant_name}_training_results")
    else:
        output_base = output_dir / f"{variant_name}_training_results"
    saved_paths = save_figure(fig, output_base, formats)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths


def main() -> None:
    args = parse_args()
    runs_dir = resolve_project_path(args.runs_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else None

    for variant_name, display_name in VARIANTS.items():
        csv_file = runs_dir / f"{variant_name}.csv"
        if not csv_file.exists():
            print(f"skipping missing_csv={csv_file}")
            continue

        saved_paths = plot_variant(
            variant_name,
            display_name,
            csv_file,
            output_dir,
            args.formats,
            show=args.show,
        )
        print(f"plotted {variant_name}: {', '.join(str(path) for path in saved_paths)}")


if __name__ == "__main__":
    main()
