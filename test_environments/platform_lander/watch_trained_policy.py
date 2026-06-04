"""Load one saved PlatformLander policy and show several animated rollouts.

The checkpoint name is intentionally hardcoded so this can be used as a simple
"watch the trained model" script after running one of the REINFORCE variants.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from vanilla_reinforce import PROJECT_ROOT, RUNS_DIR, animate, load_policy


MODEL_FILE = RUNS_DIR / "full_reinforce.pt"
ANIMATION_RUNS = 5
DEFAULT_MAX_STEPS = 700
DEFAULT_GAMMA = 0.99
DEFAULT_SEED = 0


def main() -> None:
    model_path = Path(MODEL_FILE)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {model_path}. Train it first with "
            "`python full_reinforce.py` or change MODEL_FILE in this script."
        )

    policy, checkpoint = load_policy(MODEL_FILE)
    training_args = checkpoint.get("args", {})
    args = Namespace(
        seed=int(training_args.get("seed", DEFAULT_SEED)),
        gamma=float(training_args.get("gamma", DEFAULT_GAMMA)),
        max_steps=int(training_args.get("max_steps", DEFAULT_MAX_STEPS)),
        wind=bool(training_args.get("wind", False)),
        wind_power=float(training_args.get("wind_power", 5.0)),
        animation_runs=ANIMATION_RUNS,
    )

    print(f"loaded_model={model_path}")
    animate(policy, args)


if __name__ == "__main__":
    main()
