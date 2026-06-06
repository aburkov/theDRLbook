"""Watch the saved gamma-dropped batch reward-to-go REINFORCE policy."""

from __future__ import annotations

import watch_gamma_dropped_rtg_policy as watcher


watcher.DEFAULT_MODEL_FILE = watcher.RUNS_DIR / "gamma_dropped_batch_rtg_reinforce.pt"
watcher.DEFAULT_BEST_MODEL_FILE = watcher.RUNS_DIR / "gamma_dropped_batch_rtg_reinforce_best.pt"
watcher.DESCRIPTION = "Animate a saved gamma-dropped batch reward-to-go REINFORCE policy."
watcher.TRAIN_COMMAND = (
    "python gamma_dropped_batch_rtg_reinforce.py --wind --wind-power 5.0 "
    "--gamma 0.99 --learning-rate 1e-6 --seed 42"
)
watcher.WINDOW_TITLE = "PlatformLander batch policy with per-engine fire counters"


if __name__ == "__main__":
    watcher.main()
