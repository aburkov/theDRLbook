"""Resumable Optuna search for gamma-dropped reward-to-go REINFORCE.

The objective maximizes the best ``success50`` value reached by
``gamma_dropped_rtg_reinforce.py`` at or before episode 50,000.

Examples:
    python -m pip install optuna
    python search_gamma_dropped_rtg_hyperparams.py --trials 64 --workers 4

Run the same command again to continue the same study. Completed trials are
loaded from the journal file and are not repeated.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "hparam_search" / "gamma_dropped_rtg"
DEFAULT_SCRIPT = PROJECT_ROOT / "gamma_dropped_rtg_reinforce.py"
SUCCESS_RE = re.compile(r"episode=\s*(?P<episode>\d+).*success50=\s*(?P<success>\d+)")


@dataclass(frozen=True)
class SearchConfig:
    study_name: str
    storage_file: Path
    output_dir: Path
    script: Path
    python: str
    max_episodes: int
    max_steps_choices: tuple[int, ...]
    eval_episodes: int
    print_every: int
    target_window: int
    seeds: tuple[int, ...]
    seed_aggregation: str
    prune: bool
    startup_trials: int
    prune_warmup_episodes: int
    torch_threads_per_trial: int
    verbose: bool


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(
        description=(
            "Search hyperparameters for gamma_dropped_rtg_reinforce.py. "
            "The study is persistent, so rerunning this command continues it."
        )
    )
    parser.add_argument("--trials", type=int, default=64, help="Number of new trials to run in this invocation.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count - 1, 4)),
        help="Parallel worker processes. Defaults to up to 4 CPU workers.",
    )
    parser.add_argument("--study-name", default="gamma_dropped_rtg_success50")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Optuna journal, per-trial logs, CSVs, models, and summaries.",
    )
    parser.add_argument("--storage-file", type=Path, default=None, help="Optuna journal file. Defaults under output-dir.")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-episodes", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=50_000)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--target-window", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--seed-aggregation",
        choices=("mean", "min", "max"),
        default="mean",
        help="How to combine multiple seeds inside one trial.",
    )
    parser.add_argument(
        "--max-steps-choices",
        type=int,
        nargs="+",
        default=[300, 400, 600],
        help="Candidate max-steps values.",
    )
    parser.add_argument("--sampler-seed", type=int, default=20260604)
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Optuna median pruning based on intermediate success50 logs.",
    )
    parser.add_argument("--startup-trials", type=int, default=12, help="Completed trials before pruning starts.")
    parser.add_argument("--prune-warmup-episodes", type=int, default=10_000)
    parser.add_argument(
        "--torch-threads-per-trial",
        type=int,
        default=1,
        help="Thread caps passed to each training subprocess to avoid CPU oversubscription.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every subprocess log line.")
    parser.add_argument(
        "--enqueue-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try the known baseline gamma=0.99, learning_rate=1e-6, hidden_dim=64, max_steps=400 first.",
    )
    return parser.parse_args()


def require_optuna():
    try:
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required for this search. Install it with:\n\n"
            "    python -m pip install optuna\n"
        ) from exc
    return optuna, JournalStorage, JournalFileBackend


def create_storage(storage_file: Path):
    _optuna, JournalStorage, JournalFileBackend = require_optuna()
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(storage_file)))


def create_study(config: SearchConfig, sampler_seed: int, startup_trials: int):
    optuna, _JournalStorage, _JournalFileBackend = require_optuna()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            multivariate=True,
            constant_liar=True,
        )
    if config.prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=startup_trials,
            n_warmup_steps=config.prune_warmup_episodes,
            interval_steps=config.print_every,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(
        study_name=config.study_name,
        storage=create_storage(config.storage_file),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def trial_params_already_exist(study, params: dict[str, object]) -> bool:
    for trial in study.trials:
        if trial.params == params:
            return True
    return False


def enqueue_default_trial(study, config: SearchConfig) -> None:
    params = {
        "gamma": 0.99,
        "learning_rate": 1e-6,
        "hidden_dim": 64,
        "max_steps": 400 if 400 in config.max_steps_choices else config.max_steps_choices[0],
    }
    if not trial_params_already_exist(study, params):
        study.enqueue_trial(params)


def suggest_params(trial, config: SearchConfig) -> dict[str, object]:
    return {
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 3e-5, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "max_steps": trial.suggest_categorical("max_steps", list(config.max_steps_choices)),
    }


def trial_dir(config: SearchConfig, trial_number: int, seed: int) -> Path:
    return config.output_dir / f"trial_{trial_number:05d}" / f"seed_{seed}"


def subprocess_env(config: SearchConfig) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, config.torch_threads_per_trial))
    env.setdefault("OMP_NUM_THREADS", threads)
    env.setdefault("MKL_NUM_THREADS", threads)
    env.setdefault("OPENBLAS_NUM_THREADS", threads)
    env.setdefault("VECLIB_MAXIMUM_THREADS", threads)
    env.setdefault("NUMEXPR_NUM_THREADS", threads)
    return env


def command_for_run(
    config: SearchConfig,
    params: dict[str, object],
    *,
    seed: int,
    log_file: Path,
    csv_file: Path,
    model_file: Path,
) -> list[str]:
    return [
        config.python,
        str(config.script),
        "--episodes",
        str(config.max_episodes),
        "--max-steps",
        str(params["max_steps"]),
        "--gamma",
        f"{float(params['gamma']):.10g}",
        "--learning-rate",
        f"{float(params['learning_rate']):.10g}",
        "--hidden-dim",
        str(params["hidden_dim"]),
        "--seed",
        str(seed),
        "--target-window",
        str(config.target_window),
        "--print-every",
        str(config.print_every),
        "--no-animation",
        "--log-file",
        str(log_file),
        "--csv-file",
        str(csv_file),
        "--model-file",
        str(model_file),
    ]


def best_success_from_csv(csv_file: Path, eval_episodes: int) -> int:
    best = 0
    if not csv_file.exists():
        return best
    with csv_file.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            episode = int(row["episode"])
            if episode > eval_episodes:
                break
            best = max(best, int(row["success_count"]))
    return best


def aggregate(values: Sequence[int], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    if mode == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unknown seed aggregation mode: {mode}")


def run_seed(trial, config: SearchConfig, params: dict[str, object], seed: int) -> int:
    optuna, _JournalStorage, _JournalFileBackend = require_optuna()
    run_dir = trial_dir(config, trial.number, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    stdout_file = run_dir / "stdout.log"
    csv_file = run_dir / "train.csv"
    model_file = run_dir / "policy.pt"
    command_file = run_dir / "command.txt"
    command = command_for_run(
        config,
        params,
        seed=seed,
        log_file=log_file,
        csv_file=csv_file,
        model_file=model_file,
    )
    command_file.write_text(" ".join(command) + "\n", encoding="utf-8")

    best_seen = 0
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=subprocess_env(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    try:
        with stdout_file.open("w", encoding="utf-8") as stdout_log:
            for line in process.stdout:
                stdout_log.write(line)
                stdout_log.flush()
                match = SUCCESS_RE.search(line)
                if match:
                    episode = int(match.group("episode"))
                    success = int(match.group("success"))
                    if episode <= config.eval_episodes:
                        best_seen = max(best_seen, success)
                        trial.report(best_seen, step=episode)
                        if config.verbose:
                            print(f"[trial {trial.number} seed {seed}] {line}", end="")
                        elif episode == 1 or episode % max(config.print_every * 10, config.print_every) == 0:
                            print(
                                f"[trial {trial.number} seed {seed}] "
                                f"episode={episode} best_success50={best_seen}"
                            )
                        if trial.should_prune():
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            trial.set_user_attr(f"seed_{seed}_best_success50_partial", best_seen)
                            raise optuna.TrialPruned(
                                f"Pruned at episode {episode}; best success50 so far was {best_seen}."
                            )
                elif config.verbose:
                    print(f"[trial {trial.number} seed {seed}] {line}", end="")
    finally:
        if process.poll() is None:
            process.terminate()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {return_code}: {' '.join(command)}")

    best = best_success_from_csv(csv_file, config.eval_episodes)
    trial.set_user_attr(f"seed_{seed}_best_success50", best)
    trial.set_user_attr(f"seed_{seed}_csv", str(csv_file))
    trial.set_user_attr(f"seed_{seed}_model", str(model_file))
    return best


def objective_factory(config: SearchConfig):
    def objective(trial) -> float:
        params = suggest_params(trial, config)
        trial.set_user_attr("params", params)
        seed_scores = []
        for seed in config.seeds:
            seed_scores.append(run_seed(trial, config, params, seed))
        value = aggregate(seed_scores, config.seed_aggregation)
        trial.set_user_attr("seed_scores", seed_scores)
        trial.set_user_attr("best_seed_score", max(seed_scores) if seed_scores else 0)
        trial.set_user_attr("criterion", f"{config.seed_aggregation}_best_success50_before_{config.eval_episodes}")
        print(f"[trial {trial.number}] value={value:.3f} seed_scores={seed_scores} params={params}")
        return value

    return objective


def split_trials(total_trials: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total_trials)) if total_trials > 0 else 1
    base = total_trials // workers
    remainder = total_trials % workers
    return [base + (1 if worker < remainder else 0) for worker in range(workers)]


def worker_main(payload: tuple[SearchConfig, int, int, int]) -> None:
    config, worker_id, trials_for_worker, sampler_seed = payload
    if trials_for_worker <= 0:
        return
    study = create_study(config, sampler_seed + worker_id, startup_trials=config.startup_trials)
    study.optimize(
        objective_factory(config),
        n_trials=trials_for_worker,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )


def print_best_summary(study) -> None:
    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials yet.")
        return
    print("\nBest completed trial")
    print(f"  number: {best.number}")
    print(f"  value:  {best.value}")
    print(f"  params: {best.params}")
    print(f"  attrs:  {best.user_attrs}")


def main() -> None:
    args = parse_args()
    optuna, _JournalStorage, _JournalFileBackend = require_optuna()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    output_dir = args.output_dir.resolve()
    storage_file = (args.storage_file or output_dir / "optuna_journal.log").resolve()
    config = SearchConfig(
        study_name=args.study_name,
        storage_file=storage_file,
        output_dir=output_dir,
        script=args.script.resolve(),
        python=args.python,
        max_episodes=args.max_episodes,
        max_steps_choices=tuple(args.max_steps_choices),
        eval_episodes=args.eval_episodes,
        print_every=args.print_every,
        target_window=args.target_window,
        seeds=tuple(args.seeds),
        seed_aggregation=args.seed_aggregation,
        prune=args.prune,
        startup_trials=args.startup_trials,
        prune_warmup_episodes=args.prune_warmup_episodes,
        torch_threads_per_trial=args.torch_threads_per_trial,
        verbose=args.verbose,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    study = create_study(config, args.sampler_seed, startup_trials=args.startup_trials)
    if args.enqueue_default:
        enqueue_default_trial(study, config)

    print(
        f"study={config.study_name} storage={config.storage_file} "
        f"existing_trials={len(study.trials)} new_trials={args.trials} workers={args.workers}"
    )
    print(
        "criterion=maximize "
        f"{config.seed_aggregation} best success50 at or before episode {config.eval_episodes}"
    )

    if args.trials > 0:
        per_worker = split_trials(args.trials, args.workers)
        payloads = [
            (config, worker_id, trials_for_worker, args.sampler_seed)
            for worker_id, trials_for_worker in enumerate(per_worker)
            if trials_for_worker > 0
        ]
        if len(payloads) == 1:
            worker_main(payloads[0])
        else:
            with mp.get_context("spawn").Pool(processes=len(payloads)) as pool:
                pool.map(worker_main, payloads)

    study = create_study(config, args.sampler_seed, startup_trials=args.startup_trials)
    print_best_summary(study)


if __name__ == "__main__":
    main()
