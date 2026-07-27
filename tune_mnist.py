"""Bayesian TPE hyperparameter search for MNIST PeTeR (lr and ratio).

Mirrors :mod:`tune` but only for MNIST. Maximizes mean log-likelihood on
``corrupted_datasets/mnist/sigma0.010/r0.data`` (stored as ``final_adv_test_ll``).

Search space (log-uniform)::

    lr    in [LR_LOW, LR_HIGH]
    ratio in [RATIO_LOW, RATIO_HIGH]

One Optuna study per CW-ball ``k`` in ``{1, 3, 5}``. Trials are interleaved
across Ks into one ``ProcessPoolExecutor`` job queue (``-j`` workers).

Metadata-only artifacts (no circuits / plots)::

    sweeps/tpe/k<k>/mnist/
        journal.log
        study_summary.json
        trials/trial_<n>/{config,metrics|error}.json
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from peter import DEFAULT_ITERS, VALID_K
from peter_mnist import DATASET, require_mnist_inputs, run
from prepare_mnist_data import TUNE_SIGMA, format_sigma
from sweep_io import (
    tpe_dataset_dir,
    tpe_journal_path,
    tpe_study_summary_path,
    tpe_trial_dir,
    write_summary,
)

LR_LOW = 1e-8
LR_HIGH = 0.5
RATIO_LOW = 0.1
RATIO_HIGH = 100.0
DEFAULT_N_TRIALS = 50
N_STARTUP_TRIALS = 10
_TUNE_CORRUPT = f"corrupted_datasets/mnist/sigma{format_sigma(TUNE_SIGMA)}/r0.data"
_TUNE_LABEL = f"sigma{format_sigma(TUNE_SIGMA)}/r0"

_FINISHED_STATES = (TrialState.COMPLETE, TrialState.FAIL)


def study_name(k: int) -> str:
    return f"peter_{DATASET}_k{k}"


def make_storage(k: int) -> JournalStorage:
    journal = tpe_journal_path(DATASET, k)
    journal.parent.mkdir(parents=True, exist_ok=True)
    file_path = str(journal)
    lock_obj = JournalFileOpenLock(file_path) if sys.platform == "win32" else None
    return JournalStorage(JournalFileBackend(file_path, lock_obj=lock_obj))


def make_sampler(seed: int | None) -> TPESampler:
    return TPESampler(
        multivariate=True,
        constant_liar=True,
        n_startup_trials=N_STARTUP_TRIALS,
        seed=seed,
    )


def create_or_load_study(k: int, *, seed: int | None = None) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name(k),
        storage=make_storage(k),
        direction="maximize",
        sampler=make_sampler(seed),
        load_if_exists=True,
    )


def _silence_worker_noise() -> None:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def finished_trial_count(study: optuna.Study) -> int:
    return sum(1 for t in study.get_trials(deepcopy=False) if t.state in _FINISHED_STATES)


def trial_record(trial: optuna.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "lr": trial.params.get("lr"),
        "ratio": trial.params.get("ratio"),
        "value": trial.value,
    }


def build_study_summary(
    study: optuna.Study,
    *,
    k: int,
    iters: int,
    n_trials: int,
    jobs: int,
) -> dict[str, Any]:
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    failed = [t for t in trials if t.state == TrialState.FAIL]
    best: dict[str, Any] | None = None
    if complete:
        bt = study.best_trial
        best = {
            "trial_number": bt.number,
            "lr": bt.params["lr"],
            "ratio": bt.params["ratio"],
            "final_adv_test_ll": bt.value,
            "objective_note": f"mean LL on {_TUNE_CORRUPT}",
            "recreate_cmd": (
                f"python peter_mnist.py --k {k} --lr {bt.params['lr']:g} "
                f"--ratio {bt.params['ratio']:g} --iters {iters}"
            ),
        }
    return {
        "dataset": DATASET,
        "k": k,
        "iters": iters,
        "n_trials_target": n_trials,
        "jobs": jobs,
        "study_name": study.study_name,
        "search_space": {
            "lr": {"low": LR_LOW, "high": LR_HIGH, "log": True},
            "ratio": {"low": RATIO_LOW, "high": RATIO_HIGH, "log": True},
        },
        "objective": "final_adv_test_ll",
        "objective_data": _TUNE_CORRUPT,
        "direction": "maximize",
        "n_complete": len(complete),
        "n_failed": len(failed),
        "n_finished": len(complete) + len(failed),
        "best": best,
        "trials": [trial_record(t) for t in trials],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def interleave_jobs(
    ks: list[int],
    *,
    iters: int,
    n_trials: int,
    seed: int | None,
) -> list[tuple[int, int]]:
    """Round-robin remaining trials so every k advances evenly.

    Returns a list of ``(k, iters)`` worker jobs.
    """
    remaining: list[tuple[int, int]] = []
    for k in ks:
        tpe_dataset_dir(DATASET, k).mkdir(parents=True, exist_ok=True)
        study = create_or_load_study(k, seed=seed)
        rem = max(0, n_trials - finished_trial_count(study))
        remaining.append((k, rem))

    jobs: list[tuple[int, int]] = []
    while True:
        progressed = False
        next_remaining: list[tuple[int, int]] = []
        for k, rem in remaining:
            if rem > 0:
                jobs.append((k, iters))
                next_remaining.append((k, rem - 1))
                progressed = True
            else:
                next_remaining.append((k, 0))
        remaining = next_remaining
        if not progressed:
            break
    return jobs


def _run_one_trial(args: tuple[int, int]) -> dict[str, Any]:
    """Worker: ask one trial from the k study, run peter_mnist, tell result."""
    k, iters = args
    _silence_worker_noise()

    study = create_or_load_study(k)
    trial = study.ask()
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    ratio = trial.suggest_float("ratio", RATIO_LOW, RATIO_HIGH, log=True)

    print(
        f"start  {DATASET}  k={k}  trial={trial.number}  "
        f"lr={lr:g}  ratio={ratio:g}",
        flush=True,
    )

    outcome = run(
        k=k,
        lr=lr,
        ratio=ratio,
        iters=iters,
        save_circuit=False,
        save_plot=False,
        quiet=True,
        out_dir=tpe_trial_dir(DATASET, k, trial.number),
    )

    if outcome.status != "ok" or outcome.metrics is None:
        print(
            f"done   {DATASET}  k={k}  trial={trial.number}  "
            f"lr={lr:g}  ratio={ratio:g}  failed",
            flush=True,
        )
        study.tell(trial, state=TrialState.FAIL)
        return {
            "dataset": DATASET,
            "k": k,
            "trial": trial.number,
            "status": "failed",
        }

    corrupt_ll = float(outcome.metrics.final_adv_test_ll)
    print(
        f"done   {DATASET}  k={k}  trial={trial.number}  "
        f"lr={lr:g}  ratio={ratio:g}  ok  final_corrupt_ll={corrupt_ll:.6f}",
        flush=True,
    )
    study.tell(trial, corrupt_ll)
    return {
        "dataset": DATASET,
        "k": k,
        "trial": trial.number,
        "status": "ok",
        "final_adv_test_ll": corrupt_ll,
    }


def write_all_summaries(
    ks: list[int],
    *,
    iters: int,
    n_trials: int,
    jobs: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for k in ks:
        study = create_or_load_study(k)
        summary = build_study_summary(
            study,
            k=k,
            iters=iters,
            n_trials=n_trials,
            jobs=jobs,
        )
        write_summary(tpe_study_summary_path(DATASET, k), summary)
        summaries.append(summary)
    return summaries


def print_best_table(summaries: list[dict[str, Any]]) -> None:
    print("best:", flush=True)
    for summary in summaries:
        best = summary.get("best")
        k = summary["k"]
        if best is None:
            print(f"  {DATASET}  k={k}  (no successful trials)", flush=True)
            continue
        print(
            f"  {DATASET}  k={k}  lr={best['lr']:g}  ratio={best['ratio']:g}  "
            f"final_corrupt_ll={best['final_adv_test_ll']:.6f}",
            flush=True,
        )


def run_tune(
    ks: list[int],
    iters: int,
    n_trials: int,
    jobs: int,
    *,
    seed: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    _silence_worker_noise()
    require_mnist_inputs()
    job_list = interleave_jobs(ks, iters=iters, n_trials=n_trials, seed=seed)

    print(
        f"tune_mnist: ks={ks}  iters={iters}  n_trials={n_trials}/study  jobs={jobs}  "
        f"studies={len(ks)}  queued_trials={len(job_list)}  "
        f"objective={_TUNE_LABEL} mean LL",
        flush=True,
    )

    if dry_run:
        from collections import Counter

        counts = Counter(k for k, _ in job_list)
        for k, n in sorted(counts.items()):
            print(f"dry run  {DATASET}  k={k}  remaining={n}", flush=True)
        if not counts:
            print("dry run  nothing remaining (all studies at n_trials)", flush=True)
        return [{"dataset": DATASET, "k": k, "status": "dry_run"} for k in ks]

    if not job_list:
        print("nothing to do (all studies already have n_trials finished)", flush=True)
        summaries = write_all_summaries(ks, iters=iters, n_trials=n_trials, jobs=jobs)
        print_best_table(summaries)
        return summaries

    if jobs == 1:
        for job in job_list:
            _run_one_trial(job)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_run_one_trial, job) for job in job_list]
            for future in as_completed(futures):
                future.result()

    summaries = write_all_summaries(ks, iters=iters, n_trials=n_trials, jobs=jobs)
    print_best_table(summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna TPE search over lr and ratio for MNIST PeTeR. "
            f"Maximizes mean LL on corrupted {_TUNE_LABEL}. "
            "Interleaves trials across K into one ProcessPoolExecutor queue. "
            "Writes metadata only under sweeps/tpe/ (no circuits)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"CW-ball radius (default: all of {list(VALID_K)})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"OGDA iterations per trial (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Finished trials (COMPLETE+FAIL) per k (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="TPESampler seed (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned queued trials without training",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        metavar="N",
        help="Parallel worker processes across all studies (default: number of available CPUs)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iters < 1:
        raise SystemExit("--iters must be at least 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.n_trials < 1:
        raise SystemExit("--n-trials must be at least 1")

    ks = [args.k] if args.k is not None else list(VALID_K)
    try:
        require_mnist_inputs()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    run_tune(
        ks=ks,
        iters=args.iters,
        n_trials=args.n_trials,
        jobs=args.jobs,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
