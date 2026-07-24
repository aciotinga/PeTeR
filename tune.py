"""Bayesian TPE hyperparameter search over lr and ratio (per dataset, per K).

Maximizes ``final_adv_test_ll``. Search space (log-uniform)::

    lr    in [LR_LOW, LR_HIGH]
    ratio in [RATIO_LOW, RATIO_HIGH]

Trials for every ``(dataset, k)`` are interleaved into one shared job queue so
``-j`` workers stay busy across all studies at once (not one dataset at a time).

Metadata-only artifacts (no circuits / plots)::

    sweeps/tpe/k<k>/<dataset>/
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

from peter import DEFAULT_ITERS, VALID_K, run
from sweep import discover_datasets
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

# Sentinel: one per worker so they exit cleanly (unused with ProcessPoolExecutor map).
_FINISHED_STATES = (TrialState.COMPLETE, TrialState.FAIL)


def study_name(dataset: str, k: int) -> str:
    return f"peter_{dataset}_k{k}"


def make_storage(dataset: str, k: int) -> JournalStorage:
    journal = tpe_journal_path(dataset, k)
    journal.parent.mkdir(parents=True, exist_ok=True)
    file_path = str(journal)
    # Symlink locks need elevated privileges on Windows; use open-file locks there.
    lock_obj = JournalFileOpenLock(file_path) if sys.platform == "win32" else None
    return JournalStorage(JournalFileBackend(file_path, lock_obj=lock_obj))


def make_sampler(seed: int | None) -> TPESampler:
    return TPESampler(
        multivariate=True,
        constant_liar=True,
        n_startup_trials=N_STARTUP_TRIALS,
        seed=seed,
    )


def create_or_load_study(
    dataset: str,
    k: int,
    *,
    seed: int | None = None,
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name(dataset, k),
        storage=make_storage(dataset, k),
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
    dataset: str,
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
            "recreate_cmd": (
                f"python peter.py {dataset} --k {k} --lr {bt.params['lr']:g} "
                f"--ratio {bt.params['ratio']:g} --iters {iters}"
            ),
        }
    return {
        "dataset": dataset,
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
        "direction": "maximize",
        "n_complete": len(complete),
        "n_failed": len(failed),
        "n_finished": len(complete) + len(failed),
        "best": best,
        "trials": [trial_record(t) for t in trials],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_pairs(
    ks: list[int],
    datasets: list[str] | None,
) -> list[tuple[str, int]]:
    """Return ``(dataset, k)`` targets in stable order."""
    pairs: list[tuple[str, int]] = []
    for k in ks:
        available = discover_datasets(k)
        if not available:
            print(f"warning: no runnable datasets for k={k}", flush=True)
            continue
        if datasets:
            missing = [d for d in datasets if d not in available]
            if missing:
                raise SystemExit(
                    f"Dataset(s) not runnable for k={k}: {', '.join(missing)}. "
                    f"Available: {', '.join(available)}"
                )
            selected = datasets
        else:
            selected = available
        for dataset in selected:
            pairs.append((dataset, k))
    if not pairs:
        raise SystemExit("No runnable (dataset, k) pairs found.")
    return pairs


def interleave_jobs(
    pairs: list[tuple[str, int]],
    *,
    iters: int,
    n_trials: int,
    seed: int | None,
) -> list[tuple[str, int, int]]:
    """Round-robin remaining trials so every (dataset, k) advances evenly."""
    remaining: list[tuple[str, int, int]] = []
    for dataset, k in pairs:
        tpe_dataset_dir(dataset, k).mkdir(parents=True, exist_ok=True)
        study = create_or_load_study(dataset, k, seed=seed)
        rem = max(0, n_trials - finished_trial_count(study))
        remaining.append((dataset, k, rem))

    jobs: list[tuple[str, int, int]] = []
    while True:
        progressed = False
        next_remaining: list[tuple[str, int, int]] = []
        for dataset, k, rem in remaining:
            if rem > 0:
                jobs.append((dataset, k, iters))
                next_remaining.append((dataset, k, rem - 1))
                progressed = True
            else:
                next_remaining.append((dataset, k, 0))
        remaining = next_remaining
        if not progressed:
            break
    return jobs


def _run_one_trial(args: tuple[str, int, int]) -> dict[str, Any]:
    """Worker: ask one trial from the (dataset, k) study, run peter, tell result."""
    dataset, k, iters = args
    _silence_worker_noise()

    study = create_or_load_study(dataset, k)
    trial = study.ask()
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    ratio = trial.suggest_float("ratio", RATIO_LOW, RATIO_HIGH, log=True)

    print(
        f"start  {dataset}  k={k}  trial={trial.number}  "
        f"lr={lr:g}  ratio={ratio:g}",
        flush=True,
    )

    outcome = run(
        dataset=dataset,
        k=k,
        lr=lr,
        ratio=ratio,
        iters=iters,
        save_circuit=False,
        save_plot=False,
        quiet=True,
        out_dir=tpe_trial_dir(dataset, k, trial.number),
    )

    if outcome.status != "ok" or outcome.metrics is None:
        print(
            f"done   {dataset}  k={k}  trial={trial.number}  "
            f"lr={lr:g}  ratio={ratio:g}  failed",
            flush=True,
        )
        # Never raise: a raised exception kills the pool worker under ProcessPoolExecutor.
        study.tell(trial, state=TrialState.FAIL)
        return {
            "dataset": dataset,
            "k": k,
            "trial": trial.number,
            "status": "failed",
        }

    adv = float(outcome.metrics.final_adv_test_ll)
    print(
        f"done   {dataset}  k={k}  trial={trial.number}  "
        f"lr={lr:g}  ratio={ratio:g}  ok  final_adv_ll={adv:.6f}",
        flush=True,
    )
    study.tell(trial, adv)
    return {
        "dataset": dataset,
        "k": k,
        "trial": trial.number,
        "status": "ok",
        "final_adv_test_ll": adv,
    }


def write_all_summaries(
    pairs: list[tuple[str, int]],
    *,
    iters: int,
    n_trials: int,
    jobs: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset, k in pairs:
        study = create_or_load_study(dataset, k)
        summary = build_study_summary(
            study,
            dataset=dataset,
            k=k,
            iters=iters,
            n_trials=n_trials,
            jobs=jobs,
        )
        write_summary(tpe_study_summary_path(dataset, k), summary)
        summaries.append(summary)
    return summaries


def print_best_table(summaries: list[dict[str, Any]]) -> None:
    print("best:", flush=True)
    for summary in summaries:
        best = summary.get("best")
        ds = summary["dataset"]
        k = summary["k"]
        if best is None:
            print(f"  {ds}  k={k}  (no successful trials)", flush=True)
            continue
        print(
            f"  {ds}  k={k}  lr={best['lr']:g}  ratio={best['ratio']:g}  "
            f"final_adv_ll={best['final_adv_test_ll']:.6f}",
            flush=True,
        )


def run_tune(
    ks: list[int],
    iters: int,
    n_trials: int,
    jobs: int,
    *,
    datasets: list[str] | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    _silence_worker_noise()
    pairs = collect_pairs(ks, datasets)
    job_list = interleave_jobs(pairs, iters=iters, n_trials=n_trials, seed=seed)

    print(
        f"tune: ks={ks}  iters={iters}  n_trials={n_trials}/study  jobs={jobs}  "
        f"studies={len(pairs)}  queued_trials={len(job_list)}",
        flush=True,
    )

    if dry_run:
        from collections import Counter

        counts = Counter((d, k) for d, k, _ in job_list)
        for (dataset, k), n in sorted(counts.items()):
            print(f"dry run  {dataset}  k={k}  remaining={n}", flush=True)
        if not counts:
            print("dry run  nothing remaining (all studies at n_trials)", flush=True)
        return [{"dataset": d, "k": k, "status": "dry_run"} for d, k in pairs]

    if not job_list:
        print("nothing to do (all studies already have n_trials finished)", flush=True)
        summaries = write_all_summaries(
            pairs, iters=iters, n_trials=n_trials, jobs=jobs
        )
        print_best_table(summaries)
        return summaries

    if jobs == 1:
        for job in job_list:
            _run_one_trial(job)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_run_one_trial, job) for job in job_list]
            for future in as_completed(futures):
                # Surface unexpected worker crashes; trial FAILs are returned as dicts.
                future.result()

    summaries = write_all_summaries(pairs, iters=iters, n_trials=n_trials, jobs=jobs)
    print_best_table(summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna TPE search over lr and ratio to maximize final_adv_test_ll. "
            "Interleaves trials across all (dataset, k) into one worker queue. "
            "Writes metadata only under sweeps/tpe/ (no circuits)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"CW-ball radius and adversarial dataset K (default: all of {list(VALID_K)})",
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
        help=f"Finished trials (COMPLETE+FAIL) per (dataset, k) (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Tune only this dataset (repeatable). Default: all runnable datasets",
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
    run_tune(
        ks=ks,
        iters=args.iters,
        n_trials=args.n_trials,
        jobs=args.jobs,
        datasets=args.datasets,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
