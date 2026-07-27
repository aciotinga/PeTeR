"""Grid search over lr and ratio for MNIST PeTeR.

Mirrors :mod:`sweep` but only for MNIST. Same learning-rate / ratio grid.
Objective metric in ``metrics.json`` is mean LL on sigma0.010/r0
(``final_adv_test_ll``).

Metadata-only artifacts::

    sweeps/grid/k<k>/mnist/lr=<lr>_ratio=<ratio>/{config,metrics|error}.json
    sweeps/grid/k<k>/sweep_summary_mnist.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from peter import DEFAULT_ITERS, VALID_K, RunOutcome
from peter_mnist import DATASET, require_mnist_inputs, run
from prepare_mnist_data import TUNE_SIGMA, format_sigma
from sweep import LEARNING_RATES, RATIOS
from sweep_io import (
    grid_run_dir,
    is_run_finished,
    load_error,
    load_metrics,
)

_ROOT = Path(__file__).resolve().parent
GRID_ROOT = _ROOT / "sweeps" / "grid"
_TUNE_CORRUPT = f"corrupted_datasets/mnist/sigma{format_sigma(TUNE_SIGMA)}/r0.data"
_TUNE_LABEL = f"sigma{format_sigma(TUNE_SIGMA)}/r0"


@dataclass(frozen=True)
class SweepTask:
    k: int
    lr: float
    ratio: float
    iters: int


@dataclass
class SweepRunRecord:
    dataset: str
    k: int
    lr: float
    ratio: float
    status: str
    out_dir: str | None = None
    error: str | None = None
    metrics: dict | None = None


def mnist_grid_summary_path(k: int) -> Path:
    return GRID_ROOT / f"k{k}" / "sweep_summary_mnist.json"


def is_complete(k: int, lr: float, ratio: float) -> bool:
    return is_run_finished(grid_run_dir(DATASET, k, lr, ratio))


def load_run_record(k: int, lr: float, ratio: float) -> SweepRunRecord:
    out_dir = grid_run_dir(DATASET, k, lr, ratio)
    metrics = load_metrics(out_dir)
    if metrics is not None:
        return SweepRunRecord(
            dataset=DATASET,
            k=k,
            lr=lr,
            ratio=ratio,
            status="ok",
            out_dir=str(out_dir),
            metrics=asdict(metrics),
        )
    error_payload = load_error(out_dir)
    if error_payload is not None:
        return SweepRunRecord(
            dataset=DATASET,
            k=k,
            lr=lr,
            ratio=ratio,
            status="failed",
            out_dir=str(out_dir),
            error=error_payload.get("error"),
        )
    raise FileNotFoundError(f"No finished run artifacts in {out_dir}")


def save_summary(k: int, iters: int, jobs: int, records: list[SweepRunRecord]) -> Path:
    path = mnist_grid_summary_path(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET,
        "k": k,
        "iters": iters,
        "jobs": jobs,
        "learning_rates": LEARNING_RATES,
        "ratios": RATIOS,
        "objective": "final_adv_test_ll",
        "objective_data": _TUNE_CORRUPT,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(records),
        "succeeded": sum(r.status == "ok" for r in records),
        "skipped": sum(r.status == "skipped" for r in records),
        "failed": sum(r.status == "failed" for r in records),
        "runs": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def outcome_to_record(task: SweepTask, outcome: RunOutcome) -> SweepRunRecord:
    if outcome.status == "ok":
        return SweepRunRecord(
            dataset=DATASET,
            k=task.k,
            lr=task.lr,
            ratio=task.ratio,
            status="ok",
            out_dir=str(outcome.out_dir),
            metrics=asdict(outcome.metrics) if outcome.metrics is not None else None,
        )
    return SweepRunRecord(
        dataset=DATASET,
        k=task.k,
        lr=task.lr,
        ratio=task.ratio,
        status="failed",
        out_dir=str(outcome.out_dir),
        error=outcome.error,
    )


def _run_combo_task(task: SweepTask) -> SweepRunRecord:
    print(
        f"start  {DATASET}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}",
        flush=True,
    )
    out_dir = grid_run_dir(DATASET, task.k, task.lr, task.ratio)
    outcome = run(
        k=task.k,
        lr=task.lr,
        ratio=task.ratio,
        iters=task.iters,
        save_circuit=False,
        save_plot=False,
        quiet=True,
        out_dir=out_dir,
    )
    record = outcome_to_record(task, outcome)
    if record.status == "ok" and record.metrics is not None:
        adv = record.metrics["final_adv_test_ll"]
        print(
            f"done   {DATASET}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}  "
            f"ok  final_corrupt_ll={adv:.6f}",
            flush=True,
        )
    else:
        print(
            f"done   {DATASET}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}  "
            f"{record.status}",
            flush=True,
        )
    return record


def combo_key(lr: float, ratio: float) -> tuple[float, float]:
    return lr, ratio


def run_sweep(
    k: int,
    iters: int,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
    jobs: int = 1,
) -> list[SweepRunRecord]:
    require_mnist_inputs()
    combos = [(lr, ratio) for lr in LEARNING_RATES for ratio in RATIOS]
    print(
        f"sweep_mnist: k={k}  iters={iters}  jobs={jobs}  "
        f"{len(LEARNING_RATES)} lrs x {len(RATIOS)} ratios = {len(combos)} runs  "
        f"objective={_TUNE_LABEL} mean LL",
        flush=True,
    )

    records_by_combo: dict[tuple[float, float], SweepRunRecord] = {}
    pending: list[SweepTask] = []
    skipped = 0

    for lr, ratio in combos:
        key = combo_key(lr, ratio)
        if skip_existing and is_complete(k, lr, ratio):
            prior = load_run_record(k, lr, ratio)
            skipped += 1
            records_by_combo[key] = SweepRunRecord(
                dataset=DATASET,
                k=k,
                lr=lr,
                ratio=ratio,
                status="skipped",
                out_dir=prior.out_dir,
                metrics=prior.metrics,
                error=prior.error,
            )
            continue

        if dry_run:
            print(f"dry run  {DATASET}  k={k}  lr={lr:g}  ratio={ratio:g}", flush=True)
            records_by_combo[key] = SweepRunRecord(
                dataset=DATASET, k=k, lr=lr, ratio=ratio, status="dry_run"
            )
            continue

        pending.append(SweepTask(k=k, lr=lr, ratio=ratio, iters=iters))

    if skipped:
        print(f"skipped {skipped} finished combo(s)", flush=True)

    if pending:
        if jobs == 1:
            for task in pending:
                record = _run_combo_task(task)
                records_by_combo[combo_key(task.lr, task.ratio)] = record
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(_run_combo_task, task): task for task in pending}
                for future in as_completed(futures):
                    task = futures[future]
                    record = future.result()
                    records_by_combo[combo_key(task.lr, task.ratio)] = record

    records = [records_by_combo[combo_key(lr, ratio)] for lr, ratio in combos]

    if not dry_run:
        summary_path = save_summary(k, iters, jobs, records)
        print(f"sweep summary -> {summary_path.resolve()}", flush=True)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter grid search over lr and ratio for MNIST PeTeR. "
            f"Maximizes mean LL on corrupted {_TUNE_LABEL} (recorded as final_adv_test_ll). "
            "Writes metadata only under sweeps/grid/ (no circuits)."
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
        help=f"OGDA iterations per run (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even when a grid combo directory already finished",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid without training",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        metavar="N",
        help="Parallel worker processes (default: number of available CPUs)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iters < 1:
        raise SystemExit("--iters must be at least 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    try:
        require_mnist_inputs()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    ks = [args.k] if args.k is not None else list(VALID_K)
    for k in ks:
        run_sweep(
            k=k,
            iters=args.iters,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run,
            jobs=args.jobs,
        )


if __name__ == "__main__":
    main()
