"""Grid search over learning rate and phi/theta ratio for every runnable dataset.

Learning rates are ``1e-x`` and ``5e-x`` for ``x`` in ``[5, 4, 3, 2, 1]``.
Ratios are ``[1, 2, 5, 10]``.

A dataset is included only when it has a circuit under ``example_pcs/`` and both
evaluation files required by ``peter.run`` (original test + adversarial K).

Each combination calls :func:`peter.run` with metadata-only persistence under::

    sweeps/grid/k<k>/<dataset>/lr=<lr>_ratio=<ratio>/
        config.json
        metrics.json   # or error.json

No circuits or plots are written. When the sweep finishes, a roll-up table is
saved to::

    sweeps/grid/k<k>/sweep_summary.json

Pending combos run in parallel via worker processes (``--jobs``). Each combo
writes to its own output directory, so runs do not conflict.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from robustify import resolve_eval_datasets

from peter import DEFAULT_ITERS, VALID_K, RunOutcome, run
from sweep_io import (
    grid_run_dir,
    grid_summary_path,
    is_run_finished,
    load_error,
    load_metrics,
)

_ROOT = Path(__file__).resolve().parent
_EXAMPLE_PCS = _ROOT / "example_pcs"

# Grid definition (fixed).
LR_EXPONENTS = [5, 4, 3, 2, 1]
LEARNING_RATES = [
    coeff * 10.0 ** -exp
    for exp in LR_EXPONENTS
    for coeff in (1.0, 5.0)
]
RATIOS = [1, 2, 5, 10]


@dataclass(frozen=True)
class SweepTask:
    dataset: str
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


def discover_datasets(k: int) -> list[str]:
    """Return dataset names that have a circuit and eval data for ``k``."""
    runnable: list[str] = []
    for circuit_path in sorted(_EXAMPLE_PCS.glob("*.json")):
        name = circuit_path.stem
        try:
            resolve_eval_datasets(name, k)
        except FileNotFoundError:
            continue
        runnable.append(name)
    return runnable


def is_complete(dataset: str, k: int, lr: float, ratio: float) -> bool:
    return is_run_finished(grid_run_dir(dataset, k, lr, ratio))


def load_run_record(dataset: str, k: int, lr: float, ratio: float) -> SweepRunRecord:
    out_dir = grid_run_dir(dataset, k, lr, ratio)
    metrics = load_metrics(out_dir)
    if metrics is not None:
        return SweepRunRecord(
            dataset=dataset,
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
            dataset=dataset,
            k=k,
            lr=lr,
            ratio=ratio,
            status="failed",
            out_dir=str(out_dir),
            error=error_payload.get("error"),
        )
    raise FileNotFoundError(f"No finished run artifacts in {out_dir}")


def save_summary(k: int, iters: int, jobs: int, records: list[SweepRunRecord]) -> Path:
    path = grid_summary_path(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": k,
        "iters": iters,
        "jobs": jobs,
        "learning_rates": LEARNING_RATES,
        "ratios": RATIOS,
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
            dataset=task.dataset,
            k=task.k,
            lr=task.lr,
            ratio=task.ratio,
            status="ok",
            out_dir=str(outcome.out_dir),
            metrics=asdict(outcome.metrics) if outcome.metrics is not None else None,
        )
    return SweepRunRecord(
        dataset=task.dataset,
        k=task.k,
        lr=task.lr,
        ratio=task.ratio,
        status="failed",
        out_dir=str(outcome.out_dir),
        error=outcome.error,
    )


def _run_combo_task(task: SweepTask) -> SweepRunRecord:
    """Worker entry point (must be top-level for multiprocessing on Windows)."""
    print(
        f"start  {task.dataset}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}",
        flush=True,
    )
    out_dir = grid_run_dir(task.dataset, task.k, task.lr, task.ratio)
    outcome = run(
        dataset=task.dataset,
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
            f"done   {task.dataset}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}  "
            f"ok  final_adv_ll={adv:.6f}",
            flush=True,
        )
    else:
        print(
            f"done   {task.dataset}  k={task.k}  lr={task.lr:g}  ratio={task.ratio:g}  "
            f"{record.status}",
            flush=True,
        )
    return record


def combo_key(dataset: str, lr: float, ratio: float) -> tuple[str, float, float]:
    return dataset, lr, ratio


def run_sweep(
    k: int,
    iters: int,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
    jobs: int = 1,
) -> list[SweepRunRecord]:
    datasets = discover_datasets(k)
    if not datasets:
        raise SystemExit(f"No runnable datasets found for k={k}.")

    combos = [(dataset, lr, ratio) for dataset in datasets for lr in LEARNING_RATES for ratio in RATIOS]
    print(
        f"sweep: k={k}  iters={iters}  jobs={jobs}  "
        f"{len(datasets)} datasets x {len(LEARNING_RATES)} lrs x {len(RATIOS)} ratios "
        f"= {len(combos)} runs",
        flush=True,
    )

    records_by_combo: dict[tuple[str, float, float], SweepRunRecord] = {}
    pending: list[SweepTask] = []
    skipped = 0

    for dataset, lr, ratio in combos:
        key = combo_key(dataset, lr, ratio)
        if skip_existing and is_complete(dataset, k, lr, ratio):
            prior = load_run_record(dataset, k, lr, ratio)
            skipped += 1
            records_by_combo[key] = SweepRunRecord(
                dataset=dataset,
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
            print(f"dry run  {dataset}  k={k}  lr={lr:g}  ratio={ratio:g}", flush=True)
            records_by_combo[key] = SweepRunRecord(
                dataset=dataset, k=k, lr=lr, ratio=ratio, status="dry_run"
            )
            continue

        pending.append(SweepTask(dataset=dataset, k=k, lr=lr, ratio=ratio, iters=iters))

    if skipped:
        print(f"skipped {skipped} finished combo(s)", flush=True)

    if pending:
        if jobs == 1:
            for task in pending:
                record = _run_combo_task(task)
                records_by_combo[combo_key(task.dataset, task.lr, task.ratio)] = record
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(_run_combo_task, task): task for task in pending}
                for future in as_completed(futures):
                    task = futures[future]
                    record = future.result()
                    records_by_combo[combo_key(task.dataset, task.lr, task.ratio)] = record

    records = [records_by_combo[combo_key(dataset, lr, ratio)] for dataset, lr, ratio in combos]

    if not dry_run:
        summary_path = save_summary(k, iters, jobs, records)
        print(f"sweep summary -> {summary_path.resolve()}", flush=True)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter grid search over lr and ratio for all runnable datasets. "
            "Writes metadata only under sweeps/grid/ (no circuits)."
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
        help=f"OGDA iterations per run (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even when sweeps/grid/k<k>/<dataset>/lr=..._ratio=.../ already finished",
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
