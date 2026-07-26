"""Attack every learned circuit under ``results/`` and ``rltpm_learned_pcs/``.

For each circuit, runs the greedy bit-flip attack from ``attack.py`` against the
matching ``original_datasets/<dataset>/<dataset>.test.data`` with the circuit's
K, and writes the attacked set next to the circuit:

* PeTeR:   ``results/.../<dataset>_K<k>_peter.data``
* RL-TPM:  ``rltpm_learned_pcs/.../<dataset>_K<k>_rltpm.data``

Existing outputs are skipped. Jobs run with ``-j`` worker processes.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from sparc.nodes import CircuitNode

from attack import greedy_attack, load_dataset, save_dataset

_ROOT = Path(__file__).resolve().parent
_RESULTS_ROOT = _ROOT / "results"
_RLTPM_ROOT = _ROOT / "rltpm_learned_pcs" / "hclt"
_ORIG_ROOT = _ROOT / "original_datasets"

_K_DIR_RE = re.compile(r"^k(\d+)$", re.IGNORECASE)
_RLTPM_K_RE = re.compile(r"^K(\d+)$")


@dataclass(frozen=True)
class AttackJob:
    circuit: Path
    dataset: Path
    output: Path
    k: int
    dataset_name: str
    source: str  # "peter" | "rltpm"


def original_test_path(dataset_name: str) -> Path:
    return _ORIG_ROOT / dataset_name / f"{dataset_name}.test.data"


def discover_peter_jobs() -> list[AttackJob]:
    jobs: list[AttackJob] = []
    if not _RESULTS_ROOT.is_dir():
        return jobs
    for circuit in sorted(_RESULTS_ROOT.glob("*/*/lr=*/circuit.json")):
        run_dir = circuit.parent
        k_dir = run_dir.parent
        dataset_name = k_dir.parent.name
        m = _K_DIR_RE.match(k_dir.name)
        if m is None:
            continue
        k = int(m.group(1))
        jobs.append(
            AttackJob(
                circuit=circuit,
                dataset=original_test_path(dataset_name),
                output=run_dir / f"{dataset_name}_K{k}_peter.data",
                k=k,
                dataset_name=dataset_name,
                source="peter",
            )
        )
    return jobs


def discover_rltpm_jobs() -> list[AttackJob]:
    jobs: list[AttackJob] = []
    if not _RLTPM_ROOT.is_dir():
        return jobs
    # rltpm_learned_pcs/hclt/<dataset>/<blocksize>/K<k>/hclt_*.json
    for circuit in sorted(_RLTPM_ROOT.glob("*/*/*/hclt_*.json")):
        k_dir = circuit.parent
        m = _RLTPM_K_RE.match(k_dir.name)
        if m is None:
            continue
        k = int(m.group(1))
        dataset_name = k_dir.parent.parent.name
        jobs.append(
            AttackJob(
                circuit=circuit,
                dataset=original_test_path(dataset_name),
                output=k_dir / f"{dataset_name}_K{k}_rltpm.data",
                k=k,
                dataset_name=dataset_name,
                source="rltpm",
            )
        )
    return jobs


def discover_jobs() -> list[AttackJob]:
    return discover_peter_jobs() + discover_rltpm_jobs()


def run_one(job: AttackJob) -> str:
    label = f"{job.source}  {job.dataset_name}  k={job.k}  {job.circuit.parent.name}"
    if job.output.is_file():
        msg = f"skip   {label}  ({job.output.name} exists)"
        print(msg, flush=True)
        return msg
    if not job.dataset.is_file():
        msg = f"SKIP   {label}  (missing {job.dataset})"
        print(msg, flush=True)
        return msg

    print(f"start  {label}", flush=True)
    graph = CircuitNode.load(job.circuit).compile()
    data = load_dataset(job.dataset)
    orig_ll = float(graph.log_likelihood(data).mean())
    adv = greedy_attack(graph, data, job.k)
    adv_ll = float(graph.log_likelihood(adv).mean())
    save_dataset(job.output, adv)
    msg = (
        f"done   {label}  mean_ll {orig_ll:.4f} -> {adv_ll:.4f}  "
        f"-> {job.output.name}"
    )
    print(msg, flush=True)
    return msg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Greedy-attack every circuit under results/ and rltpm_learned_pcs/ "
            "(writes attacked test sets next to each circuit)."
        ),
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        metavar="N",
        help="Parallel worker processes (default: number of available CPUs)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    jobs = discover_jobs()
    pending = [j for j in jobs if not j.output.is_file()]
    print(
        f"attack_all: queued={len(jobs)}  pending={len(pending)}  "
        f"skip_existing={len(jobs) - len(pending)}  jobs={args.jobs}",
        flush=True,
    )
    if not jobs:
        raise SystemExit("No circuits found under results/ or rltpm_learned_pcs/.")

    results: list[str] = []
    if args.jobs == 1:
        for job in jobs:
            results.append(run_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())

    print("\n=== summary ===")
    for line in sorted(results):
        print(line)


if __name__ == "__main__":
    main()
