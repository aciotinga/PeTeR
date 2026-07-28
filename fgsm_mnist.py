"""FGSM-attack the original MNIST MLE PC and robustified PeTeR PCs.

Uses the one-shot finite-difference FGSM from :mod:`fgsm` against:

* original MLE — ``mnist/hclt_mnist_blocksize4.json``
* PeTeR       — every ``results/mnist/k<k>/lr=*/circuit.json``

Writes attacked test sets to::

* MLE:   ``adversarial_datasets/K{k}/mnist.test.data``
* PeTeR: ``results/mnist/k{k}/lr=.../mnist_K{k}_peter.data``

Existing outputs are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sparc.nodes import CircuitNode

from fgsm import (
    MAX_CAND_ROWS,
    default_output,
    fgsm_attack,
    load_dataset,
    save_dataset,
)
from peter import VALID_K
from peter_mnist import circuit_path, original_test_path

_ROOT = Path(__file__).resolve().parent
_RESULTS_MNIST = _ROOT / "results" / "mnist"
_K_DIR_RE = re.compile(r"^k(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class AttackJob:
    circuit: Path
    dataset: Path
    output: Path
    k: int
    source: str  # "mle" | "peter"
    label: str


def discover_peter_jobs(ks: set[int] | None) -> list[AttackJob]:
    jobs: list[AttackJob] = []
    if not _RESULTS_MNIST.is_dir():
        return jobs
    for circuit in sorted(_RESULTS_MNIST.glob("k*/lr=*/circuit.json")):
        run_dir = circuit.parent
        k_dir = run_dir.parent
        m = _K_DIR_RE.match(k_dir.name)
        if m is None:
            continue
        k = int(m.group(1))
        if ks is not None and k not in ks:
            continue
        jobs.append(
            AttackJob(
                circuit=circuit,
                dataset=original_test_path(),
                output=run_dir / f"mnist_K{k}_peter.data",
                k=k,
                source="peter",
                label=f"peter  k={k}  {run_dir.name}",
            )
        )
    return jobs


def discover_mle_jobs(ks: set[int]) -> list[AttackJob]:
    mle = circuit_path()
    return [
        AttackJob(
            circuit=mle,
            dataset=original_test_path(),
            output=default_output(k),
            k=k,
            source="mle",
            label=f"mle    k={k}",
        )
        for k in sorted(ks)
    ]


def discover_jobs(ks: list[int] | None) -> list[AttackJob]:
    """MLE jobs for requested Ks + PeTeR jobs found under results/mnist/."""
    if ks is not None:
        k_set = set(ks)
        return discover_mle_jobs(k_set) + discover_peter_jobs(k_set)

    peter = discover_peter_jobs(None)
    k_set = {job.k for job in peter} or set(VALID_K)
    return discover_mle_jobs(k_set) + peter


def run_one(
    job: AttackJob,
    *,
    force: bool = False,
    max_cand_rows: int = MAX_CAND_ROWS,
    batch_size: int | None = None,
    n: int | None = None,
) -> str:
    if job.output.is_file() and not force:
        msg = f"skip   {job.label}  ({job.output.name} exists)"
        print(msg, flush=True)
        return msg
    if not job.circuit.is_file():
        msg = f"SKIP   {job.label}  (missing circuit {job.circuit})"
        print(msg, flush=True)
        return msg
    if not job.dataset.is_file():
        msg = f"SKIP   {job.label}  (missing {job.dataset})"
        print(msg, flush=True)
        return msg

    print(f"start  {job.label}", flush=True)
    graph = CircuitNode.load(job.circuit).compile()
    data = load_dataset(job.dataset)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if n is not None:
        data = data[:n]

    print(
        f"  FGSM {len(data)} x {data.shape[1]}  K={job.k}  "
        f"circuit={job.circuit.relative_to(_ROOT)}",
        flush=True,
    )
    orig_ll = float(graph.log_likelihood(data).mean())
    adv = fgsm_attack(
        graph,
        data,
        job.k,
        max_cand_rows=max_cand_rows,
        batch_size=batch_size,
    )
    adv_ll = float(graph.log_likelihood(adv).mean())

    delta = np.abs(adv.astype(np.int32) - data.astype(np.int32))
    mean_linf = float(delta.max(axis=1).mean()) if len(data) else 0.0
    max_linf = int(delta.max()) if data.size else 0

    save_dataset(job.output, adv)
    msg = (
        f"done   {job.label}  mean_ll {orig_ll:.6f} -> {adv_ll:.6f}  "
        f"mean_Linf={mean_linf:.4f}  max_Linf={max_linf}  "
        f"-> {job.output.relative_to(_ROOT)}"
    )
    print(msg, flush=True)
    return msg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "One-shot FGSM on the original MNIST MLE PC and every robustified "
            "PeTeR circuit under results/mnist/."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"Only this L-inf budget K (default: PeTeR Ks under results/, else {list(VALID_K)})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing adversarial outputs",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Attack only the first N test rows (default: full test set)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Rows per FD pass (0 = all rows at once)",
    )
    p.add_argument(
        "--max-cand-rows",
        type=int,
        default=MAX_CAND_ROWS,
        help=f"Max candidate rows per LL batch (default: {MAX_CAND_ROWS})",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.n is not None and args.n < 1:
        raise SystemExit("--n must be >= 1")

    ks = [args.k] if args.k is not None else None
    jobs = discover_jobs(ks)
    if not jobs:
        raise SystemExit("No MNIST attack jobs found (check mnist/ and results/mnist/).")

    pending = [j for j in jobs if args.force or not j.output.is_file()]
    print(
        f"fgsm_mnist: queued={len(jobs)}  pending={len(pending)}  "
        f"skip_existing={len(jobs) - len(pending)}  force={args.force}",
        flush=True,
    )

    batch_size = args.batch_size or None
    results: list[str] = []
    for job in jobs:
        results.append(
            run_one(
                job,
                force=args.force,
                max_cand_rows=args.max_cand_rows,
                batch_size=batch_size,
                n=args.n,
            )
        )

    print("\n=== summary ===")
    for line in results:
        print(line)


if __name__ == "__main__":
    main()
