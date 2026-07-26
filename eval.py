"""Compare MLE-PC, best PeTeR PC, and RL-TPM on original + adversarial test sets.

Each method is scored on its own adversarial set:

* MLE-PC  — ``adversarial_datasets/<dataset>_K<k>.data``
* PeTeR   — ``results/.../<dataset>_K<k>_peter.data`` (next to ``circuit.json``)
* RL-TPM  — ``rltpm_learned_pcs/.../<dataset>_K<k>_rltpm.data``

Scores are cached beside each adversarial file as ``*.eval.json`` (skip recompute
unless ``--force``).

Builds one job per ``(dataset, k)``, interleaved across Ks, and runs them with
``-j`` worker processes (same pattern as ``tune.py``).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sparc.nodes import CircuitNode

from best_sweep import best_from_grid, best_from_tpe
from peter import DEFAULT_ITERS, VALID_K, format_hyperparam_dir, run, run_output_dir
from robustify import mean_log_likelihood, resolve_circuit_path, resolve_eval_datasets
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_RLTPM_ROOT = _ROOT / "rltpm_learned_pcs" / "hclt"


def rltpm_path(dataset: str, k: int) -> Path:
    return (
        _RLTPM_ROOT
        / dataset
        / "4"
        / f"K{k}"
        / f"hclt_{dataset}_blocksize4_seed0.json"
    )


def load_binary_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def eval_cache_path(adv_path: Path) -> Path:
    """JSON cache sibling of an adversarial ``.data`` file."""
    return adv_path.with_suffix(".eval.json")


def mle_adv_path(dataset: str, k: int) -> Path:
    return _ROOT / "adversarial_datasets" / f"{dataset}_K{k}.data"


def peter_adv_path(dataset: str, k: int, lr: float, ratio: float) -> Path:
    return run_output_dir(dataset, k, lr, ratio) / f"{dataset}_K{k}_peter.data"


def rltpm_adv_path(dataset: str, k: int) -> Path:
    return rltpm_path(dataset, k).parent / f"{dataset}_K{k}_rltpm.data"


def load_eval_cache(adv_path: Path) -> tuple[float, float] | None:
    path = eval_cache_path(adv_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["orig_test_ll"]), float(payload["own_adv_ll"])


def save_eval_cache(adv_path: Path, orig_ll: float, adv_ll: float) -> None:
    path = eval_cache_path(adv_path)
    path.write_text(
        json.dumps(
            {"orig_test_ll": orig_ll, "own_adv_ll": adv_ll},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def eval_circuit(
    circuit_path: Path,
    orig: np.ndarray,
    adv: np.ndarray,
    *,
    adv_path: Path,
    force: bool = False,
) -> tuple[float, float]:
    if not force:
        cached = load_eval_cache(adv_path)
        if cached is not None:
            return cached
    graph = CircuitNode.load(circuit_path).compile()
    orig_ll = mean_log_likelihood(graph, orig)
    adv_ll = mean_log_likelihood(graph, adv)
    save_eval_cache(adv_path, orig_ll, adv_ll)
    return orig_ll, adv_ll


def best_peter_params(dataset: str, k: int) -> tuple[float, float, str] | None:
    candidates: list[tuple[float, float, float, str]] = []
    for ds, run_k, adv_ll, lr, ratio in best_from_tpe(k):
        if ds == dataset and run_k == k:
            candidates.append((lr, ratio, adv_ll, "tpe"))
    for ds, run_k, adv_ll, _orig, lr, ratio in best_from_grid(k):
        if ds == dataset and run_k == k:
            candidates.append((lr, ratio, adv_ll, "grid"))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[3] == "tpe"), reverse=True)
    lr, ratio, _adv, source = candidates[0]
    return lr, ratio, source


def iters_for(dataset: str, k: int, source: str) -> int:
    if source == "tpe":
        path = _ROOT / "sweeps" / "tpe" / f"k{k}" / dataset / "study_summary.json"
        if path.is_file():
            return int(json.loads(path.read_text(encoding="utf-8")).get("iters", DEFAULT_ITERS))
    if source == "grid":
        path = _ROOT / "sweeps" / "grid" / f"k{k}" / "sweep_summary.json"
        if path.is_file():
            return int(json.loads(path.read_text(encoding="utf-8")).get("iters", DEFAULT_ITERS))
    return DEFAULT_ITERS


def ensure_peter_circuit(dataset: str, k: int, lr: float, ratio: float, iters: int) -> Path:
    out_dir = run_output_dir(dataset, k, lr, ratio)
    circuit = out_dir / "circuit.json"
    if circuit.is_file():
        return circuit
    print(
        f"start  {dataset}  k={k}  materialize  lr={lr:g}  ratio={ratio:g}",
        flush=True,
    )
    outcome = run(
        dataset=dataset,
        k=k,
        lr=lr,
        ratio=ratio,
        iters=iters,
        save_circuit=True,
        save_plot=False,
        quiet=True,
        out_dir=out_dir,
    )
    if outcome.status != "ok" or not circuit.is_file():
        raise RuntimeError(outcome.error or "peter run failed")
    return circuit


def eval_circuit(path: Path, orig, adv) -> tuple[float, float]:
    graph = CircuitNode.load(path).compile()
    return mean_log_likelihood(graph, orig), mean_log_likelihood(graph, adv)


def collect_jobs(ks: list[int], datasets: list[str] | None) -> list[tuple[str, int]]:
    """Round-robin ``(dataset, k)`` jobs across Ks."""
    per_k: list[list[tuple[str, int]]] = []
    for k in ks:
        names = datasets if datasets else discover_datasets(k)
        if datasets:
            available = set(discover_datasets(k))
            missing = [d for d in datasets if d not in available]
            if missing:
                raise SystemExit(
                    f"Dataset(s) not runnable for k={k}: {', '.join(missing)}"
                )
        per_k.append([(d, k) for d in names])

    jobs: list[tuple[str, int]] = []
    while any(per_k):
        for bucket in per_k:
            if bucket:
                jobs.append(bucket.pop(0))
    return jobs


def eval_one(args: tuple[str, int, bool]) -> str:
    """Worker entry point. Returns a printable block (or SKIP line)."""
    dataset, k, force = args
    print(f"start  {dataset}  k={k}", flush=True)

    best = best_peter_params(dataset, k)
    if best is None:
        msg = f"{dataset}  k={k}  SKIP (no sweep/tpe winner)"
        print(f"done   {msg}", flush=True)
        return msg

    lr, ratio, source = best
    rltpm = rltpm_path(dataset, k)
    if not rltpm.is_file():
        msg = f"{dataset}  k={k}  SKIP (missing RL-TPM)"
        print(f"done   {msg}", flush=True)
        return msg

    try:
        orig, _ = resolve_eval_datasets(dataset, k)
        mle = resolve_circuit_path(dataset)
        peter = ensure_peter_circuit(dataset, k, lr, ratio, iters_for(dataset, k, source))
        mle_adv_file = mle_adv_path(dataset, k)
        peter_adv_file = peter_adv_path(dataset, k, lr, ratio)
        rltpm_adv_file = rltpm_adv_path(dataset, k)
        mle_adv = load_binary_data(mle_adv_file)
        peter_adv = load_binary_data(peter_adv_file)
        rltpm_adv = load_binary_data(rltpm_adv_file)
        mle_o, mle_a = eval_circuit(
            mle, orig, mle_adv, adv_path=mle_adv_file, force=force
        )
        peter_o, peter_a = eval_circuit(
            peter, orig, peter_adv, adv_path=peter_adv_file, force=force
        )
        rltpm_o, rltpm_a = eval_circuit(
            rltpm, orig, rltpm_adv, adv_path=rltpm_adv_file, force=force
        )
    except Exception as exc:
        msg = f"{dataset}  k={k}  SKIP ({exc})"
        print(f"done   {msg}", flush=True)
        return msg

    block = (
        f"{dataset}  k={k}  peter={source}  lr={lr:g}  ratio={ratio:g}  "
        f"dir={format_hyperparam_dir(lr, ratio)}\n"
        f"  {'method':<8}  {'orig_test':>12}  {'own_adv':>12}\n"
        f"  {'mle-pc':<8}  {mle_o:12.4f}  {mle_a:12.4f}\n"
        f"  {'peter':<8}  {peter_o:12.4f}  {peter_a:12.4f}\n"
        f"  {'rltpm':<8}  {rltpm_o:12.4f}  {rltpm_a:12.4f}"
    )
    print(f"done   {dataset}  k={k}", flush=True)
    return block


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare MLE-PC, best PeTeR, and RL-TPM on original test and each "
            "method's own adversarial set (parallel job queue)."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"Only this K (default: all of {list(VALID_K)})",
    )
    p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Only this dataset (repeatable). Default: all runnable",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        metavar="N",
        help="Parallel worker processes (default: number of available CPUs)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when a sibling .eval.json cache exists",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    ks = [args.k] if args.k is not None else list(VALID_K)
    pairs = collect_jobs(ks, args.datasets)
    jobs = [(dataset, k, args.force) for dataset, k in pairs]
    print(
        f"eval: ks={ks}  jobs={args.jobs}  queued={len(jobs)}  force={args.force}",
        flush=True,
    )

    results: list[str] = []
    if args.jobs == 1:
        for job in jobs:
            results.append(eval_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(eval_one, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())

    print("\n=== results ===")
    for block in sorted(results):
        print(block)
        print()


if __name__ == "__main__":
    main()
