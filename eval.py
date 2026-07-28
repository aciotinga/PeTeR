"""Compare MLE-PC, best PeTeR PC, and RL-TPM on original + adversarial + random.

Each method is scored on its own adversarial set:

* MLE-PC  — ``adversarial_datasets/<dataset>_K<k>.data``
* PeTeR   — ``results/.../<dataset>_K<k>_peter.data`` (next to ``circuit.json``)
* RL-TPM  — ``rltpm_learned_pcs/.../<dataset>_K<k>_rltpm.data``

Plus a shared random-corruption column: mean LL over the 10 copies under
``corrupted_datasets/K<k>/<dataset>/r0.data`` … ``r9.data`` (from
``random_corrupt.py``).

Scores are cached beside each adversarial file as ``*.eval.json`` (skip
recompute unless ``--force``). Cache fields: ``orig_test_ll``, ``own_adv_ll``,
``rand_mean_ll``.

After scoring, prints PeTeR win counts and paired Wilcoxon signed-rank tests
of PeTeR vs RL-TPM on ``own_adv`` and ``rand_mean``, separately for each K.

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
from scipy.stats import wilcoxon
from sparc.nodes import CircuitNode

from best_sweep import best_from_grid, best_from_tpe
from peter import DEFAULT_ITERS, VALID_K, format_hyperparam_dir, run, run_output_dir
from random_corrupt import NUM_COPIES, corrupt_path
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


def load_corrupt_sets(dataset: str, k: int) -> list[np.ndarray]:
    paths = [corrupt_path(dataset, k, r) for r in range(NUM_COPIES)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing corrupted datasets (run random_corrupt.py first):\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    return [load_binary_data(p) for p in paths]


def load_eval_cache(adv_path: Path) -> tuple[float, float, float] | None:
    path = eval_cache_path(adv_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "rand_mean_ll" not in payload:
        return None
    return (
        float(payload["orig_test_ll"]),
        float(payload["own_adv_ll"]),
        float(payload["rand_mean_ll"]),
    )


def save_eval_cache(
    adv_path: Path, orig_ll: float, adv_ll: float, rand_mean_ll: float
) -> None:
    path = eval_cache_path(adv_path)
    path.write_text(
        json.dumps(
            {
                "orig_test_ll": orig_ll,
                "own_adv_ll": adv_ll,
                "rand_mean_ll": rand_mean_ll,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def eval_circuit(
    circuit_path: Path,
    orig: np.ndarray,
    adv: np.ndarray,
    corrupt_sets: list[np.ndarray],
    *,
    adv_path: Path,
    force: bool = False,
) -> tuple[float, float, float]:
    if not force:
        cached = load_eval_cache(adv_path)
        if cached is not None:
            return cached
    graph = CircuitNode.load(circuit_path).compile()
    orig_ll = mean_log_likelihood(graph, orig)
    adv_ll = mean_log_likelihood(graph, adv)
    rand_mean_ll = float(
        np.mean([mean_log_likelihood(graph, rows) for rows in corrupt_sets])
    )
    save_eval_cache(adv_path, orig_ll, adv_ll, rand_mean_ll)
    return orig_ll, adv_ll, rand_mean_ll


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


# Scores: (mle_o, mle_a, mle_r, peter_o, peter_a, peter_r, rltpm_o, rltpm_a, rltpm_r)
EvalScores = tuple[float, float, float, float, float, float, float, float, float]
# Successful job payload: (k, scores)
ScoredJob = tuple[int, EvalScores]


def eval_one(args: tuple[str, int, bool]) -> tuple[str, ScoredJob | None]:
    """Worker entry point. Returns (printable block, (k, scores) or None if SKIP)."""
    dataset, k, force = args
    print(f"start  {dataset}  k={k}", flush=True)

    best = best_peter_params(dataset, k)
    if best is None:
        msg = f"{dataset}  k={k}  SKIP (no sweep/tpe winner)"
        print(f"done   {msg}", flush=True)
        return msg, None

    lr, ratio, source = best
    rltpm = rltpm_path(dataset, k)
    if not rltpm.is_file():
        msg = f"{dataset}  k={k}  SKIP (missing RL-TPM)"
        print(f"done   {msg}", flush=True)
        return msg, None

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
        corrupt_sets = load_corrupt_sets(dataset, k)
        mle_o, mle_a, mle_r = eval_circuit(
            mle, orig, mle_adv, corrupt_sets, adv_path=mle_adv_file, force=force
        )
        peter_o, peter_a, peter_r = eval_circuit(
            peter, orig, peter_adv, corrupt_sets, adv_path=peter_adv_file, force=force
        )
        rltpm_o, rltpm_a, rltpm_r = eval_circuit(
            rltpm, orig, rltpm_adv, corrupt_sets, adv_path=rltpm_adv_file, force=force
        )
    except Exception as exc:
        msg = f"{dataset}  k={k}  SKIP ({exc})"
        print(f"done   {msg}", flush=True)
        return msg, None

    block = (
        f"{dataset}  k={k}  peter={source}  lr={lr:g}  ratio={ratio:g}  "
        f"dir={format_hyperparam_dir(lr, ratio)}\n"
        f"  {'method':<8}  {'orig_test':>12}  {'own_adv':>12}  {'rand_mean':>12}\n"
        f"  {'mle-pc':<8}  {mle_o:12.4f}  {mle_a:12.4f}  {mle_r:12.4f}\n"
        f"  {'peter':<8}  {peter_o:12.4f}  {peter_a:12.4f}  {peter_r:12.4f}\n"
        f"  {'rltpm':<8}  {rltpm_o:12.4f}  {rltpm_a:12.4f}  {rltpm_r:12.4f}"
    )
    scores = (mle_o, mle_a, mle_r, peter_o, peter_a, peter_r, rltpm_o, rltpm_a, rltpm_r)
    print(f"done   {dataset}  k={k}", flush=True)
    return block, (k, scores)


def wilcoxon_peter_vs_rltpm(peter: np.ndarray, rltpm: np.ndarray) -> str:
    """Paired Wilcoxon signed-rank summary for PeTeR vs RL-TPM (higher LL better)."""
    n = len(peter)
    if n == 0:
        return "n=0  (no scored datasets)"
    diff = peter - rltpm
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    mean_diff = float(diff.mean())
    prefix = (
        f"n={n}  peter_wins={wins}  rltpm_wins={losses}  ties={ties}  "
        f"mean_diff={mean_diff:+.4f}"
    )
    if int(np.sum(diff != 0)) < 1:
        return f"{prefix}  W=n/a  p=n/a (all ties)"
    try:
        # zero_method='wilcox': drop zero differences (scipy default).
        res = wilcoxon(peter, rltpm, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        return f"{prefix}  W=n/a  p=n/a ({exc})"
    return f"{prefix}  W={float(res.statistic):.4g}  p={float(res.pvalue):.4g}"


def report_wilcoxon(
    scored: list[ScoredJob],
    ks: list[int],
    metric: str,
    peter_idx: int,
    rltpm_idx: int,
) -> None:
    """Print per-K Wilcoxon of PeTeR vs RL-TPM on one score column."""
    print(f"=== wilcoxon peter vs rltpm ({metric}, per K) ===")
    for k in ks:
        peter = np.array(
            [s[peter_idx] for kk, s in scored if kk == k], dtype=np.float64
        )
        rltpm = np.array(
            [s[rltpm_idx] for kk, s in scored if kk == k], dtype=np.float64
        )
        print(f"  k={k}  {wilcoxon_peter_vs_rltpm(peter, rltpm)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare MLE-PC, best PeTeR, and RL-TPM on original test, each "
            "method's own adversarial set, and shared random corruptions."
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

    results: list[tuple[str, ScoredJob | None]] = []
    if args.jobs == 1:
        for job in jobs:
            results.append(eval_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(eval_one, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())

    print("\n=== results ===")
    for block, _ in sorted(results, key=lambda r: r[0]):
        print(block)
        print()

    scored = [job for _, job in results if job is not None]
    n = len(scored)
    # Indices: mle_o, mle_a, mle_r, peter_o, peter_a, peter_r, rltpm_o, rltpm_a, rltpm_r
    beat_mle_adv = sum(1 for _, s in scored if s[4] > s[1])
    beat_rltpm_adv = sum(1 for _, s in scored if s[4] > s[7])
    beat_rltpm_orig = sum(1 for _, s in scored if s[3] > s[6])
    beat_mle_rand = sum(1 for _, s in scored if s[5] > s[2])
    beat_rltpm_rand = sum(1 for _, s in scored if s[5] > s[8])
    print("=== peter wins ===")
    print(f"  vs mle-pc  own_adv:   {beat_mle_adv}/{n}")
    print(f"  vs rltpm   own_adv:   {beat_rltpm_adv}/{n}")
    print(f"  vs rltpm   orig_test: {beat_rltpm_orig}/{n}")
    print(f"  vs mle-pc  rand_mean: {beat_mle_rand}/{n}")
    print(f"  vs rltpm   rand_mean: {beat_rltpm_rand}/{n}")
    # Indices: peter_a=4, peter_r=5, rltpm_a=7, rltpm_r=8
    report_wilcoxon(scored, ks, "own_adv", peter_idx=4, rltpm_idx=7)
    report_wilcoxon(scored, ks, "rand_mean", peter_idx=5, rltpm_idx=8)


if __name__ == "__main__":
    main()
