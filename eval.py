"""Compare MLE-PC, best PeTeR PC, and RL-TPM on original + adversarial test sets.

For each (dataset, k), looks up the best PeTeR ``(lr, ratio)`` from
``sweeps/tpe`` (preferred) or ``sweeps/grid``, materializes
``results/.../circuit.json`` via ``peter.run`` if missing, then prints mean
log-likelihood of MLE / PeTeR / RL-TPM on both test sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def best_peter_params(
    dataset: str,
    k: int,
) -> tuple[float, float, float, str] | None:
    """Return ``(lr, ratio, recorded_adv_ll, source)`` or None."""
    candidates: list[tuple[float, float, float, str]] = []

    for ds, run_k, adv_ll, lr, ratio in best_from_tpe(k):
        if ds == dataset and run_k == k:
            candidates.append((lr, ratio, adv_ll, "tpe"))

    for ds, run_k, adv_ll, _orig, lr, ratio in best_from_grid(k):
        if ds == dataset and run_k == k:
            candidates.append((lr, ratio, adv_ll, "grid"))

    if not candidates:
        return None
    # Prefer higher recorded adversarial LL; break ties toward TPE.
    candidates.sort(key=lambda c: (c[2], c[3] == "tpe"), reverse=True)
    lr, ratio, adv_ll, source = candidates[0]
    return lr, ratio, adv_ll, source


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
        f"  materializing peter circuit  lr={lr:g}  ratio={ratio:g}  iters={iters} ...",
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
        raise RuntimeError(
            f"failed to materialize peter circuit for {dataset} k={k}: {outcome.error}"
        )
    return circuit


def eval_circuit(path: Path, orig, adv) -> tuple[float, float]:
    graph = CircuitNode.load(path).compile()
    return mean_log_likelihood(graph, orig), mean_log_likelihood(graph, adv)


def eval_one(dataset: str, k: int) -> None:
    best = best_peter_params(dataset, k)
    if best is None:
        print(f"{dataset}  k={k}  SKIP (no sweep/tpe winner)")
        return

    lr, ratio, recorded_adv, source = best
    mle = resolve_circuit_path(dataset)
    rltpm = rltpm_path(dataset, k)
    if not rltpm.is_file():
        print(f"{dataset}  k={k}  SKIP (missing RL-TPM: {rltpm})")
        return

    try:
        orig, adv = resolve_eval_datasets(dataset, k)
    except FileNotFoundError as exc:
        print(f"{dataset}  k={k}  SKIP ({exc})")
        return

    iters = iters_for(dataset, k, source)
    try:
        peter = ensure_peter_circuit(dataset, k, lr, ratio, iters)
    except Exception as exc:
        print(f"{dataset}  k={k}  SKIP (peter: {exc})")
        return

    mle_o, mle_a = eval_circuit(mle, orig, adv)
    peter_o, peter_a = eval_circuit(peter, orig, adv)
    rltpm_o, rltpm_a = eval_circuit(rltpm, orig, adv)

    print(
        f"{dataset}  k={k}  peter={source}  lr={lr:g}  ratio={ratio:g}  "
        f"dir={format_hyperparam_dir(lr, ratio)}"
    )
    print(f"  {'method':<8}  {'orig_test':>12}  {'adv_test':>12}")
    print(f"  {'mle-pc':<8}  {mle_o:12.4f}  {mle_a:12.4f}")
    print(f"  {'peter':<8}  {peter_o:12.4f}  {peter_a:12.4f}")
    print(f"  {'rltpm':<8}  {rltpm_o:12.4f}  {rltpm_a:12.4f}")
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare MLE-PC, best PeTeR, and RL-TPM mean test log-likelihoods.",
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
        help="Only this dataset (repeatable). Default: all with example_pcs + eval data",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    ks = [args.k] if args.k is not None else list(VALID_K)

    for k in ks:
        datasets = args.datasets if args.datasets else discover_datasets(k)
        for dataset in datasets:
            eval_one(dataset, k)


if __name__ == "__main__":
    main()
