"""Emit a LaTeX table of MLE vs PeTeR log-likelihoods for DEBD at a given K.

Columns: Dataset, MLE Train LL, MLE Test LL, MLE Perturbed Test LL,
PeTeR Perturbed Test LL. By default perturbed values use each method's own
adversarial set (``own_adv_ll`` from ``eval.py`` caches); pass ``--random``
to use mean random-corruption LL (``rand_mean_ll``) instead. The better
perturbed LL is wrapped in ``\\textbf{}``.

Requires ``eval.py`` caches; computes MLE train LL on first use and caches it
beside the train split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparc.nodes import CircuitNode

from eval import best_peter_params, load_eval_cache, mle_adv_path, peter_adv_path
from peter import VALID_K
from robustify import mean_log_likelihood, resolve_circuit_path
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent


def train_path(dataset: str) -> Path:
    return _ROOT / "original_datasets" / dataset / f"{dataset}.train.data"


def train_ll_cache_path(dataset: str) -> Path:
    return train_path(dataset).with_suffix(".mle_train_ll.json")


def mle_train_ll(dataset: str) -> float:
    cache = train_ll_cache_path(dataset)
    if cache.is_file():
        return float(json.loads(cache.read_text(encoding="utf-8"))["train_ll"])
    import numpy as np

    rows = np.loadtxt(train_path(dataset), delimiter=",", dtype=np.int32)
    graph = CircuitNode.load(resolve_circuit_path(dataset)).compile()
    ll = mean_log_likelihood(graph, rows)
    cache.write_text(json.dumps({"train_ll": ll}, indent=2) + "\n", encoding="utf-8")
    return ll


def fmt(x: float, *, bold: bool = False) -> str:
    s = f"{x:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def collect_rows(
    k: int, *, random: bool = False
) -> list[tuple[str, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float]] = []
    for dataset in discover_datasets(k):
        mle = load_eval_cache(mle_adv_path(dataset, k))
        if mle is None:
            print(f"skip  {dataset}: missing MLE eval cache (run eval.py)", flush=True)
            continue
        best = best_peter_params(dataset, k)
        if best is None:
            print(f"skip  {dataset}: no PeTeR sweep winner", flush=True)
            continue
        lr, ratio, _src = best
        peter = load_eval_cache(peter_adv_path(dataset, k, lr, ratio))
        if peter is None:
            print(f"skip  {dataset}: missing PeTeR eval cache (run eval.py)", flush=True)
            continue
        if not train_path(dataset).is_file():
            print(f"skip  {dataset}: missing train split", flush=True)
            continue
        mle_o, mle_a, mle_r = mle
        _peter_o, peter_a, peter_r = peter
        mle_pert = mle_r if random else mle_a
        peter_pert = peter_r if random else peter_a
        rows.append((dataset, mle_train_ll(dataset), mle_o, mle_pert, peter_pert))
    return rows


def to_latex(rows: list[tuple[str, float, float, float, float]], k: int) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        f"% K = {k}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & Train & Test & "
        "Perturbed & \\method \\\\",
        "\\midrule",
    ]
    for dataset, train, test, mle_pert, peter_pert in rows:
        mle_wins = mle_pert >= peter_pert
        peter_wins = peter_pert >= mle_pert
        lines.append(
            f"{dataset} & {fmt(train)} & {fmt(test)} & "
            f"{fmt(mle_pert, bold=mle_wins)} & {fmt(peter_pert, bold=peter_wins)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table of MLE vs PeTeR LLs for DEBD at a given K.",
    )
    parser.add_argument("--k", type=int, required=True, choices=VALID_K)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random-corruption mean LL instead of adversarial LL",
    )
    args = parser.parse_args()
    rows = collect_rows(args.k, random=args.random)
    if not rows:
        raise SystemExit(f"No rows for k={args.k}. Run eval.py first.")
    print(to_latex(rows, args.k), end="")
    wins = sum(1 for *_, mle_pert, peter_pert in rows if peter_pert > mle_pert)
    setting = "random" if args.random else "adversarial"
    print(f"PeTeR beats MLE-PC on perturbed ({setting}, k={args.k}): {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
