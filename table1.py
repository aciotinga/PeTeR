"""Emit LaTeX Table 1: MLE-PC vs RL-TPM vs \\method on DEBD.

For each dataset and ε ∈ {1, 3, 5}:

* ``\\mathcal{T}``   — clean test log-likelihood
* ``\\mathcal{T}_a`` — each method's own adversarial set
* ``\\mathcal{T}_r`` — mean ± std over the 10 shared random corruptions

Best / second-best within each metric group are ``\\textbf`` / ``\\underline``
(higher LL better). MLE-PC under ``\\mathcal{T}`` is a multirow (same clean
LL for all ε). Writes ``figures/table1.tex``. Backfills ``rand_std_ll`` into
eval caches when missing. Pass ``--dataset`` to restrict / order rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval import (
    best_peter_params,
    ensure_eval_std,
    load_corrupt_sets,
    load_eval_cache_with_std,
    mle_adv_path,
    peter_adv_path,
    rltpm_adv_path,
    rltpm_path,
)
from peter import VALID_K, run_output_dir
from robustify import resolve_circuit_path
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "figures" / "table1.tex"
_EPSILONS = list(VALID_K)  # 1, 3, 5
_N_EPS = len(_EPSILONS)

# Per-ε scores for one method: (orig, adv, rand_mean, rand_std)
MethodScores = tuple[float, float, float, float]
# One dataset block: name, mle clean LL, and per-ε triples of method scores.
DatasetBlock = tuple[str, float, dict[int, tuple[MethodScores, MethodScores, MethodScores]]]


def fmt_num(x: float) -> str:
    return f"{x:.2f}"


def fmt_pm(mean: float, std: float) -> str:
    return f"{fmt_num(mean)}{{\\scriptsize$\\pm{fmt_num(std)}$}}"


def emphasize(s: str, *, bold: bool, under: bool) -> str:
    if bold:
        s = f"\\textbf{{{s}}}"
    elif under:
        s = f"\\underline{{{s}}}"
    return s


def rank_marks(values: list[float]) -> list[tuple[bool, bool]]:
    """Per-entry (is_best, is_second) among ``values`` (higher better).

    Ranks by the displayed 2-decimal rounding so ties match what the table shows.
    """
    keyed = [round(v, 2) for v in values]
    order = sorted(range(len(keyed)), key=lambda i: keyed[i], reverse=True)
    ranks = [0] * len(keyed)
    rank = 1
    for pos, idx in enumerate(order):
        if pos > 0 and keyed[idx] < keyed[order[pos - 1]]:
            rank = pos + 1
        ranks[idx] = rank
    return [(r == 1, r == 2) for r in ranks]


def collect_blocks(datasets: list[str] | None = None) -> list[DatasetBlock]:
    blocks: list[DatasetBlock] = []
    available = discover_datasets(_EPSILONS[0])
    if datasets is None:
        names = available
    else:
        available_set = set(available)
        missing = [d for d in datasets if d not in available_set]
        if missing:
            raise SystemExit(
                "Unknown or unrunnable dataset(s): "
                + ", ".join(missing)
                + f"\nRunnable: {', '.join(available)}"
            )
        names = datasets  # preserve caller order
    for dataset in names:
        mle_circuit = resolve_circuit_path(dataset)
        per_eps: dict[int, tuple[MethodScores, MethodScores, MethodScores]] = {}
        mle_orig: float | None = None
        skip_reason: str | None = None

        for k in _EPSILONS:
            best = best_peter_params(dataset, k)
            if best is None:
                skip_reason = f"no PeTeR sweep winner (k={k})"
                break
            lr, ratio, _src = best
            peter_circuit = run_output_dir(dataset, k, lr, ratio) / "circuit.json"
            rltpm_circuit = rltpm_path(dataset, k)
            if not peter_circuit.is_file():
                skip_reason = f"missing PeTeR circuit (k={k})"
                break
            if not rltpm_circuit.is_file():
                skip_reason = f"missing RL-TPM circuit (k={k})"
                break

            mle_adv = mle_adv_path(dataset, k)
            peter_adv = peter_adv_path(dataset, k, lr, ratio)
            rltpm_adv = rltpm_adv_path(dataset, k)
            try:
                mle_cached = load_eval_cache_with_std(mle_adv)
                peter_cached = load_eval_cache_with_std(peter_adv)
                rltpm_cached = load_eval_cache_with_std(rltpm_adv)
                if (
                    mle_cached is not None
                    and peter_cached is not None
                    and rltpm_cached is not None
                ):
                    mle_s, peter_s, rltpm_s = mle_cached, peter_cached, rltpm_cached
                else:
                    corrupt = load_corrupt_sets(dataset, k)
                    mle_s = ensure_eval_std(mle_circuit, mle_adv, corrupt)
                    peter_s = ensure_eval_std(peter_circuit, peter_adv, corrupt)
                    rltpm_s = ensure_eval_std(rltpm_circuit, rltpm_adv, corrupt)
            except FileNotFoundError as exc:
                skip_reason = f"{exc} (k={k})"
                break

            if mle_orig is None:
                mle_orig = mle_s[0]
            per_eps[k] = (mle_s, rltpm_s, peter_s)

        if skip_reason is not None or mle_orig is None or len(per_eps) != _N_EPS:
            print(f"skip  {dataset}: {skip_reason or 'incomplete'}", flush=True)
            continue
        print(f"ok    {dataset}", flush=True)
        blocks.append((dataset, mle_orig, per_eps))
    return blocks


def to_latex(blocks: list[DatasetBlock]) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "% Method column uses \\method (define in preamble).",
        "\\begin{tabular}{lcrrrrrrrrr}",
        "\\toprule",
        "Dataset & $\\epsilon$ & "
        "\\multicolumn{3}{c}{$\\mathcal{T}$} & "
        "\\multicolumn{3}{c}{$\\mathcal{T}_a$} & "
        "\\multicolumn{3}{c}{$\\mathcal{T}_r$} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "& & MLE-PC & RL-TPM & \\textbf{\\method} & "
        "MLE-PC & RL-TPM & \\textbf{\\method} & "
        "MLE-PC & RL-TPM & \\textbf{\\method} \\\\",
        "\\midrule",
    ]

    for bi, (dataset, mle_orig, per_eps) in enumerate(blocks):
        # MLE clean formatting: bold if best (or tied) at every ε, else underline
        # if second at every ε, else plain.
        mle_t_bold = True
        mle_t_under = True
        for k in _EPSILONS:
            mle_s, rltpm_s, peter_s = per_eps[k]
            marks = rank_marks([mle_orig, rltpm_s[0], peter_s[0]])
            mle_t_bold = mle_t_bold and marks[0][0]
            mle_t_under = mle_t_under and marks[0][1]
        if mle_t_bold:
            mle_t_under = False
        mle_t_cell = emphasize(fmt_num(mle_orig), bold=mle_t_bold, under=mle_t_under)

        for ei, k in enumerate(_EPSILONS):
            mle_s, rltpm_s, peter_s = per_eps[k]
            t_marks = rank_marks([mle_orig, rltpm_s[0], peter_s[0]])
            a_marks = rank_marks([mle_s[1], rltpm_s[1], peter_s[1]])
            r_marks = rank_marks([mle_s[2], rltpm_s[2], peter_s[2]])

            t_rltpm = emphasize(
                fmt_num(rltpm_s[0]), bold=t_marks[1][0], under=t_marks[1][1]
            )
            t_peter = emphasize(
                fmt_num(peter_s[0]), bold=t_marks[2][0], under=t_marks[2][1]
            )

            a_cells = [
                emphasize(fmt_num(v), bold=b, under=u)
                for v, (b, u) in zip(
                    [mle_s[1], rltpm_s[1], peter_s[1]], a_marks, strict=True
                )
            ]
            r_cells = [
                emphasize(fmt_pm(m, s), bold=b, under=u)
                for (m, s), (b, u) in zip(
                    [
                        (mle_s[2], mle_s[3]),
                        (rltpm_s[2], rltpm_s[3]),
                        (peter_s[2], peter_s[3]),
                    ],
                    r_marks,
                    strict=True,
                )
            ]

            if ei == 0:
                ds_cell = f"\\multirow{{{_N_EPS}}}{{*}}{{{dataset}}}"
                t_mle_cell = f"\\multirow{{{_N_EPS}}}{{*}}{{{mle_t_cell}}}"
            else:
                ds_cell = ""
                t_mle_cell = ""

            lines.append(
                f"{ds_cell} & {k} & {t_mle_cell} & {t_rltpm} & {t_peter} & "
                f"{a_cells[0]} & {a_cells[1]} & {a_cells[2]} & "
                f"{r_cells[0]} & {r_cells[1]} & {r_cells[2]} \\\\"
            )

        if bi < len(blocks) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write LaTeX Table 1 (MLE-PC / RL-TPM / \\method) to figures/table1.tex.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help=(
            "Include only this dataset (repeatable; order is preserved). "
            "Default: all runnable DEBD datasets."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_OUT,
        help=f"Output path (default: {_OUT})",
    )
    args = parser.parse_args()
    blocks = collect_blocks(args.datasets)
    if not blocks:
        raise SystemExit("No complete dataset blocks. Run eval.py first.")
    tex = to_latex(blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output}  ({len(blocks)} datasets)", flush=True)


if __name__ == "__main__":
    main()
