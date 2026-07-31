"""Emit a LaTeX table of \\method runtime from ``runtime_results/*.json``.

Columns: Dataset, iterations/sec, estimated wall time for 500 iterations
in minutes (``500 / iterations_per_sec / 60``). Writes ``runtime_results/table2.tex``.
Missing JSON files are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peter import DEFAULT_ITERS
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_RESULTS = _ROOT / "runtime_results"
_OUT = _RESULTS / "table2.tex"
_K = 1


def load_row(dataset: str) -> tuple[str, float, float] | None:
    path = _RESULTS / f"{dataset}.json"
    if not path.is_file():
        print(f"skip  {dataset}: missing {path.name}", flush=True)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ips = float(data["iterations_per_sec"])
    if ips <= 0:
        print(f"skip  {dataset}: non-positive iterations_per_sec", flush=True)
        return None
    time_500_min = (DEFAULT_ITERS / ips) / 60.0
    return dataset, ips, time_500_min


def collect_rows() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for dataset in discover_datasets(_K):
        row = load_row(dataset)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def fmt_ips(x: float) -> str:
    return f"{x:.2f}"


def fmt_time(x: float) -> str:
    return f"{x:.2f}"


def to_latex(rows: list[tuple[str, float, float]]) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        f"% Estimated time (minutes) for {DEFAULT_ITERS} iterations from "
        f"iterations/sec (runtime_results/, k={_K}).",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Dataset & Iters/s & Time (min) \\\\",
        "\\midrule",
    ]
    for dataset, ips, time_500_min in rows:
        lines.append(
            f"{dataset} & {fmt_ips(ips)} & {fmt_time(time_500_min)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write LaTeX runtime table from runtime_results/*.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_OUT,
        help=f"Output path (default: {_OUT})",
    )
    args = parser.parse_args()
    rows = collect_rows()
    if not rows:
        raise SystemExit("No runtime_results JSON files found.")
    tex = to_latex(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(tex, end="")
    print(f"wrote {args.output}  ({len(rows)} datasets)", flush=True)


if __name__ == "__main__":
    main()
