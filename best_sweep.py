"""Print best lr/ratio per dataset from metadata-only sweep / TPE results."""

from __future__ import annotations

import argparse
import json
from sweep_io import GRID_ROOT, TPE_ROOT, grid_summary_path


def best_from_grid(k: int | None = None) -> list[tuple[str, int, float, float, float, float]]:
    """Return (dataset, k, adv_ll, orig_ll, lr, ratio) winners from grid sweeps."""
    best: dict[tuple[str, int], tuple[float, float, float, float]] = {}

    pattern = f"k{k}/*/lr=*_ratio=*/metrics.json" if k is not None else "k*/**/lr=*_ratio=*/metrics.json"
    for metrics_path in GRID_ROOT.glob(pattern):
        # sweeps/grid/k{k}/<dataset>/lr=..._ratio=.../metrics.json
        combo_dir = metrics_path.parent
        dataset = combo_dir.parent.name
        k_dir = combo_dir.parent.parent.name
        if not k_dir.startswith("k"):
            continue
        run_k = int(k_dir[1:])
        combo = combo_dir.name
        if not combo.startswith("lr=") or "_ratio=" not in combo:
            continue
        lr_s, ratio_s = combo.removeprefix("lr=").split("_ratio=", 1)
        lr, ratio = float(lr_s), float(ratio_s)
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        adv_ll = m["final_adv_test_ll"]
        key = (dataset, run_k)
        if key not in best or adv_ll > best[key][0]:
            best[key] = (adv_ll, m["final_orig_test_ll"], lr, ratio)

    # Also consult roll-up summary if present (covers skipped/ok metrics).
    ks = [k] if k is not None else sorted(
        {int(p.parent.name[1:]) for p in GRID_ROOT.glob("k*/sweep_summary.json")}
    )
    for run_k in ks:
        summary_path = grid_summary_path(run_k)
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for run in payload.get("runs", []):
            metrics = run.get("metrics")
            if not metrics or run.get("status") not in ("ok", "skipped"):
                continue
            adv_ll = metrics["final_adv_test_ll"]
            key = (run["dataset"], run_k)
            if key not in best or adv_ll > best[key][0]:
                best[key] = (
                    adv_ll,
                    metrics["final_orig_test_ll"],
                    float(run["lr"]),
                    float(run["ratio"]),
                )

    return [
        (dataset, run_k, adv, orig, lr, ratio)
        for (dataset, run_k), (adv, orig, lr, ratio) in sorted(best.items())
    ]


def best_from_tpe(k: int | None = None) -> list[tuple[str, int, float, float, float]]:
    """Return (dataset, k, adv_ll, lr, ratio) winners from TPE study summaries."""
    rows: list[tuple[str, int, float, float, float]] = []
    pattern = f"k{k}/*/study_summary.json" if k is not None else "k*/**/study_summary.json"
    for summary_path in sorted(TPE_ROOT.glob(pattern)):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best = payload.get("best")
        if not best:
            continue
        rows.append(
            (
                payload["dataset"],
                int(payload["k"]),
                float(best["final_adv_test_ll"]),
                float(best["lr"]),
                float(best["ratio"]),
            )
        )
    return rows


def print_grid(rows: list[tuple[str, int, float, float, float, float]]) -> None:
    if not rows:
        print("No grid sweep winners found under sweeps/grid/")
        return
    print("Grid sweep best (final_adv_test_ll):")
    for dataset, run_k, adv_ll, orig_ll, lr, ratio in rows:
        print(
            f"  {dataset}  k={run_k}  lr={lr:g}  ratio={ratio:g}  "
            f"final_adv_ll={adv_ll:.4f}  final_orig_ll={orig_ll:.4f}"
        )


def print_tpe(rows: list[tuple[str, int, float, float, float]]) -> None:
    if not rows:
        print("No TPE winners found under sweeps/tpe/")
        return
    print("TPE best (final_adv_test_ll):")
    for dataset, run_k, adv_ll, lr, ratio in rows:
        print(
            f"  {dataset}  k={run_k}  lr={lr:g}  ratio={ratio:g}  "
            f"final_adv_ll={adv_ll:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print best lr/ratio per dataset from sweeps/grid and sweeps/tpe metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Only show results for this K (default: all)",
    )
    parser.add_argument(
        "--source",
        choices=("all", "grid", "tpe"),
        default="all",
        help="Which metadata source to report (default: all)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source in ("all", "grid"):
        print_grid(best_from_grid(args.k))
    if args.source == "all":
        print()
    if args.source in ("all", "tpe"):
        print_tpe(best_from_tpe(args.k))


if __name__ == "__main__":
    main()
