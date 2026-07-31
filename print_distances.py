"""Print Circuit-Wasserstein distances for MLE-on-adversarial PCs.

For each dataset under ``mle_adv_learned_pcs/``, prints:

* CW(K0, K1)
* CW(K0, K3)
* CW(K0, K5)

Circuits live at ``mle_adv_learned_pcs/<dataset>/k<k>/adv_learned.json``.

Uses the same CW settings as PeTeR training (``metric_p=1.0``, ``scale_factor=1.0``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sparc.nodes import CircuitNode
from sparc.queries import cw_distance

_ROOT = Path(__file__).resolve().parent
OUT_DIR = _ROOT / "mle_adv_learned_pcs"
_CW_KW = dict(metric_p=1.0, scale_factor=1.0)
_COMPARE_KS = (1, 3, 5)


def out_path(dataset: str, k: int) -> Path:
    return OUT_DIR / dataset / f"k{k}" / "adv_learned.json"


def discover_datasets() -> list[str]:
    if not OUT_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in OUT_DIR.iterdir()
        if d.is_dir() and out_path(d.name, 0).is_file()
    )


def cw_between(path_a: Path, path_b: Path) -> float | None:
    """Return CW distance, or None if the circuits are structurally incompatible."""
    a = CircuitNode.load(path_a).compile()
    b = CircuitNode.load(path_b).compile()
    try:
        return float(cw_distance(a, b, **_CW_KW))
    except ValueError:
        return None


def fmt_cw(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.6f}"


def collect_rows(
    compare_ks: list[int],
) -> list[tuple[str, dict[int, float | None]]]:
    rows: list[tuple[str, dict[int, float | None]]] = []
    for dataset in discover_datasets():
        k0 = out_path(dataset, 0)
        dists: dict[int, float | None] = {}
        any_ok = False
        for k in compare_ks:
            other = out_path(dataset, k)
            if not other.is_file():
                print(f"skip  {dataset}  CW(K0,K{k}): missing MLE-adv PC", flush=True)
                dists[k] = None
                continue
            dists[k] = cw_between(k0, other)
            any_ok = True
        if not any_ok:
            continue
        rows.append((dataset, dists))
        parts = "  ".join(f"CW(K0,K{k})={fmt_cw(dists[k])}" for k in compare_ks)
        print(f"done  {dataset:<14}  {parts}", flush=True)
    return rows

def print_table(
    rows: list[tuple[str, dict[int, float | None]]], compare_ks: list[int]
) -> None:
    cols = "  ".join(f"{'CW(K0,K' + str(k) + ')':>14}" for k in compare_ks)
    header = f"{'dataset':<14}  {cols}"
    print()
    print(header)
    print("-" * len(header))
    for dataset, dists in rows:
        vals = "  ".join(f"{fmt_cw(dists[k]):>14}" for k in compare_ks)
        print(f"{dataset:<14}  {vals}")


def print_means(
    rows: list[tuple[str, dict[int, float | None]]], compare_ks: list[int]
) -> None:
    """Mean CW per comparison K, excluding undefined values."""
    print()
    print("mean CW (defined only)")
    header = "  ".join(
        f"{'mean CW(K0,K' + str(k) + ')':>18}  {'n':>4}" for k in compare_ks
    )
    print(header)
    print("-" * len(header))
    parts: list[str] = []
    for k in compare_ks:
        vals = [dists[k] for _d, dists in rows if dists[k] is not None]
        mean = f"{sum(vals) / len(vals):.6f}" if vals else "undefined"
        parts.append(f"{mean:>18}  {len(vals):>4}")
    print("  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print Circuit-Wasserstein distances between the K=0 MLE PC and "
            "the K in {1, 3, 5} MLE-on-adversarial PCs."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        choices=_COMPARE_KS,
        dest="ks",
        help="Restrict to one comparison K (repeatable). Default: 1, 3, 5.",
    )
    args = parser.parse_args()
    compare_ks = args.ks if args.ks else list(_COMPARE_KS)
    rows = collect_rows(compare_ks)
    if not rows:
        raise SystemExit("No distances computed. Check that circuits exist.")
    print_table(rows, compare_ks)
    print_means(rows, compare_ks)
    print(f"\n{len(rows)} datasets")


if __name__ == "__main__":
    main()
