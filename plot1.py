"""Figure 1: Paired DEBD comparison plots (PeTeR vs RL-TPM).

Writes three PDFs over ε ∈ {1, 3, 5}:

* ``figures/plot1.pdf``  — combined 2x3 (random + adversarial)
* ``figures/plot1a.pdf`` — random corruptions only (1x3)
* ``figures/plot1b.pdf`` — adversarial corruptions only (1x3)

Each panel shows one bar per DEBD dataset (MNIST excluded):

* value = 100 * (PeTeR − RL-TPM) / |RL-TPM|
  on corrupted log-likelihood per binary feature

Positive ⇒ PeTeR higher LL (better). Uses existing ``eval.py`` caches only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from eval import best_peter_params, load_eval_cache, peter_adv_path, rltpm_adv_path
from peter import VALID_K
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "figures" / "plot1.pdf"
_OUT_A = _ROOT / "figures" / "plot1a.pdf"
_OUT_B = _ROOT / "figures" / "plot1b.pdf"
_EXCLUDE = frozenset({"mnist"})
_EPSILONS = list(VALID_K)  # 1, 3, 5

# Colors: PeTeR better / worse (higher LL better).
_WIN_COLOR = "#2a6f97"
_LOSS_COLOR = "#c1121f"

_TITLE_RANDOM = "Random Corruptions"
_TITLE_ADV = "Adversarial Corruptions"

Kind = Literal["random", "greedy", "combined"]


def n_features(dataset: str) -> int:
    """Number of binary features from the first row of the test split."""
    path = _ROOT / "original_datasets" / dataset / f"{dataset}.test.data"
    with path.open(encoding="utf-8") as f:
        line = f.readline()
    if not line.strip():
        raise FileNotFoundError(f"Empty test split: {path}")
    return line.count(",") + 1


# One scored dataset for one ε: (name, rltpm_rand, peter_rand, rltpm_adv, peter_adv, n_vars)
ScoredRow = tuple[str, float, float, float, float, int]


def collect_all() -> dict[int, list[ScoredRow]]:
    """Load PeTeR / RL-TPM caches for every DEBD dataset and ε (MNIST excluded)."""
    by_k: dict[int, list[ScoredRow]] = {k: [] for k in _EPSILONS}
    datasets = [d for d in discover_datasets(_EPSILONS[0]) if d not in _EXCLUDE]
    dims = {d: n_features(d) for d in datasets}

    for dataset in datasets:
        d = dims[dataset]
        for k in _EPSILONS:
            best = best_peter_params(dataset, k)
            if best is None:
                print(f"skip  {dataset}  k={k}: no PeTeR sweep winner", flush=True)
                continue
            lr, ratio, _src = best
            peter = load_eval_cache(peter_adv_path(dataset, k, lr, ratio))
            rltpm = load_eval_cache(rltpm_adv_path(dataset, k))
            if peter is None:
                print(f"skip  {dataset}  k={k}: missing PeTeR eval cache", flush=True)
                continue
            if rltpm is None:
                print(f"skip  {dataset}  k={k}: missing RL-TPM eval cache", flush=True)
                continue
            _po, peter_a, peter_r = peter
            _ro, rltpm_a, rltpm_r = rltpm
            by_k[k].append(
                (
                    dataset,
                    rltpm_r / d,
                    peter_r / d,
                    rltpm_a / d,
                    peter_a / d,
                    d,
                )
            )
    return by_k


def pct_diff(peter: np.ndarray, rltpm: np.ndarray) -> np.ndarray:
    """Percent improvement of PeTeR over RL-TPM: 100*(peter-rltpm)/|rltpm|."""
    return 100.0 * (peter - rltpm) / np.abs(rltpm)


def plot_panel(
    ax: plt.Axes,
    names: list[str],
    peter: np.ndarray,
    rltpm: np.ndarray,
    title: str,
) -> None:
    if len(names) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    pct = pct_diff(peter, rltpm)
    order = np.argsort(pct)  # worst → best so PeTeR wins rise to the top
    names_s = [names[i] for i in order]
    pct_s = pct[order]
    colors = [_WIN_COLOR if v > 0.0 else _LOSS_COLOR if v < 0.0 else "0.5" for v in pct_s]

    y = np.arange(len(names_s))
    ax.barh(y, pct_s, color=colors, edgecolor="0.25", linewidth=0.3, height=0.75)
    ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=10)
    ax.tick_params(axis="x", labelsize=7)

    pad = 0.08 * max(float(np.abs(pct_s).max()), 1e-9)
    ax.set_xlim(float(pct_s.min()) - pad, float(pct_s.max()) + pad)

    ax.set_title(title, fontsize=26, pad=2)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)


def _panel_arrays(
    rows: list[ScoredRow], kind: Literal["random", "greedy"]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [r[0] for r in rows]
    if kind == "random":
        rltpm = np.array([r[1] for r in rows], dtype=np.float64)
        peter = np.array([r[2] for r in rows], dtype=np.float64)
    else:
        rltpm = np.array([r[3] for r in rows], dtype=np.float64)
        peter = np.array([r[4] for r in rows], dtype=np.float64)
    return names, peter, rltpm


def _fill_row(
    axes: np.ndarray,
    by_k: dict[int, list[ScoredRow]],
    kind: Literal["random", "greedy"],
) -> None:
    for col, k in enumerate(_EPSILONS):
        names, peter, rltpm = _panel_arrays(by_k[k], kind)
        plot_panel(axes[col], names, peter, rltpm, rf"$\epsilon={k}$")


def build_figure(by_k: dict[int, list[ScoredRow]], kind: Kind) -> plt.Figure:
    if kind == "combined":
        fig = plt.figure(figsize=(11.5, 10.5), constrained_layout=True)
        subfigs = fig.subfigures(2, 1, hspace=0.08)
        for subfig, row_kind, row_title in (
            (subfigs[0], "random", _TITLE_RANDOM),
            (subfigs[1], "greedy", _TITLE_ADV),
        ):
            subfig.suptitle(row_title, fontsize=26)
            axes = subfig.subplots(1, 3)
            _fill_row(axes, by_k, row_kind)
        return fig

    # Half the combined height so each panel stays the same size.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.25), constrained_layout=True)
    row_kind: Literal["random", "greedy"] = "random" if kind == "random" else "greedy"
    fig.suptitle(_TITLE_RANDOM if row_kind == "random" else _TITLE_ADV, fontsize=26)
    _fill_row(axes, by_k, row_kind)
    return fig


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write figures/plot1.pdf (combined), plot1a.pdf (random), "
            "and plot1b.pdf (adversarial)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_OUT,
        help=f"Combined PDF path (default: {_OUT})",
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=_OUT_A,
        help=f"Random-corruptions PDF path (default: {_OUT_A})",
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=_OUT_B,
        help=f"Adversarial-corruptions PDF path (default: {_OUT_B})",
    )
    args = parser.parse_args()

    by_k = collect_all()
    save_figure(build_figure(by_k, "combined"), args.output)
    save_figure(build_figure(by_k, "random"), args.output_a)
    save_figure(build_figure(by_k, "greedy"), args.output_b)


if __name__ == "__main__":
    main()
