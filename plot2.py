"""Figure 2: Clean–robustness Pareto frontier (PeTeR vs RL-TPM vs MLE).

Two panels:

* Left:  shared random flips (``rand_mean_ll``)
* Right: each method's own greedy adversarial set (``own_adv_ll``)

For each robustification radius ε ∈ {1, 3, 5}:

* x = median clean degradation vs MLE-PC
      (MLE orig − method orig) / n_vars
* y = median robust improvement vs MLE-PC
      (method corrupt − MLE corrupt) / n_vars

Medians are over DEBD datasets (MNIST excluded). Error bars are
percentile bootstrap 95% CIs over datasets. Deltas are relative to the
non-robust MLE-PC baseline (not plotted). Better models sit toward the
upper-left.

Uses existing ``eval.py`` caches only. Writes ``figures/plot2.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval import (
    best_peter_params,
    load_eval_cache,
    mle_adv_path,
    peter_adv_path,
    rltpm_adv_path,
)
from peter import VALID_K
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "figures" / "plot2.pdf"
_EXCLUDE = frozenset({"mnist"})
_EPSILONS = list(VALID_K)  # 1, 3, 5
_N_BOOT = 10_000
_SEED = 0
_CI = (2.5, 97.5)

_PETER_COLOR = "#2a6f97"
_RLTPM_COLOR = "#c1121f"

# Manual ε-label offsets in points (dx, dy). Edit these to nudge overlapping labels.
# Keys: (panel, method, ε) with panel in {"random", "adversarial"},
# method in {"peter", "rltpm"}. Missing keys fall back to _DEFAULT_LABEL_OFFSET.
_DEFAULT_LABEL_OFFSET: tuple[float, float] = (6, 5)
_LABEL_OFFSETS: dict[tuple[str, str, int], tuple[float, float]] = {
    ("random", "peter", 1): (0, -25),
    ("random", "peter", 3): (10, -10),
    ("random", "peter", 5): (6, 5),
    ("random", "rltpm", 1): (6, -5),
    ("random", "rltpm", 3): (6, -15),
    ("random", "rltpm", 5): (6, 5),
    ("adversarial", "peter", 1): (6, -5),
    ("adversarial", "peter", 3): (6, -10),
    ("adversarial", "peter", 5): (-20, 15),
    ("adversarial", "rltpm", 1): (7, 3),
    ("adversarial", "rltpm", 3): (10, -5),
    ("adversarial", "rltpm", 5): (6, 5),
}

# Per-dataset deltas for one method at one ε: (clean_deg, rob_imp)
DatasetDeltas = tuple[float, float]


def n_features(dataset: str) -> int:
    """Number of binary features from the first row of the test split."""
    path = _ROOT / "original_datasets" / dataset / f"{dataset}.test.data"
    with path.open(encoding="utf-8") as f:
        line = f.readline()
    if not line.strip():
        raise FileNotFoundError(f"Empty test split: {path}")
    return line.count(",") + 1


def collect_deltas(
    *,
    random: bool,
) -> dict[str, dict[int, list[DatasetDeltas]]]:
    """Per-method, per-ε lists of (clean_deg, rob_imp) over DEBD datasets.

    ``clean_deg = (mle_orig - method_orig) / n`` (positive ⇒ worse on clean).
    ``rob_imp  = (method_pert - mle_pert) / n`` (positive ⇒ better on corrupt).
    """
    out: dict[str, dict[int, list[DatasetDeltas]]] = {
        "peter": {k: [] for k in _EPSILONS},
        "rltpm": {k: [] for k in _EPSILONS},
    }
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
            mle = load_eval_cache(mle_adv_path(dataset, k))
            peter = load_eval_cache(peter_adv_path(dataset, k, lr, ratio))
            rltpm = load_eval_cache(rltpm_adv_path(dataset, k))
            if mle is None:
                print(f"skip  {dataset}  k={k}: missing MLE eval cache", flush=True)
                continue
            if peter is None:
                print(f"skip  {dataset}  k={k}: missing PeTeR eval cache", flush=True)
                continue
            if rltpm is None:
                print(f"skip  {dataset}  k={k}: missing RL-TPM eval cache", flush=True)
                continue

            mle_o, mle_a, mle_r = mle
            peter_o, peter_a, peter_r = peter
            rltpm_o, rltpm_a, rltpm_r = rltpm
            mle_p = mle_r if random else mle_a
            peter_p = peter_r if random else peter_a
            rltpm_p = rltpm_r if random else rltpm_a

            out["peter"][k].append(
                ((mle_o - peter_o) / d, (peter_p - mle_p) / d)
            )
            out["rltpm"][k].append(
                ((mle_o - rltpm_o) / d, (rltpm_p - mle_p) / d)
            )
    return out


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    n_boot: int = _N_BOOT,
    seed: int = _SEED,
    ci: tuple[float, float] = _CI,
) -> tuple[float, float, float]:
    """Return (median, lo, hi) with percentile bootstrap CI over ``values``."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(values))
    if n == 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boots, ci)
    return med, float(lo), float(hi)


def aggregate(
    deltas: dict[str, dict[int, list[DatasetDeltas]]],
) -> dict[str, dict[int, tuple[float, float, float, float, float, float]]]:
    """Per method/ε: (x, x_lo, x_hi, y, y_lo, y_hi) from bootstrap medians."""
    agg: dict[str, dict[int, tuple[float, float, float, float, float, float]]] = {}
    for method, by_k in deltas.items():
        agg[method] = {}
        for k, rows in by_k.items():
            if not rows:
                continue
            xs = np.array([r[0] for r in rows], dtype=np.float64)
            ys = np.array([r[1] for r in rows], dtype=np.float64)
            # Independent seeds so x/y CIs are not artificially coupled.
            x_med, x_lo, x_hi = bootstrap_median_ci(xs, seed=_SEED + k)
            y_med, y_lo, y_hi = bootstrap_median_ci(ys, seed=_SEED + 1000 + k)
            agg[method][k] = (x_med, x_lo, x_hi, y_med, y_lo, y_hi)
    return agg


def plot_panel(
    ax: plt.Axes,
    agg: dict[str, dict[int, tuple[float, float, float, float, float, float]]],
    title: str,
    panel: str,
) -> None:
    styles = {
        "peter": {
            "color": _PETER_COLOR,
            "marker": "o",
            "label": "PeTeR",
        },
        "rltpm": {
            "color": _RLTPM_COLOR,
            "marker": "^",
            "label": "RL-TPM",
        },
    }

    for method, style in styles.items():
        by_k = agg.get(method, {})
        ks = [k for k in _EPSILONS if k in by_k]
        if not ks:
            continue
        xs = np.array([by_k[k][0] for k in ks], dtype=np.float64)
        x_lo = np.array([by_k[k][1] for k in ks], dtype=np.float64)
        x_hi = np.array([by_k[k][2] for k in ks], dtype=np.float64)
        ys = np.array([by_k[k][3] for k in ks], dtype=np.float64)
        y_lo = np.array([by_k[k][4] for k in ks], dtype=np.float64)
        y_hi = np.array([by_k[k][5] for k in ks], dtype=np.float64)

        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=1.4,
            alpha=0.85,
            zorder=2,
        )
        ax.errorbar(
            xs,
            ys,
            xerr=[xs - x_lo, x_hi - xs],
            yerr=[ys - y_lo, y_hi - ys],
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.3,
            zorder=3,
        )
        ax.scatter(
            xs,
            ys,
            s=110,
            marker=style["marker"],
            color=style["color"],
            edgecolors="0.15",
            linewidths=0.5,
            zorder=5,
            label=style["label"],
        )
        for k, x, y in zip(ks, xs, ys):
            dx, dy = _LABEL_OFFSETS.get((panel, method, k), _DEFAULT_LABEL_OFFSET)
            ax.annotate(
                rf"$\epsilon={k}$",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=24,
                color=style["color"],
            )

    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="0.6", linewidth=0.8, zorder=1)
    ax.set_xlabel("Clean degradation (median ΔLL / var)", fontsize=14)
    ax.set_ylabel("Robust improvement (median ΔLL / var)", fontsize=14)
    ax.set_title(title, fontsize=26, pad=14)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=12)


def build_figure() -> plt.Figure:
    deltas_rand = collect_deltas(random=True)
    deltas_adv = collect_deltas(random=False)
    agg_rand = aggregate(deltas_rand)
    agg_adv = aggregate(deltas_adv)

    for name, deltas in (("random", deltas_rand), ("adversarial", deltas_adv)):
        for method, by_k in deltas.items():
            for k, rows in by_k.items():
                print(f"{name}  {method}  eps={k}: n={len(rows)}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    plot_panel(axes[0], agg_rand, r"Random Corruptions", panel="random")
    plot_panel(axes[1], agg_adv, r"Adversarial Corruptions", panel="adversarial")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write figures/plot2.pdf (clean–robustness Pareto frontier)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_OUT,
        help=f"Output PDF path (default: {_OUT})",
    )
    args = parser.parse_args()

    fig = build_figure()
    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
