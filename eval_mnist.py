"""Evaluate best MNIST PeTeR (and MLE baseline) on corrupted MNIST.

Picks the hyperparameter winner with highest ``final_adv_test_ll`` (mean LL on
``sigma0.010/r0`` during tuning) from TPE and/or grid sweeps.  If
``results/mnist/k<k>/lr=..._ratio=.../circuit.json`` is missing, materializes it
via :func:`peter_mnist.run`.

Scores MLE (``mnist/hclt_mnist_blocksize4.json``) and PeTeR on:

* original test
* each sigma in ``{0.001, ..., 0.010}``: mean LL over ``r0.data`` … ``r9.data``
* FGSM sets from ``fgsm_mnist.py`` (both models on each set):
  - MLE FGSM — ``adversarial_datasets/K{k}/mnist.test.data``
  - PeTeR FGSM — ``results/mnist/k{k}/.../mnist_K{k}_peter.data``

Statistical testing (PeTeR vs MLE on corrupted sets)
----------------------------------------------------
For each sigma, both models are scored on the same ``NUM_COPIES`` independent
Gaussian-noise replicates.  The unit of observation is the *per-replicate mean
log-likelihood* (not a single pooled mean).  Paired differences are

    d_r = LL_PeTeR(r) - LL_MLE(r),  r = 0 .. NUM_COPIES-1

and we test H0: E[d] = 0 (two-sided) with:

1. Wilcoxon signed-rank (primary; nonparametric, matches ``eval.py``)
2. Paired Student t-test (secondary; assumes roughly normal d_r)

Because there are ``len(SIGMAS)`` sigma levels, raw p-values are adjusted with
Holm–Bonferroni (family-wise, separately for Wilcoxon and for t-tests).  A
sigma is marked significant at alpha=0.05 when the Holm-adjusted p-value is
below alpha.

Writes a summary JSON under ``results/mnist/k<k>/.../eval_summary.json`` (incl.
per-replicate LLs + significance) and a dropoff plot
``mnist/eval_k<k>_dropoff.png`` (x-axis = sigma * 256).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sparc.nodes import CircuitNode

from best_sweep import best_from_grid, best_from_tpe
from fgsm import default_output
from peter import DEFAULT_ITERS, VALID_K, format_hyperparam_dir, run_output_dir
from peter_mnist import DATASET, circuit_path, original_test_path, run
from prepare_mnist_data import NUM_COPIES, SIGMAS, TUNE_SIGMA, corrupt_path, format_sigma
from robustify import mean_log_likelihood
from sweep_io import TPE_ROOT

_ROOT = Path(__file__).resolve().parent
_MNIST_DIR = _ROOT / "mnist"
_TUNE_LABEL = f"sigma{format_sigma(TUNE_SIGMA)}"
_SIGMA_AXIS_SCALE = 256.0
_ALPHA = 0.05


def load_binary_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    data = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def best_peter_params(k: int) -> tuple[float, float, float, str] | None:
    """Return ``(lr, ratio, tune_sigma_ll, source)`` for the best MNIST winner."""
    candidates: list[tuple[float, float, float, str]] = []
    for ds, run_k, adv_ll, lr, ratio in best_from_tpe(k):
        if ds == DATASET and run_k == k:
            candidates.append((lr, ratio, adv_ll, "tpe"))
    for ds, run_k, adv_ll, _orig, lr, ratio in best_from_grid(k):
        if ds == DATASET and run_k == k:
            candidates.append((lr, ratio, adv_ll, "grid"))
    if not candidates:
        return None
    # Prefer higher tune-sigma LL; ties prefer TPE (same as eval.py).
    candidates.sort(key=lambda c: (c[2], c[3] == "tpe"), reverse=True)
    lr, ratio, adv_ll, source = candidates[0]
    return lr, ratio, adv_ll, source


def iters_for(k: int, source: str) -> int:
    if source == "tpe":
        path = TPE_ROOT / f"k{k}" / DATASET / "study_summary.json"
        if path.is_file():
            return int(json.loads(path.read_text(encoding="utf-8")).get("iters", DEFAULT_ITERS))
    if source == "grid":
        path = _ROOT / "sweeps" / "grid" / f"k{k}" / "sweep_summary_mnist.json"
        if path.is_file():
            return int(json.loads(path.read_text(encoding="utf-8")).get("iters", DEFAULT_ITERS))
    return DEFAULT_ITERS


def ensure_peter_circuit(k: int, lr: float, ratio: float, iters: int) -> Path:
    out_dir = run_output_dir(DATASET, k, lr, ratio)
    circuit = out_dir / "circuit.json"
    if circuit.is_file():
        print(f"  circuit exists: {circuit.relative_to(_ROOT)}", flush=True)
        return circuit
    print(
        f"  materializing PeTeR circuit  k={k}  lr={lr:g}  ratio={ratio:g}  iters={iters}",
        flush=True,
    )
    outcome = run(
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
        raise RuntimeError(outcome.error or "peter_mnist run failed")
    print(f"  saved {circuit.relative_to(_ROOT)}", flush=True)
    return circuit


def load_corrupt_sets(sigma: float) -> list[np.ndarray]:
    paths = [corrupt_path(sigma, r) for r in range(NUM_COPIES)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing corrupted MNIST sets (run prepare_mnist_data.py first):\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    return [load_binary_data(p) for p in paths]


def sigma_label(sigma: float) -> str:
    return f"sigma{format_sigma(sigma)}"


def peter_fgsm_path(out_dir: Path, k: int) -> Path:
    return out_dir / f"mnist_K{k}_peter.data"


def fgsm_score_keys() -> list[str]:
    return [
        "mle_on_mle_fgsm",
        "peter_on_mle_fgsm",
        "mle_on_peter_fgsm",
        "peter_on_peter_fgsm",
    ]


def score_fgsm(
    mle_circuit: Path,
    peter_circuit: Path,
    *,
    k: int,
    out_dir: Path,
) -> dict[str, Any] | None:
    """Mean LL of MLE and PeTeR on each model's FGSM set (from fgsm_mnist)."""
    mle_adv_path = default_output(k)
    peter_adv_path = peter_fgsm_path(out_dir, k)
    missing = [p for p in (mle_adv_path, peter_adv_path) if not p.is_file()]
    if missing:
        print(
            "  FGSM datasets missing (run fgsm_mnist.py first):\n"
            + "\n".join(f"    {p}" for p in missing),
            flush=True,
        )
        return None

    mle_adv = load_binary_data(mle_adv_path)
    peter_adv = load_binary_data(peter_adv_path)
    print(
        f"  scoring FGSM  mle_n={len(mle_adv)}  peter_n={len(peter_adv)}  K={k}",
        flush=True,
    )
    mle_graph = CircuitNode.load(mle_circuit).compile()
    peter_graph = CircuitNode.load(peter_circuit).compile()
    return {
        "k": k,
        "mle_fgsm_path": str(mle_adv_path),
        "peter_fgsm_path": str(peter_adv_path),
        "n_mle_fgsm": int(len(mle_adv)),
        "n_peter_fgsm": int(len(peter_adv)),
        "mle_on_mle_fgsm": mean_log_likelihood(mle_graph, mle_adv),
        "peter_on_mle_fgsm": mean_log_likelihood(peter_graph, mle_adv),
        "mle_on_peter_fgsm": mean_log_likelihood(mle_graph, peter_adv),
        "peter_on_peter_fgsm": mean_log_likelihood(peter_graph, peter_adv),
    }


def eval_on_all(
    circuit: Path,
    original: np.ndarray,
    corrupt_by_sigma: dict[float, list[np.ndarray]],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Return (mean scores, per-sigma replicate mean LLs)."""
    graph = CircuitNode.load(circuit).compile()
    scores: dict[str, float] = {
        "orig_test_ll": mean_log_likelihood(graph, original),
    }
    replicate_ll: dict[str, list[float]] = {}
    for sigma, sets in corrupt_by_sigma.items():
        label = sigma_label(sigma)
        lls = [mean_log_likelihood(graph, rows) for rows in sets]
        replicate_ll[label] = [float(x) for x in lls]
        scores[f"{label}_mean_ll"] = float(np.mean(lls))
        if abs(sigma - TUNE_SIGMA) < 1e-12:
            scores[f"{_TUNE_LABEL}_r0_ll"] = float(lls[0])
    return scores, replicate_ll


def expected_score_keys() -> list[str]:
    return ["orig_test_ll"] + [f"{sigma_label(s)}_mean_ll" for s in SIGMAS]


def expected_replicate_labels() -> list[str]:
    return [sigma_label(s) for s in SIGMAS]


def holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Holm–Bonferroni adjusted p-values (None entries stay None)."""
    valid = [(i, float(p)) for i, p in enumerate(pvalues) if p is not None]
    adjusted: list[float | None] = [None] * len(pvalues)
    if not valid:
        return adjusted
    m = len(valid)
    order = sorted(valid, key=lambda ip: ip[1])
    running = 0.0
    for rank, (idx, p) in enumerate(order):
        # rank 0 => multiplier m; rank m-1 => multiplier 1
        raw = min(1.0, (m - rank) * p)
        running = max(running, raw)
        adjusted[idx] = running
    return adjusted


def _safe_wilcoxon(peter: np.ndarray, mle: np.ndarray) -> tuple[float | None, float | None]:
    diff = peter - mle
    if int(np.sum(diff != 0)) < 1:
        return None, None
    try:
        res = wilcoxon(peter, mle, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None, None
    return float(res.statistic), float(res.pvalue)


def _safe_ttest(peter: np.ndarray, mle: np.ndarray) -> tuple[float | None, float | None]:
    if len(peter) < 2:
        return None, None
    if float(np.std(peter - mle)) == 0.0:
        return None, None
    res = ttest_rel(peter, mle, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def paired_sigma_tests(
    mle_replicate_ll: dict[str, list[float]],
    peter_replicate_ll: dict[str, list[float]],
    *,
    alpha: float = _ALPHA,
) -> dict[str, Any]:
    """Paired Wilcoxon + t-tests per sigma; Holm correction across sigmas."""
    labels = expected_replicate_labels()
    per_sigma: list[dict[str, Any]] = []
    wilcox_ps: list[float | None] = []
    ttest_ps: list[float | None] = []

    for label in labels:
        mle = np.asarray(mle_replicate_ll[label], dtype=np.float64)
        peter = np.asarray(peter_replicate_ll[label], dtype=np.float64)
        if mle.shape != peter.shape or mle.size != NUM_COPIES:
            raise ValueError(
                f"Expected {NUM_COPIES} paired replicate LLs for {label}, "
                f"got mle={mle.shape} peter={peter.shape}"
            )
        diff = peter - mle
        w_stat, w_p = _safe_wilcoxon(peter, mle)
        t_stat, t_p = _safe_ttest(peter, mle)
        wilcox_ps.append(w_p)
        ttest_ps.append(t_p)
        per_sigma.append(
            {
                "sigma": label,
                "n": int(mle.size),
                "mle_mean": float(mle.mean()),
                "peter_mean": float(peter.mean()),
                "mean_diff": float(diff.mean()),
                "peter_wins": int(np.sum(diff > 0)),
                "mle_wins": int(np.sum(diff < 0)),
                "ties": int(np.sum(diff == 0)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "ttest_stat": t_stat,
                "ttest_p": t_p,
            }
        )

    wilcox_holm = holm_adjust(wilcox_ps)
    ttest_holm = holm_adjust(ttest_ps)
    by_sigma: dict[str, Any] = {}
    for row, w_h, t_h in zip(per_sigma, wilcox_holm, ttest_holm):
        row["wilcoxon_p_holm"] = w_h
        row["ttest_p_holm"] = t_h
        row["wilcoxon_significant"] = w_h is not None and w_h < alpha
        row["ttest_significant"] = t_h is not None and t_h < alpha
        by_sigma[row["sigma"]] = row

    return {
        "unit": "per-replicate mean log-likelihood",
        "n_replicates": NUM_COPIES,
        "alternative": "two-sided",
        "alpha": alpha,
        "multiple_testing": "holm",
        "primary_test": "wilcoxon_signed_rank",
        "secondary_test": "paired_ttest",
        "by_sigma": by_sigma,
    }


def sigma_cache_is_current(payload: dict) -> bool:
    """True when cached scores + per-replicate LLs cover every current sigma."""
    keys = expected_score_keys()
    labels = expected_replicate_labels()
    for side in ("mle", "peter"):
        scores = payload.get(side)
        if not isinstance(scores, dict):
            return False
        if any(k not in scores for k in keys):
            return False
    for side in ("mle_replicate_ll", "peter_replicate_ll"):
        reps = payload.get(side)
        if not isinstance(reps, dict):
            return False
        for label in labels:
            vals = reps.get(label)
            if not isinstance(vals, list) or len(vals) != NUM_COPIES:
                return False
    return True


def fgsm_cache_is_current(payload: dict) -> bool:
    fgsm = payload.get("fgsm")
    if not isinstance(fgsm, dict):
        return False
    return all(key in fgsm for key in fgsm_score_keys())


def cache_is_current(payload: dict) -> bool:
    return sigma_cache_is_current(payload) and fgsm_cache_is_current(payload)


def print_table(
    k: int,
    lr: float,
    ratio: float,
    source: str,
    mle_scores: dict[str, float],
    peter_scores: dict[str, float],
) -> None:
    print(
        f"\nmnist  k={k}  peter={source}  lr={lr:g}  ratio={ratio:g}  "
        f"dir={format_hyperparam_dir(lr, ratio)}"
    )
    headers = ["set", "mle-pc", "peter", "delta"]
    print(f"  {headers[0]:<22}  {headers[1]:>12}  {headers[2]:>12}  {headers[3]:>10}")

    for key in expected_score_keys():
        mle = mle_scores[key]
        peter = peter_scores[key]
        label = "original" if key == "orig_test_ll" else key.removesuffix("_mean_ll")
        print(f"  {label:<22}  {mle:12.4f}  {peter:12.4f}  {peter - mle:+10.4f}")


def print_fgsm(fgsm: dict[str, Any] | None) -> None:
    if fgsm is None:
        print("\n=== FGSM ===\n  (skipped — missing datasets; run fgsm_mnist.py)", flush=True)
        return
    print(
        f"\n=== FGSM K={fgsm['k']}  "
        f"(mle_n={fgsm['n_mle_fgsm']}  peter_n={fgsm['n_peter_fgsm']}) ==="
    )
    headers = ["set", "mle-pc", "peter", "delta"]
    print(f"  {headers[0]:<22}  {headers[1]:>12}  {headers[2]:>12}  {headers[3]:>10}")
    rows = [
        ("fgsm_mle", fgsm["mle_on_mle_fgsm"], fgsm["peter_on_mle_fgsm"]),
        ("fgsm_peter", fgsm["mle_on_peter_fgsm"], fgsm["peter_on_peter_fgsm"]),
    ]
    for label, mle, peter in rows:
        print(f"  {label:<22}  {mle:12.4f}  {peter:12.4f}  {peter - mle:+10.4f}")
    print(
        f"  own_adv:  mle={fgsm['mle_on_mle_fgsm']:.4f}  "
        f"peter={fgsm['peter_on_peter_fgsm']:.4f}  "
        f"delta={fgsm['peter_on_peter_fgsm'] - fgsm['mle_on_mle_fgsm']:+.4f}"
    )


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{p:.4g}"


def print_significance(significance: dict[str, Any]) -> None:
    alpha = significance["alpha"]
    print(
        f"\n=== paired tests peter vs mle-pc "
        f"(n={significance['n_replicates']} replicates / sigma, "
        f"two-sided, Holm alpha={alpha}) ==="
    )
    print(
        f"  {'sigma':<12}  {'mean_d':>9}  {'W':>6}  {'p_W':>9}  {'p_W_h':>9}  "
        f"{'t':>8}  {'p_t':>9}  {'p_t_h':>9}  {'sig_W':>5}"
    )
    for label in expected_replicate_labels():
        row = significance["by_sigma"][label]
        sig = "yes" if row["wilcoxon_significant"] else "no"
        w_stat = "n/a" if row["wilcoxon_stat"] is None else f"{row['wilcoxon_stat']:.4g}"
        t_stat = "n/a" if row["ttest_stat"] is None else f"{row['ttest_stat']:.4g}"
        print(
            f"  {label:<12}  {row['mean_diff']:+9.4f}  {w_stat:>6}  "
            f"{_fmt_p(row['wilcoxon_p']):>9}  {_fmt_p(row['wilcoxon_p_holm']):>9}  "
            f"{t_stat:>8}  {_fmt_p(row['ttest_p']):>9}  {_fmt_p(row['ttest_p_holm']):>9}  "
            f"{sig:>5}"
        )
        print(
            f"    peter_wins={row['peter_wins']}  mle_wins={row['mle_wins']}  "
            f"ties={row['ties']}  t_sig={'yes' if row['ttest_significant'] else 'no'}"
        )


def dropoff_plot_path(k: int) -> Path:
    return _MNIST_DIR / f"eval_k{k}_dropoff.png"


def save_dropoff_plot(
    k: int,
    mle_scores: dict[str, float],
    peter_scores: dict[str, float],
    *,
    lr: float,
    ratio: float,
) -> Path:
    """Plot mean LL vs sigma*256 for MLE and PeTeR; save under mnist/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    xs = [sigma * _SIGMA_AXIS_SCALE for sigma in SIGMAS]
    mle_ys = [mle_scores[f"{sigma_label(s)}_mean_ll"] for s in SIGMAS]
    peter_ys = [peter_scores[f"{sigma_label(s)}_mean_ll"] for s in SIGMAS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, mle_ys, marker="o", linewidth=1.5, label="mle-pc")
    ax.plot(xs, peter_ys, marker="o", linewidth=1.5, label="peter")
    ax.set(
        xlabel=r"$\sigma \times 256$",
        ylabel="mean log-likelihood",
        title=(
            f"MNIST corruption dropoff  k={k}  "
            f"lr={lr:g}  ratio={ratio:g}"
        ),
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = dropoff_plot_path(k)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def report_scores(
    k: int,
    lr: float,
    ratio: float,
    source: str,
    mle_scores: dict[str, float],
    peter_scores: dict[str, float],
    significance: dict[str, Any],
    fgsm: dict[str, Any] | None,
) -> None:
    print_table(k, lr, ratio, source, mle_scores, peter_scores)
    print_fgsm(fgsm)
    print_significance(significance)
    plot_path = save_dropoff_plot(
        k, mle_scores, peter_scores, lr=lr, ratio=ratio
    )
    print(f"  wrote {plot_path.relative_to(_ROOT)}", flush=True)


def ensure_significance(payload: dict) -> dict[str, Any]:
    """Compute or refresh significance block from cached replicate LLs."""
    significance = paired_sigma_tests(
        payload["mle_replicate_ll"],
        payload["peter_replicate_ll"],
    )
    payload["significance"] = significance
    return significance


def eval_k(k: int, *, force: bool = False) -> dict | None:
    print(f"start  mnist  k={k}", flush=True)
    best = best_peter_params(k)
    if best is None:
        print(f"done   mnist  k={k}  SKIP (no sweep/tpe winner)", flush=True)
        return None

    lr, ratio, tune_ll, source = best
    print(
        f"  best from {source}: lr={lr:g}  ratio={ratio:g}  "
        f"tune_{_TUNE_LABEL}_ll={tune_ll:.6f}",
        flush=True,
    )

    out_dir = run_output_dir(DATASET, k, lr, ratio)
    summary_path = out_dir / "eval_summary.json"
    mle_circuit = circuit_path()
    if not mle_circuit.is_file():
        raise FileNotFoundError(f"Missing MLE circuit: {mle_circuit}")
    peter_circuit = ensure_peter_circuit(k, lr, ratio, iters_for(k, source))

    if summary_path.is_file() and not force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if sigma_cache_is_current(payload):
            print(f"  using cached sigma scores {summary_path.relative_to(_ROOT)}", flush=True)
            significance = ensure_significance(payload)
            if not fgsm_cache_is_current(payload):
                payload["fgsm"] = score_fgsm(
                    mle_circuit, peter_circuit, k=k, out_dir=out_dir
                )
            summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            report_scores(
                k,
                lr,
                ratio,
                source,
                payload["mle"],
                payload["peter"],
                significance,
                payload.get("fgsm"),
            )
            print(f"done   mnist  k={k}", flush=True)
            return payload
        print(
            f"  stale cache (need per-replicate LLs / sigma keys) -> recomputing  "
            f"{summary_path.relative_to(_ROOT)}",
            flush=True,
        )

    original = load_binary_data(original_test_path())
    corrupt_by_sigma = {sigma: load_corrupt_sets(sigma) for sigma in SIGMAS}

    print("  scoring MLE...", flush=True)
    mle_scores, mle_replicate_ll = eval_on_all(mle_circuit, original, corrupt_by_sigma)
    print("  scoring PeTeR...", flush=True)
    peter_scores, peter_replicate_ll = eval_on_all(
        peter_circuit, original, corrupt_by_sigma
    )

    significance = paired_sigma_tests(mle_replicate_ll, peter_replicate_ll)
    fgsm = score_fgsm(mle_circuit, peter_circuit, k=k, out_dir=out_dir)

    payload = {
        "dataset": DATASET,
        "k": k,
        "source": source,
        "lr": lr,
        "ratio": ratio,
        "tune_final_adv_test_ll": tune_ll,
        "mle_circuit": str(mle_circuit),
        "peter_circuit": str(peter_circuit),
        "sigmas": list(SIGMAS),
        "mle": mle_scores,
        "peter": peter_scores,
        "mle_replicate_ll": mle_replicate_ll,
        "peter_replicate_ll": peter_replicate_ll,
        "significance": significance,
        "fgsm": fgsm,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {summary_path.relative_to(_ROOT)}", flush=True)

    report_scores(
        k, lr, ratio, source, mle_scores, peter_scores, significance, fgsm
    )
    print(f"done   mnist  k={k}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate best MNIST PeTeR (by tune-sigma LL) and MLE on "
            "original + all corrupted sigma levels + FGSM sets from "
            "fgsm_mnist.py. Materializes the PeTeR circuit if missing. "
            "Reports paired Wilcoxon / t-tests over the 10 corruption "
            "replicates per sigma (Holm-corrected)."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"Only this CW-ball K (default: all of {list(VALID_K)})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when eval_summary.json already exists",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    ks = [args.k] if args.k is not None else list(VALID_K)
    print(f"eval_mnist: ks={ks}  force={args.force}", flush=True)

    results: list[dict] = []
    for k in ks:
        try:
            payload = eval_k(k, force=args.force)
        except Exception as exc:
            print(f"done   mnist  k={k}  SKIP ({exc})", flush=True)
            continue
        if payload is not None:
            results.append(payload)

    if not results:
        raise SystemExit("No MNIST evaluations completed.")


if __name__ == "__main__":
    main()
