"""Evaluate best MNIST PeTeR (and MLE baseline) on corrupted MNIST.

Picks the hyperparameter winner with highest ``final_adv_test_ll`` (mean LL on
``sigma0.010/r0`` during tuning) from TPE and/or grid sweeps.  If
``results/mnist/k<k>/lr=..._ratio=.../circuit.json`` is missing, materializes it
via :func:`peter_mnist.run`.

Scores MLE (``mnist/hclt_mnist_blocksize4.json``) and PeTeR on:

* original test
* each sigma in ``{0.001, ..., 0.010}``: mean LL over ``r0.data`` … ``r9.data``

Writes a summary JSON under ``results/mnist/k<k>/.../eval_summary.json`` and a
dropoff plot ``mnist/eval_k<k>_dropoff.png`` (x-axis = sigma * 256).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sparc.nodes import CircuitNode

from best_sweep import best_from_grid, best_from_tpe
from peter import DEFAULT_ITERS, VALID_K, format_hyperparam_dir, run_output_dir
from peter_mnist import DATASET, circuit_path, original_test_path, run
from prepare_mnist_data import NUM_COPIES, SIGMAS, TUNE_SIGMA, corrupt_path, format_sigma
from robustify import mean_log_likelihood
from sweep_io import TPE_ROOT

_ROOT = Path(__file__).resolve().parent
_MNIST_DIR = _ROOT / "mnist"
_TUNE_LABEL = f"sigma{format_sigma(TUNE_SIGMA)}"
_SIGMA_AXIS_SCALE = 256.0


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


def mean_ll_over_sets(graph, sets: list[np.ndarray]) -> float:
    return float(np.mean([mean_log_likelihood(graph, rows) for rows in sets]))


def eval_on_all(
    circuit: Path,
    original: np.ndarray,
    corrupt_by_sigma: dict[float, list[np.ndarray]],
) -> dict[str, float]:
    graph = CircuitNode.load(circuit).compile()
    scores: dict[str, float] = {
        "orig_test_ll": mean_log_likelihood(graph, original),
    }
    for sigma, sets in corrupt_by_sigma.items():
        scores[f"sigma{format_sigma(sigma)}_mean_ll"] = mean_ll_over_sets(graph, sets)
        if abs(sigma - TUNE_SIGMA) < 1e-12:
            scores[f"{_TUNE_LABEL}_r0_ll"] = mean_log_likelihood(graph, sets[0])
    return scores


def expected_score_keys() -> list[str]:
    return ["orig_test_ll"] + [f"sigma{format_sigma(s)}_mean_ll" for s in SIGMAS]


def cache_is_current(payload: dict) -> bool:
    """True when cached mle/peter scores include every current sigma key."""
    keys = expected_score_keys()
    for side in ("mle", "peter"):
        scores = payload.get(side)
        if not isinstance(scores, dict):
            return False
        if any(k not in scores for k in keys):
            return False
    return True


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
    mle_ys = [mle_scores[f"sigma{format_sigma(s)}_mean_ll"] for s in SIGMAS]
    peter_ys = [peter_scores[f"sigma{format_sigma(s)}_mean_ll"] for s in SIGMAS]

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
) -> None:
    print_table(k, lr, ratio, source, mle_scores, peter_scores)
    plot_path = save_dropoff_plot(
        k, mle_scores, peter_scores, lr=lr, ratio=ratio
    )
    print(f"  wrote {plot_path.relative_to(_ROOT)}", flush=True)


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
    if summary_path.is_file() and not force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if cache_is_current(payload):
            print(f"  using cached {summary_path.relative_to(_ROOT)}", flush=True)
            report_scores(k, lr, ratio, source, payload["mle"], payload["peter"])
            print(f"done   mnist  k={k}", flush=True)
            return payload
        print(
            f"  stale cache (sigma keys changed) -> recomputing  "
            f"{summary_path.relative_to(_ROOT)}",
            flush=True,
        )

    mle_circuit = circuit_path()
    if not mle_circuit.is_file():
        raise FileNotFoundError(f"Missing MLE circuit: {mle_circuit}")

    peter_circuit = ensure_peter_circuit(k, lr, ratio, iters_for(k, source))
    original = load_binary_data(original_test_path())
    corrupt_by_sigma = {sigma: load_corrupt_sets(sigma) for sigma in SIGMAS}

    print("  scoring MLE...", flush=True)
    mle_scores = eval_on_all(mle_circuit, original, corrupt_by_sigma)
    print("  scoring PeTeR...", flush=True)
    peter_scores = eval_on_all(peter_circuit, original, corrupt_by_sigma)

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
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {summary_path.relative_to(_ROOT)}", flush=True)

    report_scores(k, lr, ratio, source, mle_scores, peter_scores)
    print(f"done   mnist  k={k}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate best MNIST PeTeR (by tune-sigma LL) and MLE on "
            "original + all corrupted sigma levels. Materializes the PeTeR "
            "circuit if missing."
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
