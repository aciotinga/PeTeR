"""Run PeTeR on the MNIST HCLT (single-run API for CLI and sweeps).

Mirrors :mod:`peter` but loads ``mnist/hclt_mnist_blocksize4.json`` and
evaluates on::

    original_datasets/mnist/mnist.test.data
    corrupted_datasets/mnist/sigma0.1/r0.data

``RunMetrics.final_adv_test_ll`` / ``best_adv_test_ll`` store mean log-likelihood
on the sigma0.1/r0 corrupted set (the tuning objective).

Default full-run artifacts land under::

    results/mnist/k<k>/lr=<lr>_ratio=<ratio>/
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sparc.nodes import CircuitNode

from peter import (
    DEFAULT_ITERS,
    DEFAULT_LR,
    DEFAULT_RATIO,
    DETERMINISTIC,
    EVAL_EVERY,
    THETA_NUM_SAMPLES,
    VALID_K,
    WARM_START_ITERS,
    LikelihoodCurveRecorder,
    RunConfig,
    RunOutcome,
    error_path,
    metrics_path,
    run_output_dir,
    save_json,
    save_run_error,
)
from prepare_mnist_data import corrupt_path
from robustify import ETA_LAMBDA, run_dro_ogda

_ROOT = Path(__file__).resolve().parent
DATASET = "mnist"
_CIRCUIT = _ROOT / "mnist" / "hclt_mnist_blocksize4.json"
_ORIG_TEST = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"
_CORRUPT_EVAL = corrupt_path(0.1, 0)


def circuit_path() -> Path:
    return _CIRCUIT


def original_test_path() -> Path:
    return _ORIG_TEST


def corrupt_eval_path() -> Path:
    return _CORRUPT_EVAL


def require_mnist_inputs() -> None:
    missing = [p for p in (_CIRCUIT, _ORIG_TEST, _CORRUPT_EVAL) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "MNIST inputs missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nTrain/export the circuit and run prepare_mnist_data.py first."
        )


def load_eval_datasets() -> tuple:
    import numpy as np

    require_mnist_inputs()
    original = np.loadtxt(_ORIG_TEST, delimiter=",", dtype=np.int32)
    corrupt = np.loadtxt(_CORRUPT_EVAL, delimiter=",", dtype=np.int32)
    if original.ndim == 1:
        original = original.reshape(1, -1)
    if corrupt.ndim == 1:
        corrupt = corrupt.reshape(1, -1)
    if original.shape != corrupt.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs corrupt {corrupt.shape}. "
            "Re-run prepare_mnist_data.py so both use the same number of rows."
        )
    return original, corrupt


def _log(quiet: bool, *args, **kwargs) -> None:
    if not quiet:
        print(*args, **kwargs)


def run(
    k: int,
    lr: float,
    ratio: float,
    iters: int,
    *,
    save_circuit: bool = True,
    save_plot: bool = True,
    quiet: bool = False,
    out_dir: Path | None = None,
) -> RunOutcome:
    """Execute one MNIST PeTeR run and persist artifacts.

    On failure, writes ``error.json`` and returns ``status="failed"`` (does not raise).
    """
    if k not in VALID_K:
        raise ValueError(f"k must be one of {VALID_K}, got {k}")

    require_mnist_inputs()
    resolved_out = out_dir if out_dir is not None else run_output_dir(DATASET, k, lr, ratio)
    resolved_out.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        dataset=DATASET,
        k=k,
        lr=lr,
        ratio=ratio,
        iters=iters,
        theta_num_samples=THETA_NUM_SAMPLES,
        warm_start_iters=WARM_START_ITERS,
        eval_every=EVAL_EVERY,
        deterministic=DETERMINISTIC,
        eta_lambda=ETA_LAMBDA,
        circuit_source=str(_CIRCUIT.resolve()),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    save_json(resolved_out / "config.json", asdict(config))

    try:
        return _run_training(
            k=k,
            lr=lr,
            ratio=ratio,
            iters=iters,
            out_dir=resolved_out,
            save_circuit=save_circuit,
            save_plot=save_plot,
            quiet=quiet,
        )
    except Exception as exc:
        save_run_error(resolved_out, exc)
        _log(quiet, f"\nrun failed: {type(exc).__name__}: {exc}")
        _log(quiet, f"error details -> {error_path(resolved_out).resolve()}")
        return RunOutcome(
            out_dir=resolved_out,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )


def _run_training(
    *,
    k: int,
    lr: float,
    ratio: float,
    iters: int,
    out_dir: Path,
    save_circuit: bool,
    save_plot: bool,
    quiet: bool,
) -> RunOutcome:
    _log(quiet, f"dataset={DATASET!r}  k={k}  lr={lr:g}  ratio={ratio:g}  iters={iters}")
    _log(quiet, f"loading circuit from {_CIRCUIT}")
    p_hat = CircuitNode.load(_CIRCUIT)
    _log(quiet, f"  nodes in scope: {len(p_hat.scope_as_list())}")

    _log(quiet, "loading eval datasets (original + sigma0.1/r0)")
    original_data, corrupt_data = load_eval_datasets()
    _log(
        quiet,
        f"  original test: {len(original_data)} rows, "
        f"corrupt sigma0.1/r0: {len(corrupt_data)} rows",
    )
    _log(quiet, f"results -> {out_dir.resolve()}")

    # LikelihoodCurveRecorder's "adv" series is the corrupt LL objective.
    recorder = LikelihoodCurveRecorder(DATASET, k, lr, ratio, enable_plot=save_plot)

    p_theta, final_lambda = run_dro_ogda(
        p_hat,
        k=float(k),
        num_iters=iters,
        lr=lr,
        ratio=ratio,
        eta_lambda=ETA_LAMBDA,
        warm_start_iters=WARM_START_ITERS,
        theta_num_samples=THETA_NUM_SAMPLES,
        deterministic=DETERMINISTIC,
        eval_every=EVAL_EVERY,
        original_data=original_data,
        adversarial_data=corrupt_data,
        plotter=recorder,
        quiet=quiet,
    )

    if save_circuit:
        circuit_out = out_dir / "circuit.json"
        p_theta.save(circuit_out)
        _log(quiet, f"\nsaved circuit to {circuit_out.resolve()}")

    if save_plot:
        chart_out = out_dir / "likelihood.png"
        recorder.save(chart_out)
        _log(quiet, f"saved chart to {chart_out.resolve()}")
    else:
        recorder.close()

    metrics = recorder.summarize(final_lambda)
    save_json(metrics_path(out_dir), asdict(metrics))

    _log(quiet, f"\nfinal lambda={final_lambda:.4f}")
    _log(
        quiet,
        f"final test LL: orig={metrics.final_orig_test_ll:.6f}  "
        f"corrupt(sigma0.1/r0)={metrics.final_adv_test_ll:.6f}",
    )
    return RunOutcome(out_dir=out_dir, status="ok", metrics=metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PeTeR on MNIST with tunable lr/ratio. "
            "Eval target is mean LL on corrupted sigma0.1/r0. "
            "Saves under results/mnist/."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        required=True,
        help="CW-ball radius",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Theta learning rate eta_theta (default: {DEFAULT_LR:g})",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Phi/theta LR ratio; eta_phi = lr * ratio (default: {DEFAULT_RATIO:g})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"OGDA iterations after warm start (default: {DEFAULT_ITERS})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    if args.iters < 1:
        raise SystemExit("--iters must be at least 1")

    outcome = run(k=args.k, lr=args.lr, ratio=args.ratio, iters=args.iters)
    if outcome.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
