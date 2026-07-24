"""Run PeTeR with fixed algorithm settings and tunable LR / ratio.

This script is meant for hyperparameter sweeps.  Only ``--lr`` and ``--ratio``
vary between runs; everything else is held constant so differences in
likelihood curves and saved circuits reflect those two knobs alone.

For a given ``--dataset``, the adversarial evaluation set uses the same K as
the CW-ball radius (``--k`` replaces both ``k`` and ``dataset_k`` from
``robustify.py``).

Results are written under::

    results/<dataset>/k<k>/lr=<lr>_ratio=<ratio>/
        config.json       # full run metadata for reproducibility
        likelihood.png    # original vs adversarial test log-likelihood (success only)
        circuit.json      # robustified PC (gcw-circuit-v1, success only)
        metrics.json      # final scalar metrics (success only)
        error.json        # failure record with traceback (failed runs)

Callers (grid sweep / TPE tuner) may pass ``save_circuit=False``,
``save_plot=False``, ``quiet=True``, and a custom ``out_dir`` to persist only
metadata under ``sweeps/``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from sparc.nodes import CircuitNode

from robustify import (
    ETA_LAMBDA,
    resolve_circuit_path,
    resolve_eval_datasets,
    run_dro_ogda,
)

_ROOT = Path(__file__).resolve().parent
_RESULTS_ROOT = _ROOT / "results"

# Fixed algorithm settings (not exposed on the CLI).
THETA_NUM_SAMPLES = 100
WARM_START_ITERS = 100
EVAL_EVERY = 1
DETERMINISTIC = True

DEFAULT_LR = 1e-3
DEFAULT_RATIO = 3.0
DEFAULT_ITERS = 500
VALID_K = (1, 3, 5)


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce or compare a run."""

    dataset: str
    k: int
    lr: float
    ratio: float
    iters: int
    theta_num_samples: int
    warm_start_iters: int
    eval_every: int
    deterministic: bool
    eta_lambda: float
    circuit_source: str
    started_at: str


@dataclass
class RunMetrics:
    """Scalar outcomes for quick comparison across a hyperparameter grid."""

    final_lambda: float
    final_orig_test_ll: float
    final_adv_test_ll: float
    best_orig_test_ll: float
    best_adv_test_ll: float
    best_orig_iter: int
    best_adv_iter: int


@dataclass
class RunOutcome:
    """Result of a single PeTeR run (success or recorded failure)."""

    out_dir: Path
    status: Literal["ok", "failed"]
    metrics: RunMetrics | None = None
    error: str | None = None
    error_type: str | None = None


def metrics_path(out_dir: Path) -> Path:
    return out_dir / "metrics.json"


def error_path(out_dir: Path) -> Path:
    return out_dir / "error.json"


def is_run_finished(out_dir: Path) -> bool:
    """True when a prior run completed successfully or failed in a recorded way."""
    return metrics_path(out_dir).is_file() or error_path(out_dir).is_file()


def format_hyperparam_dir(lr: float, ratio: float) -> str:
    """Stable, filesystem-safe folder name for one (lr, ratio) pair."""
    return f"lr={lr:g}_ratio={ratio:g}"


def run_output_dir(dataset: str, k: int, lr: float, ratio: float) -> Path:
    return _RESULTS_ROOT / dataset / f"k{k}" / format_hyperparam_dir(lr, ratio)


class LikelihoodCurveRecorder:
    """Record eval points during training; optionally save a PNG at the end."""

    def __init__(
        self,
        dataset: str,
        k: int,
        lr: float,
        ratio: float,
        *,
        enable_plot: bool = True,
    ) -> None:
        self._enable_plot = enable_plot
        self._iters: list[int] = []
        self._orig_lls: list[float] = []
        self._adv_lls: list[float] = []
        self._plt = None
        self._fig = None
        self._ax = None
        self._line_orig = None
        self._line_adv = None

        if enable_plot:
            import matplotlib.pyplot as plt

            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(9, 5))
            (self._line_orig,) = self._ax.plot(
                [], [], label=f"original test ({dataset})", linewidth=1.5
            )
            (self._line_adv,) = self._ax.plot(
                [], [], label=f"adversarial K={k} ({dataset})", linewidth=1.5
            )
            self._ax.set(
                xlabel="iteration",
                ylabel="mean log-likelihood",
                title=f"{dataset}  k={k}  lr={lr:g}  ratio={ratio:g}",
            )
            self._ax.legend(loc="best")
            self._ax.grid(True, alpha=0.3)
            self._fig.tight_layout()

    def update(self, it: int, orig_ll: float, adv_ll: float) -> None:
        self._iters.append(it)
        self._orig_lls.append(orig_ll)
        self._adv_lls.append(adv_ll)
        if self._enable_plot and self._line_orig is not None and self._line_adv is not None:
            self._line_orig.set_data(self._iters, self._orig_lls)
            self._line_adv.set_data(self._iters, self._adv_lls)
            self._ax.relim()
            self._ax.autoscale_view()

    def summarize(self, final_lambda: float) -> RunMetrics:
        if not self._iters:
            raise RuntimeError("No evaluation points were recorded during training.")
        best_orig_idx = int(np.argmax(self._orig_lls))
        best_adv_idx = int(np.argmax(self._adv_lls))
        return RunMetrics(
            final_lambda=final_lambda,
            final_orig_test_ll=self._orig_lls[-1],
            final_adv_test_ll=self._adv_lls[-1],
            best_orig_test_ll=self._orig_lls[best_orig_idx],
            best_adv_test_ll=self._adv_lls[best_adv_idx],
            best_orig_iter=self._iters[best_orig_idx],
            best_adv_iter=self._iters[best_adv_idx],
        )

    def save(self, path: Path) -> None:
        if not self._enable_plot or self._fig is None or self._plt is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        self._plt.close(self._fig)

    def close(self) -> None:
        if self._enable_plot and self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_run_error(out_dir: Path, exc: BaseException) -> None:
    save_json(
        error_path(out_dir),
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _log(quiet: bool, *args, **kwargs) -> None:
    if not quiet:
        print(*args, **kwargs)


def run(
    dataset: str,
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
    """Execute one PeTeR run and persist artifacts.

    On numerical or other runtime failure, writes ``error.json`` into the run
    directory and returns ``RunOutcome(status="failed")`` instead of raising.

    Parameters
    ----------
    save_circuit:
        When True (default), write ``circuit.json`` (CLI / final runs).
    save_plot:
        When True (default), write ``likelihood.png``.
    quiet:
        When True, suppress console output (for sweep/tuner workers).
    out_dir:
        Override the default ``results/<dataset>/k<k>/lr=..._ratio=.../`` path.
    """
    if k not in VALID_K:
        raise ValueError(f"k must be one of {VALID_K}, got {k}")

    circuit_path = resolve_circuit_path(dataset)
    resolved_out = out_dir if out_dir is not None else run_output_dir(dataset, k, lr, ratio)
    resolved_out.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        dataset=dataset,
        k=k,
        lr=lr,
        ratio=ratio,
        iters=iters,
        theta_num_samples=THETA_NUM_SAMPLES,
        warm_start_iters=WARM_START_ITERS,
        eval_every=EVAL_EVERY,
        deterministic=DETERMINISTIC,
        eta_lambda=ETA_LAMBDA,
        circuit_source=str(circuit_path),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    save_json(resolved_out / "config.json", asdict(config))

    try:
        return _run_training(
            dataset=dataset,
            k=k,
            lr=lr,
            ratio=ratio,
            iters=iters,
            circuit_path=circuit_path,
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
    dataset: str,
    k: int,
    lr: float,
    ratio: float,
    iters: int,
    circuit_path: Path,
    out_dir: Path,
    save_circuit: bool,
    save_plot: bool,
    quiet: bool,
) -> RunOutcome:
    _log(quiet, f"dataset={dataset!r}  k={k}  lr={lr:g}  ratio={ratio:g}  iters={iters}")
    _log(quiet, f"loading circuit from {circuit_path}")
    p_hat = CircuitNode.load(circuit_path)
    _log(quiet, f"  nodes in scope: {len(p_hat.scope_as_list())}")

    _log(quiet, f"loading eval datasets for {dataset!r} (K={k})")
    original_data, adversarial_data = resolve_eval_datasets(dataset, k)
    _log(
        quiet,
        f"  original test: {len(original_data)} rows, "
        f"adversarial K={k}: {len(adversarial_data)} rows",
    )
    _log(quiet, f"results -> {out_dir.resolve()}")

    recorder = LikelihoodCurveRecorder(
        dataset, k, lr, ratio, enable_plot=save_plot
    )

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
        adversarial_data=adversarial_data,
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
        f"adv={metrics.final_adv_test_ll:.6f}",
    )
    return RunOutcome(out_dir=out_dir, status="ok", metrics=metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PeTeR with tunable learning rate and phi/theta ratio. "
            "Saves the likelihood curve and robustified circuit under results/."
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (basename of example_pcs/<dataset>.json, e.g. nips, accidents)",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        required=True,
        help="CW-ball radius and adversarial dataset K (same value for both)",
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

    outcome = run(
        dataset=args.dataset,
        k=args.k,
        lr=args.lr,
        ratio=args.ratio,
        iters=args.iters,
    )
    if outcome.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
