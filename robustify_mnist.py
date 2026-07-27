"""Robustify the MNIST HCLT with PeTeR and live LL plots.

Loads ``mnist/hclt_mnist_blocksize4.json``, runs DRO-OGDA with tunable
``--k``, ``--lr``, ``--ratio``, ``--iters``, etc., and evaluates every
iteration on:

* ``original_datasets/mnist/mnist.test.data``
* ``corrupted_datasets/mnist/sigma0.010/r0.data``

A live matplotlib window tracks both mean log-likelihoods.  The robustified
circuit (and a final PNG of the curve) are written under ``mnist/``.
Checkpoints are saved every ``--checkpoint-every`` iterations (default 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sparc.nodes import CircuitNode

from peter import DEFAULT_ITERS, DEFAULT_LR, DEFAULT_RATIO, WARM_START_ITERS
from prepare_mnist_data import TUNE_SIGMA, corrupt_path, format_sigma
from robustify import (
    ETA_LAMBDA,
    THETA_NUM_SAMPLES,
    mean_log_likelihood,
    run_dro_ogda,
)

_ROOT = Path(__file__).resolve().parent
_MNIST_DIR = _ROOT / "mnist"
_DEFAULT_CIRCUIT = _MNIST_DIR / "hclt_mnist_blocksize4.json"
_ORIG_TEST = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"
_CORRUPT_EVAL = corrupt_path(TUNE_SIGMA, 0)
_DEFAULT_CHECKPOINT_EVERY = 5
_TUNE_SIGMA_LABEL = f"sigma{format_sigma(TUNE_SIGMA)}"


class MnistLiveLikelihoodPlot:
    """Live plot of original vs tune-sigma corrupted mean log-likelihood."""

    def __init__(self, *, k: float, lr: float, ratio: float) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(
                "Live plotting requires matplotlib. Install it with: pip install matplotlib"
            ) from exc

        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        (self._line_orig,) = self.ax.plot([], [], label="original test", linewidth=1.5)
        (self._line_corrupt,) = self.ax.plot(
            [], [], label=f"corrupted {_TUNE_SIGMA_LABEL} / r0", linewidth=1.5
        )
        self.ax.set(
            xlabel="iteration",
            ylabel="mean log-likelihood",
            title=f"MNIST PeTeR  k={k:g}  lr={lr:g}  ratio={ratio:g}",
        )
        self.ax.legend(loc="best")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self._iters: list[int] = []
        self._orig_lls: list[float] = []
        self._corrupt_lls: list[float] = []

    def update(self, it: int, orig_ll: float, corrupt_ll: float) -> None:
        self._iters.append(it)
        self._orig_lls.append(orig_ll)
        self._corrupt_lls.append(corrupt_ll)
        self._line_orig.set_data(self._iters, self._orig_lls)
        self._line_corrupt.set_data(self._iters, self._corrupt_lls)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._plt.pause(0.05)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def hold_open(self) -> None:
        self._plt.ioff()
        self._plt.show()


def load_binary_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def default_output_path(k: float, lr: float, ratio: float) -> Path:
    return _MNIST_DIR / f"hclt_mnist_robust_k{k:g}_lr={lr:g}_ratio={ratio:g}.json"


def checkpoint_path(out_path: Path, it: int) -> Path:
    return out_path.with_name(f"{out_path.stem}_iter{it:04d}{out_path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Robustify the MNIST circuit with PeTeR; live-plot original vs "
            f"{_TUNE_SIGMA_LABEL}/r0 log-likelihood; save under mnist/."
        ),
    )
    p.add_argument(
        "--circuit",
        type=Path,
        default=_DEFAULT_CIRCUIT,
        help=f"Input circuit JSON (default: {_DEFAULT_CIRCUIT.relative_to(_ROOT)})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output circuit path (default: mnist/hclt_mnist_robust_k...json)",
    )
    p.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="CW-ball radius (default: 1)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Theta learning rate eta_theta (default: {DEFAULT_LR:g})",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Phi/theta LR ratio; eta_phi = lr * ratio (default: {DEFAULT_RATIO:g})",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"OGDA iterations after warm start (default: {DEFAULT_ITERS})",
    )
    p.add_argument(
        "--warm-start-iters",
        type=int,
        default=WARM_START_ITERS,
        help=f"Q-only iterations before theta updates (default: {WARM_START_ITERS})",
    )
    p.add_argument(
        "--theta-num-samples",
        type=int,
        default=THETA_NUM_SAMPLES,
        help=f"Samples from Q_phi per theta update (default: {THETA_NUM_SAMPLES})",
    )
    p.add_argument(
        "--eta-lambda",
        type=float,
        default=ETA_LAMBDA,
        help=f"Dual (lambda) step size (default: {ETA_LAMBDA:g})",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=1,
        metavar="N",
        help="Eval / update plot every N iterations (default: 1)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=_DEFAULT_CHECKPOINT_EVERY,
        metavar="N",
        help=(
            f"Save a circuit checkpoint every N OGDA iterations "
            f"(default: {_DEFAULT_CHECKPOINT_EVERY}; 0 disables)"
        ),
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Seed theta sampling by iteration (default: on)",
    )
    p.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="Disable deterministic theta sampling",
    )
    p.add_argument(
        "--no-plot-hold",
        action="store_true",
        help="Close the plot window immediately after training (do not block)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.k < 0:
        raise SystemExit("--k must be non-negative")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    if args.iters < 1:
        raise SystemExit("--iters must be at least 1")
    if args.warm_start_iters < 0:
        raise SystemExit("--warm-start-iters must be non-negative")
    if args.theta_num_samples < 1:
        raise SystemExit("--theta-num-samples must be at least 1")
    if args.eta_lambda <= 0:
        raise SystemExit("--eta-lambda must be positive")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be at least 1")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be non-negative")

    circuit_path = args.circuit.resolve()
    if not circuit_path.is_file():
        raise SystemExit(f"Circuit not found: {circuit_path}")

    out_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(args.k, args.lr, args.ratio)
    )
    plot_path = out_path.with_suffix(".likelihood.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {circuit_path.name} from {circuit_path.parent}")
    p_hat = CircuitNode.load(circuit_path)
    print(f"  nodes in scope: {len(p_hat.scope_as_list())}")

    print("loading eval datasets:")
    print(f"  original:  {_ORIG_TEST.relative_to(_ROOT)}")
    print(f"  corrupted: {_CORRUPT_EVAL.relative_to(_ROOT)}")
    original_data = load_binary_data(_ORIG_TEST)
    corrupted_data = load_binary_data(_CORRUPT_EVAL)
    if original_data.shape != corrupted_data.shape:
        raise SystemExit(
            f"Shape mismatch: original {original_data.shape} vs "
            f"corrupted {corrupted_data.shape}. Re-run prepare_mnist_data.py "
            "so both use the same number of rows."
        )
    print(f"  rows={original_data.shape[0]}  dims={original_data.shape[1]}")

    # Warm up LL once so the first live-plot update is not dominated by compile cost.
    g0 = p_hat.compile()
    print(
        f"  initial LL: orig={mean_log_likelihood(g0, original_data):.6f}  "
        f"corrupt={mean_log_likelihood(g0, corrupted_data):.6f}"
    )
    if args.checkpoint_every > 0:
        print(f"  checkpoints every {args.checkpoint_every} iters -> {out_path.parent}")

    plotter = MnistLiveLikelihoodPlot(k=args.k, lr=args.lr, ratio=args.ratio)

    def on_iter(it: int, node: CircuitNode, _lam: float) -> None:
        if args.checkpoint_every <= 0 or it % args.checkpoint_every != 0:
            return
        ckpt = checkpoint_path(out_path, it)
        node.save(ckpt)
        print(f"  checkpoint iter {it:3d} -> {ckpt.name}")

    p_theta, lam = run_dro_ogda(
        p_hat,
        k=args.k,
        num_iters=args.iters,
        lr=args.lr,
        ratio=args.ratio,
        eta_lambda=args.eta_lambda,
        warm_start_iters=args.warm_start_iters,
        theta_num_samples=args.theta_num_samples,
        deterministic=args.deterministic,
        eval_every=args.eval_every,
        original_data=original_data,
        adversarial_data=corrupted_data,
        plotter=plotter,
        on_iter=on_iter,
    )

    p_theta.save(out_path)
    print(f"\nsaved robustified circuit to {out_path.resolve()}")

    plotter.save(plot_path)
    print(f"saved likelihood plot to {plot_path.resolve()}")
    print(f"final lambda={lam:.4f}")

    if not args.no_plot_hold:
        plotter.hold_open()


if __name__ == "__main__":
    main()
