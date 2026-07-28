"""Robustify a saved PC via PeTeR.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparc import CompiledCircuit
from sparc.nodes import CircuitNode
from sparc.optim import apply_grads, global_grad_norm
from sparc.queries import (
    cw_distance,
    cw_distance_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

_ROOT = Path(__file__).resolve().parent

# Defaults
K = 1.0
LR = 1e-3
RATIO = 3.0
ETA_LAMBDA = 10.0
LAMBDA_MAX = 1000.0
LAMBDA_LEAK = 1.0
ALPHA = 1.0
WARM_START_ITERS = 5
THETA_NUM_SAMPLES = 100


# ---------------------------------------------------------------------------
# Paths and evaluation data
# ---------------------------------------------------------------------------

def resolve_circuit_path(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    for candidate in (_ROOT / "example_pcs" / name, _ROOT / "example_pcs" / f"{name}.json"):
        if candidate.is_file():
            return candidate.resolve()
    choices = sorted(p.name for p in (_ROOT / "example_pcs").glob("*.json"))
    raise FileNotFoundError(
        f"Circuit not found: {name!r}. "
        f"Pass a path or a basename under example_pcs/ ({', '.join(choices)})."
    )


def circuit_path_to_dataset_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("hclt_"):
        rest = stem[len("hclt_"):]
        for sep in ("_blocksize", "_seed"):
            if sep in rest:
                return rest.split(sep)[0]
        return rest.split("_")[0]
    return stem


def resolve_eval_datasets(dataset_name: str, dataset_k: int) -> tuple[np.ndarray, np.ndarray]:
    original = _ROOT / "original_datasets" / dataset_name / f"{dataset_name}.test.data"
    adversarial = _ROOT / "adversarial_datasets" / f"{dataset_name}_K{dataset_k}.data"
    missing = [p for p in (original, adversarial) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Evaluation dataset(s) not found:\n" + "\n".join(f"  {p}" for p in missing)
        )
    return np.loadtxt(original, delimiter=",", dtype=np.int32), np.loadtxt(
        adversarial, delimiter=",", dtype=np.int32
    )


def mean_log_likelihood(graph: CompiledCircuit, rows: np.ndarray) -> float:
    return float(graph.log_likelihood(rows).mean())


# ---------------------------------------------------------------------------
# Compiled circuit players
# ---------------------------------------------------------------------------

@dataclass
class CompiledPlayer:
    """Mutable circuit root with a cached CompiledCircuit for fast queries."""

    node: CircuitNode
    graph: CompiledCircuit

    @classmethod
    def clone_from(cls, root: CircuitNode) -> CompiledPlayer:
        node = root.clone()
        return cls(node=node, graph=node.compile())

    def apply_grads(self, grads, lr: float, *, ascent: bool = False) -> None:
        apply_grads(self.node, grads, lr, ascent=ascent)
        self.graph.refresh_parameters()


# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------

class LiveLikelihoodPlot:
    def __init__(self, dataset_name: str, dataset_k: int) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(
                "Live plotting requires matplotlib. Install it with: pip install matplotlib"
            ) from exc
        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        (self._line_orig,) = self.ax.plot([], [], label=f"original test ({dataset_name})")
        (self._line_adv,) = self.ax.plot([], [], label=f"adversarial K={dataset_k} ({dataset_name})")
        self.ax.set(xlabel="iteration", ylabel="mean log-likelihood")
        self.ax.legend(loc="best")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self._iters: list[int] = []
        self._orig_lls: list[float] = []
        self._adv_lls: list[float] = []

    def update(self, it: int, orig_ll: float, adv_ll: float) -> None:
        self._iters.append(it)
        self._orig_lls.append(orig_ll)
        self._adv_lls.append(adv_ll)
        self._line_orig.set_data(self._iters, self._orig_lls)
        self._line_adv.set_data(self._iters, self._adv_lls)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._plt.pause(0.05)

    def hold_open(self) -> None:
        self._plt.ioff()
        self._plt.show()


# ---------------------------------------------------------------------------
# Gradients and OGDA
# ---------------------------------------------------------------------------

def _parts(grads) -> tuple[dict, dict]:
    if hasattr(grads, "sum_grads"):
        return grads.sum_grads, grads.cat_grads
    return grads


def _blend(sum_a, cat_a, sum_b, cat_b, w_a: float, w_b: float) -> tuple[dict, dict]:
    out_sum, out_cat = {}, {}
    for out, a, b in ((out_sum, sum_a, sum_b), (out_cat, cat_a, cat_b)):
        for nid in set(a) | set(b):
            out[nid] = w_a * np.asarray(a.get(nid, 0.0)) + w_b * np.asarray(b.get(nid, 0.0))
    return out_sum, out_cat


def combine_phi_grads(logexp_grads, cw_grads, lam: float) -> tuple[dict, dict]:
    """Normalized convex combination of phi descent directions."""
    n_e = global_grad_norm(logexp_grads)
    n_c = global_grad_norm(cw_grads)
    w = lam / (1.0 + lam)
    s_e = (1.0 - w) / n_e if n_e > 0 else 0.0
    s_c = w / n_c if n_c > 0 else 0.0
    le_s, le_c = _parts(logexp_grads)
    cw_s, cw_c = _parts(cw_grads)
    return _blend(le_s, le_c, cw_s, cw_c, s_e, s_c)


def ogda_dir(curr, prev) -> tuple[dict, dict]:
    sum_c, cat_c = _parts(curr)
    if prev is None:
        return sum_c, cat_c
    sum_p, cat_p = _parts(prev)
    return _blend(sum_c, cat_c, sum_p, cat_p, 2.0, -1.0)


def ogda_scalar(curr: float, prev: float | None) -> float:
    return curr if prev is None else 2.0 * curr - prev


def ogda_step(
    p_theta: CompiledPlayer,
    q_phi: CompiledPlayer,
    p_hat_g: CompiledCircuit,
    lam: float,
    prev: dict,
    *,
    k: float,
    eta_theta: float,
    eta_phi: float,
    eta_lambda: float,
    theta_num_samples: int,
    it: int,
    deterministic: bool,
    update_theta: bool,
) -> tuple[float, float, float]:
    """One OGDA step; return (lam, log_e, cw)."""
    _, _, grad_phi_logexp = log_exp_query_and_grad(p_theta.graph, q_phi.graph)
    cw_val, cw_grads = cw_distance_and_grad(
        p_hat_g, q_phi.graph, metric_p=1.0, scale_factor=1.0
    )
    phi_grads = combine_phi_grads(grad_phi_logexp, cw_grads, lam)
    violation = ALPHA * (cw_val - k)

    q_phi.apply_grads(ogda_dir(phi_grads, prev.get("phi")), eta_phi, ascent=False)

    lam = min(
        LAMBDA_MAX,
        max(0.0, LAMBDA_LEAK * lam + eta_lambda * ogda_scalar(violation, prev.get("viol"))),
    )

    if update_theta:
        samples = q_phi.graph.sample(
            theta_num_samples, seed=it if deterministic else None
        )
        _, grad_theta = p_theta.graph.mean_log_likelihood_and_grad(samples)
        p_theta.apply_grads(ogda_dir(grad_theta, prev.get("theta")), eta_theta, ascent=True)
        prev["theta"] = grad_theta

    prev["phi"] = phi_grads
    prev["viol"] = violation
    return lam, log_exp_query(p_theta.graph, q_phi.graph), cw_val


def run_dro_ogda(
    p_hat: CircuitNode,
    *,
    k: float = K,
    num_iters: int = 20,
    lr: float = LR,
    ratio: float = RATIO,
    eta_lambda: float = ETA_LAMBDA,
    warm_start_iters: int = WARM_START_ITERS,
    theta_num_samples: int = THETA_NUM_SAMPLES,
    deterministic: bool = False,
    eval_every: int | None = None,
    original_data: np.ndarray | None = None,
    adversarial_data: np.ndarray | None = None,
    plotter: LiveLikelihoodPlot | None = None,
    on_iter: Callable[[int, CircuitNode, float], None] | None = None,
    quiet: bool = False,
    start_iter: int = 0,
    p_theta_init: CircuitNode | None = None,
    lam_init: float = 0.0,
) -> tuple[CircuitNode, float]:
    """Run DRO-OGDA.

    ``p_hat`` is the CW-ball reference (typically the MLE).  To resume, pass
    ``start_iter`` / ``p_theta_init`` / ``lam_init`` from a checkpoint; warm
    start is skipped when ``start_iter > 0``.
    """
    if start_iter < 0:
        raise ValueError(f"start_iter must be non-negative, got {start_iter}")
    if start_iter >= num_iters:
        raise ValueError(
            f"start_iter ({start_iter}) must be < num_iters ({num_iters})"
        )

    eta_theta, eta_phi = lr, lr * ratio
    p_hat_g = p_hat.compile()
    theta_src = p_theta_init if p_theta_init is not None else p_hat
    p_theta = CompiledPlayer.clone_from(theta_src)
    # Without a saved Q, warm-start the adversary from the current theta.
    q_phi = CompiledPlayer.clone_from(theta_src)
    lam = float(lam_init)
    prev: dict = {}
    do_eval = original_data is not None and adversarial_data is not None
    cw_kw = dict(metric_p=1.0, scale_factor=1.0)

    if not quiet:
        resume_note = f"  resume_from={start_iter}" if start_iter > 0 else ""
        print(
            f"initial: log(E)={log_exp_query(p_theta.graph, q_phi.graph):.6f}  "
            f"CW={cw_distance(p_hat_g, q_phi.graph, **cw_kw):.6f}  "
            f"lr={lr:g}  ratio={ratio:g}  (eta_theta={eta_theta:g}  eta_phi={eta_phi:g})"
            f"{resume_note}"
        )

    def eval_at(it: int) -> None:
        orig_ll = mean_log_likelihood(p_theta.graph, original_data)
        adv_ll = mean_log_likelihood(p_theta.graph, adversarial_data)
        if not quiet:
            print(f"  eval iter {it:3d}: orig_test_ll={orig_ll:.6f}  adv_test_ll={adv_ll:.6f}")
        if plotter is not None:
            plotter.update(it, orig_ll, adv_ll)

    if do_eval:
        eval_at(start_iter)

    step_kw = dict(
        k=k,
        eta_theta=eta_theta,
        eta_phi=eta_phi,
        eta_lambda=eta_lambda,
        theta_num_samples=theta_num_samples,
        deterministic=deterministic,
    )

    if start_iter == 0 and warm_start_iters > 0:
        if not quiet:
            print(f"  warm start: {warm_start_iters} Q-only iteration(s)")
        for w in range(1, warm_start_iters + 1):
            lam, log_e, cw = ogda_step(
                p_theta, q_phi, p_hat_g, lam, prev, it=w, update_theta=False, **step_kw
            )
            if not quiet:
                print(
                    f"  [warm-start {w}/{warm_start_iters}] "
                    f"log(E)={log_e:.6f}  CW={cw:.6f}  violation={cw - k:+.6f}  lambda={lam:.4f}"
                )

    for it in range(start_iter + 1, num_iters + 1):
        lam, log_e, cw = ogda_step(
            p_theta, q_phi, p_hat_g, lam, prev, it=it, update_theta=True, **step_kw
        )
        if not quiet:
            print(
                f"  iter {it:3d}: log(E)={log_e:.6f}  CW={cw:.6f}  "
                f"violation={cw - k:+.6f}  lambda={lam:.4f}"
            )
        if do_eval and eval_every is not None and it % eval_every == 0:
            eval_at(it)
        if on_iter is not None:
            on_iter(it, p_theta.node, lam)

    return p_theta.node, lam


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Robustify a PC with sample-based DRO using OGDA.")
    parser.add_argument("circuit", help="Basename or path to a gcw-circuit-v1 JSON")
    parser.add_argument("--k", type=float, default=K, help="CW-ball radius")
    parser.add_argument("--iters", type=int, default=20, help="OGDA iterations")
    parser.add_argument("-o", "--output", help="Save robustified circuit (gcw-circuit-v1 JSON)")
    parser.add_argument(
        "--dataset-k", type=int, choices=(1, 3, 5), default=None,
        help="Adversarial dataset K; enables eval/plot every --eval-every iterations",
    )
    parser.add_argument(
        "--eval-every", type=int, default=1, metavar="N",
        help="Eval test log-likelihood every N iterations (default: 5, requires --dataset-k)",
    )
    parser.add_argument(
        "--warm-start-iters", type=int, default=WARM_START_ITERS,
        help=f"Q-only iterations before theta updates (default: {WARM_START_ITERS})",
    )
    parser.add_argument(
        "--theta-num-samples", type=int, default=THETA_NUM_SAMPLES,
        help=f"Samples from Q_phi per theta update (default: {THETA_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use iteration number as RNG seed for theta sampling (reproducible runs)",
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help=f"Theta learning rate eta_theta (default: {LR:g})",
    )
    parser.add_argument(
        "--ratio", type=float, default=RATIO,
        help=f"Phi/theta LR ratio; eta_phi = lr * ratio (default: {RATIO:g})",
    )
    parser.add_argument(
        "--eta-lambda", type=float, default=ETA_LAMBDA,
        help=f"Dual (lambda) step size (default: {ETA_LAMBDA:g})",
    )
    args = parser.parse_args()

    if args.dataset_k is not None and args.eval_every <= 0:
        parser.error("--eval-every must be a positive integer")
    if args.warm_start_iters < 0:
        parser.error("--warm-start-iters must be non-negative")
    if args.theta_num_samples < 1:
        parser.error("--theta-num-samples must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.ratio <= 0:
        parser.error("--ratio must be positive")
    if args.eta_lambda <= 0:
        parser.error("--eta-lambda must be positive")

    path = resolve_circuit_path(args.circuit)
    print(f"loading {path.name} from {path.parent}")
    p_hat = CircuitNode.load(path)
    print(f"  nodes in scope: {len(p_hat.scope_as_list())}")

    original_data = adversarial_data = None
    plotter = None
    if args.dataset_k is not None:
        dataset_name = circuit_path_to_dataset_name(path)
        print(f"loading eval datasets for {dataset_name!r} (K={args.dataset_k})")
        original_data, adversarial_data = resolve_eval_datasets(dataset_name, args.dataset_k)
        print(
            f"  original test: {len(original_data)} rows, "
            f"adversarial K={args.dataset_k}: {len(adversarial_data)} rows"
        )
        plotter = LiveLikelihoodPlot(dataset_name, args.dataset_k)

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
        eval_every=args.eval_every if args.dataset_k is not None else None,
        original_data=original_data,
        adversarial_data=adversarial_data,
        plotter=plotter,
    )

    if args.output:
        out = Path(args.output)
        p_theta.save(out)
        print(f"\nsaved robustified circuit to {out.resolve()}")

    print(f"\nfinal lambda={lam:.4f}")
    if plotter is not None:
        plotter.hold_open()


if __name__ == "__main__":
    main()
