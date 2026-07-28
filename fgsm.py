"""One-shot FGSM attack on MNIST against a SPARC PC via finite-difference grads.

Approximates per-pixel log-likelihood gradients with central finite differences,
then applies a single Fast Gradient Sign step under an L-infinity pixel budget K::

    g_i = LL(x with x_i+1) - LL(x with x_i-1)   # clamped to [0, 255]
    x'_i = clip(x_i - K * sign(g_i), 0, 255)

Writes::

    adversarial_datasets/K{k}/mnist.test.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sparc.nodes import CircuitNode

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CIRCUIT = _ROOT / "mnist" / "hclt_mnist_blocksize4.json"
_DEFAULT_DATASET = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"

# Cap candidates materialized per LL call so high-dim data stays in memory.
MAX_CAND_ROWS = 1 << 16
PIXEL_MIN = 0
PIXEL_MAX = 255


def default_output(k: int) -> Path:
    return _ROOT / "adversarial_datasets" / f"K{k}" / "mnist.test.data"


def load_dataset(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def save_dataset(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%d")


def finite_difference_scores(
    graph,
    data: np.ndarray,
    *,
    max_cand_rows: int = MAX_CAND_ROWS,
) -> np.ndarray:
    """Return g[n, d] with g_i = LL(x_i+1) - LL(x_i-1), neighbors clipped to [0, 255]."""
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    n, d = data.shape
    if n == 0:
        return np.empty((0, d), dtype=np.float64)

    scores = np.empty((n, d), dtype=np.float64)
    cols_chunk = max(1, max_cand_rows // max(1, n))

    for c0 in range(0, d, cols_chunk):
        c1 = min(c0 + cols_chunk, d)
        c = c1 - c0
        cols = np.arange(c0, c1)

        # (n, c, d) candidates: bump column c0+j by +1 / -1 then clip
        rep_plus = np.broadcast_to(data[:, None, :], (n, c, d)).copy()
        rep_minus = rep_plus.copy()
        bumped = data[:, cols].astype(np.int32)
        rep_plus[:, np.arange(c), cols] = np.clip(bumped + 1, PIXEL_MIN, PIXEL_MAX)
        rep_minus[:, np.arange(c), cols] = np.clip(bumped - 1, PIXEL_MIN, PIXEL_MAX)

        ll_plus = graph.log_likelihood(rep_plus.reshape(n * c, d)).reshape(n, c)
        ll_minus = graph.log_likelihood(rep_minus.reshape(n * c, d)).reshape(n, c)
        scores[:, c0:c1] = ll_plus - ll_minus

    return scores


def fgsm_attack(
    graph,
    data: np.ndarray,
    k: int,
    *,
    max_cand_rows: int = MAX_CAND_ROWS,
    batch_size: int | None = None,
) -> np.ndarray:
    """One-shot FGSM: x' = clip(x - K * sign(g), 0, 255) with FD scores g."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")

    n = data.shape[0]
    if n == 0:
        return data.copy()

    bs = n if batch_size is None or batch_size <= 0 else batch_size
    adv = np.empty_like(data)

    for r0 in range(0, n, bs):
        r1 = min(r0 + bs, n)
        batch = data[r0:r1]
        g = finite_difference_scores(graph, batch, max_cand_rows=max_cand_rows)
        step = k * np.sign(g).astype(np.int32)
        adv[r0:r1] = np.clip(batch.astype(np.int32) - step, PIXEL_MIN, PIXEL_MAX)

    return adv.astype(np.int32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot FGSM (finite-difference grads) minimizing SPARC LL "
            "under L-infinity pixel budget K on MNIST."
        )
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="L-infinity pixel budget (absolute brightness change per pixel)",
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=_DEFAULT_CIRCUIT,
        help=f"SPARC circuit JSON (default: {_DEFAULT_CIRCUIT})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"Clean MNIST CSV (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: adversarial_datasets/K{k}/mnist.test.data)",
    )
    parser.add_argument(
        "--max-cand-rows",
        type=int,
        default=MAX_CAND_ROWS,
        help=f"Max candidate rows per LL batch (default: {MAX_CAND_ROWS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Rows per FD pass (0 = all rows at once)",
    )
    args = parser.parse_args()

    if args.k < 1:
        raise SystemExit(f"k must be >= 1, got {args.k}")
    if not args.circuit.is_file():
        raise SystemExit(f"Circuit not found: {args.circuit}")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    output = args.output if args.output is not None else default_output(args.k)

    print(f"loading circuit {args.circuit}", flush=True)
    graph = CircuitNode.load(args.circuit).compile()
    print(f"loading dataset {args.dataset}", flush=True)
    data = load_dataset(args.dataset)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    print(
        f"FGSM attacking {len(data)} x {data.shape[1]} with K={args.k}",
        flush=True,
    )

    orig_ll = float(graph.log_likelihood(data).mean())
    adv = fgsm_attack(
        graph,
        data,
        args.k,
        max_cand_rows=args.max_cand_rows,
        batch_size=args.batch_size or None,
    )
    adv_ll = float(graph.log_likelihood(adv).mean())

    delta = np.abs(adv.astype(np.int32) - data.astype(np.int32))
    mean_linf = float(delta.max(axis=1).mean()) if len(data) else 0.0
    mean_l1 = float(delta.sum(axis=1).mean()) if len(data) else 0.0
    changed_frac = float((delta > 0).mean()) if data.size else 0.0
    max_linf = int(delta.max()) if data.size else 0

    save_dataset(output, adv)
    print(
        f"saved {output}  mean_ll {orig_ll:.6f} -> {adv_ll:.6f}  "
        f"mean_Linf={mean_linf:.4f}  max_Linf={max_linf}  "
        f"mean_L1={mean_l1:.2f}  changed_frac={changed_frac:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
