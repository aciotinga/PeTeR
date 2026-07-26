"""Greedy Hamming-ball bit-flip attack against a SPARC circuit.

For each datapoint, up to ``k`` times: evaluate every single-bit flip with
batched log-likelihood, apply the flip that most decreases LL, and stop early
for that row if no flip helps. Re-flips of previously flipped bits are allowed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sparc.nodes import CircuitNode

# Cap candidates materialized per LL call so high-dim data stays in memory.
MAX_CAND_ROWS = 1 << 16


def load_dataset(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def save_dataset(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%d")


def greedy_attack(
    graph,
    data: np.ndarray,
    k: int,
    *,
    max_cand_rows: int = MAX_CAND_ROWS,
) -> np.ndarray:
    """Return a copy of ``data`` with at most ``k`` greedy bit flips per row."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    if k == 0:
        return data.copy()

    adv = data.copy()
    n, d = adv.shape
    cols_chunk = max(1, max_cand_rows // max(1, n))
    rows_idx = np.arange(n)

    for _ in range(k):
        current_ll = graph.log_likelihood(adv)
        cand_ll = np.empty((n, d), dtype=np.float64)

        for c0 in range(0, d, cols_chunk):
            c1 = min(c0 + cols_chunk, d)
            c = c1 - c0
            # (n, c, d) candidates: flip column c0+j in slice j
            rep = np.broadcast_to(adv[:, None, :], (n, c, d)).copy()
            cols = np.arange(c0, c1)
            rep[:, np.arange(c), cols] ^= 1
            cand_ll[:, c0:c1] = graph.log_likelihood(rep.reshape(n * c, d)).reshape(n, c)

        best_i = np.argmin(cand_ll, axis=1)
        best_ll = cand_ll[rows_idx, best_i]
        accept = best_ll < current_ll
        if not np.any(accept):
            break
        flip_rows = rows_idx[accept]
        adv[flip_rows, best_i[accept]] ^= 1

    return adv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy k-bit flip attack minimizing SPARC log-likelihood."
    )
    parser.add_argument("circuit", type=Path, help="Path to SPARC circuit JSON")
    parser.add_argument("dataset", type=Path, help="Path to CSV binary dataset")
    parser.add_argument("output", type=Path, help="Path to write attacked dataset")
    parser.add_argument("k", type=int, help="Max bit flips per datapoint (Hamming budget)")
    parser.add_argument(
        "--max-cand-rows",
        type=int,
        default=MAX_CAND_ROWS,
        help=f"Max candidate rows per LL batch (default: {MAX_CAND_ROWS})",
    )
    args = parser.parse_args()

    if args.k < 0:
        raise SystemExit(f"k must be non-negative, got {args.k}")
    if not args.circuit.is_file():
        raise SystemExit(f"Circuit not found: {args.circuit}")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    print(f"loading circuit {args.circuit}")
    graph = CircuitNode.load(args.circuit).compile()
    print(f"loading dataset {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"attacking {len(data)} x {data.shape[1]} with k={args.k}")

    orig_ll = float(graph.log_likelihood(data).mean())
    adv = greedy_attack(graph, data, args.k, max_cand_rows=args.max_cand_rows)
    adv_ll = float(graph.log_likelihood(adv).mean())
    flips = int((adv != data).sum())
    max_ham = int((adv != data).sum(axis=1).max()) if len(data) else 0

    save_dataset(args.output, adv)
    print(
        f"saved {args.output}  mean_ll {orig_ll:.6f} -> {adv_ll:.6f}  "
        f"total_flips={flips}  max_hamming={max_ham}"
    )


if __name__ == "__main__":
    main()
