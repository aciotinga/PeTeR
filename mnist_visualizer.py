"""Save a grid of random MNIST images across corruption levels.

For each ``corrupted_datasets/mnist/sigma*/`` directory, randomly samples
``--num`` images (default 4) from ``r0.data`` and writes a single figure
under ``mnist/``.  The same row indices are used at every sigma so columns
are comparable.  An optional top row shows the matching original digits.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_CORRUPT_ROOT = _ROOT / "corrupted_datasets" / "mnist"
_ORIG_PATH = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"
_DEFAULT_OUT = _ROOT / "mnist" / "corruption_grid.png"

_IMG_SIDE = 28
_SIGMA_DIR_RE = re.compile(r"^sigma(\d+(?:\.\d+)?)$")


def discover_sigma_dirs() -> list[tuple[float, Path]]:
    if not _CORRUPT_ROOT.is_dir():
        raise FileNotFoundError(f"Missing {_CORRUPT_ROOT}")
    found: list[tuple[float, Path]] = []
    for d in sorted(_CORRUPT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = _SIGMA_DIR_RE.match(d.name)
        if m is None:
            continue
        found.append((float(m.group(1)), d))
    found.sort(key=lambda x: x[0])
    if not found:
        raise FileNotFoundError(f"No sigma* directories under {_CORRUPT_ROOT}")
    return found


def load_rows(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    data = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != _IMG_SIDE * _IMG_SIDE:
        raise ValueError(
            f"Expected {_IMG_SIDE * _IMG_SIDE} features, got {data.shape[1]} in {path}"
        )
    return data


def to_image(row: np.ndarray) -> np.ndarray:
    return row.reshape(_IMG_SIDE, _IMG_SIDE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Save a grid of randomly sampled MNIST images, one row per "
            "corruption sigma (4 samples by default)."
        ),
    )
    p.add_argument(
        "--num",
        type=int,
        default=4,
        metavar="N",
        help="Images to sample per corruption level (default: 4)",
    )
    p.add_argument(
        "--replicate",
        type=int,
        default=0,
        metavar="R",
        help="Which rR.data file to sample from (default: 0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for row sampling (default: 0)",
    )
    p.add_argument(
        "--no-original",
        action="store_true",
        help="Do not include an original-test top row",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output image path (default: {_DEFAULT_OUT.relative_to(_ROOT)})",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.num < 1:
        raise SystemExit("--num must be at least 1")
    if args.replicate < 0:
        raise SystemExit("--replicate must be non-negative")

    sigma_dirs = discover_sigma_dirs()
    datasets: list[tuple[str, np.ndarray]] = []

    for sigma, d in sigma_dirs:
        path = d / f"r{args.replicate}.data"
        datasets.append((f"sigma={sigma:g}", load_rows(path)))

    n_rows = datasets[0][1].shape[0]
    for label, data in datasets[1:]:
        if data.shape[0] != n_rows:
            raise SystemExit(
                f"Row-count mismatch: {datasets[0][0]} has {n_rows}, "
                f"{label} has {data.shape[0]}"
            )

    if args.num > n_rows:
        raise SystemExit(f"--num={args.num} exceeds dataset size {n_rows}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_rows, size=args.num, replace=False)
    indices.sort()

    rows: list[tuple[str, list[np.ndarray]]] = []
    if not args.no_original:
        if not _ORIG_PATH.is_file():
            raise SystemExit(
                f"Original test set not found at {_ORIG_PATH}. "
                "Re-run prepare_mnist_data.py or pass --no-original."
            )
        original = load_rows(_ORIG_PATH)
        if original.shape[0] != n_rows:
            raise SystemExit(
                f"Original has {original.shape[0]} rows but corrupted has {n_rows}"
            )
        rows.append(("original", [to_image(original[i]) for i in indices]))

    for label, data in datasets:
        rows.append((label, [to_image(data[i]) for i in indices]))

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    n_levels = len(rows)
    n_cols = args.num
    fig, axes = plt.subplots(
        n_levels,
        n_cols,
        figsize=(1.6 * n_cols, 1.6 * n_levels),
        squeeze=False,
    )
    for r, (label, images) in enumerate(rows):
        for c, img in enumerate(images):
            ax = axes[r][c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=18)
            if r == 0:
                ax.set_title(f"idx {indices[c]}", fontsize=9)

    fig.suptitle(
        f"MNIST corruptions  (r{args.replicate}, seed={args.seed})",
        fontsize=12,
    )
    fig.tight_layout()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.resolve()}")
    print(f"  sampled indices: {', '.join(map(str, indices.tolist()))}")
    print(f"  levels: {', '.join(label for label, _ in rows)}")


if __name__ == "__main__":
    main()
