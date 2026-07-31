"""Save a grid comparing original, sigma=0.01, and MLE FGSM adversarials.

Randomly samples ``--num`` images (default 4) and writes a single figure
under ``mnist/`` with three rows:

* original test digits
* Gaussian corruption at ``sigma=0.010`` (``r0.data`` by default)
* matching rows from the MLE FGSM attack (``adversarial_datasets/K{k}/``)

The same row indices are used in every row so columns are comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fgsm import default_output
from prepare_mnist_data import corrupt_path, format_sigma
from peter import VALID_K

_ROOT = Path(__file__).resolve().parent
_ORIG_PATH = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"
_DEFAULT_OUT = _ROOT / "mnist" / "corruption_grid.png"
_COMPARE_SIGMA = 0.010

_IMG_SIDE = 28


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
            "Save a grid of randomly sampled MNIST images: original, "
            f"sigma={format_sigma(_COMPARE_SIGMA)}, and MLE FGSM adversarials."
        ),
    )
    p.add_argument(
        "--num",
        type=int,
        default=4,
        metavar="N",
        help="Images to sample (default: 4)",
    )
    p.add_argument(
        "--replicate",
        type=int,
        default=0,
        metavar="R",
        help=f"Which rR.data file to use for sigma={format_sigma(_COMPARE_SIGMA)} (default: 0)",
    )
    p.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=1,
        help="L-inf budget K for the MLE FGSM dataset (default: 1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for row sampling (default: 0)",
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

    sigma_path = corrupt_path(_COMPARE_SIGMA, args.replicate)
    adv_path = default_output(args.k)

    datasets: list[tuple[str, np.ndarray]] = [
        ("Unperturbed", load_rows(_ORIG_PATH)),
        (f"Random Corruption", load_rows(sigma_path)),
        (f"Adversarial Corruption", load_rows(adv_path)),
    ]

    n_rows = min(data.shape[0] for _, data in datasets)
    for label, data in datasets:
        if data.shape[0] != n_rows:
            print(
                f"warning: {label} has {data.shape[0]} rows; "
                f"using first {n_rows} shared indices",
                flush=True,
            )

    if args.num > n_rows:
        raise SystemExit(f"--num={args.num} exceeds shared dataset size {n_rows}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_rows, size=args.num, replace=False)
    indices.sort()

    rows: list[tuple[str, list[np.ndarray]]] = [
        (label, [to_image(data[i]) for i in indices]) for label, data in datasets
    ]

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
                ax.set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=18,
                    fontsize=16,
                )
            if r == 0:
                ax.set_title(f"idx {indices[c]}", fontsize=9)

    
    fig.tight_layout()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.resolve()}")
    print(f"  sampled indices: {', '.join(map(str, indices.tolist()))}")
    print(f"  levels: {', '.join(label for label, _ in rows)}")
    print(f"  adv source: {adv_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
