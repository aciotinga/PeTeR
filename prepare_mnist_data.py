"""Export MNIST test data and Gaussian-noise corrupted copies.

Writes the (optionally truncated) MNIST test set to::

    original_datasets/mnist/mnist.test.data

Also writes a peter_mnist-only truncated original and its TUNE_SIGMA
corrupt copies (``n=PETER_EVAL_N``)::

    original_datasets/mnist/mnist.test.n500.data
    corrupted_datasets/mnist/sigma0.100/r0.data
    ...
    corrupted_datasets/mnist/sigma0.100/r9.data

Then, for each sigma in ``{0.001, 0.002, ..., 0.010}`` and each of 10 replicates,
writes::

    corrupted_datasets/mnist/sigma<s>/r0.data
    ...
    corrupted_datasets/mnist/sigma<s>/r9.data

Corruption per pixel (independent noise per datapoint)::

    clip(round(x + 255 * N(0, sigma^2)), 0, 255)

Every run overwrites all outputs.  ``--n`` limits to the first N test rows
but does not appear in any path names.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_ORIG_PATH = _ROOT / "original_datasets" / "mnist" / "mnist.test.data"
# Truncated original used only by peter_mnist (matches TUNE_SIGMA corrupt size).
_ORIG_PETER_PATH = _ROOT / "original_datasets" / "mnist" / "mnist.test.n500.data"
_CORRUPT_ROOT = _ROOT / "corrupted_datasets" / "mnist"
_MNIST_CACHE = _ROOT / "data" / "MNIST" / "raw"

_TEST_IMAGES_URL = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
)
_TEST_IMAGES_NAME = "t10k-images-idx3-ubyte.gz"

NUM_COPIES = 10
SIGMAS = tuple(round(0.001 * i, 3) for i in range(1, 11))
# Upper end of SIGMAS; used as the PeTeR tune / live-eval corruption target.
TUNE_SIGMA = 0.1
# Row count for peter_mnist original + TUNE_SIGMA corrupt eval sets.
PETER_EVAL_N = 500


def format_sigma(sigma: float) -> str:
    return f"{sigma:.3f}"


def sigma_dir(sigma: float) -> Path:
    return _CORRUPT_ROOT / f"sigma{format_sigma(sigma)}"


def corrupt_path(sigma: float, replicate: int) -> Path:
    return sigma_dir(sigma) / f"r{replicate}.data"


def original_peter_path() -> Path:
    return _ORIG_PETER_PATH


def seed_for(sigma: float, replicate: int) -> int:
    digest = hashlib.md5(
        f"mnist|sigma{format_sigma(sigma)}|r{replicate}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def save_dataset(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%d")


def _download_test_images() -> Path:
    _MNIST_CACHE.mkdir(parents=True, exist_ok=True)
    path = _MNIST_CACHE / _TEST_IMAGES_NAME
    if path.is_file():
        return path
    print(f"downloading {_TEST_IMAGES_URL} ...", flush=True)
    urllib.request.urlretrieve(_TEST_IMAGES_URL, path)
    return path


def load_mnist_test(n: int | None) -> np.ndarray:
    gz_path = _download_test_images()
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.int32)

    if n is not None:
        if n < 1:
            raise ValueError(f"--n must be at least 1, got {n}")
        if n > len(data):
            raise ValueError(f"--n={n} exceeds MNIST test size {len(data)}")
        data = data[:n]
    return data


def gaussian_corrupt(
    data: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add per-pixel Gaussian noise: clip(round(x + 255 * N(0, sigma^2)), 0, 255)."""
    noise = rng.normal(loc=0.0, scale=sigma, size=data.shape)
    noisy = np.rint(data.astype(np.float64) + 255.0 * noise)
    return np.clip(noisy, 0, 255).astype(np.int32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Save MNIST test set under original_datasets/mnist and "
            f"{NUM_COPIES} Gaussian-noise copies per sigma under corrupted_datasets/mnist."
        ),
    )
    p.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N MNIST test datapoints (default: all 10000)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("loading MNIST test set...", flush=True)
    data = load_mnist_test(args.n)
    print(f"  shape={data.shape}  range=[{data.min()}, {data.max()}]", flush=True)

    save_dataset(_ORIG_PATH, data)
    print(f"wrote  {_ORIG_PATH.relative_to(_ROOT)}", flush=True)

    if len(data) >= PETER_EVAL_N:
        peter_rows = data[:PETER_EVAL_N]
        save_dataset(_ORIG_PETER_PATH, peter_rows)
        print(
            f"wrote  {_ORIG_PETER_PATH.relative_to(_ROOT)}  "
            f"(peter_mnist original, n={PETER_EVAL_N})",
            flush=True,
        )
        for replicate in range(NUM_COPIES):
            out = corrupt_path(TUNE_SIGMA, replicate)
            rng = np.random.default_rng(seed_for(TUNE_SIGMA, replicate))
            corrupted = gaussian_corrupt(peter_rows, TUNE_SIGMA, rng)
            save_dataset(out, corrupted)
            print(
                f"wrote  {out.relative_to(_ROOT)}  "
                f"(sigma={format_sigma(TUNE_SIGMA)}  r{replicate}  "
                f"n={PETER_EVAL_N})",
                flush=True,
            )
    else:
        print(
            f"skip peter_mnist n={PETER_EVAL_N} originals/corrupt "
            f"(only {len(data)} rows available)",
            flush=True,
        )

    for sigma in SIGMAS:
        for replicate in range(NUM_COPIES):
            out = corrupt_path(sigma, replicate)
            rng = np.random.default_rng(seed_for(sigma, replicate))
            corrupted = gaussian_corrupt(data, sigma, rng)
            save_dataset(out, corrupted)
            print(
                f"wrote  {out.relative_to(_ROOT)}  "
                f"(sigma={format_sigma(sigma)}  r{replicate})",
                flush=True,
            )

    print(
        f"\ndone: original + {len(SIGMAS)} sigmas x {NUM_COPIES} copies "
        f"({data.shape[0]} rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
