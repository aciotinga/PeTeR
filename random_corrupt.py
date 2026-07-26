"""Create shared random bit-flip corruptions of each original test set.

For every ``original_datasets/<name>/<name>.test.data`` and each K in
``{1, 3, 5}``, writes 10 independent copies under::

    corrupted_datasets/K<k>/<name>/r0.data
    ...
    corrupted_datasets/K<k>/<name>/r9.data

Each datapoint gets exactly ``K`` bit flips **with replacement** (the same
feature may be chosen more than once, which is a no-op on the second flip).
Seeds are fixed per ``(dataset, k, replicate)`` so regenerations match.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from peter import VALID_K

_ROOT = Path(__file__).resolve().parent
_ORIG_ROOT = _ROOT / "original_datasets"
_CORRUPT_ROOT = _ROOT / "corrupted_datasets"

NUM_COPIES = 10


def load_dataset(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def save_dataset(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%d")


def original_test_path(dataset: str) -> Path:
    return _ORIG_ROOT / dataset / f"{dataset}.test.data"


def corrupt_dir(dataset: str, k: int) -> Path:
    return _CORRUPT_ROOT / f"K{k}" / dataset


def corrupt_path(dataset: str, k: int, replicate: int) -> Path:
    return corrupt_dir(dataset, k) / f"r{replicate}.data"


def seed_for(dataset: str, k: int, replicate: int) -> int:
    """Deterministic 32-bit seed for ``(dataset, k, replicate)``."""
    digest = hashlib.md5(f"{dataset}|K{k}|r{replicate}".encode()).hexdigest()
    return int(digest[:8], 16)


def discover_datasets() -> list[str]:
    if not _ORIG_ROOT.is_dir():
        return []
    names: list[str] = []
    for d in sorted(_ORIG_ROOT.iterdir()):
        if d.is_dir() and original_test_path(d.name).is_file():
            names.append(d.name)
    return names


def random_bit_flips(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Apply ``k`` with-replacement bit flips per row."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    out = data.copy()
    if k == 0 or out.size == 0:
        return out
    n, d = out.shape
    rows = np.arange(n)
    for _ in range(k):
        cols = rng.integers(0, d, size=n)
        out[rows, cols] ^= 1
    return out


def make_one(args: tuple[str, int, int, bool]) -> str:
    dataset, k, replicate, force = args
    out = corrupt_path(dataset, k, replicate)
    label = f"{dataset}  k={k}  r{replicate}"
    if out.is_file() and not force:
        msg = f"skip   {label}  ({out.name} exists)"
        print(msg, flush=True)
        return msg

    src = original_test_path(dataset)
    if not src.is_file():
        msg = f"SKIP   {label}  (missing {src})"
        print(msg, flush=True)
        return msg

    print(f"start  {label}", flush=True)
    data = load_dataset(src)
    rng = np.random.default_rng(seed_for(dataset, k, replicate))
    corrupted = random_bit_flips(data, k, rng)
    save_dataset(out, corrupted)
    msg = f"done   {label}  -> {out.relative_to(_ROOT)}"
    print(msg, flush=True)
    return msg


def collect_jobs(
    ks: list[int],
    datasets: list[str] | None,
    force: bool,
) -> list[tuple[str, int, int, bool]]:
    names = datasets if datasets else discover_datasets()
    if not names:
        raise SystemExit("No original test datasets found under original_datasets/.")
    if datasets:
        available = set(discover_datasets())
        missing = [d for d in datasets if d not in available]
        if missing:
            raise SystemExit(f"Dataset(s) not found: {', '.join(missing)}")
    return [
        (dataset, k, r, force)
        for k in ks
        for dataset in names
        for r in range(NUM_COPIES)
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Write shared random bit-flip corruptions under corrupted_datasets/ "
            f"(K flips with replacement, {NUM_COPIES} copies per dataset/K)."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        choices=VALID_K,
        default=None,
        help=f"Only this K (default: all of {list(VALID_K)})",
    )
    p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Only this dataset (repeatable). Default: all with a .test.data",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        metavar="N",
        help="Parallel worker processes (default: number of available CPUs)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing r*.data files",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    ks = [args.k] if args.k is not None else list(VALID_K)
    jobs = collect_jobs(ks, args.datasets, args.force)
    pending = [j for j in jobs if args.force or not corrupt_path(j[0], j[1], j[2]).is_file()]
    print(
        f"random_corrupt: ks={ks}  jobs={args.jobs}  queued={len(jobs)}  "
        f"pending={len(pending)}  force={args.force}",
        flush=True,
    )

    results: list[str] = []
    if args.jobs == 1:
        for job in jobs:
            results.append(make_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(make_one, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())

    print("\n=== summary ===")
    for line in sorted(results):
        print(line)


if __name__ == "__main__":
    main()
