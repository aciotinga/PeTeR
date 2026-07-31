"""Learn HCLT block-size-4 MLE PCs on PeTeR adversarial sets (and clean test for K=0).

Mirrors ``learn_rltpm.py`` hyperparameters / init (structure from original train,
seed=0, EM schedule) but fits on fixed data with no online corruption:

* K=0  — ``original_datasets/<dataset>/<dataset>.test.data``
* K>=1 — ``results/<dataset>/k<k>/<hyperparams>/<dataset>_K<k>_peter.data``
  (exactly one hyperparam directory required)

By default runs all of K in {0, 1, 3, 5}. With ``-j N``, every (dataset, K)
pair is one job in a shared queue so workers stay busy across K values.

Writes::

    mle_adv_learned_pcs/<dataset>/k<k>/adv_learned.json
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# pyjuice calls random.randint(0, 1e8) when compiling Triton kernels; Py>=3.12 rejects float bounds
_randint = random.randint
random.randint = lambda a, b: _randint(int(a), int(b))

import numpy as np
import pyjuice as juice
import torch
from torch.utils.data import DataLoader, TensorDataset

from learn_rltpm import serialize_circuit

DEBD_DIR = Path("original_datasets")
RESULTS_DIR = Path("results")
OUT_DIR = Path("mle_adv_learned_pcs")
BLOCK_SIZE = 4
SEED = 0
EPOCHS = 1000
BATCH_SIZE = 512
DEFAULT_KS = (0, 1, 3, 5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.float32)


def load_split(dataset: str, split: str) -> np.ndarray:
    path = DEBD_DIR / dataset / f"{dataset}.{split}.data"
    return load_csv(path)


def debd_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.train.data").is_file()
    )


def out_path(dataset: str, k: int) -> Path:
    return OUT_DIR / dataset / f"k{k}" / "adv_learned.json"


def resolve_peter_adv_path(dataset: str, k: int) -> Path:
    """Return the unique PeTeR adversarial data path for ``dataset`` at strength ``k``."""
    k_dir = RESULTS_DIR / dataset / f"k{k}"
    if not k_dir.is_dir():
        raise FileNotFoundError(f"Missing PeTeR results directory: {k_dir}")

    hp_dirs = sorted(p for p in k_dir.iterdir() if p.is_dir())
    if len(hp_dirs) == 0:
        raise FileNotFoundError(f"No hyperparam directories under {k_dir}")
    if len(hp_dirs) > 1:
        names = ", ".join(p.name for p in hp_dirs)
        raise RuntimeError(
            f"Expected exactly one hyperparam directory under {k_dir}, "
            f"found {len(hp_dirs)}: {names}"
        )

    path = hp_dirs[0] / f"{dataset}_K{k}_peter.data"
    if not path.is_file():
        raise FileNotFoundError(f"Missing PeTeR adversarial data: {path}")
    return path


def fit_data_path(dataset: str, k: int) -> Path:
    if k == 0:
        path = DEBD_DIR / dataset / f"{dataset}.test.data"
        if not path.is_file():
            raise FileNotFoundError(f"Missing clean test data: {path}")
        return path
    return resolve_peter_adv_path(dataset, k)


def learn_and_save(dataset: str, k: int, device: torch.device) -> None:
    dst = out_path(dataset, k)
    if dst.exists():
        print(f"Skipping {dataset} ({dst} exists)")
        return

    fit_path = fit_data_path(dataset, k)
    print(f"[{dataset} K{k}] structure=train  fit={fit_path}")

    seed_everything(SEED)
    train_data = torch.from_numpy(load_split(dataset, "train")).to(device).int()
    fit_data = torch.from_numpy(load_csv(fit_path)).to(device).int()

    fit_batch = min(BATCH_SIZE, len(fit_data))
    fit_loader = DataLoader(
        TensorDataset(fit_data),
        batch_size=fit_batch,
        shuffle=False,
        drop_last=len(fit_data) > fit_batch,
    )

    # Structure + random init from original train under SEED (shared across K).
    ns = juice.structures.HCLT(
        train_data, num_latents=BLOCK_SIZE, input_node_params={"num_cats": 2}
    )
    pc = juice.compile(ns)
    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr=0.1, pseudocount=0.1, method="EM")
    scheduler = juice.optim.CircuitScheduler(
        optimizer,
        method="multi_linear",
        lrs=[0.9, 0.1, 0.05],
        milestone_steps=[0, len(fit_loader) * 100, len(fit_loader) * 500],
    )

    for batch in fit_loader:
        x = batch[0].to(device)
        pc(x, record_cudagraph=True).mean().backward()
        break

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_ll = 0.0
        for batch in fit_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            lls = pc(x)
            lls.mean().backward()
            train_ll += lls.mean().item()
            optimizer.step()
            scheduler.step()
        train_ll /= len(fit_loader)
        t1 = time.time()

        print(
            f"[{dataset} K{k} epoch {epoch}/{EPOCHS}] "
            f"fit LL {train_ll:.2f} ({t1 - t0:.1f}s)"
        )

    pc.update_parameters()

    dst.parent.mkdir(parents=True, exist_ok=True)
    serialize_circuit(pc, dst, block_size=BLOCK_SIZE)
    print(f"Saved {dst}")


def build_jobs(datasets: list[str], ks: list[int]) -> list[tuple[str, int]]:
    """Flat (dataset, K) queue — workers drain this continuously."""
    return [(dataset, k) for k in ks for dataset in datasets]


def _job_subprocess_cmd(dataset: str, k: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--k",
        str(k),
        "--datasets",
        dataset,
        "--jobs",
        "1",
    ]


def _run_job_subprocess(dataset: str, k: int) -> tuple[str, int, int]:
    cmd = _job_subprocess_cmd(dataset, k)
    print(f"starting {dataset} K{k}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    return dataset, k, completed.returncode


def run_jobs_parallel(jobs: list[tuple[str, int]], max_workers: int) -> int:
    """Run one isolated subprocess per (dataset, K), at most ``max_workers`` at a time."""
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_job_subprocess, dataset, k): (dataset, k)
            for dataset, k in jobs
        }
        for future in as_completed(futures):
            dataset, k = futures[future]
            _, _, returncode = future.result()
            if returncode == 0:
                print(f"finished {dataset} K{k}", flush=True)
            else:
                failures += 1
                print(f"FAILED {dataset} K{k} (exit {returncode})", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn MLE HCLT PCs on PeTeR adversarial datasets "
            "(K=0: clean test) for DEBD. Default: all K in "
            f"{list(DEFAULT_KS)} as a continuous (dataset, K) job queue."
        )
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help=(
            f"Corruption strength(s) (0 = clean original test). "
            f"Default: {' '.join(str(k) for k in DEFAULT_KS)}."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="DEBD dataset names (default: all with train splits).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run up to N (dataset, K) jobs in parallel via separate "
            "subprocesses (default: 1)."
        ),
    )
    args = parser.parse_args()

    ks = list(args.k) if args.k is not None else list(DEFAULT_KS)
    if any(k < 0 for k in ks):
        parser.error("--k values must be non-negative")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    datasets = args.datasets if args.datasets else debd_datasets()
    job_list = build_jobs(datasets, ks)
    pending = [(d, k) for d, k in job_list if not out_path(d, k).exists()]
    print(
        f"learn_mle_adv: datasets={len(datasets)}  K={ks}  "
        f"queued={len(job_list)}  pending={len(pending)}  "
        f"skip_existing={len(job_list) - len(pending)}  jobs={args.jobs}",
        flush=True,
    )

    if args.jobs > 1 and len(job_list) > 1:
        failures = run_jobs_parallel(job_list, args.jobs)
        if failures:
            raise SystemExit(f"{failures} job(s) failed")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running sequentially on {device}", flush=True)
    for dataset, k in job_list:
        print(f"\n=== {dataset} K{k} ===", flush=True)
        learn_and_save(dataset, k, device)


if __name__ == "__main__":
    main()
