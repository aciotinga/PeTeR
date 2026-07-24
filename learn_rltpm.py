"""Learn adversarially robust HCLT block-size-4 PCs on all DEBD datasets (seed=0) using RL-TPM.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# pyjuice calls random.randint(0, 1e8) when compiling Triton kernels; Py>=3.12 rejects float bounds
_randint = random.randint
random.randint = lambda a, b: _randint(int(a), int(b))

import numpy as np
import pyjuice as juice
import torch
from torch.utils.data import DataLoader, TensorDataset

DEBD_DIR = Path("original_datasets")
OUT_DIR = Path("rltpm_learned_pcs")
BLOCK_SIZE = 4
SEED = 0
EPOCHS = 1000
BATCH_SIZE = 512

# Cap the number of candidate rows materialized per corruption forward pass so
# greedy_corrupt stays within GPU memory even for high-dimensional datasets.
MAX_CAND_ROWS = 1 << 16
GCW_CIRCUIT_FORMAT = "gcw-circuit-v1"


@dataclass
class _SerNode:
    kind: str
    children: list["_SerNode"] = field(default_factory=list)
    scope: list[int] | None = None
    params: list[float] | None = None


def _build_ser_nodes(
    circuit,
    block_size: int,
    scope_offset: int,
    cache: dict,
) -> list[_SerNode]:
    if circuit in cache:
        return cache[circuit]

    if circuit.is_sum():
        child_vals = _build_ser_nodes(circuit.chs[0], block_size, scope_offset, cache)
        params = circuit.get_params().numpy()
        value = [
            _SerNode("sum", children=list(child_vals), params=params[0, i, :].tolist())
            for i in range(len(params[0]))
        ]
    elif circuit.is_prod():
        child_vals = [
            _build_ser_nodes(child, block_size, scope_offset, cache) for child in circuit.chs
        ]
        value = [
            _SerNode("product", children=[c[i] for c in child_vals])
            for i in range(block_size)
        ]
    elif circuit.is_input():
        params = circuit.get_params().numpy().reshape(block_size, -1)
        scope_var = list(circuit.scope)[0] + scope_offset
        value = [
            _SerNode(
                "categorical",
                children=[],
                scope=[scope_var],
                params=params[i, :].tolist(),
            )
            for i in range(block_size)
        ]
    else:
        raise NotImplementedError("Invalid node type")

    cache[circuit] = value
    return value


def _ser_tree_to_payload(root: _SerNode) -> dict[str, Any]:
    id_to_idx: dict[int, int] = {}
    nodes: list[_SerNode] = []

    def visit(node: _SerNode) -> int:
        key = id(node)
        if key in id_to_idx:
            return id_to_idx[key]
        for child in node.children:
            visit(child)
        idx = len(nodes)
        id_to_idx[key] = idx
        nodes.append(node)
        return idx

    root_idx = visit(root)
    records: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        rec: dict[str, Any] = {
            "id": i,
            "kind": node.kind,
            "children": [id_to_idx[id(child)] for child in node.children],
        }
        if node.kind == "sum":
            rec["params"] = node.params
        elif node.kind == "categorical":
            rec["scope"] = node.scope
            rec["params"] = node.params
        records.append(rec)

    return {
        "format": GCW_CIRCUIT_FORMAT,
        "backend": "numpy",
        "root": root_idx,
        "nodes": records,
    }


def serialize_circuit(
    pc,
    path: Path,
    *,
    block_size: int = BLOCK_SIZE,
    scope_offset: int = 0,
    indent: int = 2,
) -> None:
    """Serialize a PyJuice PC directly to gcw-circuit-v1 JSON."""
    cache: dict = {}
    root = _build_ser_nodes(pc.root_ns, block_size, scope_offset, cache)[0]
    payload = _ser_tree_to_payload(root)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(dataset: str, split: str) -> np.ndarray:
    path = DEBD_DIR / dataset / f"{dataset}.{split}.data"
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.float32)


def debd_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.train.data").is_file()
    )


def out_path(dataset: str, k: int) -> Path:
    return (
        OUT_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


@torch.no_grad()
def greedy_corrupt(pc, batch: torch.Tensor, k: int) -> torch.Tensor:
    """Batched greedy bit-flip corruption of ``batch`` against ``pc``.

    Torch port of the per-row search in adversarial_dataset_generation.py: for
    each of ``k`` steps, find (per row) the single bit flip that most decreases
    the log-likelihood under ``pc`` and apply it. Faithfully reproduces the
    reference quirk that a bit is *always* flipped each step, defaulting to bit 0
    when no flip lowers the LL.
    """
    batch = batch.clone()
    n, d = batch.shape
    rows = torch.arange(n, device=batch.device)
    cols_chunk = max(1, MAX_CAND_ROWS // max(1, n))

    for _ in range(k):
        current_ll = pc(batch)
        cand_ll = torch.empty((n, d), device=batch.device, dtype=current_ll.dtype)

        for c0 in range(0, d, cols_chunk):
            c1 = min(c0 + cols_chunk, d)
            c = c1 - c0
            rep = batch.unsqueeze(1).expand(n, c, d).clone()
            cols = torch.arange(c0, c1, device=batch.device)
            rep[:, torch.arange(c, device=batch.device), cols] ^= 1
            lls = pc(rep.reshape(n * c, d))
            cand_ll[:, c0:c1] = lls.reshape(n, c)

        min_ll, best_i = cand_ll.min(dim=1)
        accept = min_ll < current_ll
        best_i = torch.where(accept, best_i, torch.zeros_like(best_i))
        batch[rows, best_i] ^= 1

    return batch


def learn_and_save(dataset: str, k: int, device: torch.device) -> None:
    dst = out_path(dataset, k)
    if dst.exists():
        print(f"Skipping {dataset} ({dst} exists)")
        return

    seed_everything(SEED)
    train_data = torch.from_numpy(load_split(dataset, "train")).to(device).int()
    valid_data = torch.from_numpy(load_split(dataset, "valid")).to(device).int()

    train_batch = min(BATCH_SIZE, len(train_data))
    valid_batch = min(BATCH_SIZE, len(valid_data))
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=train_batch,
        shuffle=False,
        drop_last=len(train_data) > train_batch,
    )
    valid_loader = DataLoader(
        TensorDataset(valid_data),
        batch_size=valid_batch,
        shuffle=False,
        drop_last=False,
    )

    ns = juice.structures.HCLT(train_data, num_latents=BLOCK_SIZE, input_node_params={"num_cats": 2})
    pc = juice.compile(ns)
    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr=0.1, pseudocount=0.1, method="EM")
    scheduler = juice.optim.CircuitScheduler(
        optimizer,
        method="multi_linear",
        lrs=[0.9, 0.1, 0.05],
        milestone_steps=[0, len(train_loader) * 100, len(train_loader) * 500],
    )

    for batch in train_loader:
        x = batch[0].to(device)
        pc(x, record_cudagraph=True).mean().backward()
        break

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            x_adv = greedy_corrupt(pc, x, k)
            optimizer.zero_grad()
            lls = pc(x_adv)
            lls.mean().backward()
            train_ll += lls.mean().item()
            optimizer.step()
            scheduler.step()
        train_ll /= len(train_loader)

        t1 = time.time()
        valid_ll = 0.0
        for batch in valid_loader:
            x = batch[0].to(device)
            valid_ll += pc(x).mean().item()
        valid_ll /= len(valid_loader)
        t2 = time.time()

        print(
            f"[{dataset} K{k} epoch {epoch}/{EPOCHS}] "
            f"adv train LL {train_ll:.2f}, valid LL {valid_ll:.2f} "
            f"({t1 - t0:.1f}s train, {t2 - t1:.1f}s valid)"
        )

    pc.update_parameters()

    dst.parent.mkdir(parents=True, exist_ok=True)
    serialize_circuit(pc, dst, block_size=BLOCK_SIZE)
    print(f"Saved {dst}")


def _dataset_subprocess_cmd(dataset: str, k: int) -> list[str]:
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


def _run_dataset_subprocess(dataset: str, k: int) -> tuple[str, int]:
    cmd = _dataset_subprocess_cmd(dataset, k)
    print(f"starting {dataset}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return dataset, completed.returncode


def run_datasets_parallel(datasets: list[str], k: int, jobs: int) -> int:
    """Run one isolated subprocess per dataset, at most ``jobs`` at a time."""
    failures = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_dataset_subprocess, dataset, k): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset = futures[future]
            _, returncode = future.result()
            if returncode == 0:
                print(f"finished {dataset}")
            else:
                failures += 1
                print(f"FAILED {dataset} (exit {returncode})")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn adversarially robust HCLT PCs on DEBD datasets."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of greedy bit-flip steps per batch (corruption strength).",
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
        help="Run up to N datasets in parallel via separate subprocesses (default: 1).",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else debd_datasets()
    print(
        f"Found {len(datasets)} DEBD datasets on {device} "
        f"(K={args.k}, jobs={args.jobs})"
    )

    if args.jobs > 1 and len(datasets) > 1:
        failures = run_datasets_parallel(datasets, args.k, args.jobs)
        if failures:
            raise SystemExit(f"{failures} dataset run(s) failed")
        return

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        learn_and_save(dataset, args.k, device)


if __name__ == "__main__":
    main()