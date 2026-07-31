"""Time PeTeR OGDA throughput over DEBD datasets.

Uses K=1 and optimal lr/ratio from the sweep (TPE preferred over grid).
Matches materialization warm-start / samples / deterministic seeds.
Stops at 250 full OGDA iters or 30 seconds, whichever first.
No eval, no per-iteration logging. Warm-start sits outside the timed window.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sparc.nodes import CircuitNode

from eval import best_peter_params
from peter import DETERMINISTIC, THETA_NUM_SAMPLES, WARM_START_ITERS
from robustify import (
    ETA_LAMBDA,
    CompiledPlayer,
    log_exp_query,
    ogda_step,
    resolve_circuit_path,
)
from sweep import discover_datasets

_ROOT = Path(__file__).resolve().parent
_OUT_ROOT = _ROOT / "runtime_results"
_K = 1
_MAX_ITERS = 250
_DURATION_S = 30.0


def benchmark(dataset: str, lr: float, ratio: float) -> dict:
    p_hat = CircuitNode.load(resolve_circuit_path(dataset))
    eta_theta, eta_phi = lr, lr * ratio
    p_hat_g = p_hat.compile()
    p_theta = CompiledPlayer.clone_from(p_hat)
    q_phi = CompiledPlayer.clone_from(p_hat)
    lam = 0.0
    prev: dict = {}
    step_kw = dict(
        k=float(_K),
        eta_theta=eta_theta,
        eta_phi=eta_phi,
        eta_lambda=ETA_LAMBDA,
        theta_num_samples=THETA_NUM_SAMPLES,
        deterministic=DETERMINISTIC,
    )

    # Touch once so compile/cache costs sit outside the timed window.
    _ = log_exp_query(p_theta.graph, q_phi.graph)

    # Same Q-only warm-start as peter / materialization (carries prev into timed loop).
    for w in range(1, WARM_START_ITERS + 1):
        lam, _, _ = ogda_step(
            p_theta, q_phi, p_hat_g, lam, prev, it=w, update_theta=False, **step_kw
        )
    print(
        f"  warm start done ({WARM_START_ITERS} Q-only iters)  lambda={lam:.4f}",
        flush=True,
    )

    t0 = time.perf_counter()
    deadline = t0 + _DURATION_S
    iters = 0
    while iters < _MAX_ITERS and time.perf_counter() < deadline:
        iters += 1
        lam, _, _ = ogda_step(
            p_theta, q_phi, p_hat_g, lam, prev, it=iters, update_theta=True, **step_kw
        )
    elapsed = time.perf_counter() - t0
    stopped = "iters" if iters >= _MAX_ITERS else "time"
    return {
        "dataset": dataset,
        "k": _K,
        "lr": lr,
        "ratio": ratio,
        "warm_start_iters": WARM_START_ITERS,
        "max_iters": _MAX_ITERS,
        "max_seconds": _DURATION_S,
        "stopped_by": stopped,
        "seconds": elapsed,
        "iterations": iters,
        "iterations_per_sec": iters / elapsed if elapsed > 0 else 0.0,
        "final_lambda": lam,
    }


def main() -> None:
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets(_K)
    print(
        f"timing {len(datasets)} datasets  k={_K}  "
        f"stop at {_MAX_ITERS} iters or {_DURATION_S:g}s  "
        f"warm_start={WARM_START_ITERS}"
    )

    for dataset in datasets:
        best = best_peter_params(dataset, _K)
        if best is None:
            print(f"skip  {dataset}: no sweep winner")
            continue
        lr, ratio, source = best
        print(f"start {dataset}  lr={lr:g}  ratio={ratio:g}  ({source})", flush=True)
        result = benchmark(dataset, lr, ratio)
        result["source"] = source
        out = _OUT_ROOT / f"{dataset}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"done  {dataset}: {result['iterations']} iterations in "
            f"{result['seconds']:.4f}s  ({result['iterations_per_sec']:.3f} iters/s, "
            f"stopped_by={result['stopped_by']})  -> {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
