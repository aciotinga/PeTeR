"""Shared paths and helpers for metadata-only hyperparameter sweeps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from peter import (
    RunMetrics,
    error_path,
    format_hyperparam_dir,
    is_run_finished,
    metrics_path,
    save_json,
)

_ROOT = Path(__file__).resolve().parent
SWEEPS_ROOT = _ROOT / "sweeps"
GRID_ROOT = SWEEPS_ROOT / "grid"
TPE_ROOT = SWEEPS_ROOT / "tpe"


def grid_run_dir(dataset: str, k: int, lr: float, ratio: float) -> Path:
    """``sweeps/grid/k{k}/<dataset>/lr=<lr>_ratio=<ratio>/``"""
    return GRID_ROOT / f"k{k}" / dataset / format_hyperparam_dir(lr, ratio)


def grid_summary_path(k: int) -> Path:
    return GRID_ROOT / f"k{k}" / "sweep_summary.json"


def tpe_dataset_dir(dataset: str, k: int) -> Path:
    """``sweeps/tpe/k{k}/<dataset>/``"""
    return TPE_ROOT / f"k{k}" / dataset


def tpe_journal_path(dataset: str, k: int) -> Path:
    return tpe_dataset_dir(dataset, k) / "journal.log"


def tpe_study_summary_path(dataset: str, k: int) -> Path:
    return tpe_dataset_dir(dataset, k) / "study_summary.json"


def tpe_trial_dir(dataset: str, k: int, trial_number: int) -> Path:
    return tpe_dataset_dir(dataset, k) / "trials" / f"trial_{trial_number}"


def load_metrics(out_dir: Path) -> RunMetrics | None:
    path = metrics_path(out_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunMetrics(**payload)


def load_error(out_dir: Path) -> dict | None:
    path = error_path(out_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(path: Path, payload: dict) -> Path:
    payload = {
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(path, payload)
    return path


__all__ = [
    "GRID_ROOT",
    "SWEEPS_ROOT",
    "TPE_ROOT",
    "grid_run_dir",
    "grid_summary_path",
    "is_run_finished",
    "load_error",
    "load_metrics",
    "metrics_path",
    "error_path",
    "save_json",
    "tpe_dataset_dir",
    "tpe_journal_path",
    "tpe_study_summary_path",
    "tpe_trial_dir",
    "write_summary",
]
