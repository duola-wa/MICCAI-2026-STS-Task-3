"""Portable project paths shared by the command-line entry points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY_NAME = "MICCAI-Chllenge-STS26-Task3"


def _environment_path(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def default_data_root() -> Path:
    """Find the dataset on Ubuntu and on the original Windows workstation.

    Search order:
    1. ``MMDENTAL_DATA_ROOT``;
    2. a dataset directory inside the project (the Ubuntu layout);
    3. a dataset directory next to the project (the original Windows layout).
    """
    configured = _environment_path("MMDENTAL_DATA_ROOT")
    if configured is not None:
        return configured

    candidates = [
        PROJECT_ROOT / DATA_DIRECTORY_NAME,
        PROJECT_ROOT.parent / DATA_DIRECTORY_NAME,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Return the preferred Ubuntu layout so error messages point to the expected
    # location even before the data has been copied.
    return candidates[0]


def default_cache_dir() -> Path:
    configured = _environment_path("MMDENTAL_CACHE_DIR")
    return configured if configured is not None else PROJECT_ROOT / "cache" / "views_s12_224"


def default_segmentation_dir(data_root: Optional[Path] = None) -> Path:
    """Return the flat nnU-Net prediction directory containing ``<case>.nii.gz``."""
    configured = _environment_path("MMDENTAL_SEGMENTATION_DIR")
    if configured is not None:
        return configured
    root = Path(data_root) if data_root is not None else default_data_root()
    return root / "prediction"


def default_runs_dir() -> Path:
    configured = _environment_path("MMDENTAL_RUNS_DIR")
    return configured if configured is not None else PROJECT_ROOT / "runs"


def default_predictions_dir() -> Path:
    configured = _environment_path("MMDENTAL_PREDICTIONS_DIR")
    return configured if configured is not None else PROJECT_ROOT / "predictions"


def default_work_dir() -> Path:
    configured = _environment_path("MMDENTAL_WORK_DIR")
    return configured if configured is not None else PROJECT_ROOT / "work"
