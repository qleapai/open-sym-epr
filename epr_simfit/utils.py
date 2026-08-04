"""Small utility helpers shared across modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_finite_arrays(*arrays: Iterable[float]) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    if not arrs:
        return []
    mask = np.ones_like(arrs[0], dtype=bool)
    for arr in arrs:
        mask &= np.isfinite(arr)
    return [arr[mask] for arr in arrs]


def normalize_maxabs(y: Iterable[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    scale = np.nanmax(np.abs(arr)) if arr.size else 0.0
    if not np.isfinite(scale) or scale == 0:
        return arr.copy()
    return arr / scale


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
