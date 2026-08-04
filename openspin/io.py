"""Load experimental cw-EPR spectra from text / CSV / ASC files."""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd


def parse_spectrum(text: str, field_col: int = 0, intensity_col: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (field, intensity) spectrum from arbitrary text.

    Skips non-numeric header lines, auto-detects the delimiter, and converts a
    Gauss field axis to mT when the values look like Gauss (median > 1000).
    Returns (field_mT, intensity).
    """
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\s,;\t]+", line)
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        raise ValueError("No numeric two-column data found in the file.")
    arr = np.array([r[: max(field_col, intensity_col) + 1] for r in rows if len(r) > max(field_col, intensity_col)])
    field = arr[:, field_col].astype(float)
    inten = arr[:, intensity_col].astype(float)
    # Gauss -> mT
    if np.median(np.abs(field)) > 1000:
        field = field / 10.0
    order = np.argsort(field)
    return field[order], inten[order]


def normalise(y: np.ndarray, method: str = "max") -> np.ndarray:
    """Normalise a spectrum for fitting (max-absolute by default)."""
    y = np.asarray(y, dtype=float)
    if method == "max":
        s = np.nanmax(np.abs(y))
    elif method == "ptp":
        s = np.nanmax(y) - np.nanmin(y)
    else:
        s = 1.0
    return y / s if s and np.isfinite(s) else y


def to_dataframe(field_mT: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Field_mT": field_mT, "Intensity": intensity})
