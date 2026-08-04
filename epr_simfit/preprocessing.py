"""Spectrum preprocessing: crop, baseline correction, normalization, field alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags, savgol_filter

from .utils import clean_finite_arrays


@dataclass
class PreprocessResult:
    field_mT: np.ndarray
    raw: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    processed: np.ndarray
    display: np.ndarray
    mask: np.ndarray
    settings: dict


def estimate_baseline(field: np.ndarray, y: np.ndarray, method: str = "none", polynomial_order: int = 1) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "none":
        return np.zeros_like(y)
    if method == "constant":
        edge = max(3, int(0.08 * len(y)))
        return np.full_like(y, float(np.median(np.r_[y[:edge], y[-edge:]])))

    edge = max(4, int(0.12 * len(y)))
    idx = np.r_[np.arange(edge), np.arange(len(y) - edge, len(y))]
    if method == "linear edge":
        order = 1
    elif method == "polynomial edge":
        order = max(1, min(int(polynomial_order), 5))
    else:
        return np.zeros_like(y)
    coeff = np.polyfit(field[idx], y[idx], order)
    return np.polyval(coeff, field)


def normalize_signal(y: np.ndarray, method: str = "max absolute", field: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if method == "none":
        return y.copy()
    if method == "max absolute":
        scale = np.nanmax(np.abs(y)) if y.size else 0.0
    elif method == "peak-to-peak":
        scale = np.nanmax(y) - np.nanmin(y) if y.size else 0.0
    elif method == "area":
        if field is None:
            scale = np.trapz(np.abs(y))
        else:
            scale = np.trapz(np.abs(y), np.asarray(field, dtype=float))
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale == 0:
        return y.copy()
    return y / scale


def auto_align_spectra(
    exp_intensity: np.ndarray,
    ref_intensity: np.ndarray,
    field_mT: np.ndarray,
    max_shift_mT: float = 5.0,
) -> float:
    """Return the field shift (mT) needed to move the experimental dominant peak onto the simulated peak.

    Both spectra must be sampled on the same field_mT grid.
    Strategy:
      1. Primary: match the positions of the global |max| peak in each spectrum.
      2. Refinement: cross-correlation within ±max_shift_mT to sub-sample precision.
    A positive return value shifts the experimental field axis up (spectrum moves left on plot).
    """
    exp = np.asarray(exp_intensity, dtype=float)
    ref = np.asarray(ref_intensity, dtype=float)
    field = np.asarray(field_mT, dtype=float)

    if len(exp) < 3 or len(ref) < 3 or len(field) < 3:
        return 0.0

    # ── Step 1: locate dominant peak in each spectrum ──────────────────────
    i_exp = int(np.argmax(np.abs(exp)))
    i_ref = int(np.argmax(np.abs(ref)))
    coarse_shift = float(field[i_ref] - field[i_exp])
    coarse_shift = float(np.clip(coarse_shift, -max_shift_mT, max_shift_mT))

    # ── Step 2: cross-correlation refinement in a window around coarse peak ─
    step = float(np.median(np.abs(np.diff(field)))) if len(field) > 1 else 0.1
    if step <= 0:
        return round(coarse_shift, 4)

    window = max(3, int(round(max_shift_mT / step)))
    lo_exp = max(0, i_exp - window)
    hi_exp = min(len(exp), i_exp + window + 1)
    lo_ref = max(0, i_ref - window)
    hi_ref = min(len(ref), i_ref + window + 1)

    exp_win = exp[lo_exp:hi_exp].copy()
    ref_win = ref[lo_ref:hi_ref].copy()

    for arr in (exp_win, ref_win):
        sc = np.nanmax(np.abs(arr))
        if sc > 0:
            arr /= sc

    corr = correlate(exp_win, ref_win, mode="full")
    lags = correlation_lags(len(exp_win), len(ref_win), mode="full")
    max_lag_samples = max(1, int(round(max_shift_mT / step)))
    valid = np.abs(lags) <= max_lag_samples
    if valid.any():
        best_lag = int(lags[valid][np.argmax(corr[valid])])
    else:
        best_lag = 0

    fine_shift = coarse_shift - best_lag * step
    fine_shift = float(np.clip(fine_shift, -max_shift_mT, max_shift_mT))
    return round(fine_shift, 4)


def preprocess_spectrum(
    field_mT,
    intensity,
    crop_min: float | None = None,
    crop_max: float | None = None,
    field_shift_mT: float = 0.0,
    baseline_method: str = "linear edge",
    polynomial_order: int = 2,
    normalization: str = "max absolute",
    smooth_display: bool = False,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    invert: bool = False,
) -> PreprocessResult:
    field, raw = clean_finite_arrays(field_mT, intensity)
    if invert:
        raw = -raw
    if field_shift_mT:
        field = field + float(field_shift_mT)
    mask = np.ones_like(field, dtype=bool)
    if crop_min is not None:
        mask &= field >= float(crop_min)
    if crop_max is not None:
        mask &= field <= float(crop_max)
    field = field[mask]
    raw = raw[mask]
    baseline = estimate_baseline(field, raw, baseline_method, polynomial_order)
    corrected = raw - baseline
    processed = normalize_signal(corrected, normalization, field)
    display = processed.copy()
    if smooth_display and len(display) >= 7:
        window = int(smooth_window)
        if window % 2 == 0:
            window += 1
        window = min(window, len(display) - (1 - len(display) % 2))
        if window >= 5:
            display = savgol_filter(display, window, min(int(smooth_polyorder), window - 2))
    return PreprocessResult(
        field_mT=field,
        raw=raw,
        baseline=baseline,
        corrected=corrected,
        processed=processed,
        display=display,
        mask=mask,
        settings={
            "crop_min": crop_min,
            "crop_max": crop_max,
            "baseline_method": baseline_method,
            "polynomial_order": polynomial_order,
            "field_shift_mT": field_shift_mT,
            "normalization": normalization,
            "smooth_display": smooth_display,
            "invert": invert,
        },
    )
