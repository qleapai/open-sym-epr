"""Model comparison metrics for fitted presets."""

from __future__ import annotations

import pandas as pd

from .constants import delta_bic_label
from .fitter import FitResult, fit_spectrum


def compare_models(
    field_mT,
    intensity,
    presets: list[str],
    mw_frequency_GHz: float = 9.85,
    mode: str = "weights only",
    baseline_order: int = 0,
) -> tuple[pd.DataFrame, dict[str, FitResult]]:
    results: dict[str, FitResult] = {}
    rows = []
    for preset in presets:
        fit = fit_spectrum(field_mT, intensity, preset=preset, mw_frequency_GHz=mw_frequency_GHz, mode=mode, baseline_order=baseline_order)
        results[preset] = fit
        rows.append({"model": preset, "k": fit.n_parameters, **fit.metrics})
    df = pd.DataFrame(rows)
    if not df.empty:
        min_aic = df["AIC"].min()
        min_bic = df["BIC"].min()
        df["Delta_AIC"] = df["AIC"] - min_aic
        df["Delta_BIC"] = df["BIC"] - min_bic
    return df, results


def bic_improvement(simple_fit: FitResult, candidate_fit: FitResult) -> dict[str, float | str]:
    delta = simple_fit.metrics["BIC"] - candidate_fit.metrics["BIC"]
    return {"Delta_BIC_simple_minus_candidate": delta, "support": delta_bic_label(delta)}
