"""Batch / time-series / catalyst-series processing with kinetic export.

Applies a single fixed model to a series of spectra (e.g. a kinetic time series
or a catalyst/condition series), tracks the fitted component fractions across
the series, and produces kinetic tables and plots (fraction vs time/index).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fitter import fit_spectrum
from .io import parse_epr_text
from .preprocessing import preprocess_spectrum
from .spin_models import SpinComponent


@dataclass
class BatchSpectrum:
    """One spectrum in a series, with an ordering coordinate (time/index/label)."""
    label: str
    coordinate: float          # time, index, potential, dose, etc.
    field_mT: np.ndarray
    intensity: np.ndarray


@dataclass
class BatchResult:
    coordinate_name: str
    fractions: pd.DataFrame    # rows: spectra; cols: coordinate + component fractions
    weights: pd.DataFrame      # absolute weights
    metrics: pd.DataFrame      # per-spectrum fit metrics
    errors: pd.DataFrame       # per-spectrum fraction standard errors
    fits: list = field(default_factory=list)


def batch_from_texts(
    named_texts: list[tuple[str, float, str]],
    mw_frequency_GHz: float = 9.85,
) -> list[BatchSpectrum]:
    """Parse and preprocess a list of (label, coordinate, file_text) into spectra."""
    out: list[BatchSpectrum] = []
    for label, coord, text in named_texts:
        parsed = parse_epr_text(text, filename=label, mw_frequency_override=mw_frequency_GHz)
        prep = preprocess_spectrum(parsed.dataframe["Field_mT"], parsed.dataframe["Intensity_raw"])
        out.append(BatchSpectrum(label=label, coordinate=float(coord),
                                 field_mT=prep.field_mT, intensity=prep.processed))
    return out


def run_batch(
    spectra: list[BatchSpectrum],
    components: list[SpinComponent],
    mw_frequency_GHz: float = 9.85,
    mode: str = "weights only",
    baseline_order: int = 0,
    coordinate_name: str = "coordinate",
    n_orientations: int = 600,
    max_nfev: int = 400,
) -> BatchResult:
    """Fit the SAME fixed model to every spectrum and assemble kinetic tables."""
    frac_rows, weight_rows, metric_rows, err_rows, fits = [], [], [], [], []
    for sp in spectra:
        fit = fit_spectrum(
            sp.field_mT, sp.intensity, components=[c.clone() for c in components],
            mw_frequency_GHz=mw_frequency_GHz, mode=mode, baseline_order=baseline_order,
            n_orientations=n_orientations, max_nfev=max_nfev,
        )
        fits.append(fit)
        cf = fit.component_fractions.set_index("component")
        frac_rows.append({coordinate_name: sp.coordinate, "label": sp.label,
                          **{cid: cf.loc[cid, "fraction"] for cid in cf.index}})
        weight_rows.append({coordinate_name: sp.coordinate, "label": sp.label,
                            **{cid: cf.loc[cid, "weight"] for cid in cf.index}})
        err_rows.append({coordinate_name: sp.coordinate, "label": sp.label,
                         **{cid: cf.loc[cid, "fraction_std_error"] for cid in cf.index}})
        metric_rows.append({coordinate_name: sp.coordinate, "label": sp.label, **fit.metrics})

    return BatchResult(
        coordinate_name=coordinate_name,
        fractions=pd.DataFrame(frac_rows),
        weights=pd.DataFrame(weight_rows),
        metrics=pd.DataFrame(metric_rows),
        errors=pd.DataFrame(err_rows),
        fits=fits,
    )


def kinetic_plot_png(result: BatchResult, title: str = "Component kinetics") -> bytes:
    """Render component fraction vs coordinate as a PNG (with error bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.fractions.sort_values(result.coordinate_name)
    err = result.errors.set_index(result.coordinate_name)
    x = df[result.coordinate_name].to_numpy()
    comp_cols = [c for c in df.columns if c not in (result.coordinate_name, "label")]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cid in comp_cols:
        y = df[cid].to_numpy() * 100.0
        try:
            ye = err.loc[df[result.coordinate_name], cid].to_numpy() * 100.0
        except Exception:  # noqa: BLE001
            ye = None
        ax.errorbar(x, y, yerr=ye, marker="o", capsize=3, lw=1.6, ms=5, label=cid)
    ax.set_xlabel(result.coordinate_name)
    ax.set_ylabel("Spectral fraction (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# Backwards-compatible helper retained for older callers.
def analyze_batch(
    named_texts: list[tuple[str, str, str]],
    mw_frequency_GHz: float = 9.85,
    presets: tuple[str, ...] = ("M1_water_dmso", "M2_water_dmso_o2", "N1_n2rr_dmso_corrected"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy preset-sweep batch analysis (kept for compatibility)."""
    from .model_library import components_for_preset

    metric_rows, fraction_rows = [], []
    for filename, condition, text in named_texts:
        parsed = parse_epr_text(text, filename=filename, mw_frequency_override=mw_frequency_GHz)
        prep = preprocess_spectrum(parsed.dataframe["Field_mT"], parsed.dataframe["Intensity_raw"])
        for preset in presets:
            fit = fit_spectrum(prep.field_mT, prep.processed, preset=preset,
                               components=components_for_preset(preset), mw_frequency_GHz=mw_frequency_GHz)
            metric_rows.append({"filename": filename, "condition": condition, "model": preset, **fit.metrics})
            for _, row in fit.component_fractions.iterrows():
                fraction_rows.append({"filename": filename, "condition": condition, "model": preset, **row.to_dict()})
    return pd.DataFrame(metric_rows), pd.DataFrame(fraction_rows)
