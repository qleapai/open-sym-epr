"""Plotly and Matplotlib plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Cursor-readout configuration (an ADDITIVE overlay of information on hover — it never
# removes the default point). The app sets these once per run; call sites need not change.
#   HOVER_READOUT: "xy" (field+intensity), "g" (field+g-value), or "both".
#   MW_FREQUENCY_GHZ: microwave frequency used to convert each field point to a g-value.
HOVER_READOUT = "both"
MW_FREQUENCY_GHZ: float | None = None
# g = _PLANCK_GHZ_MT * nu_GHz / B_mT   (h / muB, scaled to GHz and mT)
_PLANCK_GHZ_MT = 6.62607015e-34 * 1e12 / 9.2740100783e-24  # ~71.447


def _g_values(field, mw_GHz):
    B = np.asarray(field, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(B > 0, _PLANCK_GHZ_MT * float(mw_GHz) / B, np.nan)


def _hover_template(readout: str, has_g: bool) -> str:
    xy = "Field: %{x:.3f} mT<br>Intensity: %{y:.4g}"
    g = "g: %{customdata:.5f}"
    if readout == "g" and has_g:
        body = "Field: %{x:.3f} mT<br>" + g
    elif readout == "both" and has_g:
        body = xy + "<br>" + g
    else:
        body = xy
    return body + "<extra>%{fullData.name}</extra>"


def spectrum_figure(field, traces: dict[str, np.ndarray], title: str = "EPR spectrum",
                    readout: str | None = None, mw_frequency_GHz: float | None = None) -> go.Figure:
    readout = (readout or HOVER_READOUT or "xy").lower()
    mw = mw_frequency_GHz if mw_frequency_GHz is not None else MW_FREQUENCY_GHZ
    gvals = _g_values(field, mw) if (mw and readout in ("g", "both")) else None
    tmpl = _hover_template(readout, gvals is not None)
    fig = go.Figure()
    for name, y in traces.items():
        fig.add_trace(go.Scatter(x=field, y=y, mode="lines", name=name,
                                 customdata=gvals, hovertemplate=tmpl))
    fig.update_layout(
        title=title,
        xaxis_title="Magnetic field / mT",
        yaxis_title="Intensity / a.u.",
        template="plotly_white",
        height=430,
        margin=dict(l=54, r=24, t=54, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    # Crosshair spikes so a readout appears wherever the cursor sits on a trace.
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikedash="dot", spikecolor="#888")
    fig.update_yaxes(showspikes=True, spikethickness=1, spikedash="dot", spikecolor="#888")
    return fig


def residual_figure(field, residual, title: str = "Residual") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=field, y=residual, mode="lines", name="Residual"))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#555")
    fig.update_layout(
        title=title,
        xaxis_title="Magnetic field / mT",
        yaxis_title="Experimental - fit",
        template="plotly_white",
        height=290,
        margin=dict(l=54, r=24, t=54, b=48),
    )
    return fig


def component_figure(field, component_curves: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> go.Figure:
    fig = go.Figure()
    for cid, curve in component_curves.items():
        weight = 1.0 if weights is None else weights.get(cid, 1.0)
        fig.add_trace(go.Scatter(x=field, y=weight * curve, mode="lines", name=cid))
    fig.update_layout(
        title="Component decomposition",
        xaxis_title="Magnetic field / mT",
        yaxis_title="Weighted component / a.u.",
        template="plotly_white",
        height=430,
        margin=dict(l=54, r=24, t=54, b=48),
    )
    return fig


def comparison_bar_figure(comparison: pd.DataFrame, metric: str = "BIC") -> go.Figure:
    fig = go.Figure()
    if comparison is not None and not comparison.empty:
        fig.add_trace(go.Bar(x=comparison["model"], y=comparison[metric], name=metric))
    fig.update_layout(
        title=f"Model comparison ({metric})",
        xaxis_title="Model",
        yaxis_title=metric,
        template="plotly_white",
        height=360,
        margin=dict(l=54, r=24, t=54, b=72),
    )
    return fig


def heatmap_figure(table: pd.DataFrame, title: str = "Component fraction heatmap") -> go.Figure:
    if table.empty:
        return go.Figure()
    pivot = table.pivot_table(index="condition", columns="component", values="fraction", fill_value=0.0)
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_white", height=420, margin=dict(l=90, r=24, t=54, b=90))
    return fig
