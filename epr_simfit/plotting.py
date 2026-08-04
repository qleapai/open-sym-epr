"""Plotly and Matplotlib plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def spectrum_figure(field, traces: dict[str, np.ndarray], title: str = "EPR spectrum") -> go.Figure:
    fig = go.Figure()
    for name, y in traces.items():
        fig.add_trace(go.Scatter(x=field, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Magnetic field / mT",
        yaxis_title="Intensity / a.u.",
        template="plotly_white",
        height=430,
        margin=dict(l=54, r=24, t=54, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
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
