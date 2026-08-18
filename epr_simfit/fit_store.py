"""Global fit registry (Open-Sym-EPR Pro).

A session-wide store of completed fits so that *any* tab can save a fit and *any*
plotting tab can load one back — to overlay it on a spectrum, export it, or reuse
its refined parameters as the starting model for a further optimisation.

Entries are kept in ``st.session_state["fit_registry"]`` as an ordered dict keyed by
a unique display name. Each entry holds the field axis, the experimental and fitted
curves (for overlay/export), the refined ``SpinComponent`` list and weights (for
"use as starting model"), plus provenance (source tab, timestamp, R²). Nothing here
imports Streamlit; the caller passes ``st.session_state`` in as ``state``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

REGISTRY_KEY = "fit_registry"


def registry(state) -> dict[str, dict]:
    """Return the ordered registry dict, creating it on first use."""
    return state.setdefault(REGISTRY_KEY, {})


def _unique_name(state, name: str) -> str:
    base = (name or "fit").strip() or "fit"
    reg = registry(state)
    if base not in reg:
        return base
    i = 2
    while f"{base} ({i})" in reg:
        i += 1
    return f"{base} ({i})"


def add(state, name: str, *, field_mT, experimental, fit_total, components, weights,
        mw_frequency_GHz: float, R2: float, source: str, n_parameters: int = 0,
        extra: dict | None = None) -> str:
    """Store one fit; returns the (possibly de-duplicated) name it was stored under."""
    key = _unique_name(state, name)
    registry(state)[key] = {
        "name": key,
        "source": source,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "field_mT": np.asarray(field_mT, dtype=float),
        "experimental": np.asarray(experimental, dtype=float),
        "fit_total": np.asarray(fit_total, dtype=float),
        "components": [c.clone() for c in components],
        "weights": dict(weights or {}),
        "mw_frequency_GHz": float(mw_frequency_GHz),
        "R2": float(R2) if R2 is not None else float("nan"),
        "n_parameters": int(n_parameters),
        "extra": dict(extra or {}),
    }
    return key


def add_fitresult(state, name: str, fit, mw_frequency_GHz: float, source: str,
                  extra: dict | None = None) -> str:
    """Convenience: store an ``epr_simfit.fitter.FitResult``."""
    return add(
        state, name,
        field_mT=fit.field_mT, experimental=fit.experimental, fit_total=fit.fit_total,
        components=fit.components, weights=fit.weights, mw_frequency_GHz=mw_frequency_GHz,
        R2=fit.metrics.get("R2", float("nan")), source=source,
        n_parameters=getattr(fit, "n_parameters", 0), extra=extra,
    )


def names(state) -> list[str]:
    return list(registry(state).keys())


def get(state, name: str) -> dict | None:
    return registry(state).get(name)


def delete(state, name: str) -> None:
    registry(state).pop(name, None)


def summary_rows(state) -> list[dict[str, Any]]:
    """Compact table rows for a saved-fit manager."""
    rows = []
    for e in registry(state).values():
        rows.append({
            "name": e["name"], "source": e["source"],
            "R²": round(e["R2"], 4) if np.isfinite(e["R2"]) else None,
            "components": len(e["components"]), "saved": e["timestamp"],
        })
    return rows


def overlay_traces(state, selected: list[str], target_field, *, include_experimental=False) -> dict:
    """Return {label: y} traces for the selected saved fits, interpolated onto
    ``target_field`` so they can be added to a plot that uses a different axis.
    Regions outside a saved fit's own field range are left as NaN (not drawn)."""
    tf = np.asarray(target_field, dtype=float)
    out: dict[str, np.ndarray] = {}
    for name in selected:
        e = registry(state).get(name)
        if e is None:
            continue
        src, sy = e["field_mT"], e["fit_total"]
        y = np.interp(tf, src, sy, left=np.nan, right=np.nan)
        out[f"[saved] {name}"] = y
        if include_experimental:
            out[f"[saved-exp] {name}"] = np.interp(tf, src, e["experimental"], left=np.nan, right=np.nan)
    return out


def export_csv(state, name: str) -> str | None:
    """Origin-ready CSV (Field_mT, experimental, fit, residual) for one saved fit."""
    e = registry(state).get(name)
    if e is None:
        return None
    B, y, f = e["field_mT"], e["experimental"], e["fit_total"]
    res = y - f
    lines = ["Field_mT,experimental,fit,residual"]
    for i in range(len(B)):
        lines.append(f"{B[i]:.6g},{y[i]:.6g},{f[i]:.6g},{res[i]:.6g}")
    return "\n".join(lines)
