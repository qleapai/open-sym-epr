"""Publication-quality export bundles (Open-Sym-EPR Pro).

Builds a single ZIP containing, for each selected result (main fit, automated
mixture fit, and/or manual mixture): the decomposed data as CSV (field,
experimental, total, residual, and each weighted component), a parameter/ratio
table as CSV, and 600-dpi publication figures (overlay + decomposition).
Nothing here imports Streamlit; the app hands in plain dict payloads.
"""
from __future__ import annotations

import re
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "result"


def _fig_png(field, traces: dict, title: str, ylabel: str = "dχ″/dB (norm.)", dpi: int = 600) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Arial with a DejaVu fallback so the Unicode minus renders (see project notes).
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for nm, y in traces.items():
        ax.plot(np.asarray(field, float), np.asarray(y, float), lw=1.3, label=nm)
    ax.set_xlabel("Magnetic field / mT")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _data_frame(field, experimental, total, components: dict) -> pd.DataFrame:
    data = {"Field_mT": np.asarray(field, float)}
    if experimental is not None:
        data["experimental"] = np.asarray(experimental, float)
    if total is not None:
        data["total"] = np.asarray(total, float)
    if experimental is not None and total is not None:
        data["residual"] = np.asarray(experimental, float) - np.asarray(total, float)
    for cn, cv in components.items():
        data[cn] = np.asarray(cv, float)
    return pd.DataFrame(data)


def build_selection_zip(items: dict, meta: dict | None = None) -> bytes:
    """items: {display_name: payload}. Each payload:
        kind: "fit" | "mixture" | "manual"
        field_mT, total, components {name: weighted_curve}
        experimental (optional), R2 (optional), params_df (optional DataFrame)
    Returns the ZIP bytes."""
    buf = BytesIO()
    manifest = ["Open-Sym-EPR — publication export", "=" * 36, ""]
    if meta:
        manifest += [f"{k}: {v}" for k, v in meta.items()] + [""]
    with ZipFile(buf, "w", ZIP_DEFLATED) as z:
        for name, p in items.items():
            folder = _safe(name)
            field = p["field_mT"]
            comps = p.get("components", {})
            total = p.get("total")
            exp = p.get("experimental")
            kind = p.get("kind", "fit")
            r2 = p.get("R2")

            z.writestr(f"{folder}/data.csv", _data_frame(field, exp, total, comps).to_csv(index=False))
            if p.get("params_df") is not None:
                z.writestr(f"{folder}/parameters.csv", p["params_df"].to_csv(index=False))

            overlay = {}
            if exp is not None:
                overlay["experimental"] = exp
            if total is not None:
                overlay["composite" if kind == "manual" else "fit"] = total
            if overlay:
                ttl = name + (f"  (R²={r2:.3f})" if isinstance(r2, (int, float)) else "")
                z.writestr(f"{folder}/figure_overlay.png", _fig_png(field, overlay, ttl))
            if comps:
                z.writestr(f"{folder}/figure_decomposition.png",
                           _fig_png(field, comps, f"{name} — decomposition"))
            manifest.append(f"- {name} [{kind}]: data.csv"
                            + (", parameters.csv" if p.get("params_df") is not None else "")
                            + (", figure_overlay.png" if overlay else "")
                            + (", figure_decomposition.png" if comps else ""))
        z.writestr("README.txt", "\n".join(manifest))
    return buf.getvalue()
