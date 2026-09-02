"""Export any Plotly figure as CSV / PNG / SVG (Open-Sym-EPR Pro).

Kaleido (Plotly's static-image engine) is not bundled, so images are re-rendered
from the figure's own traces with matplotlib — lightweight and dependency-free
beyond matplotlib, which is already required. Works for line (Scatter) and bar
figures; every trace becomes a column in the CSV and a series in the image.
"""
from __future__ import annotations

from io import BytesIO, StringIO

import numpy as np
import pandas as pd


def _title(fig, axis: str | None = None) -> str:
    lay = fig.layout
    if axis == "x":
        return (lay.xaxis.title.text if lay.xaxis and lay.xaxis.title else None) or "x"
    if axis == "y":
        return (lay.yaxis.title.text if lay.yaxis and lay.yaxis.title else None) or "y"
    return (lay.title.text if lay.title else None) or "figure"


def fig_to_csv(fig) -> str:
    """Wide CSV: the x column plus one column per trace (y), when traces share x;
    otherwise x/y column pairs per trace."""
    traces = list(fig.data)
    if not traces:
        return ""
    xname = _title(fig, "x")
    x0 = traces[0].x
    same = x0 is not None and all(t.x is not None and len(t.x) == len(x0) for t in traces)
    if same:
        data = {xname: list(x0)}
        for i, t in enumerate(traces):
            nm = t.name or f"trace{i + 1}"
            base, k = nm, 1
            while nm in data:
                k += 1
                nm = f"{base}_{k}"
            data[nm] = list(t.y) if t.y is not None else [np.nan] * len(x0)
        return pd.DataFrame(data).to_csv(index=False)
    buf = StringIO()
    frames = []
    for i, t in enumerate(traces):
        nm = t.name or f"trace{i + 1}"
        xs = list(t.x) if t.x is not None else []
        ys = list(t.y) if t.y is not None else []
        frames.append(pd.DataFrame({f"{nm}_x": xs, f"{nm}_y": ys}))
    pd.concat(frames, axis=1).to_csv(buf, index=False)
    return buf.getvalue()


def fig_to_image(fig, fmt: str = "png", dpi: int = 600) -> bytes:
    """Re-render the figure's traces with matplotlib and return PNG or SVG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]  # Unicode-minus safe
    mfig, ax = plt.subplots(figsize=(6.8, 4.3))
    has_label = False
    for t in fig.data:
        nm = t.name
        has_label = has_label or bool(nm)
        xs = list(t.x) if t.x is not None else []
        ys = list(t.y) if t.y is not None else []
        if getattr(t, "type", "") == "bar":
            ax.bar([str(v) for v in xs], ys, label=nm)
        else:
            ax.plot(xs, ys, lw=1.3, label=nm)
    ax.set_xlabel(_title(fig, "x"))
    ax.set_ylabel(_title(fig, "y"))
    ttl = _title(fig)
    if ttl and ttl != "figure":
        ax.set_title(ttl)
    if has_label:
        ax.legend(frameon=False, fontsize=8)
    mfig.tight_layout()
    buf = BytesIO()
    mfig.savefig(buf, format=fmt, dpi=dpi)
    plt.close(mfig)
    return buf.getvalue()
