"""Publication-ready DOCX report (Open-Sym-EPR Pro).

Produces a clean, colour-free Word report for one or more decomposition results:
a manuscript methods paragraph, a publication figure (experimental + fit overlay
and the component decomposition) with a detailed caption, and a formatted table
of the fitted spin-Hamiltonian parameters and spectral fractions of every
decomposed spin-adduct component — with proper italic g, subscripted a and ΔBpp,
and superscripted R². No Streamlit dependency; the app hands in plain dict entries.

Each entry:
    name, kind ("fit"/"mixture"/"manual"),
    field_mT, total, experimental (optional), curves {label: weighted_curve},
    rows [{component, assignment, g, avals [(isotope,label,A_mT)], linewidth_mT,
           eta, fraction_pct}],
    R2, nrmse, aic, bic, n_params (optional), methods_text (optional).
"""
from __future__ import annotations

from io import BytesIO

import numpy as np


# ── rich-text helpers (real Word sub/superscripts) ──────────────────────────────
def _run(p, text, *, italic=False, bold=False, sub=False, sup=False, size=None):
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    if sub:
        r.font.subscript = True
    if sup:
        r.font.superscript = True
    if size is not None:
        from docx.shared import Pt
        r.font.size = Pt(size)
    return r


def _sym(p, kind):
    """Emit a formatted scientific symbol into paragraph/cell p."""
    if kind == "g":
        _run(p, "g", italic=True)
    elif kind == "R2":
        _run(p, "R", italic=True); _run(p, "2", sup=True)
    elif kind == "dBpp":
        _run(p, "ΔB"); _run(p, "pp", sub=True); _run(p, " (mT)")
    elif kind == "aN":
        _run(p, "a", italic=True); _run(p, "N", sub=True); _run(p, " (G)")
    elif kind == "aiso":
        _run(p, "a", italic=True); _run(p, "iso", sub=True); _run(p, " (G)")


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _iso_label(isotope: str) -> str:
    """'14N' -> '¹⁴N' (mass number as Unicode superscript)."""
    num = "".join(ch for ch in isotope if ch.isdigit())
    el = "".join(ch for ch in isotope if not ch.isdigit())
    return num.translate(_SUP) + el


def _avals_text(avals) -> str:
    return "; ".join(f"{_iso_label(iso)} {round(A * 10, 2)}" for iso, _lab, A in avals) or "—"


def _report_figure(entry) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    B = np.asarray(entry["field_mT"], float)
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.4), dpi=300, sharex=True)
    if entry.get("experimental") is not None:
        ax[0].plot(B, entry["experimental"], color="0.15", lw=1.2, label="experimental")
    ax[0].plot(B, entry["total"], color="#b22222", lw=1.4,
               label=("composite" if entry.get("kind") == "manual" else "fit"))
    ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylabel("dχ″/dB (norm.)")
    for label, curve in entry.get("curves", {}).items():
        ax[1].plot(B, np.asarray(curve, float), lw=1.0, label=label)
    ax[1].legend(frameon=False, fontsize=7); ax[1].set_ylabel("component")
    ax[1].set_xlabel("Magnetic field / mT")
    fig.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=300); plt.close(fig)
    return buf.getvalue()


def _fmt(v):
    """Compact publication formatting for a table cell value."""
    try:
        import math
        f = float(v)
        if math.isnan(f):
            return "—"
        if math.isinf(f):
            return "∞" if f > 0 else "−∞"
        if f == int(f) and abs(f) < 1e6:
            return str(int(f))
        return f"{f:.5g}"
    except (TypeError, ValueError, OverflowError):
        return "" if v is None else str(v)


def _add_df_table(doc, df, tab_no, caption_text):
    """Render a whole DataFrame as a bordered, colour-free Word table with a caption."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    tcap = doc.add_paragraph()
    _run(tcap, f"Table {tab_no}. ", bold=True)
    _run(tcap, caption_text)
    cols = [str(c) for c in df.columns]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for j, c in enumerate(cols):
        _run(table.rows[0].cells[j].paragraphs[0], c, bold=True)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, c in enumerate(df.columns):
            cells[j].text = _fmt(row[c])


def build_report_docx(entries: list[dict], meta: dict | None = None) -> bytes:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph()
    _run(title, "EPR spin-adduct decomposition — publication report", bold=True, size=15)
    if meta:
        mp = doc.add_paragraph()
        _run(mp, " · ".join(f"{k}: {v}" for k, v in meta.items()), size=9)

    fig_no = tab_no = 0
    for e in entries:
        doc.add_paragraph()
        h = doc.add_paragraph()
        _run(h, e["name"], bold=True, size=12)

        if e.get("methods_text"):
            mp = doc.add_paragraph()
            _run(mp, "Methods. ", bold=True)
            mp.add_run(e["methods_text"])

        # ── Figure ──
        fig_no += 1
        doc.add_picture(BytesIO(_report_figure(e)), width=Inches(6.1))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        _run(cap, f"Figure {fig_no}. ", bold=True)
        _run(cap, "Continuous-wave X-band EPR spectrum of ")
        _run(cap, e["name"])
        _run(cap, " (experimental, black) and the ")
        _run(cap, "composite simulation" if e.get("kind") == "manual" else "least-squares fit")
        _run(cap, " (red), decomposed into ")
        _run(cap, f"{len(e.get('rows', []))}")
        _run(cap, " spin-adduct components (lower panel). ")
        if e.get("R2") is not None:
            _run(cap, "The fit reproduced ")
            _sym(cap, "R2"); _run(cap, f" = {e['R2']:.3f} of the spectral variance.")

        # ── Table ──
        tab_no += 1
        tcap = doc.add_paragraph()
        _run(tcap, f"Table {tab_no}. ", bold=True)
        _run(tcap, "Isotropic spin-Hamiltonian parameters — ")
        _sym(tcap, "g"); _run(tcap, " factor, hyperfine coupling constants ")
        _run(tcap, "a", italic=True); _run(tcap, ", peak-to-peak linewidth ")
        _run(tcap, "ΔB"); _run(tcap, "pp", sub=True)
        _run(tcap, " — and relative spectral fractions of the decomposed components. "
                   "Assignments are candidate identifications requiring independent validation "
                   "(isotope labelling, concentration series, and chemical controls).")

        rows = e.get("rows", [])
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        table.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hdr = table.rows[0].cells
        _run(hdr[0].paragraphs[0], "Component", bold=True)
        _run(hdr[1].paragraphs[0], "Assignment", bold=True)
        p = hdr[2].paragraphs[0]; _run(p, "", bold=True); _sym(p, "g")
        p = hdr[3].paragraphs[0]; _sym(p, "aiso")
        p = hdr[4].paragraphs[0]; _sym(p, "dBpp")
        p = hdr[5].paragraphs[0]; _run(p, "Fraction (%)", bold=True)
        for r in rows:
            cells = table.add_row().cells
            cells[0].text = str(r.get("component", ""))
            cells[1].text = str(r.get("assignment", ""))
            cells[2].text = f"{r.get('g', float('nan')):.5f}"
            cells[3].text = _avals_text(r.get("avals", []))
            cells[4].text = f"{r.get('linewidth_mT', float('nan')):.3f}"
            cells[5].text = f"{r.get('fraction_pct', 0.0):.1f}"

        # ── Detailed fitted-parameter table (every parameter, value, error, bounds) ──
        if e.get("param_df") is not None and len(e["param_df"]):
            tab_no += 1
            _add_df_table(doc, e["param_df"], tab_no,
                          "Complete list of fitted spin-Hamiltonian parameters — value, standard error, "
                          "search bounds, and whether each was optimised or held fixed. Hyperfine (A) and "
                          "linewidth are in mT; weights are in arbitrary units.")
        # ── Component-fraction table ──
        if e.get("fraction_df") is not None and len(e["fraction_df"]):
            tab_no += 1
            _add_df_table(doc, e["fraction_df"], tab_no,
                          "Relative spectral fractions (integrated contribution) of each decomposed component.")
        # ── Monte-Carlo (bootstrap) uncertainties ──
        if e.get("mc_df") is not None and len(e["mc_df"]):
            tab_no += 1
            _add_df_table(doc, e["mc_df"], tab_no,
                          "Monte-Carlo (bootstrap) parameter uncertainties from refitting many noise "
                          "realisations: best value, standard deviation, and the 95% confidence interval.")

        if any(k in e for k in ("R2", "nrmse", "aic", "bic")):
            gp = doc.add_paragraph()
            _run(gp, "Goodness of fit: ", bold=True)
            _sym(gp, "R2"); _run(gp, f" = {e.get('R2', float('nan')):.4f}; ")
            _run(gp, f"normalised RMSE = {e.get('nrmse', float('nan')):.4f}; "
                     f"AIC = {e.get('aic', float('nan')):.1f}; BIC = {e.get('bic', float('nan')):.1f}"
                     + (f"; {e['n_params']} free parameters." if e.get("n_params") else "."))

    doc.add_paragraph()
    note = doc.add_paragraph()
    _run(note, "Note. ", bold=True)
    _run(note, "Fit quality is not chemical proof. All assignments require validation with authentic "
               "standards, controls, and isotope/substitution experiments. Nitrogen-centred adducts "
               "require ¹⁵N confirmation.", italic=True)

    buf = BytesIO(); doc.save(buf)
    return buf.getvalue()
