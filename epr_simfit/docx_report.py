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


import re as _re


def _species_runs(name: str):
    """Split a species/adduct name into (text, style) runs so chemical formulas render
    with subscripts (PBN-CH3 -> PBN-CH₃) and isotope mass numbers with superscripts
    (Nitroxide-14N -> Nitroxide-¹⁴N). style ∈ {None, 'sub', 'sup'}."""
    runs, i = [], 0
    for m in _re.finditer(r"\d+", name):
        s, en = m.span()
        if s > i:
            runs.append((name[i:s], None))
        pre = name[s - 1] if s > 0 else ""
        nxt = name[en] if en < len(name) else ""
        if pre in ("", " ", "-", "(", "/", "·") and nxt[:1].isupper():
            runs.append((m.group(), "sup"))          # isotope prefix, e.g. 14N
        elif pre.isalpha() or pre == ")":
            runs.append((m.group(), "sub"))           # formula subscript, e.g. CH3
        else:
            runs.append((m.group(), None))
        i = en
    if i < len(name):
        runs.append((name[i:], None))
    return runs


def _add_species(paragraph, name: str, *, bold=False):
    for text, style in _species_runs(name):
        _run(paragraph, text, bold=bold, sub=(style == "sub"), sup=(style == "sup"))


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _iso_label(isotope: str) -> str:
    """'14N' -> '¹⁴N' (mass number as Unicode superscript)."""
    num = "".join(ch for ch in isotope if ch.isdigit())
    el = "".join(ch for ch in isotope if not ch.isdigit())
    return num.translate(_SUP) + el


def _avals_text(avals) -> str:
    return "; ".join(f"{_iso_label(iso)} {round(A * 10, 2)}" for iso, _lab, A in avals) or "—"


def _num(v, fmt=".5g"):
    try:
        f = float(v)
        return format(f, fmt)
    except (TypeError, ValueError):
        return "—"


def _val_err(v, e, fmt=".5g"):
    """'value ± error' (Unicode ±); error omitted when missing/non-finite."""
    import math
    s = _num(v, fmt)
    try:
        ef = float(e)
        if math.isfinite(ef) and ef > 0:
            return f"{s} ± {format(ef, '.2g')}"
    except (TypeError, ValueError):
        pass
    return s


def _avals_text_err(avals) -> str:
    """Hyperfine list in Gauss with per-nucleus error: '¹⁴N 15.2 ± 0.2; ¹H 2.8'."""
    parts = []
    for item in avals:
        iso, _lab, A = item[0], item[1], item[2]
        err = item[3] if len(item) > 3 else None
        g = A * 10.0
        ge = (err * 10.0) if (err is not None) else None
        parts.append(f"{_iso_label(iso)} {_val_err(g, ge, '.4g')}")
    return "; ".join(parts) or "—"


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


def _add_methods_section(doc):
    """Methods section: models + fitting statistics with properly formatted equations."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc.add_paragraph()
    h = doc.add_paragraph()
    _run(h, "Methods — models, equations, and statistics", bold=True, size=13)

    p = doc.add_paragraph()
    p.add_run("Continuous-wave EPR spectra were simulated from the isotropic spin Hamiltonian and fitted to "
              "the experimental first-derivative spectrum by bounded non-linear least squares "
              "(scipy.optimize.least_squares, trust-region reflective) in Open-Sym-EPR. Each paramagnetic "
              "component was described by an isotropic g factor, one or more isotropic hyperfine coupling "
              "constants, and a peak-to-peak linewidth with a pseudo-Voigt lineshape; the composite spectrum "
              "is the weighted sum of the component first-derivative lineshapes. Relative spectral fractions "
              "are the normalised component weights; standard errors are from the covariance matrix, "
              "optionally refined by Monte-Carlo (bootstrap) refitting.")

    def eq(parts, label=None):
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for text, style in parts:
            _run(q, text, italic=(style == "i"), sub=(style == "sub"), sup=(style == "sup"))
        if label:
            _run(q, "          (" + label + ")")

    _run(doc.add_paragraph(), "Spin Hamiltonian", bold=True)
    eq([("Ĥ = μ", None), ("B", "sub"), (" ", None), ("B", "i"), ("0", "sub"), (" ", None),
        ("g", "i"), (" Ŝ  +  Σ", None), ("i", "sub"), ("  Ŝ · ", None), ("A", "i"), ("i", "sub"),
        (" · Î", None), ("i", "sub")], "1")

    _run(doc.add_paragraph(), "Pseudo-Voigt lineshape and composite spectrum", bold=True)
    eq([("V(", None), ("B", "i"), (") = η L(", None), ("B", "i"), (") + (1 − η) G(", None),
        ("B", "i"), (")", None)], "2")
    eq([("ŷ(", None), ("B", "i"), (") = Σ", None), ("i", "sub"), (" ", None), ("w", "i"), ("i", "sub"),
        (" V", None), ("i", "sub"), ("′(", None), ("B", "i"), (")", None)], "3")

    _run(doc.add_paragraph(), "Objective function and goodness-of-fit statistics", bold=True)
    eq([("χ", "i"), ("2", "sup"), (" = Σ", None), ("i", "sub"), (" [ ", None), ("y", "i"), ("i", "sub"),
        (" − ŷ", None), ("i", "sub"), ("(θ) ]", None), ("2", "sup")], "4")
    eq([("R", "i"), ("2", "sup"), (" = 1 − Σ", None), ("i", "sub"), ("(", None), ("y", "i"), ("i", "sub"),
        ("−ŷ", None), ("i", "sub"), (")", None), ("2", "sup"), (" / Σ", None), ("i", "sub"), ("(", None),
        ("y", "i"), ("i", "sub"), ("−ȳ)", None), ("2", "sup")], "5")
    eq([("NRMSE = RMSE / (", None), ("y", "i"), ("max", "sub"), (" − ", None), ("y", "i"), ("min", "sub"),
        ("),   RMSE = √(RSS / ", None), ("n", "i"), (")", None)], "6")
    eq([("AIC = 2", None), ("k", "i"), (" + ", None), ("n", "i"), (" ln(RSS/", None), ("n", "i"), ("),   ",
        None), ("BIC = ", None), ("k", "i"), (" ln(", None), ("n", "i"), (") + ", None), ("n", "i"),
        (" ln(RSS/", None), ("n", "i"), (")", None)], "7")

    d = doc.add_paragraph()
    _run(d, "where ", italic=True)
    _run(d, "μ"); _run(d, "B", sub=True)
    d.add_run(" is the Bohr magneton, ")
    _run(d, "B", italic=True); _run(d, "0", sub=True)
    d.add_run(" the static magnetic field, ")
    _run(d, "g", italic=True)
    d.add_run(" the g factor, Ŝ and Î the electron and nuclear spin operators, ")
    _run(d, "A", italic=True); _run(d, "i", sub=True)
    d.add_run(" the isotropic hyperfine coupling of nucleus i, η the Lorentzian/Gaussian mixing, ")
    _run(d, "w", italic=True); _run(d, "i", sub=True)
    d.add_run(" the component weight, θ the fitted-parameter vector, ")
    _run(d, "y", italic=True); _run(d, "i", sub=True)
    d.add_run(" and ŷ")
    _run(d, "i", sub=True)
    d.add_run(" the experimental and simulated intensities, ȳ their mean, RSS the residual sum of squares, ")
    _run(d, "k", italic=True)
    d.add_run(" the number of free parameters, and ")
    _run(d, "n", italic=True)
    d.add_run(" the number of data points.")


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

        # ── Results narrative ──
        if e.get("results_text"):
            rp = doc.add_paragraph()
            _run(rp, "Results and discussion. ", bold=True)
            rp.add_run(e["results_text"])

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
        _run(tcap, "Combined isotropic spin-Hamiltonian parameters of the decomposed components — ")
        _sym(tcap, "g"); _run(tcap, " factor, hyperfine coupling constants ")
        _run(tcap, "a", italic=True); _run(tcap, " (G), peak-to-peak linewidth ")
        _run(tcap, "ΔB"); _run(tcap, "pp", sub=True)
        _run(tcap, " (mT), pseudo-Voigt mixing η, and relative spectral fraction — with standard "
                   "errors (± 1σ) where a parameter was optimised. Assignments are candidate "
                   "identifications requiring independent validation (isotope labelling, concentration "
                   "series, and chemical controls).")

        rows = e.get("rows", [])
        table = doc.add_table(rows=1, cols=7)
        table.style = "Table Grid"
        table.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hdr = table.rows[0].cells
        _run(hdr[0].paragraphs[0], "Component", bold=True)
        _run(hdr[1].paragraphs[0], "Assignment", bold=True)
        p = hdr[2].paragraphs[0]; _sym(p, "g")
        p = hdr[3].paragraphs[0]; _sym(p, "aiso")
        p = hdr[4].paragraphs[0]; _sym(p, "dBpp")
        p = hdr[5].paragraphs[0]; _run(p, "η (L/G)")
        p = hdr[6].paragraphs[0]; _run(p, "Fraction (%)", bold=True)
        for r in rows:
            cells = table.add_row().cells
            _add_species(cells[0].paragraphs[0], str(r.get("component", "")))
            _add_species(cells[1].paragraphs[0], str(r.get("assignment", "")))
            cells[2].text = _val_err(r.get("g"), r.get("g_err"), ".5f")
            cells[3].text = _avals_text_err(r.get("avals", []))
            cells[4].text = _val_err(r.get("linewidth_mT"), r.get("lw_err"), ".3f")
            cells[5].text = _num(r.get("eta", 0.5), ".2f")
            cells[6].text = _val_err(r.get("fraction_pct", 0.0), r.get("fraction_err"), ".1f")

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

        # ── Goodness-of-fit interpretation ──
        _gof = e.get("gof") or {}
        if _gof:
            ip = doc.add_paragraph()
            _run(ip, "Interpretation. ", bold=True)
            ip.add_run(" ".join(x for x in (_gof.get("r2_note"), _gof.get("nrmse_note"),
                                            _gof.get("overall")) if x))

        # ── Per-species interpretation ──
        _species = e.get("species") or []
        if _species:
            sp = doc.add_paragraph()
            _run(sp, "Detected paramagnetic species (candidate assignments). ", bold=True)
            for d in _species:
                b = doc.add_paragraph(style="List Bullet")
                _add_species(b, str(d.get("name", "")), bold=True)
                _run(b, f" — {d.get('fraction_pct', 0.0):.1f}% ({d.get('confidence', '')}); ")
                _sym(b, "g"); _run(b, f" = {d.get('g', float('nan')):.5f}, ")
                _run(b, "ΔB"); _run(b, "pp", sub=True)
                _run(b, f" = {d.get('linewidth_mT', float('nan')):.3f} mT, hyperfine {d.get('nuclei_str', '')}. ")
                if d.get("interpretation"):
                    _run(b, str(d["interpretation"]))
                if d.get("warning"):
                    _run(b, "  " + str(d["warning"]), italic=True)

        # ── Per-spectrum methods note ──
        if e.get("methods_text"):
            mp = doc.add_paragraph()
            _run(mp, "Methods (this spectrum). ", bold=True)
            mp.add_run(e["methods_text"])

    # ── Global Methods, models, and equations ──
    _add_methods_section(doc)

    doc.add_paragraph()
    note = doc.add_paragraph()
    _run(note, "Note. ", bold=True)
    _run(note, "Fit quality is not chemical proof. All assignments require validation with authentic "
               "standards, controls, and isotope/substitution experiments. Nitrogen-centred adducts "
               "require ¹⁵N confirmation.", italic=True)

    buf = BytesIO(); doc.save(buf)
    return buf.getvalue()
