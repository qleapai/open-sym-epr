from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from epr_simfit.about import ABOUT_TEXT, DETAILED_INFO, SHORT_CREDIT
from epr_simfit.constants import DEFAULT_MW_FREQUENCY_GHZ
from epr_simfit.demo_data import DEMO_CONDITIONS, generate_demo_text
from epr_simfit.export import build_export_zip, fit_components_dataframe
from epr_simfit.fitter import fit_spectrum
from epr_simfit.io import parse_epr_text
from epr_simfit.metadata_parser import metadata_table
from epr_simfit import mixtures as mixtures_mod
from epr_simfit import fit_store
from epr_simfit.model_comparison import compare_models
from epr_simfit.model_library import MODEL_DESCRIPTIONS, MODEL_PRESETS, component_table, default_components
from epr_simfit.model_suggester import ExperimentContext, suggest_models
from epr_simfit.plotting import comparison_bar_figure, component_figure, residual_figure, spectrum_figure
from epr_simfit.interpretation import (
    SCIENCE_FIT,
    SCIENCE_IMPORT,
    SCIENCE_MODELBUILDER,
    SCIENCE_PREPROCESS,
    assess_fit_quality,
    intermediates_dataframe,
    publication_methods_paragraph,
    publication_parameters_table,
    suggest_fit_improvements,
    suggest_intermediates,
)
from epr_simfit.preprocessing import preprocess_spectrum
from epr_simfit.report import generate_report_html, generate_report_text
from epr_simfit.simulator import simulate_model
from epr_simfit.spin_models import Nucleus, SpinComponent
from epr_simfit.user_models import components_from_text, components_to_json


APP_DIR = Path(__file__).resolve().parent
WHITE_PAPER = APP_DIR / "docs" / "WHITE_PAPER.md"
CITATION = APP_DIR / "CITATION.cff"
CUSTOM_MODEL_DOC = APP_DIR / "docs" / "CUSTOM_MODEL_FORMAT.md"
REFERENCE_SPECTRA_DOC = APP_DIR / "docs" / "REFERENCE_SPECTRA.md"
LOGO     = APP_DIR / "assets" / "logo.svg"
LOGO_BANNER = APP_DIR / "assets" / "logo_banner.svg"

st.set_page_config(page_title="Open-Sym-EPR", page_icon="⚛️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:0.65rem 0.8rem;}
    .credit-note {font-size:0.9rem;color:#44515c;margin-top:-0.35rem;margin-bottom:1rem;}
    .science-note {background:#eef6ff;border:1px solid #bfdbfe;border-radius:8px;padding:0.8rem 1rem;color:#1e3a5f;}
    .warning-note {background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:0.8rem 1rem;color:#7c2d12;}
    .quality-excellent {background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:0.8rem 1rem;color:#14532d;}
    .quality-good {background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;padding:0.8rem 1rem;color:#1e3a5f;}
    .quality-moderate {background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:0.8rem 1rem;color:#78350f;}
    .quality-poor {background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:0.8rem 1rem;color:#7f1d1d;}
    </style>
    """,
    unsafe_allow_html=True,
)


def uploaded_text(uploaded) -> tuple[str | None, str | None]:
    if uploaded is None:
        return None, None
    return uploaded.getvalue().decode("utf-8", errors="replace"), uploaded.name


def slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower()).strip("_")
    return clean or "custom_component"


def parse_nuclei_text(text: str) -> list[Nucleus]:
    nuclei: list[Nucleus] = []
    if not text or not str(text).strip():
        return nuclei
    for item in re.split(r"[;,]", str(text)):
        item = item.strip()
        if not item:
            continue
        match = re.match(r"([A-Za-z0-9]+)\s*[:=]\s*([0-9.]+)", item)
        if not match:
            continue
        isotope, a_value = match.groups()
        nuclei.append(Nucleus(isotope=isotope, A_mT=float(a_value), label=isotope))
    return nuclei


def custom_components_from_table(table: pd.DataFrame) -> list[SpinComponent]:
    components: list[SpinComponent] = []
    for _, row in table.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        cid = "custom_" + slug(name)
        components.append(
            SpinComponent(
                component_id=cid,
                display_name=name,
                radical_assignment=str(row.get("assignment", name)),
                category=str(row.get("category", "custom")),
                g=float(row.get("g", 2.003)),
                g_bounds=(float(row.get("g_min", 1.95)), float(row.get("g_max", 2.20))),
                nuclei=parse_nuclei_text(str(row.get("nuclei", ""))),
                linewidth_mT=float(row.get("linewidth_mT", 0.2)),
                linewidth_bounds=(float(row.get("lw_min", 0.01)), float(row.get("lw_max", 10.0))),
                eta=float(row.get("eta", 0.5)),
                weight=float(row.get("weight", 0.3)),
                interpretation="User-defined component.",
                warning="Custom component: verify g, hyperfine, and linewidth bounds against literature or controls.",
            )
        )
    return components


def apply_component_edits(selected_ids: list[str], edited: pd.DataFrame, custom_components: list[SpinComponent]) -> list[SpinComponent]:
    library = default_components()
    custom_by_id = {component.component_id: component for component in custom_components}
    components: list[SpinComponent] = []
    for cid in selected_ids:
        if cid in library:
            components.append(library[cid].clone())
        elif cid in custom_by_id:
            components.append(custom_by_id[cid].clone())
    by_id = {component.component_id: component for component in components}
    for _, row in edited.iterrows():
        cid = row.get("component ID")
        if cid in by_id:
            comp = by_id[cid]
            comp.g = float(row.get("g", comp.g))
            comp.linewidth_mT = float(row.get("linewidth mT", comp.linewidth_mT))
            comp.eta = float(row.get("eta", comp.eta))
            comp.weight = float(row.get("weight", comp.weight))
    return components


def anisotropic_editor_frame(components: list[SpinComponent]) -> pd.DataFrame:
    """Build the editable anisotropic-parameter table for the selected components."""
    rows = []
    for comp in components:
        gx, gy, gz = comp.g_principal()
        rows.append({
            "component ID": comp.component_id,
            "S (spin)": float(comp.spin_S),
            "gx": round(gx, 5),
            "gy": round(gy, 5),
            "gz": round(gz, 5),
            "D (MHz)": float(comp.D_MHz),
            "E (MHz)": float(comp.E_MHz),
            "mode": comp.mode,
        })
    cols = ["component ID", "S (spin)", "gx", "gy", "gz", "D (MHz)", "E (MHz)", "mode"]
    return pd.DataFrame(rows, columns=cols)


def apply_anisotropic_edits(components: list[SpinComponent], edited: pd.DataFrame) -> list[SpinComponent]:
    """Apply anisotropic (g-tensor, S, D, E, mode) edits to the selected components."""
    by_id = {c.component_id: c for c in components}
    for _, row in edited.iterrows():
        cid = row.get("component ID")
        comp = by_id.get(cid)
        if comp is None:
            continue
        try:
            comp.spin_S = float(row.get("S (spin)", comp.spin_S))
            gx = float(row.get("gx")); gy = float(row.get("gy")); gz = float(row.get("gz"))
            comp.g_tensor = (gx, gy, gz)
            comp.g = (gx + gy + gz) / 3.0
            comp.D_MHz = float(row.get("D (MHz)", comp.D_MHz))
            comp.E_MHz = float(row.get("E (MHz)", comp.E_MHz))
            mode = str(row.get("mode", comp.mode)).strip().lower()
            comp.mode = mode if mode in ("auto", "isotropic", "powder") else "auto"
        except (TypeError, ValueError):
            continue
    return components


# ── Header banner ────────────────────────────────────────────────────────────
if LOGO_BANNER.exists():
    st.image(str(LOGO_BANNER), use_container_width=True)
else:
    st.title("Open-Sym-EPR")
st.caption("General high-field cw-EPR simulation, fitting, model comparison, and publication-ready export.")
st.markdown(f"<div class='credit-note'>{SHORT_CREDIT}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='science-note'>Open-Sym-EPR uses isotropic high-field cw-EPR line simulations. "
    "It is scientifically useful for screening, mixture fitting, and transparent reporting, "
    "but anisotropic powder spectra, tensor-resolved g/A analysis, saturation behavior, and exchange/correlation effects require specialist EPR treatment.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    # Logo in sidebar
    if LOGO.exists():
        st.image(str(LOGO), width=120)
    st.header("Experiment metadata")
    project_title = st.text_input("Project title", value="Untitled EPR study")
    sample_name = st.text_input("Sample name", value="Sample 1")
    solvent = st.text_input("Solvent / matrix", value="water, buffer, frozen glass, solid, or custom")
    catalyst_material = st.text_input("Catalyst/material/combination", value="custom catalyst or material")
    atmosphere = st.text_input("Atmosphere / gas", value="N2, Ar, air, O2, vacuum, or custom")
    condition_text = st.text_input("Condition", value="light/dark, temperature, reaction time, pH, potential, dose, etc.")
    additives = st.text_area("Additives / electrolyte / spin probe", value="", height=70)
    sample_class = st.selectbox(
        "Sample class",
        [
            "unknown/general",
            "organic radical",
            "nitroxide/spin label",
            "ROS/spin trap",
            "N2RR / nitrogen reduction",
            "CO2RR / carbon dioxide reduction",
            "electrocatalysis",
            "photocatalysis",
            "HER / hydrogen evolution",
            "OER / ORR oxygen electrochemistry",
            "reference standards / calibration",
            "transition metal",
            "solid defect/broad signal",
        ],
        index=0,
    )
    pbn_used = st.checkbox("PBN or DMSO-specific spin trapping", value=False)
    st.divider()
    with st.expander("💾 Project — save / resume", expanded=False):
        from epr_simfit import project as _proj
        proj_up = st.file_uploader("Open project (.simepr.json)", type=["json"], key="project_upload")
        if proj_up is not None and st.button("↩ Load project", key="do_load_project"):
            try:
                obj = _proj.load_project(proj_up.getvalue())
                st.session_state["_project_file_text"] = obj.get("spectrum_text")
                st.session_state["_project_filename"] = obj.get("filename") or "project_spectrum.asc"
                if obj.get("microwave_frequency_GHz"):
                    st.session_state["mw_freq_input"] = float(obj["microwave_frequency_GHz"])
                if obj.get("components_json"):
                    from epr_simfit.user_models import components_from_json
                    _c, _ = components_from_json(obj["components_json"])
                    st.session_state["uploaded_model_components"] = _c
                st.session_state["_project_loaded_msg"] = _proj.project_summary(obj)
                st.rerun()
            except Exception as _pe:  # noqa: BLE001
                st.error(f"Could not load project: {_pe}")
        if st.session_state.get("_project_loaded_msg"):
            st.success("Loaded — " + st.session_state["_project_loaded_msg"])
        st.caption("The **Save project** button is at the bottom of the page (captures your current work).")
    st.header("Data")
    uploaded = st.file_uploader("Upload EPR file (ASC/TXT/CSV)", type=["asc", "txt", "dat", "csv"])
    bruker_files = st.file_uploader(
        "…or Bruker BES3T (.DTA + .DSC)", type=["dta", "dsc"], accept_multiple_files=True,
        help="Upload BOTH the .DTA (binary data) and .DSC (descriptor) files of a Bruker Xepr/Elexsys spectrum.",
    )
    demo_choice = st.selectbox("Or load demo", ["none"] + list(DEMO_CONDITIONS.keys()), index=0)
    st.session_state.setdefault("mw_freq_input", DEFAULT_MW_FREQUENCY_GHZ)
    mw_freq = st.number_input("Microwave frequency / GHz", min_value=1.0, max_value=300.0, step=0.01, key="mw_freq_input",
                              help="X-band ≈ 9.4 GHz, Q-band ≈ 34 GHz, W-band ≈ 94 GHz. Multifrequency-aware: anisotropic patterns scale with frequency.")
    field_unit = st.selectbox("Field unit", ["Auto", "Gauss", "mT"], index=0)
    advanced = st.checkbox("Advanced fitting controls", value=False)
    st.divider()
    st.header("Powder engine")
    powder_quality = st.select_slider(
        "Orientation accuracy",
        options=["Fast (600)", "Standard (1000)", "High (2000)", "Very high (4000)"],
        value="Standard (1000)",
        help="Orientations for powder averaging of anisotropic / high-spin components. More = smoother, slower.",
    )
    n_orient = {"Fast (600)": 600, "Standard (1000)": 1000, "High (2000)": 2000, "Very high (4000)": 4000}[powder_quality]

file_text, filename = uploaded_text(uploaded)
# resume from a loaded project when no new file is uploaded
if file_text is None and not bruker_files and st.session_state.get("_project_file_text"):
    file_text = st.session_state["_project_file_text"]
    filename = st.session_state.get("_project_filename", "project_spectrum.asc")
st.session_state["_cur_file_text"] = file_text
st.session_state["_cur_filename"] = filename
bruker_meta = None
# Bruker BES3T pair takes priority when both .DTA and .DSC are uploaded.
if file_text is None and bruker_files:
    dsc = next((f for f in bruker_files if f.name.lower().endswith(".dsc")), None)
    dta = next((f for f in bruker_files if f.name.lower().endswith(".dta")), None)
    if dsc is not None and dta is not None:
        try:
            from epr_simfit.bruker import bruker_to_text, load_bes3t
            _bf, _bi, bruker_meta = load_bes3t(dsc.getvalue().decode("latin-1"), dta.getvalue())
            file_text = bruker_to_text(_bf, _bi, bruker_meta)
            filename = bruker_meta.get("title") or dta.name
            if bruker_meta.get("microwave_frequency_GHz"):
                mw_freq = float(bruker_meta["microwave_frequency_GHz"])
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Bruker import failed: {exc}")
    else:
        st.sidebar.warning("Upload BOTH the .DTA and .DSC files for a Bruker spectrum.")
if file_text is None and demo_choice != "none":
    demo_condition, demo_preset, demo_weights = DEMO_CONDITIONS[demo_choice]
    file_text = generate_demo_text(demo_condition, demo_preset, demo_weights)
    filename = demo_choice

context = ExperimentContext(
    analysis_mode="General EPR fitting",
    sample_class=sample_class,
    condition=condition_text,
    catalyst=catalyst_material,
    solvent=solvent,
    pbn_used=pbn_used,
)
suggested = suggest_models(context)

parsed = None
prep = None
parse_error = None
if file_text:
    try:
        parsed_initial = parse_epr_text(file_text, filename=filename, field_unit=field_unit, mw_frequency_override=mw_freq)
    except Exception as exc:  # noqa: BLE001
        parsed_initial = None
        parse_error = str(exc)
else:
    parsed_initial = None

# Apply any programmatic field-shift written to _pending_field_shift BEFORE the
# slider widget (key="field_shift_mT") is instantiated.  Streamlit forbids writing
# to a widget-bound key after the widget renders, so all button handlers stage their
# update here and call st.rerun() to let this block commit it first.
if "_pending_field_shift" in st.session_state:
    st.session_state["field_shift_mT"] = st.session_state.pop("_pending_field_shift")


def save_fit_ui(source: str, key: str, *, field_mT, experimental, fit_total, components,
                weights, R2, n_parameters=0, extra=None, default_name=None):
    """Render a name box + 'Save to global store' button. Any tab can call this to
    push a completed fit into the session-wide registry (fit_store)."""
    dn = default_name or f"{source} {source_seq(source)}"
    c1, c2 = st.columns([3, 1])
    nm = c1.text_input("Save this fit as", value=dn, key=f"save_name_{key}",
                       label_visibility="collapsed", placeholder="name this fit")
    if c2.button("💾 Save to store", key=f"save_btn_{key}"):
        saved = fit_store.add(st.session_state, nm, field_mT=field_mT, experimental=experimental,
                              fit_total=fit_total, components=components, weights=weights,
                              mw_frequency_GHz=mw_freq, R2=R2, source=source,
                              n_parameters=n_parameters, extra=extra)
        st.toast(f"Saved '{saved}' to the global fit store.", icon="💾")


def source_seq(source: str) -> int:
    """Small running counter so default names don't collide."""
    n = sum(1 for k in fit_store.names(st.session_state)) + 1
    return n


def overlay_ui(target_field, key: str):
    """Multiselect of saved fits; returns overlay traces to merge into a plot dict."""
    saved = fit_store.names(st.session_state)
    if not saved:
        return {}
    picks = st.multiselect("Overlay saved fits from the global store", saved, key=f"overlay_{key}")
    inc_exp = False
    if picks:
        inc_exp = st.checkbox("also overlay their experimental traces", key=f"overlay_exp_{key}")
    return fit_store.overlay_traces(st.session_state, picks, target_field, include_experimental=inc_exp)


# ── Global fit store manager (sidebar) — load any fit into any plotting tab ──────
with st.sidebar.expander("🗂 Saved fits (global store)", expanded=False):
    _reg_names = fit_store.names(st.session_state)
    if not _reg_names:
        st.caption("No saved fits yet. Run a fit (Fit, ML-assisted, or Adduct mixture) "
                   "and click 💾 Save to store.")
    else:
        st.dataframe(pd.DataFrame(fit_store.summary_rows(st.session_state)),
                     hide_index=True, width="stretch")
        _sel = st.selectbox("Select a saved fit", _reg_names, key="store_select")
        _b1, _b2 = st.columns(2)
        if _b1.button("↪ Use as start model", key="store_use_start",
                      help="Load this fit's refined components as the starting model in "
                           "Model builder, so you can run a further optimisation from it."):
            _e = fit_store.get(st.session_state, _sel)
            if _e:
                st.session_state["uploaded_model_components"] = [c.clone() for c in _e["components"]]
                st.session_state["uploaded_model_meta"] = {"from_saved_fit": _sel, "R2": _e["R2"]}
                st.session_state["_pending_preselect"] = [c.component_id for c in _e["components"]]
                if _e.get("mw_frequency_GHz"):
                    st.session_state["mw_freq_input"] = float(_e["mw_frequency_GHz"])
                st.session_state["_start_model_msg"] = _sel
                st.rerun()
        if _b2.button("🗑 Delete", key="store_delete"):
            fit_store.delete(st.session_state, _sel)
            st.rerun()
        _csv = fit_store.export_csv(st.session_state, _sel)
        if _csv:
            st.download_button("⬇ CSV (Origin-ready)", data=_csv,
                               file_name=f"{_sel}.csv", mime="text/csv", key="store_csv")
if st.session_state.pop("_start_model_msg", None):
    st.sidebar.success("Loaded saved fit into Model builder as the starting model. "
                       "Open **Model builder** → adjust if needed → **Fit** to optimise further.")

tabs = st.tabs(["Import", "Metadata", "Preprocess", "Model builder", "Fit", "Compare", "Export", "Batch / kinetics", "References", "Solvers", "ML-assisted fit", "Adduct mixture", "White paper / citation"])

with tabs[0]:
    st.subheader("Import")
    with st.expander("📖 Scientific background — cw-EPR fundamentals", expanded=False):
        st.markdown(SCIENCE_IMPORT)
    if parse_error:
        st.error(parse_error)
    if parsed_initial is None:
        st.info("Upload an EPR text/ASC/CSV file or load a demo.")
    else:
        col_opts = list(range(parsed_initial.detected_columns))
        c1, c2, c3, c4 = st.columns(4)
        field_col = c1.selectbox("Field column", col_opts, index=min(parsed_initial.field_col, len(col_opts) - 1))
        intensity_col = c2.selectbox("Intensity column", col_opts, index=min(1, len(col_opts) - 1))
        manual_unit = c3.selectbox("Manual field unit", ["Auto", "Gauss", "mT"], index=["Auto", "Gauss", "mT"].index(field_unit))
        manual_mw = c4.number_input("Manual microwave GHz", min_value=1.0, max_value=300.0, value=float(mw_freq), step=0.01)
        parsed = parse_epr_text(
            file_text,
            filename=filename,
            field_col=int(field_col),
            intensity_col=int(intensity_col),
            field_unit=manual_unit,
            mw_frequency_override=float(manual_mw),
        )
        context.metadata = parsed.metadata
        suggested = suggest_models(context)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("File", parsed.filename or "uploaded")
        m2.metric("Rows", f"{parsed.numeric_rows:,}")
        m3.metric("Columns", parsed.detected_columns)
        m4.metric("Field unit", parsed.detected_field_unit)
        m5.metric("MW freq", f"{parsed.metadata.get('microwave_frequency_GHz', manual_mw):.4g} GHz")
        for warning in parsed.warnings:
            st.warning(warning)
        st.dataframe(pd.DataFrame(metadata_table(parsed.metadata)), width="stretch")
        with st.expander("Header text"):
            st.text(parsed.header_text or "(no non-numeric header detected)")
        st.dataframe(parsed.dataframe.head(500), width="stretch")
        st.plotly_chart(spectrum_figure(parsed.dataframe["Field_mT"], {"raw": parsed.dataframe["Intensity_raw"]}, "Raw EPR spectrum"), width="stretch")

with tabs[1]:
    st.subheader("Metadata and suggested models")
    meta_rows = [
        ("Project", project_title),
        ("Sample", sample_name),
        ("Solvent/matrix", solvent),
        ("Catalyst/material", catalyst_material),
        ("Atmosphere/gas", atmosphere),
        ("Condition", condition_text),
        ("Additives/spin probe", additives),
        ("Sample class", sample_class),
    ]
    st.dataframe(pd.DataFrame(meta_rows, columns=["field", "value"]), width="stretch")
    if pbn_used and "dmso" in (solvent + additives).lower():
        st.markdown(
            "<div class='warning-note'>PBN/DMSO was indicated. DMSO-derived carbon radical adducts must be considered before assigning other spin-trapped species.</div>",
            unsafe_allow_html=True,
        )
    cards = []
    for model in suggested["suggested_models"] + [m for m in MODEL_PRESETS if m not in suggested["suggested_models"]]:
        cards.append(
            {
                "model": model,
                "suggested": model in suggested["suggested_models"],
                "description": MODEL_DESCRIPTIONS.get(model, ""),
                "components": ", ".join(MODEL_PRESETS[model]),
            }
        )
    st.dataframe(pd.DataFrame(cards), width="stretch")

with tabs[2]:
    st.subheader("Preprocess")
    with st.expander("📖 Scientific background — baseline correction, normalisation & field alignment", expanded=False):
        st.markdown(SCIENCE_PREPROCESS)
    if parsed is None:
        st.info("Import a spectrum first.")
    else:
        raw_field = parsed.dataframe["Field_mT"].to_numpy()
        raw_intensity = parsed.dataframe["Intensity_raw"].to_numpy()

        # Reset shift when a new file is loaded
        _file_key = filename or ""
        if st.session_state.get("_preproc_file") != _file_key:
            st.session_state["field_shift_mT"] = 0.0
            st.session_state["_preproc_file"] = _file_key
            st.session_state.pop("sim_preview", None)
        if "field_shift_mT" not in st.session_state:
            st.session_state["field_shift_mT"] = 0.0

        field_shift = float(st.session_state["field_shift_mT"])
        shifted_min = float(raw_field.min()) + field_shift
        shifted_max = float(raw_field.max()) + field_shift

        pc1, pc2, pc3, pc4 = st.columns(4)
        crop_min, crop_max = pc1.slider("Fitting window / mT", shifted_min, shifted_max, (shifted_min, shifted_max))
        baseline_method = pc2.selectbox("Baseline correction", ["none", "constant", "linear edge", "polynomial edge"], index=2)
        poly_order = pc2.number_input("Polynomial order", min_value=1, max_value=5, value=2)
        norm_method = pc3.selectbox("Normalization", ["none", "max absolute", "peak-to-peak", "area"], index=1)
        invert = pc3.checkbox("Invert intensity", value=False)
        smooth = pc4.checkbox("Display smoothing", value=False)

        prep = preprocess_spectrum(
            raw_field,
            raw_intensity,
            crop_min=crop_min,
            crop_max=crop_max,
            field_shift_mT=field_shift,
            baseline_method=baseline_method,
            polynomial_order=int(poly_order),
            normalization=norm_method,
            smooth_display=smooth,
            invert=invert,
        )

        preprocess_title = "Preprocessing" + (f"  ·  field shift {field_shift:+.3f} mT" if field_shift else "")
        st.plotly_chart(
            spectrum_figure(
                prep.field_mT,
                {"raw window": prep.raw, "baseline corrected": prep.corrected, "processed": prep.processed, "display": prep.display},
                preprocess_title,
            ),
            width="stretch",
        )

        # ── Field alignment ──────────────────────────────────────────────────
        with st.expander("Field alignment — shift experimental axis to match model peak", expanded=(field_shift != 0.0)):
            st.markdown(
                "Compensate for spectrometer field-axis calibration offsets. "
                "Build a model in the **Model builder** tab, then click **Auto-align** to "
                "compute the optimal shift via cross-correlation. "
                "Fine-tune manually with the slider."
            )
            acol1, acol2, acol3 = st.columns([4, 2, 1])
            acol1.slider(
                "Field shift / mT",
                min_value=-10.0,
                max_value=10.0,
                step=0.01,
                format="%.2f",
                key="field_shift_mT",
                help="Shifts the experimental field axis. Positive = axis moves up (spectrum shifts left on plot).",
            )
            sim_preview = st.session_state.get("sim_preview")
            auto_disabled = sim_preview is None
            auto_help = (
                "Compute the optimal field shift by cross-correlating the preprocessed experimental "
                "spectrum with the current model simulation."
                if not auto_disabled
                else "Build a model in the Model builder tab first."
            )
            if acol2.button("Auto-align with model", disabled=auto_disabled, help=auto_help):
                sim_field_prev, sim_total_prev = sim_preview  # type: ignore[misc]
                if len(prep.field_mT) > 1 and len(sim_total_prev) > 1:
                    from epr_simfit.preprocessing import auto_align_spectra  # lazy import avoids hot-reload cache issues
                    sim_on_grid = np.interp(prep.field_mT, sim_field_prev, sim_total_prev)
                    delta = auto_align_spectra(prep.processed, sim_on_grid, prep.field_mT)
                    st.session_state["_pending_field_shift"] = round(field_shift + delta, 2)
                    st.rerun()
            if acol3.button("Reset", help="Reset field shift to zero"):
                st.session_state["_pending_field_shift"] = 0.0
                st.rerun()
            if field_shift:
                st.info(f"Active shift: **{field_shift:+.3f} mT** — the experimental field axis has been offset by this amount.")
            elif auto_disabled:
                st.caption("Build a model in the Model builder tab, then click Auto-align to set the shift automatically.")

            # ── Live alignment preview ──────────────────────────────────────
            _sp = st.session_state.get("sim_preview")
            if _sp is not None:
                _sf, _st2 = _sp
                _sim_live = np.interp(prep.field_mT, _sf, _st2)
                st.plotly_chart(
                    spectrum_figure(
                        prep.field_mT,
                        {"experimental (shifted)": prep.processed, "simulation": _sim_live},
                        f"Live alignment preview  ·  shift = {field_shift:+.3f} mT",
                    ),
                    use_container_width=True,
                )

with tabs[3]:
    st.subheader("Model builder")
    with st.expander("📖 Scientific background — isotropic cw-EPR simulation", expanded=False):
        st.markdown(SCIENCE_MODELBUILDER)
    st.dataframe(pd.DataFrame(component_table()), width="stretch")
    preset_names = list(MODEL_PRESETS.keys()) + ["custom"]
    default_preset = suggested.get("recommended_model", "G0_single_line")
    preset = st.selectbox("Preset", preset_names, index=preset_names.index(default_preset) if default_preset in preset_names else 0)
    default_ids = MODEL_PRESETS.get(preset, suggested["suggested_models"][:1])
    st.caption("Custom nuclei syntax: `14N:1.55; 1H:0.30; 63Cu:8.5` where hyperfine values are in mT.")

    with st.expander("Upload custom model library (JSON, YAML, CSV)", expanded=False):
        st.markdown(
            "Upload a reusable Open-Sym-EPR model pack (`.simepr.json`/`.json`), YAML model pack, "
            "or CSV table. CSV columns can include: `name, component_id, assignment, category, "
            "g, g_min, g_max, nuclei, linewidth_mT, lw_min, lw_max, eta, weight, spin_S, gx, gy, gz, D_MHz, E_MHz, mode`."
        )
        model_upload = st.file_uploader(
            "Custom model file",
            type=["json", "simepr", "yaml", "yml", "csv"],
            key="custom_model_library_upload",
        )
        if model_upload is not None:
            try:
                uploaded_text_value = model_upload.getvalue().decode("utf-8", errors="replace")
                uploaded_components, uploaded_meta = components_from_text(uploaded_text_value, model_upload.name)
                st.session_state["uploaded_model_components"] = uploaded_components
                st.session_state["uploaded_model_meta"] = uploaded_meta
                st.success(f"Loaded {len(uploaded_components)} uploaded component(s) from {model_upload.name}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load custom model library: {exc}")
        uploaded_components = st.session_state.get("uploaded_model_components", [])
        uploaded_meta = st.session_state.get("uploaded_model_meta", {})
        if uploaded_components:
            st.caption(f"Uploaded model metadata: {uploaded_meta}")
            st.dataframe(pd.DataFrame([component.to_table_row() for component in uploaded_components]), width="stretch")
            if st.button("Clear uploaded model library"):
                st.session_state.pop("uploaded_model_components", None)
                st.session_state.pop("uploaded_model_meta", None)
                st.rerun()

    custom_default = pd.DataFrame(
        [
            {
                "name": "",
                "assignment": "",
                "category": "custom",
                "g": 2.003,
                "g_min": 1.95,
                "g_max": 2.20,
                "nuclei": "",
                "linewidth_mT": 0.20,
                "lw_min": 0.01,
                "lw_max": 10.0,
                "eta": 0.5,
                "weight": 0.3,
            }
        ]
    )
    custom_table = st.data_editor(custom_default, num_rows="dynamic", width="stretch")
    custom_components = custom_components_from_table(custom_table)
    uploaded_components = st.session_state.get("uploaded_model_components", [])
    custom_pool = [*uploaded_components, *custom_components]
    component_ids = list(default_components().keys()) + [component.component_id for component in custom_pool]
    # Keyed selection so it can be driven programmatically (e.g. "Use as start model"
    # from the global fit store) while still resetting when the preset changes.
    if st.session_state.get("_mb_last_preset") != preset:
        st.session_state["_mb_last_preset"] = preset
        st.session_state["model_component_ids"] = [cid for cid in default_ids if cid in component_ids]
    if "_pending_preselect" in st.session_state:
        _pend = st.session_state.pop("_pending_preselect")
        st.session_state["model_component_ids"] = [cid for cid in _pend if cid in component_ids]
    st.session_state.setdefault("model_component_ids", [cid for cid in default_ids if cid in component_ids])
    st.session_state["model_component_ids"] = [c for c in st.session_state["model_component_ids"] if c in component_ids]
    selected_ids = st.multiselect("Components to simulate/fit", component_ids, key="model_component_ids")
    rows = []
    all_by_id = {**default_components(), **{component.component_id: component for component in custom_pool}}
    for cid in selected_ids:
        rows.append(all_by_id[cid].to_table_row())
    edit_cols = ["component ID", "name", "category", "g", "linewidth mT", "eta", "weight", "interpretation", "warning"]
    edited = st.data_editor(pd.DataFrame(rows, columns=edit_cols), disabled=["component ID", "name", "category", "interpretation", "warning"], width="stretch")
    selected_components = apply_component_edits(selected_ids, edited, custom_pool)
    if selected_components:
        st.download_button(
            "Download selected components as Open-Sym-EPR model pack",
            data=components_to_json(selected_components, name=f"{project_title}_{sample_name}_model").encode("utf-8"),
            file_name="Open-Sym-EPR_custom_model.simepr.json",
            mime="application/json",
        )

    # ── Anisotropic / high-spin parameters (tensor g, S, zero-field splitting) ──
    with st.expander("⚛ Anisotropic / high-spin parameters (g-tensor, spin S, zero-field splitting)", expanded=False):
        st.markdown(
            "Set the **g-tensor** principal values (gx, gy, gz), electron **spin S**, and "
            "**zero-field splitting** D, E (MHz) for rigid-limit, powder, frozen-solution, and "
            "high-spin systems. When gx≠gy≠gz, S>½, or D/E≠0, the component is solved by full "
            "spin-Hamiltonian diagonalisation with powder averaging. "
            "`mode`: *auto* (detect), *isotropic* (force fast path), *powder* (force averaging)."
        )
        if selected_components:
            aniso_frame = anisotropic_editor_frame(selected_components)
            aniso_edited = st.data_editor(
                aniso_frame, disabled=["component ID"], width="stretch", key="aniso_editor",
                column_config={
                    "S (spin)": st.column_config.NumberColumn(help="Electron spin: 0.5, 1, 1.5, 2, 2.5 ...", min_value=0.5, max_value=3.5, step=0.5),
                    "D (MHz)": st.column_config.NumberColumn(help="Axial zero-field splitting (S>1/2 only)"),
                    "E (MHz)": st.column_config.NumberColumn(help="Rhombic zero-field splitting (|E/D| ≤ 1/3)"),
                },
            )
            selected_components = apply_anisotropic_edits(selected_components, aniso_edited)
            _any_aniso = any(c.is_anisotropic() for c in selected_components)
            if _any_aniso:
                st.info(f"Powder averaging active for anisotropic/high-spin components ({n_orient} orientations). "
                        "Increase orientation accuracy in the sidebar for smoother patterns.")
        else:
            st.caption("Select components above to edit their anisotropic parameters.")

    if parsed is not None and prep is not None and selected_components:
        weights = {component.component_id: component.weight for component in selected_components}
        _need_powder = any(c.is_anisotropic() for c in selected_components)
        if _need_powder:
            with st.spinner(f"Simulating powder pattern ({n_orient} orientations)..."):
                sim_total, sim_curves = simulate_model(prep.field_mT, selected_components, weights=weights, mw_frequency_GHz=mw_freq, n_orientations=n_orient)
        else:
            sim_total, sim_curves = simulate_model(prep.field_mT, selected_components, weights=weights, mw_frequency_GHz=mw_freq, n_orientations=n_orient)
        # Cache for auto-alignment in the Preprocess tab
        st.session_state["sim_preview"] = (prep.field_mT.copy(), sim_total.copy())
        st.session_state["selected_components"] = selected_components
        st.session_state["n_orient"] = n_orient
        traces = {"experimental processed": prep.processed, "simulation total": sim_total}
        traces.update({cid: weights.get(cid, 1.0) * curve for cid, curve in sim_curves.items()})
        traces.update(overlay_ui(prep.field_mT, "model_tab"))
        st.plotly_chart(spectrum_figure(prep.field_mT, traces, "Manual simulation"), width="stretch")

with tabs[4]:
    st.subheader("Fit")
    with st.expander("📖 Scientific background — fitting algorithm, parameters & statistics", expanded=False):
        st.markdown(SCIENCE_FIT)
    if parsed is None or prep is None or not selected_components:
        st.info("Import, preprocess, and select components first.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        mode = fc1.selectbox("Fit mode", ["weights only", "weights + linewidths", "weights + linewidths + g"], index=0)
        baseline_order = fc2.selectbox("Baseline in fit", [0, 1], index=0)
        max_eval = fc3.number_input("Max evaluations", min_value=100, max_value=5000, value=700, step=100)
        mcc1, mcc2, mcc3 = st.columns(3)
        mc_on = mcc1.checkbox("Monte-Carlo error bars", value=False,
                              help="Bootstrap uncertainties by refitting noise realisations — captures nonlinearity "
                                   "and parameter correlations, unlike linearised errors. Slower (recommended for publication).")
        mc_n = mcc2.number_input("MC realisations", min_value=20, max_value=500, value=50, step=10, disabled=not mc_on)
        mc_noise = mcc3.selectbox("MC noise model", ["gaussian", "residual"], disabled=not mc_on,
                                  help="gaussian: add Gaussian noise of the residual std; residual: resample fit residuals.")

        # ── Spectral masking ──────────────────────────────────────────────────
        with st.expander("Spectral masking — exclude field regions from fit optimization", expanded=False):
            st.markdown(
                "Define field ranges to **exclude** from the least-squares optimization. "
                "Useful when peaks in those regions are artefacts, solvent lines, or belong to an unmodelled species. "
                "Excluded regions are still shown in the overlay plot so you can inspect them."
            )
            _mask_default = pd.DataFrame({"From (mT)": pd.Series([], dtype=float), "To (mT)": pd.Series([], dtype=float)})
            _mask_state = st.session_state.get("fit_masks_df", _mask_default.to_dict("records"))
            _mask_edited = st.data_editor(
                pd.DataFrame(_mask_state) if _mask_state else _mask_default,
                num_rows="dynamic",
                use_container_width=True,
                key="mask_editor",
                column_config={
                    "From (mT)": st.column_config.NumberColumn("From (mT)", format="%.2f"),
                    "To (mT)": st.column_config.NumberColumn("To (mT)", format="%.2f"),
                },
            )
            st.session_state["fit_masks_df"] = _mask_edited.to_dict("records")
            _fit_masks = [
                (float(r["From (mT)"]), float(r["To (mT)"]))
                for r in st.session_state["fit_masks_df"]
                if r.get("From (mT)") is not None and r.get("To (mT)") is not None
                and float(r["From (mT)"]) < float(r["To (mT)"])
            ]

        # Build boolean keep-mask for fitting
        _fit_keep = np.ones(len(prep.field_mT), dtype=bool)
        for _mlo, _mhi in _fit_masks:
            _fit_keep &= ~((prep.field_mT >= _mlo) & (prep.field_mT <= _mhi))
        _n_excl = int((~_fit_keep).sum())
        if _n_excl:
            st.caption(f"ℹ {_n_excl} of {len(prep.field_mT)} data points are excluded from the fit by spectral masks.")

        # ── Run fit ───────────────────────────────────────────────────────────
        _auto_refit = st.session_state.pop("_auto_refit", False)
        if st.button("Run fit", type="primary") or _auto_refit:
            _use_field = prep.field_mT[_fit_keep]
            _use_exp   = prep.processed[_fit_keep]
            _spinner_msg = "Re-fitting with aligned field..." if _auto_refit else (
                f"Fitting on {len(_use_field)} points ({_n_excl} masked)..." if _n_excl else "Fitting..."
            )
            with st.spinner(_spinner_msg):
                fit = fit_spectrum(
                    _use_field,
                    _use_exp,
                    preset=preset,
                    components=selected_components,
                    mw_frequency_GHz=mw_freq,
                    mode=mode,
                    baseline_order=int(baseline_order),
                    max_nfev=int(max_eval),
                    n_orientations=st.session_state.get("n_orient", 1000),
                    n_monte_carlo=int(mc_n) if mc_on else 0,
                    mc_method=mc_noise if mc_on else "gaussian",
                )
                # Re-evaluate on FULL grid for display (masked fit gives parameters only)
                if _n_excl > 0:
                    from dataclasses import replace as _dc_replace
                    _full_sim, _full_curves = simulate_model(
                        prep.field_mT, fit.components, fit.weights, mw_freq, fit.baseline0, fit.baseline1
                    )
                    fit = _dc_replace(
                        fit,
                        field_mT=prep.field_mT,
                        experimental=prep.processed,
                        fit_total=_full_sim,
                        residual=prep.processed - _full_sim,
                        component_curves=_full_curves,
                    )
            st.session_state["fit"] = fit

        fit = st.session_state.get("fit")
        if fit:
            # ── Fit overlay ──────────────────────────────────────────────────
            _overlay_traces = {"experimental": fit.experimental, "fit": fit.fit_total}
            if _fit_masks:
                _mask_show = np.full_like(fit.experimental, np.nan)
                for _mlo, _mhi in _fit_masks:
                    _mi = (fit.field_mT >= _mlo) & (fit.field_mT <= _mhi)
                    _mask_show[_mi] = fit.experimental[_mi]
                _overlay_traces["excluded (masked)"] = _mask_show
            _overlay_traces.update(overlay_ui(fit.field_mT, "fit_tab"))
            _overlay_fig = spectrum_figure(fit.field_mT, _overlay_traces, "Fit overlay")
            st.plotly_chart(_overlay_fig, use_container_width=True)

            save_fit_ui("Fit", "fit_tab", field_mT=fit.field_mT, experimental=fit.experimental,
                        fit_total=fit.fit_total, components=fit.components, weights=fit.weights,
                        R2=fit.metrics.get("R2", float("nan")), n_parameters=fit.n_parameters)

            # ── Auto-align from fit overlay ──────────────────────────────────
            _cur_shift = float(st.session_state.get("field_shift_mT", 0.0))
            _oa1, _oa2 = st.columns([2, 5])
            if _oa1.button("Auto-align peak to fit",
                           help="Shift experimental axis to the fitted peak, then refit."):
                from epr_simfit.preprocessing import auto_align_spectra
                _delta = auto_align_spectra(fit.experimental, fit.fit_total, fit.field_mT)
                if abs(_delta) > 0.005:
                    st.session_state["_pending_field_shift"] = round(_cur_shift + _delta, 2)
                    st.session_state["_auto_refit"] = True
                    st.rerun()
                else:
                    st.toast("Peaks are already aligned.", icon="✅")
            _oa2.caption(f"Active field shift: **{_cur_shift:+.2f} mT** — adjust in Preprocess → Field alignment.")

            # ── Residual plot + noise-floor band ─────────────────────────────
            st.markdown("**Residual  (experimental − fit)**")
            _rc1, _rc2 = st.columns([4, 1])
            _res_pct = _rc2.slider("Noise floor %", 0, 50, 0, 1,
                help="Gray band at ±N% of peak residual. Features outside the band are likely real unmodelled signal.")
            _res_fig = residual_figure(fit.field_mT, fit.residual)
            if _res_pct > 0:
                _abs_max = float(np.nanmax(np.abs(fit.residual))) if fit.residual.size else 1.0
                _thresh  = _abs_max * _res_pct / 100.0
                _res_fig.add_hrect(y0=-_thresh, y1=_thresh,
                                   fillcolor="rgba(180,180,180,0.25)",
                                   line_width=1, line_color="rgba(150,150,150,0.5)",
                                   annotation_text=f"±{_res_pct}% noise floor",
                                   annotation_position="top right")
            _rc1.plotly_chart(_res_fig, use_container_width=True)

            # ── Goodness of fit ───────────────────────────────────────────────
            st.subheader("Goodness of fit")
            fq = assess_fit_quality(fit.metrics)
            qc = st.columns(6)
            qc[0].metric("Status",       f"{fq.icon} {fq.label}")
            qc[1].metric("R²",           f"{fit.metrics['R2']:.4f}")
            qc[2].metric("Norm. RMSE",   f"{fit.metrics['normalized RMSE']:.4f}")
            qc[3].metric("RMSE",         f"{fit.metrics['RMSE']:.4g}")
            qc[4].metric("AIC",          f"{fit.metrics['AIC']:.1f}")
            qc[5].metric("BIC",          f"{fit.metrics['BIC']:.1f}")
            st.markdown(f"**{fq.r2_note}**")
            st.caption(fq.nrmse_note)
            st.info(fq.overall)

            # ── Improvement suggestions ───────────────────────────────────────
            _suggestions = suggest_fit_improvements(fit, fit_mode=mode)
            if _suggestions:
                with st.expander(
                    f"💡 Suggestions to improve this fit  ({len(_suggestions)} item{'s' if len(_suggestions)!=1 else ''})",
                    expanded=any(s.priority == "High" for s in _suggestions),
                ):
                    for _sg in _suggestions:
                        _col_icon = {"High": "🔴", "Medium": "🟡", "Low": "🔵", "Info": "✅"}.get(_sg.priority, "ℹ️")
                        st.markdown(f"**{_col_icon} [{_sg.priority}] {_sg.title}**")
                        st.caption(f"*Why:* {_sg.reason}")
                        st.caption(f"*What to do:* {_sg.action}")
                        st.divider()

            # ── Component decomposition ───────────────────────────────────────
            _comp_fig = component_figure(fit.field_mT, fit.component_curves, fit.weights)
            st.plotly_chart(_comp_fig, use_container_width=True)

            # ── Extend model and refit ────────────────────────────────────────
            _fitted_ids = [c.component_id for c in fit.components]
            _extendable = [cid for cid in component_ids if cid not in _fitted_ids]
            with st.expander("➕ Extend model — add components and refit", expanded=False):
                st.caption(
                    "Pick additional components to add to the fitted model and immediately refit. "
                    "Useful when the residual shows unaccounted peaks."
                )
                _extra_ids = st.multiselect("Components to add", _extendable, key="extend_component_ids")
                if _extra_ids:
                    _extra_comps = [all_by_id[cid].clone() for cid in _extra_ids if cid in all_by_id]
                    _extended_comps = fit.components + _extra_comps
                    _ex1, _ex2 = st.columns([2, 5])
                    if _ex1.button("Refit with extended model", type="primary", key="btn_refit_extended"):
                        with st.spinner(f"Refitting with {len(_extended_comps)} components..."):
                            _ext_fit = fit_spectrum(
                                prep.field_mT[_fit_keep], prep.processed[_fit_keep],
                                preset=preset, components=_extended_comps,
                                mw_frequency_GHz=mw_freq, mode=mode,
                                baseline_order=int(baseline_order), max_nfev=int(max_eval),
                                n_orientations=st.session_state.get("n_orient", 1000),
                            )
                            if _n_excl > 0:
                                from dataclasses import replace as _dc_replace
                                _fs2, _fc2 = simulate_model(
                                    prep.field_mT, _ext_fit.components, _ext_fit.weights,
                                    mw_freq, _ext_fit.baseline0, _ext_fit.baseline1,
                                )
                                _ext_fit = _dc_replace(_ext_fit,
                                    field_mT=prep.field_mT, experimental=prep.processed,
                                    fit_total=_fs2, residual=prep.processed - _fs2,
                                    component_curves=_fc2)
                        st.session_state["fit"] = _ext_fit
                        st.rerun()
                    _ex2.caption(
                        f"Will add: {', '.join(_extra_ids)}. "
                        f"New model has {len(_extended_comps)} components vs current {len(fit.components)}."
                    )

            # ── Save to comparison ────────────────────────────────────────────
            _fs_val     = float(st.session_state.get("field_shift_mT", 0.0))
            _save_label = f"{fit.preset}  shift={_fs_val:+.2f}mT  R²={fit.metrics['R2']:.3f}"
            sc1, sc2 = st.columns([2, 5])
            if sc1.button("💾 Save fit to comparison", help="Add to the Compare tab for side-by-side model selection."):
                if "saved_fits" not in st.session_state:
                    st.session_state["saved_fits"] = {}
                _n = len(st.session_state["saved_fits"]) + 1
                st.session_state["saved_fits"][f"[{_n}] {_save_label}"] = fit
                st.toast(f"Saved as '[{_n}] {_save_label}'", icon="💾")
            sc2.caption(f"Will be saved as: **{_save_label}**")

            # ── Export plots ──────────────────────────────────────────────────
            with st.expander("📥 Export plots", expanded=False):
                st.caption("Download interactive HTML files — open in any browser, zoom, pan, hover for values.")
                _ep1, _ep2, _ep3 = st.columns(3)
                _ep1.download_button(
                    "Fit overlay (HTML)",
                    data=_overlay_fig.to_html(include_plotlyjs="cdn").encode(),
                    file_name="Open-Sym-EPR_fit_overlay.html", mime="text/html",
                )
                _ep2.download_button(
                    "Residual (HTML)",
                    data=_res_fig.to_html(include_plotlyjs="cdn").encode(),
                    file_name="Open-Sym-EPR_residual.html", mime="text/html",
                )
                _ep3.download_button(
                    "Components (HTML)",
                    data=_comp_fig.to_html(include_plotlyjs="cdn").encode(),
                    file_name="Open-Sym-EPR_components.html", mime="text/html",
                )

            # ── Detected intermediates ────────────────────────────────────────
            st.subheader("Detected paramagnetic intermediates / species")
            _it1, _it2 = st.columns([3, 1])
            _interm_thresh = _it2.slider("Min fraction %", 1, 30, 3, 1,
                help="Only show components contributing at least this fraction of total spectral intensity.")
            _it1.caption(
                f"Components ≥ {_interm_thresh}% shown. "
                "Assignments are candidate identifications — validate with controls and isotope labelling."
            )
            intermediates = suggest_intermediates(fit, threshold_pct=float(_interm_thresh))
            if intermediates:
                st.dataframe(intermediates_dataframe(intermediates), use_container_width=True)
                for d in intermediates:
                    with st.expander(f"{d.rank}. {d.name}  —  {d.fraction_pct:.1f}%  ({d.confidence})", expanded=False):
                        st.markdown(f"**Assignment:** {d.assignment}  |  **Category:** {d.category}")
                        st.markdown(f"**g-value:** `{d.g:.5f}`  |  **ΔBpp:** `{d.linewidth_mT:.4f} mT`  |  **Hyperfine:** {d.nuclei_str}")
                        if d.interpretation:
                            st.info(d.interpretation)
                        if d.warning:
                            st.warning(d.warning)
            else:
                st.info(f"No components exceed {_interm_thresh}%. Lower the slider or add more components.")

            # ── Publication-ready parameters ──────────────────────────────────
            st.subheader("Publication-ready parameters")
            pub_df = publication_parameters_table(fit)
            st.dataframe(pub_df, use_container_width=True)
            with st.expander("Detailed fitted parameter table (raw)", expanded=False):
                st.dataframe(fit.parameters, use_container_width=True)
                st.dataframe(fit.component_fractions, use_container_width=True)

            # ── Monte-Carlo / bootstrap uncertainties ─────────────────────────
            if getattr(fit, "mc_errors", None) is not None:
                st.subheader("Monte-Carlo uncertainties (bootstrap refits)")
                st.caption(
                    "Publication-grade error bars: many noise realisations were refitted; the spread "
                    "captures nonlinearity and parameter correlations, unlike the linearised (asymptotic) "
                    "standard errors. Report the 95% confidence interval [2.5%, 97.5%]."
                )
                _mc = fit.mc_errors.copy()
                _mc_show = pd.DataFrame({
                    "component": _mc["component"],
                    "parameter": _mc["parameter"],
                    "value": _mc["value"].map(lambda v: f"{v:.5g}"),
                    "MC σ": _mc["mc_std"].map(lambda v: f"{v:.3g}"),
                    "95% CI": [f"[{lo:.5g}, {hi:.5g}]" for lo, hi in zip(_mc["ci_2.5%"], _mc["ci_97.5%"])],
                    "linear SE": _mc["linear_std_error"].map(lambda v: f"{v:.3g}" if np.isfinite(v) else "—"),
                })
                st.dataframe(_mc_show, use_container_width=True)
                st.download_button("⬇ Monte-Carlo uncertainties (CSV)", _mc.to_csv(index=False).encode(),
                                   "Open-Sym-EPR_monte_carlo_uncertainties.csv", "text/csv")
                st.caption("If MC σ greatly exceeds the linear SE, the linearised error was over-optimistic "
                           "(poorly constrained or strongly correlated parameter).")

            # ── Methods paragraph ─────────────────────────────────────────────
            st.subheader("Publication methods paragraph")
            st.caption("Copy into your manuscript methods or supplementary information.")
            _methods_text = publication_methods_paragraph(fit, mw_freq, float(st.session_state.get("field_shift_mT", 0.0)))
            st.text_area("Methods paragraph", _methods_text, height=220, label_visibility="collapsed")
            st.download_button(
                "⬇ Download publication parameters (CSV)",
                data=pub_df.to_csv(index=False).encode(),
                file_name="Open-Sym-EPR_publication_parameters.csv", mime="text/csv",
            )
            st.warning("Fit quality is not chemical proof. Validate with standards, controls, isotope/substitution tests, and chemistry-specific constraints.")

with tabs[5]:
    st.subheader("Compare")
    st.caption(
        "Save fits using **💾 Save fit to comparison** in the Fit tab after each run "
        "(different models, modes, alignments). This tab compares them side-by-side."
    )

    _saved_fits: dict = st.session_state.get("saved_fits", {})

    if not _saved_fits:
        st.info(
            "No fits saved yet.  \n"
            "1. Go to the **Fit** tab and run a fit.  \n"
            "2. Click **💾 Save fit to comparison**.  \n"
            "3. Repeat with different models or settings.  \n"
            "4. Return here to compare them by BIC / AIC / R²."
        )
    else:
        # ── Metrics table ─────────────────────────────────────────────────────
        _sf_rows = []
        for _lbl, _sf in _saved_fits.items():
            _sf_rows.append({"Model": _lbl, "n params": _sf.n_parameters, **{k: v for k, v in _sf.metrics.items()}})
        _sf_df = pd.DataFrame(_sf_rows)
        _sf_df["ΔBIC"] = (_sf_df["BIC"] - _sf_df["BIC"].min()).round(2)
        _sf_df["ΔAIC"] = (_sf_df["AIC"] - _sf_df["AIC"].min()).round(2)
        _sf_df = _sf_df.sort_values("BIC").reset_index(drop=True)

        # Highlight best row
        _best_label = _sf_df.iloc[0]["Model"]
        st.markdown(f"**Best model by BIC: {_best_label}** — ΔBIC > 10 is strong evidence for preference.")
        st.dataframe(_sf_df, use_container_width=True)

        # ── BIC bar chart ─────────────────────────────────────────────────────
        _cmp_for_bar = _sf_df.rename(columns={"Model": "model"})[["model", "BIC", "AIC", "R2", "ΔBIC", "ΔAIC"]]
        _metric_sel = st.radio("Plot metric", ["BIC", "AIC", "R2"], horizontal=True, index=0)
        st.plotly_chart(comparison_bar_figure(_cmp_for_bar, _metric_sel), use_container_width=True)

        # ── Overlay all saved fits on experimental spectrum ───────────────────
        if parsed is not None and prep is not None:
            with st.expander("Overlay all saved fits on experimental spectrum", expanded=True):
                _ov_traces = {"experimental": prep.processed}
                for _lbl, _sf in _saved_fits.items():
                    _ov_traces[_lbl] = np.interp(prep.field_mT, _sf.field_mT, _sf.fit_total)
                _ov_fig = spectrum_figure(prep.field_mT, _ov_traces, "All saved fits vs experimental")
                st.plotly_chart(_ov_fig, use_container_width=True)
                st.download_button(
                    "⬇ Export comparison overlay (HTML)",
                    data=_ov_fig.to_html(include_plotlyjs="cdn").encode(),
                    file_name="Open-Sym-EPR_comparison_overlay.html", mime="text/html",
                )

        # ── BIC interpretation guide ──────────────────────────────────────────
        with st.expander("ΔBIC interpretation (Kass & Raftery, 1995)", expanded=False):
            st.markdown("""
| ΔBIC | Evidence against higher-BIC model |
|------|----------------------------------|
| 0 – 2 | Not worth more than a bare mention |
| 2 – 6 | Positive (weak) evidence |
| 6 – 10 | Strong evidence |
| > 10 | Very strong evidence |

Lower BIC = statistically preferred. BIC penalises additional parameters more heavily than AIC.
Use model comparison as a guide; chemical validation always takes precedence.
""")

        # ── Download comparison table ──────────────────────────────────────────
        _dc1, _dc2 = st.columns([2, 5])
        _dc1.download_button(
            "⬇ Comparison table (CSV)",
            data=_sf_df.to_csv(index=False).encode(),
            file_name="Open-Sym-EPR_model_comparison.csv", mime="text/csv",
        )
        if _dc2.button("🗑 Clear all saved fits"):
            st.session_state["saved_fits"] = {}
            st.rerun()

    # ── Batch preset comparison (secondary) ──────────────────────────────────
    with st.expander("Run batch preset comparison on current aligned spectrum (advanced)", expanded=False):
        st.caption(
            "Fits the selected library presets on the current preprocessed + aligned data. "
            "Results appear only here and are NOT added to the saved-fit comparison above."
        )
        if parsed is None or prep is None:
            st.info("Import and preprocess first.")
        else:
            defaults2 = ["G0_single_line"] + [m for m in suggested["suggested_models"] if m != "G0_single_line"][:3]
            compare_presets = st.multiselect("Presets", list(MODEL_PRESETS.keys()), default=list(dict.fromkeys(defaults2)), key="cmp_presets2")
            compare_mode = st.selectbox("Fit mode", ["weights only", "weights + linewidths"], index=0, key="cmp_mode2")
            if st.button("Run batch comparison"):
                with st.spinner("Running batch comparison on aligned spectrum..."):
                    comparison, _ = compare_models(prep.field_mT, prep.processed, compare_presets, mw_frequency_GHz=mw_freq, mode=compare_mode)
                st.session_state["batch_comparison"] = comparison
            _bc = st.session_state.get("batch_comparison")
            if _bc is not None and not _bc.empty:
                st.dataframe(_bc, use_container_width=True)
                st.plotly_chart(comparison_bar_figure(_bc, "BIC"), use_container_width=True)

with tabs[6]:
    st.subheader("Export")
    fit = st.session_state.get("fit")
    comparison = st.session_state.get("comparison")
    processed = None
    if parsed is not None and prep is not None:
        processed = pd.DataFrame({"Field_mT": prep.field_mT, "Intensity_raw": prep.raw, "Intensity_processed": prep.processed})
    evidence_summary = {
        "evidence_class": "GENERAL_EPR_FIT",
        "explanation": "Open-Sym-EPR reports candidate spectral decompositions using high-field isotropic cw-EPR models.",
        "warnings": ["Validate chemical assignments independently; anisotropic tensor cases require specialist EPR analysis."],
        "recommended_manuscript_wording": "The cw-EPR spectrum was fitted using Open-Sym-EPR high-field isotropic model components, and candidate assignments were evaluated with model comparison and chemical controls.",
    }
    export_context = {
        "project_title": project_title,
        "sample_name": sample_name,
        "solvent_matrix": solvent,
        "catalyst_material_combination": catalyst_material,
        "atmosphere_gas": atmosphere,
        "condition": condition_text,
        "additives_spin_probe": additives,
        "sample_class": sample_class,
        "software": "Open-Sym-EPR",
    }
    report_txt = generate_report_text(
        parsed.metadata if parsed else {},
        context,
        prep.settings if prep else {},
        fit.preset if fit else preset if "preset" in locals() else "not fitted",
        fit.parameters if fit else None,
        comparison,
        evidence_summary,
    )
    report_txt = "User-entered experiment metadata\n" + str(export_context) + "\n\n" + report_txt
    st.text_area("Report preview", report_txt, height=320)
    if fit:
        st.dataframe(fit_components_dataframe(fit).head(300), width="stretch")
    if st.button("Prepare Open-Sym-EPR export package"):
        with st.spinner("Preparing CSVs, fitted metrics, plot data, figures, report, and citation files..."):
            zip_bytes = build_export_zip(
                fit=fit,
                processed=processed,
                comparison=comparison,
                evidence_summary=evidence_summary,
                report_txt=report_txt,
                report_html=generate_report_html(report_txt),
                config=export_context,
                mw_frequency_GHz=mw_freq,
            )
            extra_docs = [WHITE_PAPER, CITATION, CUSTOM_MODEL_DOC, REFERENCE_SPECTRA_DOC]
            if any(path.exists() for path in extra_docs):
                from io import BytesIO
                from zipfile import ZIP_DEFLATED, ZipFile

                buf = BytesIO()
                with ZipFile(BytesIO(zip_bytes)) as zin, ZipFile(buf, "w", ZIP_DEFLATED) as zout:
                    for item in zin.infolist():
                        zout.writestr(item, zin.read(item.filename))
                    if WHITE_PAPER.exists():
                        zout.writestr("citation/WHITE_PAPER.md", WHITE_PAPER.read_text(encoding="utf-8"))
                    if CITATION.exists():
                        zout.writestr("citation/CITATION.cff", CITATION.read_text(encoding="utf-8"))
                    if CUSTOM_MODEL_DOC.exists():
                        zout.writestr("docs/CUSTOM_MODEL_FORMAT.md", CUSTOM_MODEL_DOC.read_text(encoding="utf-8"))
                    if REFERENCE_SPECTRA_DOC.exists():
                        zout.writestr("docs/REFERENCE_SPECTRA.md", REFERENCE_SPECTRA_DOC.read_text(encoding="utf-8"))
                zip_bytes = buf.getvalue()
            st.session_state["export_zip"] = zip_bytes
    st.caption("Export includes fit metrics, model metrics, all plotted datasets as CSV, figures, reports, citation files, EasySpin script, and ORCA templates where a fit is available.")
    st.download_button("Download Open-Sym-EPR export ZIP", data=st.session_state.get("export_zip", b""), file_name="Open-Sym-EPR_export.zip", mime="application/zip", disabled="export_zip" not in st.session_state)

with tabs[7]:
    st.subheader("Batch / kinetics — fixed model across a spectrum series")
    st.caption(
        "Apply one fixed model to a time series, catalyst series, potential series, or dose series. "
        "Open-Sym-EPR fits every spectrum with the same components and plots component fractions vs your "
        "ordering coordinate (time, index, potential, etc.) with uncertainty bars."
    )
    if not selected_components:
        st.info("Build a model in the **Model builder** tab first — the batch uses those exact components.")
    else:
        st.markdown(f"**Active model:** {', '.join(c.component_id for c in selected_components)}")
        bc1, bc2, bc3 = st.columns(3)
        batch_coord_name = bc1.text_input("Coordinate name", value="time (min)")
        batch_mode = bc2.selectbox("Batch fit mode", ["weights only", "weights + linewidths"], index=0, key="batch_mode")
        batch_baseline = bc3.selectbox("Baseline", [0, 1], index=0, key="batch_baseline")
        batch_files = st.file_uploader(
            "Upload the spectrum series (multiple files)", type=["asc", "txt", "dat", "csv"],
            accept_multiple_files=True, key="batch_uploader",
        )
        st.caption("Coordinate is taken from each file's numeric prefix/suffix if present, otherwise the upload order (0,1,2,...).")
        if batch_files and st.button("Run batch / kinetics", type="primary"):
            from epr_simfit.batch import BatchSpectrum, run_batch, kinetic_plot_png

            spectra = []
            for idx, f in enumerate(batch_files):
                text = f.getvalue().decode("utf-8", errors="replace")
                m = re.search(r"([-+]?\d+\.?\d*)", f.name)
                coord = float(m.group(1)) if m else float(idx)
                try:
                    parsed_b = parse_epr_text(text, filename=f.name, mw_frequency_override=mw_freq)
                    prep_b = preprocess_spectrum(parsed_b.dataframe["Field_mT"], parsed_b.dataframe["Intensity_raw"])
                    spectra.append(BatchSpectrum(label=f.name, coordinate=coord,
                                                 field_mT=prep_b.field_mT, intensity=prep_b.processed))
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Skipped {f.name}: {exc}")
            if spectra:
                with st.spinner(f"Fitting {len(spectra)} spectra with the fixed model..."):
                    result = run_batch(
                        spectra, selected_components, mw_frequency_GHz=mw_freq,
                        mode=batch_mode, baseline_order=int(batch_baseline),
                        coordinate_name=batch_coord_name,
                        n_orientations=st.session_state.get("n_orient", 600),
                    )
                st.session_state["batch_result"] = result
        result = st.session_state.get("batch_result")
        if result is not None:
            st.markdown("**Component fractions vs coordinate**")
            from epr_simfit.batch import kinetic_plot_png
            st.image(kinetic_plot_png(result, title="Open-Sym-EPR component kinetics"), use_container_width=True)
            st.markdown("**Fraction table**")
            st.dataframe(result.fractions, use_container_width=True)
            st.markdown("**Per-spectrum fit metrics**")
            st.dataframe(result.metrics, use_container_width=True)
            kc1, kc2, kc3 = st.columns(3)
            kc1.download_button("⬇ Fractions (CSV)", data=result.fractions.to_csv(index=False).encode(),
                                file_name="Open-Sym-EPR_batch_fractions.csv", mime="text/csv")
            kc2.download_button("⬇ Weights (CSV)", data=result.weights.to_csv(index=False).encode(),
                                file_name="Open-Sym-EPR_batch_weights.csv", mime="text/csv")
            kc3.download_button("⬇ Metrics (CSV)", data=result.metrics.to_csv(index=False).encode(),
                                file_name="Open-Sym-EPR_batch_metrics.csv", mime="text/csv")
            st.download_button("⬇ Kinetic plot (PNG)", data=kinetic_plot_png(result),
                               file_name="Open-Sym-EPR_kinetics.png", mime="image/png")

with tabs[8]:
    st.subheader("Reference standards — validate against known systems")
    st.caption(
        "Literature reference standards (DPPH, TEMPO, Mn(II), Cu(II), vanadyl, PBN-OH) "
        "simulated natively for validation and cross-checking."
    )
    from epr_simfit.reference_library import reference_standards

    refs = reference_standards()
    ref_key = st.selectbox("Reference standard", list(refs.keys()),
                           format_func=lambda k: refs[k].name)
    ref = refs[ref_key]
    st.markdown(f"**{ref.name}** — {ref.note}")
    rcomp = ref.component
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("g (avg)", f"{(sum(rcomp.g_principal())/3):.5f}")
    rc2.metric("Spin S", f"{rcomp.spin_S:g}")
    rc3.metric("Engine", "powder" if rcomp.is_anisotropic() else "isotropic")
    rfield = np.linspace(ref.field_window_mT[0], ref.field_window_mT[1], 1500)
    with st.spinner("Simulating reference standard..."):
        rsim, _ = simulate_model(rfield, [rcomp], {rcomp.component_id: 1.0},
                                 mw_frequency_GHz=ref.mw_freq_GHz,
                                 n_orientations=st.session_state.get("n_orient", 1000))
    st.plotly_chart(spectrum_figure(rfield, {ref.name: rsim}, f"{ref.name} (Open-Sym-EPR simulation)"),
                    use_container_width=True)
    st.download_button("⬇ Reference spectrum (CSV)",
                       data=pd.DataFrame({"Field_mT": rfield, "Intensity": rsim}).to_csv(index=False).encode(),
                       file_name=f"OpenSymEPR_reference_{ref_key}.csv", mime="text/csv")

with tabs[9]:
    st.subheader("Solvers — native Python spin-Hamiltonian engine, running live in this tab")
    from epr_simfit import native_solvers as nsv
    import plotly.graph_objects as go

    def _native_fig(x, traces, xlabel, ylabel, title):
        fig = go.Figure()
        for nm, y in traces.items():
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nm))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                          template="plotly_white", height=430,
                          margin=dict(l=58, r=24, t=54, b=48),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        return fig

    if not nsv.AVAILABLE:
        st.error(
            "Native solver engine (Open-Sym-EPR) is not importable, so native solvers cannot run here.\n\n"
            f"Reason: `{nsv.IMPORT_ERROR}`\n\n"
            "Install it into this environment with:  `pip install -e E:\\Open-Sym-EPR`"
        )
    else:
        st.success(
            "Every solver below is a native-Python implementation (Open-Sym-EPR engine, validated "
            "against analytical limits) and runs **live in this tab — no MATLAB and no scripts**. "
            "Set the shared spin system once, then run any solver."
        )
        cat = nsv.native_catalog()
        st.dataframe(pd.DataFrame([
            {"Solver": s.name, "Experiment": s.experiment, "Title": s.title,
             "Native engine": s.engine, "Status": "Runs live ✅"}
            for s in cat.values()
        ]), use_container_width=True, hide_index=True)

        # ── 1. Shared spin system ────────────────────────────────────────────
        st.markdown("#### 1 · Spin system (shared by every solver)")
        sc1, sc2, sc3 = st.columns([1.1, 1, 1])
        with sc1:
            g_mode = st.radio("g-factor", ["isotropic", "anisotropic tensor"], horizontal=True, key="es_gmode")
            if g_mode == "isotropic":
                g_iso = st.number_input("g", value=2.0060, format="%.5f", key="es_giso")
                g_tensor = None
            else:
                gg = st.columns(3)
                gx = gg[0].number_input("gx", value=2.0080, format="%.5f", key="es_gx")
                gy = gg[1].number_input("gy", value=2.0061, format="%.5f", key="es_gy")
                gz = gg[2].number_input("gz", value=2.0027, format="%.5f", key="es_gz")
                g_iso, g_tensor = None, (gx, gy, gz)
        with sc2:
            es_S = st.selectbox("Electron spin S", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], index=0, key="es_S")
            es_D = st.number_input("D / MHz (ZFS, S>½)", value=0.0, format="%.1f", key="es_D")
            es_E = st.number_input("E / MHz (ZFS, S>½)", value=0.0, format="%.1f", key="es_E")
        with sc3:
            nuc_text = st.text_area("Nuclei  (isotope:A in mT)", value="14N:1.55",
                                    height=90, key="es_nuc",
                                    help="One per line or ';'-separated. Isotropic: `14N:1.55`. "
                                         "Anisotropic tensor (for ESEEM): `14N:0.3,0.3,1.0`. "
                                         "Known isotopes: 1H 2H 13C 14N 15N 19F 31P 27Al 51V 55Mn 63Cu 65Cu.")
        try:
            es_system = nsv.build_system(g_iso=g_iso, g_tensor=g_tensor, S=float(es_S),
                                         D_MHz=es_D, E_MHz=es_E, nuclei_text=nuc_text)
            es_sys_ok = True
        except Exception as exc:  # noqa: BLE001
            es_sys_ok = False
            st.error(f"Spin system invalid: {exc}")

        # default field window around resonance for the current microwave frequency
        _g0 = (g_iso if g_iso is not None else (sum(g_tensor) / 3.0)) if es_sys_ok else 2.006
        try:
            _bres = float(__import__("openspin").isotropic_resonance_field_mT(mw_freq, _g0))
        except Exception:  # noqa: BLE001
            _bres = 1000.0 * 6.626e-34 * mw_freq * 1e9 / (_g0 * 9.274e-24)
        _fmin_d, _fmax_d = round(_bres - 12, 1), round(_bres + 12, 1)

        if es_sys_ok:
            st.markdown("#### 2 · Solvers (each runs live; download the result as CSV)")
            sub = st.tabs([
                "garlic · cw (solution)", "pepper · cw (solid)", "salt · ENDOR",
                "saffron · ESEEM", "curry · magnetometry",
            ])

            # garlic
            with sub[0]:
                st.caption("Isotropic / fast-motion cw-EPR — liquid-solution spectra (cw solution · garlic).")
                g1, g2, g3, g4 = st.columns(4)
                fmin = g1.number_input("Field min / mT", value=_fmin_d, key="gar_fmin")
                fmax = g2.number_input("Field max / mT", value=_fmax_d, key="gar_fmax")
                glw = g3.number_input("Linewidth ΔBpp / mT", value=0.10, format="%.3f", key="gar_lw")
                geta = g4.slider("η (Lorentz↔Gauss)", 0.0, 1.0, 0.5, key="gar_eta")
                if st.button("▶ Run garlic", type="primary", key="run_garlic"):
                    with st.spinner("Running native garlic…"):
                        st.session_state["es_garlic"] = nsv.run_garlic(es_system, fmin, fmax, mw_freq, glw, geta)
                r = st.session_state.get("es_garlic")
                if r:
                    st.plotly_chart(_native_fig(r["x"], {"garlic": r["y"]}, r["xlabel"], r["ylabel"],
                                                f"garlic — isotropic cw-EPR (ν = {mw_freq} GHz)"),
                                    use_container_width=True)
                    st.download_button("⬇ garlic spectrum (CSV)",
                                       pd.DataFrame({"Field_mT": r["x"], "Intensity": r["y"]}).to_csv(index=False).encode(),
                                       "native_garlic.csv", "text/csv", key="dl_garlic")

            # pepper
            with sub[1]:
                st.caption("Solid-state powder cw-EPR with full g/A/D tensors and high spin (cw powder · pepper).")
                p1, p2, p3, p4 = st.columns(4)
                pfmin = p1.number_input("Field min / mT", value=_fmin_d, key="pep_fmin")
                pfmax = p2.number_input("Field max / mT", value=_fmax_d, key="pep_fmax")
                plw = p3.number_input("Linewidth ΔBpp / mT", value=0.30, format="%.3f", key="pep_lw")
                peta = p4.slider("η (Lorentz↔Gauss)", 0.0, 1.0, 0.5, key="pep_eta")
                pn = st.select_slider("Powder orientations", options=[300, 600, 1000, 2000, 4000],
                                      value=2000, key="pep_n")
                if st.button("▶ Run pepper", type="primary", key="run_pepper"):
                    with st.spinner("Running native pepper (powder average)…"):
                        st.session_state["es_pepper"] = nsv.run_pepper(es_system, pfmin, pfmax, mw_freq, plw, peta, n_orient=pn)
                r = st.session_state.get("es_pepper")
                if r:
                    st.plotly_chart(_native_fig(r["x"], {"pepper": r["y"]}, r["xlabel"], r["ylabel"],
                                                f"pepper — powder cw-EPR (ν = {mw_freq} GHz)"),
                                    use_container_width=True)
                    st.download_button("⬇ pepper spectrum (CSV)",
                                       pd.DataFrame({"Field_mT": r["x"], "Intensity": r["y"]}).to_csv(index=False).encode(),
                                       "native_pepper.csv", "text/csv", key="dl_pepper")

            # salt / ENDOR
            with sub[2]:
                st.caption("Orientation-selective / powder ENDOR (ENDOR · salt). Needs at least one nucleus.")
                if not es_system.nuclei:
                    st.warning("Add a nucleus in the shared spin system above to simulate ENDOR.")
                else:
                    s1, s2, s3, s4 = st.columns(4)
                    sfield = s1.number_input("Static field / mT", value=float(round(_bres, 1)), key="salt_field")
                    srfmax = s2.number_input("RF max / MHz", value=30.0, key="salt_rfmax")
                    srflw = s3.number_input("RF linewidth / MHz", value=0.20, format="%.3f", key="salt_lw")
                    sn = s4.select_slider("Orientations", options=[300, 600, 1000, 1500, 3000], value=1500, key="salt_n")
                    if st.button("▶ Run salt", type="primary", key="run_salt"):
                        with st.spinner("Running native salt (ENDOR)…"):
                            st.session_state["es_salt"] = nsv.run_salt(es_system, sfield, srfmax, mw_freq, srflw, n_orient=sn)
                    r = st.session_state.get("es_salt")
                    if r:
                        st.plotly_chart(_native_fig(r["x"], {"ENDOR": r["y"]}, r["xlabel"], r["ylabel"],
                                                    f"salt — ENDOR at {sfield} mT"), use_container_width=True)
                        st.caption("Peaks at |ν_n ± A/2| (weak coupling) or |A/2 ± ν_n| (strong coupling).")
                        st.download_button("⬇ ENDOR (CSV)",
                                           pd.DataFrame({"RF_MHz": r["x"], "Intensity": r["y"]}).to_csv(index=False).encode(),
                                           "native_salt_endor.csv", "text/csv", key="dl_salt")

            # saffron / ESEEM
            with sub[3]:
                st.caption("Pulse EPR — 2-pulse and 3-pulse ESEEM, time + frequency domain (pulse ESEEM · saffron). "
                           "Needs a nucleus with **anisotropic** hyperfine (use tensor syntax `14N:0.3,0.3,1.0`).")
                if not es_system.nuclei:
                    st.warning("Add an anisotropic nucleus in the shared spin system above to simulate ESEEM.")
                else:
                    f1, f2, f3, f4 = st.columns(4)
                    efield = f1.number_input("Field / mT", value=float(round(_bres, 1)), key="saf_field")
                    taumax = f2.number_input("Max delay / µs", value=4.0, key="saf_taumax")
                    eseq = f3.selectbox("Sequence", ["2-pulse (vs τ)", "3-pulse (vs T)"], key="saf_seq")
                    taufix = f4.number_input("τ (3-pulse) / µs", value=0.20, format="%.3f", key="saf_taufix")
                    sn2 = st.select_slider("Orientations", options=[200, 400, 800, 1600], value=800, key="saf_n")
                    if st.button("▶ Run saffron", type="primary", key="run_saffron"):
                        with st.spinner("Running native saffron (ESEEM)…"):
                            st.session_state["es_saffron"] = nsv.run_saffron(es_system, efield, taumax, eseq, taufix, n_orient=sn2)
                    r = st.session_state.get("es_saffron")
                    if r:
                        cc1, cc2 = st.columns(2)
                        cc1.plotly_chart(_native_fig(r["t"], {"echo": r["echo"]}, "delay / µs", "echo amplitude",
                                                     "saffron — ESEEM time domain"), use_container_width=True)
                        cc2.plotly_chart(_native_fig(r["freq"], {"|FFT|": r["fft"]}, "frequency / MHz", "amplitude",
                                                     "saffron — ESEEM frequency domain"), use_container_width=True)
                        st.download_button("⬇ ESEEM time trace (CSV)",
                                           pd.DataFrame({"time_us": r["t"], "echo": r["echo"]}).to_csv(index=False).encode(),
                                           "native_saffron_eseem.csv", "text/csv", key="dl_saffron")

            # curry / magnetometry
            with sub[4]:
                st.caption("Magnetometry — magnetic susceptibility χT(T) and magnetisation M(B) (magnetometry · curry).")
                cmode = st.radio("Observable", ["χT vs T", "M vs B"], horizontal=True, key="cur_mode")
                if cmode == "χT vs T":
                    m1, m2, m3 = st.columns(3)
                    tmax = m1.number_input("T max / K", value=300.0, key="cur_tmax")
                    chifield = m2.number_input("Measuring field / T", value=0.5, format="%.2f", key="cur_chifield")
                    cn = m3.select_slider("Orientations", options=[50, 100, 200, 400], value=200, key="cur_n1")
                    bmax_v, mtemp_v = 7.0, 2.0
                else:
                    m1, m2, m3 = st.columns(3)
                    bmax_v = m1.number_input("B max / T", value=7.0, key="cur_bmax")
                    mtemp_v = m2.number_input("Temperature / K", value=2.0, format="%.1f", key="cur_mtemp")
                    cn = m3.select_slider("Orientations", options=[50, 100, 200, 400], value=200, key="cur_n2")
                    tmax, chifield = 300.0, 0.5
                if st.button("▶ Run curry", type="primary", key="run_curry"):
                    with st.spinner("Running native curry (magnetometry)…"):
                        st.session_state["es_curry"] = (cmode, nsv.run_curry(
                            es_system, cmode, tmax, chifield, bmax_v, mtemp_v, n_orient=cn))
                rc = st.session_state.get("es_curry")
                if rc:
                    _mode, r = rc
                    st.plotly_chart(_native_fig(r["x"], {_mode: r["y"]}, r["xlabel"], r["ylabel"],
                                                f"curry — {_mode}"), use_container_width=True)
                    _g0c = g_iso if g_iso is not None else (sum(g_tensor) / 3.0)
                    if _mode.startswith("χT"):
                        st.caption(f"High-T Curie limit ≈ 0.12505·g²·S(S+1) = "
                                   f"{0.12505 * _g0c**2 * es_S * (es_S + 1):.3f} cm³ K mol⁻¹")
                    else:
                        st.caption(f"Saturation ≈ g·S = {_g0c * es_S:.2f} µ_B")
                    st.download_button("⬇ curry data (CSV)",
                                       pd.DataFrame({r["xlabel"]: r["x"], r["ylabel"]: r["y"]}).to_csv(index=False).encode(),
                                       "native_curry.csv", "text/csv", key="dl_curry")

with tabs[10]:
    st.subheader("ML-assisted fit — physics-grounded neural initialisation")
    st.caption("A neural network trained **only on spin-Hamiltonian–simulated spectra** predicts an "
               "initial estimate; the **physics least-squares fit then refines and validates it**, and an "
               "out-of-distribution check guards against the network extrapolating. The reported "
               "parameters and R² come from the physics fit — the network only chooses the starting point.")
    try:
        from epr_simfit import ml_fitting as mlf
        _ml_ok = mlf._OSP
    except Exception as _mlexc:  # noqa: BLE001
        _ml_ok = False
        st.error(f"ML module unavailable: {_mlexc}")
    if _ml_ok:
        with st.expander("How this is scientifically constrained", expanded=False):
            st.markdown(
                "- **Training data are physics, not labels you supply.** Spectra are generated by the "
                "spin-Hamiltonian forward model over physically realistic ranges; the network learns the "
                "inverse map spectrum→parameters.\n"
                "- **The physics fit is authoritative.** The network's output seeds `esfit`, which refines "
                "g, hyperfine, linewidth and reports the R² and uncertainties.\n"
                "- **Out-of-distribution guard.** The spectrum is reconstructed with the forward model at "
                "the predicted parameters; a poor reconstruction flags the input as outside the training "
                "family, so the network cannot silently hallucinate.\n"
                "- **Scope:** isotropic ¹⁴N nitroxide with optional β-proton (nitroxide triplets and "
                "PBN/DMPO triplet-of-doublets adducts).")
        mc1, mc2, mc3 = st.columns(3)
        n_train = mc1.select_slider("Training spectra", options=[2000, 4000, 6000, 10000], value=6000, key="ml_ntr")
        if mc2.button("Train / load model", key="ml_train"):
            with st.spinner("Training the neural initialiser on physics-simulated spectra…"):
                st.session_state["ml_est"] = mlf.get_or_train_default(mw_GHz=float(mw_freq), n_samples=int(n_train), force=True)
            st.success("Model ready (cached).")
        if "ml_est" not in st.session_state and mlf.default_model_path().exists():
            try:
                st.session_state["ml_est"] = mlf.load_estimator(mlf.default_model_path())
            except Exception:  # noqa: BLE001
                pass
        est = st.session_state.get("ml_est")
        mc3.metric("Model", "ready ✅" if est else "not trained")

        if est is None:
            st.info("Click **Train / load model** to build the neural initialiser (a few seconds).")
        elif prep is None:
            st.warning("Load and preprocess a spectrum in the Import/Preprocess tabs first.")
        else:
            ml_mode = st.radio(
                "Mode",
                ["ML-assisted (NN → physics refine, recommended)", "ML-driven (NN only, instant)"],
                key="ml_mode",
                help="ML-assisted: the network seeds the least-squares fit, which refines and validates "
                     "the parameters (authoritative R²). ML-driven: the network predicts all parameters "
                     "directly and the forward model reconstructs the spectrum — instant, but no "
                     "optimisation; the reconstruction R² shows whether to trust it.")
            driven = ml_mode.startswith("ML-driven")
            if st.button("🤖 Run ML fit", type="primary", key="ml_run"):
                if driven:
                    with st.spinner("Neural prediction → forward-model reconstruction…"):
                        dres = mlf.ml_driven_fit(est, prep.field_mT, prep.processed, mw_GHz=float(mw_freq))
                    st.session_state["ml_dres"] = dres; st.session_state["ml_res"] = None
                else:
                    with st.spinner("NN initialisation → physics refinement…"):
                        res = mlf.ml_assisted_fit(est, prep.field_mT, prep.processed, mw_GHz=float(mw_freq))
                    st.session_state["ml_res"] = res; st.session_state["ml_dres"] = None

            # ── ML-driven result ──
            dres = st.session_state.get("ml_dres")
            if driven and dres is not None:
                if dres.ood_flag:
                    st.error(f"⚠ Poor reconstruction (R²={dres.R2:.2f}) — the network's parameters do not "
                             "reproduce this spectrum. Use ML-assisted or classical fitting.")
                else:
                    st.success(f"Network reconstruction matches the data (R²={dres.R2:.2f}).")
                st.markdown("**Parameters predicted entirely by the network**")
                p = dres.params
                st.write({"g": round(p["g"], 5), "a(¹⁴N) / G": round(p["aN_G"], 2),
                          "a(βH) / G": round(p["aH_G"], 2), "ΔBpp / mT": round(p["lw_mT"], 3),
                          "R² (reconstruction)": round(dres.R2, 4)})
                st.plotly_chart(spectrum_figure(prep.field_mT,
                                                {"experimental": prep.processed / (np.max(np.abs(prep.processed)) or 1),
                                                 "ML-driven reconstruction": dres.fit},
                                                f"ML-driven fit (reconstruction R²={dres.R2:.3f})"),
                                use_container_width=True)
                st.caption(dres.note + "  ·  ML-driven uses no least-squares optimisation; for quantitative "
                           "work prefer ML-assisted.")

            # ── ML-assisted result ──
            res = st.session_state.get("ml_res")
            if (not driven) and res is not None:
                if res.ood_flag:
                    st.error(f"⚠ Out-of-distribution (reconstruction R²={res.ood_R2:.2f}). "
                             "The spectrum is unlike the network's training family — treat the NN estimate "
                             "with caution and rely on the physics fit.")
                else:
                    st.success(f"In-distribution (reconstruction R²={res.ood_R2:.2f}). NN initialisation accepted.")
                cca, ccb = st.columns(2)
                cca.markdown("**Neural initial estimate**")
                cca.write({k: round(v, 4) for k, v in res.nn_estimate.items()})
                if res.refined is not None:
                    m = res.refined.metrics
                    ccb.markdown("**Physics-refined fit (authoritative)**")
                    rv = dict(zip(res.refined.params["parameter"], res.refined.params["value"]))
                    ccb.write({"g": round(rv.get("g_iso", float("nan")), 5),
                               "a(¹⁴N) / G": round(rv.get("A0_iso", float("nan")) * 10, 2),
                               **({"a(βH) / G": round(rv["A1_iso"] * 10, 2)} if "A1_iso" in rv else {}),
                               "ΔBpp / mT": round(rv.get("lw", float("nan")), 3),
                               "R²": round(m["R2"], 4)})
                    _ml_traces = {"experimental": res.refined.experimental, "ML→physics fit": res.refined.fit}
                    _ml_traces.update(overlay_ui(res.refined.field_mT, "ml_tab"))
                    st.plotly_chart(spectrum_figure(res.refined.field_mT, _ml_traces,
                                                f"ML-assisted fit (R²={m['R2']:.3f})"), use_container_width=True)
                    # Build a SpinComponent from the refined parameters so this fit can be
                    # saved to the global store and reused as a starting model.
                    _ml_nuclei = [Nucleus(isotope="14N", A_mT=float(rv.get("A0_iso", 0.15)), label="N")]
                    if "A1_iso" in rv:
                        _ml_nuclei.append(Nucleus(isotope="1H", A_mT=float(rv["A1_iso"]), label="beta H"))
                    _ml_comp = SpinComponent(component_id="ml_refined", display_name="ML-refined radical",
                                             radical_assignment="ML-assisted fit", category="ml",
                                             g=float(rv.get("g_iso", 2.003)), nuclei=_ml_nuclei,
                                             linewidth_mT=float(rv.get("lw", 0.1)))
                    save_fit_ui("ML", "ml_tab", field_mT=res.refined.field_mT,
                                experimental=res.refined.experimental, fit_total=res.refined.fit,
                                components=[_ml_comp], weights={"ml_refined": 1.0}, R2=m["R2"])
                st.caption(res.note)

with tabs[11]:
    st.subheader("Spin-adduct mixture — manual ratios & automated recovery")
    st.markdown(
        "Compose any set of spin-trap adducts (or other components) at **user-defined ratios**, "
        "simulate the composite spectrum with its stacked contributions, and — when an experimental "
        "spectrum is loaded — **automatically recover** the ratios by least-squares fitting. The "
        "automated fit can also refine linewidths, g, and **hyperfine (esfit-style)** so a modelled "
        "triplet can match a measured one whose splitting differs from the library defaults."
    )
    _lib = default_components()
    _lib_ids = list(_lib.keys())
    _default_mix = [c.component_id for c in st.session_state.get("selected_components", [])
                    if c.component_id in _lib_ids] or ["pbn_oh", "pbn_nh2_candidate"]
    _default_mix = [cid for cid in _default_mix if cid in _lib_ids]
    mix_ids = st.multiselect("Adducts in the mixture", _lib_ids, default=_default_mix,
                             key="mix_component_ids")
    if not mix_ids:
        st.info("Select at least one component to build a mixture.")
    else:
        mix_components = [_lib[cid].clone() for cid in mix_ids]
        if any(_lib[cid].warning for cid in mix_ids):
            _warned = [f"**{_lib[cid].display_name}** — {_lib[cid].warning}" for cid in mix_ids if _lib[cid].warning]
            st.warning("Candidate/assignment cautions:\n\n- " + "\n- ".join(_warned))

        st.markdown("**Manual ratios** (relative; normalised to 100 %)")
        mcols = st.columns(min(4, len(mix_ids)))
        manual = {}
        for i, cid in enumerate(mix_ids):
            with mcols[i % len(mcols)]:
                manual[cid] = st.number_input(_lib[cid].display_name, min_value=0.0, max_value=100.0,
                                              value=round(100.0 / len(mix_ids), 1), step=1.0,
                                              key=f"mix_ratio_{cid}")
        norm = mixtures_mod.normalise_ratios(manual)
        _mix_mw = mw_freq if "mw_freq" in dir() else DEFAULT_MW_FREQUENCY_GHZ
        if parsed is not None and prep is not None:
            mix_field = prep.field_mT
        else:
            _c0 = 1e3 * 6.62607015e-34 * _mix_mw * 1e9 / (2.0023 * 9.2740100783e-24)  # centre (mT)
            mix_field = np.linspace(_c0 - 10.0, _c0 + 10.0, 1200)
        total, weighted = mixtures_mod.simulate_mixture(mix_field, mix_components, ratios=manual,
                                                        mw_frequency_GHz=_mix_mw, n_orientations=n_orient)
        traces = {"composite": total}
        traces.update({_lib[cid].display_name: weighted.get(cid, np.zeros_like(mix_field)) for cid in mix_ids})
        st.plotly_chart(spectrum_figure(mix_field, traces, "Manual mixture (composite + stacked)"), width="stretch")
        st.dataframe(pd.DataFrame([
            {"component": _lib[cid].display_name, "assignment": _lib[cid].radical_assignment,
             "manual ratio %": round(100.0 * norm[cid], 1)} for cid in mix_ids
        ]), hide_index=True, width="stretch")

        st.markdown("---")
        st.markdown("**Automated ratio recovery** (fit the mixture to the loaded spectrum)")
        if parsed is None or prep is None:
            st.info("Import and preprocess a spectrum (Import / Preprocess tabs) to enable automated fitting.")
        else:
            extra = st.multiselect("Also refine (beyond ratios)", ["linewidths", "g", "hyperfine"],
                                   default=["linewidths", "g", "hyperfine"], key="mix_fit_extra",
                                   help="'hyperfine' is esfit-style a-value refinement so a modelled "
                                        "triplet can match a measured splitting (e.g. 17 G).")
            fit_mode = "weights" + ("" if not extra else " + " + " + ".join(extra))
            if st.button("Fit mixture ratios to spectrum", key="mix_fit_btn"):
                with st.spinner("Fitting mixture ..."):
                    st.session_state["mixture_fit"] = mixtures_mod.fit_mixture_ratios(
                        prep.field_mT, prep.processed, [c.clone() for c in mix_components],
                        mw_frequency_GHz=_mix_mw, mode=fit_mode, max_nfev=1200, n_orientations=n_orient)
            res = st.session_state.get("mixture_fit")
            if res:
                st.metric("Fit R²", f"{res['R2']:.3f}")
                rows = mixtures_mod.mixture_table(res["fit"].components, res["fractions"])
                for r in rows:
                    r["recovered a-values (G)"] = ", ".join(
                        str(v) for v in res["hyperfine_G"].get(
                            next((c.component_id for c in res["fit"].components if c.display_name == r["component"]), ""), []))
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                _mix_traces = {"experimental": res["fit"].experimental, "mixture fit": res["fit"].fit_total}
                _mix_traces.update(overlay_ui(res["fit"].field_mT, "mix_tab"))
                st.plotly_chart(spectrum_figure(res["fit"].field_mT, _mix_traces,
                                                f"Automated mixture fit (R²={res['R2']:.3f})"), width="stretch")
                save_fit_ui("Mixture", "mix_tab", field_mT=res["fit"].field_mT,
                            experimental=res["fit"].experimental, fit_total=res["fit"].fit_total,
                            components=res["fit"].components, weights=res["fit"].weights,
                            R2=res["R2"], n_parameters=res["fit"].n_parameters,
                            extra={"percent": res["percent"]})
                if res["R2"] < 0.6:
                    st.warning(
                        "Low R²: this component set does not adequately reproduce the spectrum. "
                        "A poor mixture fit is **not** evidence for the chosen adducts — try a different "
                        "model (e.g. a bare nitroxide triplet) and confirm any N-centred adduct with ¹⁵N labelling."
                    )

with tabs[12]:
    st.subheader("White paper / citation")
    st.write(ABOUT_TEXT)
    st.markdown("**Developer**")
    st.write(DETAILED_INFO)
    if WHITE_PAPER.exists():
        whitepaper = WHITE_PAPER.read_text(encoding="utf-8")
        st.download_button("Download Open-Sym-EPR white paper", data=whitepaper, file_name="Open-Sym-EPR_WHITE_PAPER.md", mime="text/markdown")
        st.markdown(whitepaper)
    if CITATION.exists():
        citation = CITATION.read_text(encoding="utf-8")
        st.download_button("Download CITATION.cff", data=citation, file_name="CITATION.cff", mime="text/plain")
        st.code(citation, language="yaml")
    if CUSTOM_MODEL_DOC.exists():
        custom_doc = CUSTOM_MODEL_DOC.read_text(encoding="utf-8")
        st.download_button("Download custom model format guide", data=custom_doc, file_name="CUSTOM_MODEL_FORMAT.md", mime="text/markdown")
    if REFERENCE_SPECTRA_DOC.exists():
        reference_doc = REFERENCE_SPECTRA_DOC.read_text(encoding="utf-8")
        st.download_button("Download reference spectra plan", data=reference_doc, file_name="REFERENCE_SPECTRA.md", mime="text/markdown")

# ── Save project (captures the current working session; restore via sidebar → Project) ──
st.divider()
st.subheader("💾 Save project")
st.caption("Download your current session — spectrum, microwave frequency, selected/custom models, and the "
           "latest fit summary — as a single file. Reopen it later from the sidebar **Project — save / resume** "
           "panel to continue exactly where you left off.")
from epr_simfit import project as _projS
from epr_simfit.user_models import components_to_json as _c2j
try:
    _proj_comps = selected_components
except NameError:
    _proj_comps = []
_proj_fit = st.session_state.get("fit")
_proj_fit_summary = {}
if _proj_fit is not None and getattr(_proj_fit, "metrics", None):
    _proj_fit_summary = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in _proj_fit.metrics.items()}
try:
    _proj_bytes = _projS.save_project(
        microwave_frequency_GHz=float(mw_freq),
        spectrum_text=st.session_state.get("_cur_file_text"),
        filename=st.session_state.get("_cur_filename"),
        components_json=_c2j(_proj_comps, name="project_models") if _proj_comps else None,
        fit_summary=_proj_fit_summary,
        preprocess={"field_shift_mT": float(st.session_state.get("field_shift_mT", 0.0))},
    )
    st.download_button("⬇ Save project (.simepr.json)", data=_proj_bytes,
                       file_name="simepr_project.simepr.json", mime="application/json",
                       disabled=st.session_state.get("_cur_file_text") is None,
                       help="Loads a spectrum first (Data panel) to enable saving.")
except Exception as _pse:  # noqa: BLE001
    st.caption(f"(Project save unavailable: {_pse})")
