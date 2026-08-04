"""Open-Sym-EPR — native-Python EPR / ENDOR / ESEEM / magnetometry GUI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import openspin as osp
from openspin import science

st.set_page_config(page_title="Open-Sym-EPR", page_icon="🧲", layout="wide")

st.title("🧲 Open-Sym-EPR")
st.caption("Native-Python spin-Hamiltonian simulation — cw-EPR, ENDOR, ESEEM, and magnetometry. No MATLAB required.")
with st.expander("📖 What Open-Sym-EPR computes", expanded=False):
    st.markdown(science.INTRO)


# ── Sidebar: spin-system builder ──────────────────────────────────────────────
def line_figure(x, traces: dict, xlabel: str, ylabel: str, title: str):
    fig = go.Figure()
    for name, y in traces.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title,
                      height=430, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


def _default_nuc_df():
    return pd.DataFrame([{"isotope": "14N", "A_iso_mT": 1.55, "Ax_mT": None, "Ay_mT": None, "Az_mT": None}])


with st.sidebar:
    st.header("Spin system")

    # ── Spin-trap adduct presets + project save / resume ──
    with st.expander("🧪 Spin-trap adducts & 💾 project", expanded=False):
        import json as _json
        from openspin import spin_trap_models as _stm
        raw = {f"[{m['trap']}] {m['name']}": m for m in _stm.all_spin_trap_raw()}
        pick = st.selectbox("Spin-trap adduct model (PBN / DMSO / DMPO / POBN / DEPMPO)",
                            ["(none)"] + list(raw), key="adduct_pick")
        if pick != "(none)":
            m = raw[pick]
            hf = ", ".join(f"{iso} {a*10:.1f} G" for iso, a in m["nuclei_mT"]) or "no resolved hyperfine (broad)"
            st.caption(f"g = {m['g']:.4f} · {hf}" + (f"\n\n{m['note']}" if m['note'] else ""))
            if st.button("Load this adduct into the spin system", key="load_adduct"):
                st.session_state["osp_gx"] = st.session_state["osp_gy"] = st.session_state["osp_gz"] = m["g"]
                rows = [{"isotope": iso, "A_iso_mT": a, "Ax_mT": None, "Ay_mT": None, "Az_mT": None}
                        for iso, a in m["nuclei_mT"]]
                st.session_state["osp_nuc_df"] = pd.DataFrame(rows) if rows else _default_nuc_df()
                st.session_state.pop("nuc_editor", None)
                st.rerun()
        st.divider()
        proj_up = st.file_uploader("Open project (.json)", type=["json"], key="osp_proj_up")
        if proj_up is not None and st.button("↩ Load project", key="osp_load_proj"):
            try:
                o = _json.loads(proj_up.getvalue().decode("utf-8"))
                assert o.get("schema") == "Open-Sym-EPR.project", "not an Open-Sym-EPR project"
                st.session_state["osp_gx"], st.session_state["osp_gy"], st.session_state["osp_gz"] = [float(v) for v in o["g"]]
                st.session_state["osp_S"] = float(o["S"]); st.session_state["osp_D"] = float(o["D"])
                st.session_state["osp_E"] = float(o["E"]); st.session_state["osp_mw"] = float(o["mw_freq"])
                st.session_state["osp_nuc_df"] = pd.DataFrame(o["nuclei"]) if o.get("nuclei") else _default_nuc_df()
                st.session_state.pop("nuc_editor", None)
                st.rerun()
            except Exception as _e:  # noqa: BLE001
                st.error(f"Load failed: {_e}")

    for _k, _v in [("osp_S", 0.5), ("osp_gx", 2.0060), ("osp_gy", 2.0060), ("osp_gz", 2.0060),
                   ("osp_D", 0.0), ("osp_E", 0.0), ("osp_mw", 9.5)]:
        st.session_state.setdefault(_k, _v)
    s_spin = st.selectbox("Electron spin S", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], key="osp_S")
    st.markdown("**g-tensor**")
    gcols = st.columns(3)
    gx = gcols[0].number_input("gx", format="%.5f", key="osp_gx")
    gy = gcols[1].number_input("gy", format="%.5f", key="osp_gy")
    gz = gcols[2].number_input("gz", format="%.5f", key="osp_gz")
    st.markdown("**Zero-field splitting (MHz)** — for S > ½")
    zcols = st.columns(2)
    D_MHz = zcols[0].number_input("D", format="%.1f", key="osp_D")
    E_MHz = zcols[1].number_input("E", format="%.1f", key="osp_E")

    st.markdown("**Nuclei** (A in mT)")
    st.caption("Iso A or anisotropic Ax/Ay/Az. Leave Ax blank for isotropic.")
    st.session_state.setdefault("osp_nuc_df", _default_nuc_df())
    nuc_table = st.data_editor(st.session_state["osp_nuc_df"], num_rows="dynamic", use_container_width=True, key="nuc_editor")

    st.divider()
    mw_freq = st.number_input("Microwave frequency / GHz", min_value=1.0, max_value=300.0, step=0.1, key="osp_mw")
    n_orient = st.select_slider("Powder orientations", options=[300, 600, 1000, 2000, 4000], value=1000)

    # ── Save project ──
    import json as _json2
    _osp_proj = {"schema": "Open-Sym-EPR.project", "g": [float(gx), float(gy), float(gz)], "S": float(s_spin),
                 "D": float(D_MHz), "E": float(E_MHz), "mw_freq": float(mw_freq),
                 "nuclei": nuc_table.to_dict("records") if hasattr(nuc_table, "to_dict") else []}
    st.download_button("💾 Save project (.json)", data=_json2.dumps(_osp_proj, indent=2).encode(),
                       file_name="openspin_project.json", mime="application/json")


def build_system() -> osp.SpinSystem:
    nuclei = []
    for _, row in nuc_table.iterrows():
        iso = str(row.get("isotope", "")).strip()
        if not iso:
            continue
        ax = row.get("Ax_mT")
        if ax is not None and not (isinstance(ax, float) and np.isnan(ax)):
            tensor = (float(row["Ax_mT"]), float(row["Ay_mT"]), float(row["Az_mT"]))
            nuclei.append(osp.nucleus(iso, A_tensor_mT=tensor))
        else:
            a = row.get("A_iso_mT", 0.0)
            a = 0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else float(a)
            nuclei.append(osp.nucleus(iso, a))
    return osp.spin_system(g=(gx, gy, gz), nuclei=nuclei, S=float(s_spin), D_MHz=float(D_MHz), E_MHz=float(E_MHz))


system = build_system()

tabs = st.tabs(["cw-EPR (garlic/pepper)", "Fit experimental data", "ENDOR (salt)", "ESEEM (saffron)", "Magnetometry (curry)", "Theory / validation"])

# ── cw-EPR ────────────────────────────────────────────────────────────────────
with tabs[0]:
    with st.expander("📖 cw-EPR background", expanded=False):
        st.markdown(science.CW)
    c1, c2, c3, c4 = st.columns(4)
    fmin = c1.number_input("Field min / mT", value=330.0)
    fmax = c2.number_input("Field max / mT", value=350.0)
    lw = c3.number_input("Linewidth ΔBpp / mT", value=0.2, format="%.3f")
    eta = c4.slider("η (Lorentz↔Gauss)", 0.0, 1.0, 0.5)
    field = np.linspace(fmin, fmax, 2000)
    with st.spinner("Simulating cw-EPR..."):
        spec, used = osp.cw_auto(system, field, mw_freq, lw, eta, n_orient)
    st.info(f"Engine used: **{used}** ({'anisotropic powder' if used == 'pepper' else 'isotropic solution'}).")
    st.plotly_chart(line_figure(field, {"cw-EPR": spec}, "Magnetic field / mT", "dχ″/dB (a.u.)",
                                f"cw-EPR simulation ({used}, ν = {mw_freq} GHz)"), use_container_width=True)
    st.download_button("⬇ Download spectrum (CSV)",
                       pd.DataFrame({"Field_mT": field, "Intensity": spec}).to_csv(index=False).encode(),
                       "openspin_cw.csv", "text/csv")

# ── Fit experimental data ─────────────────────────────────────────────────────
with tabs[1]:
    with st.expander("📖 How fitting works", expanded=False):
        st.markdown(science.FITTING)
    up = st.file_uploader("Upload experimental spectrum (2 columns: field, intensity)",
                          type=["txt", "csv", "dat", "asc"], key="fit_upload")
    bruker_up = st.file_uploader("…or Bruker BES3T (.DTA + .DSC)", type=["dta", "dsc"],
                                 accept_multiple_files=True, key="fit_bruker",
                                 help="Upload BOTH the .DTA and .DSC files.")
    from openspin import parse_spectrum, normalise, esfit, default_vary
    from openspin.bruker import load_bes3t
    xf = yf = None
    if bruker_up:
        dsc = next((f for f in bruker_up if f.name.lower().endswith(".dsc")), None)
        dta = next((f for f in bruker_up if f.name.lower().endswith(".dta")), None)
        if dsc is not None and dta is not None:
            try:
                xf, yf, bmeta = load_bes3t(dsc.getvalue().decode("latin-1"), dta.getvalue())
                yf = normalise(yf, "max")
                st.success(f"Bruker: {bmeta.get('title', '')} — {len(xf)} pts, "
                           f"ν = {bmeta.get('microwave_frequency_GHz', mw_freq):.4f} GHz")
                if bmeta.get("microwave_frequency_GHz"):
                    st.caption(f"Set microwave frequency to {bmeta['microwave_frequency_GHz']:.4f} GHz in the sidebar to match.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Bruker import failed: {exc}")
        else:
            st.warning("Upload BOTH the .DTA and .DSC files.")
    elif up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
        try:
            xf, yf = parse_spectrum(text)
            yf = normalise(yf, "max")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse file: {exc}")
    if up is None and not bruker_up:
        st.info("Upload a two-column spectrum or a Bruker BES3T pair to fit. The sidebar spin system is the starting model.")
    if True:
        if xf is not None:
            st.success(f"Loaded {len(xf)} points, field {xf.min():.1f}–{xf.max():.1f} mT.")
            st.markdown("### Choose parameters to fit")
            st.caption("Tick a parameter to vary it; set the initial guess and bounds.")
            base_vary = default_vary(system)
            # let the user add g-tensor and per-nucleus tensor knobs
            knob_defs = dict(base_vary)
            if any(abs(a - system.g_principal[0]) > 1e-9 for a in system.g_principal):
                gx0, gy0, gz0 = system.g_principal
                knob_defs.pop("g_iso", None)
                knob_defs["gx"] = (gx0, gx0 - 0.02, gx0 + 0.02)
                knob_defs["gy"] = (gy0, gy0 - 0.02, gy0 + 0.02)
                knob_defs["gz"] = (gz0, gz0 - 0.02, gz0 + 0.02)
            if system.S > 0.5:
                knob_defs["D"] = (system.D_MHz, system.D_MHz - 2000, system.D_MHz + 2000)
                knob_defs["E"] = (system.E_MHz, max(system.E_MHz - 500, 0), system.E_MHz + 500)

            editor_rows = []
            for name, (v0, l0, h0) in knob_defs.items():
                editor_rows.append({"fit": name in ("g_iso", "gx", "lw", "scale") or name.endswith("_iso"),
                                    "parameter": name, "initial": v0, "lower": l0, "upper": h0})
            vary_df = st.data_editor(pd.DataFrame(editor_rows), use_container_width=True,
                                     disabled=["parameter"], key="vary_editor")

            fc1, fc2, fc3 = st.columns(3)
            do_global = fc1.checkbox("Global pre-search (robust)", value=True)
            n_global = fc2.number_input("Pre-search samples", 50, 2000, 400, step=50)
            fit_orient = fc3.number_input("Powder orientations (fit)", 100, 2000, 400, step=100)

            mc1, mc2, mc3 = st.columns(3)
            do_mc = mc1.checkbox("Monte-Carlo error bars", value=False,
                                 help="Bootstrap uncertainties by refitting many noise realisations — "
                                      "captures nonlinearity and parameter correlations, unlike linearised errors. Slower.")
            n_mc = mc2.number_input("MC realisations", 20, 500, 50, step=10, disabled=not do_mc)
            mc_method = mc3.selectbox("MC noise model", ["gaussian", "residual"], disabled=not do_mc,
                                      help="gaussian: add Gaussian noise of the residual std; "
                                           "residual: resample fit residuals (distribution-free).")

            if st.button("🎯 Run fit", type="primary"):
                vary = {r["parameter"]: (float(r["initial"]), float(r["lower"]), float(r["upper"]))
                        for _, r in vary_df.iterrows() if bool(r["fit"])}
                if not vary:
                    st.warning("Tick at least one parameter to fit.")
                else:
                    msg = "Fitting model to data..." if not do_mc else f"Fitting + {int(n_mc)} Monte-Carlo refits..."
                    with st.spinner(msg):
                        res = esfit(system, xf, yf, vary, mw_freq, eta=0.5,
                                    n_orientations=int(fit_orient), max_nfev=200,
                                    global_search=int(n_global) if do_global else 0,
                                    n_monte_carlo=int(n_mc) if do_mc else 0, mc_method=mc_method)
                    st.session_state["fit_result"] = res
            res = st.session_state.get("fit_result")
            if res is not None:
                m = res.metrics
                cols = st.columns(5)
                cols[0].metric("R²", f"{m['R2']:.4f}")
                cols[1].metric("RMSE", f"{m['RMSE']:.4g}")
                cols[2].metric("norm. RMSE", f"{m['normalized RMSE']:.4f}")
                cols[3].metric("AIC", f"{m['AIC']:.1f}")
                cols[4].metric("BIC", f"{m['BIC']:.1f}")
                st.plotly_chart(line_figure(res.field_mT,
                                            {"experimental": res.experimental, "fit": res.fit},
                                            "Magnetic field / mT", "intensity (a.u.)",
                                            f"Fit overlay (engine: {res.engine})"), use_container_width=True)
                st.plotly_chart(line_figure(res.field_mT, {"residual": res.residual},
                                            "Magnetic field / mT", "exp − fit", "Residual"),
                                use_container_width=True)
                st.markdown("### Fitted parameters")
                show = res.params.copy()
                show["value ± linear SE"] = [f"{v:.5g} ± {e:.2g}" if np.isfinite(e) else f"{v:.5g}"
                                             for v, e in zip(show["value"], show["std_error"])]
                st.dataframe(show[["parameter", "value ± linear SE", "lower", "upper"]], use_container_width=True)
                st.caption("Linear SE = asymptotic Jacobian-covariance error (first-pass approximation).")

                if res.mc_errors is not None:
                    st.markdown("### Monte-Carlo uncertainties (bootstrap refits)")
                    st.caption("Refitting many noise realisations; captures nonlinearity and parameter "
                               "correlations. 95% CI = [2.5%, 97.5%] of the refit distribution.")
                    mc = res.mc_errors.copy()
                    mc_show = pd.DataFrame({
                        "parameter": mc["parameter"],
                        "value": mc["value"].map(lambda v: f"{v:.5g}"),
                        "MC σ": mc["mc_std"].map(lambda v: f"{v:.2g}"),
                        "95% CI": [f"[{lo:.5g}, {hi:.5g}]" for lo, hi in zip(mc["ci_2.5%"], mc["ci_97.5%"])],
                        "linear SE": mc["linear_std_error"].map(lambda v: f"{v:.2g}" if np.isfinite(v) else "—"),
                    })
                    st.dataframe(mc_show, use_container_width=True)
                    st.caption("If MC σ ≫ linear SE, the parameter is poorly constrained or strongly correlated — "
                               "report the Monte-Carlo interval in publications.")

                st.caption(f"Field shift {res.field_shift_mT:+.3f} mT · amplitude scale {res.scale:.3f} · {res.message}")
                dl1, dl2 = st.columns(2)
                dl1.download_button("⬇ Fit overlay (CSV)",
                                    pd.DataFrame({"Field_mT": res.field_mT, "experimental": res.experimental,
                                                  "fit": res.fit, "residual": res.residual}).to_csv(index=False).encode(),
                                    "openspin_fit.csv", "text/csv")
                dl2.download_button("⬇ Fitted parameters (CSV)", res.params.to_csv(index=False).encode(),
                                    "openspin_fit_params.csv", "text/csv")

# ── ENDOR ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    with st.expander("📖 ENDOR background", expanded=False):
        st.markdown(science.ENDOR)
    e1, e2, e3 = st.columns(3)
    endor_field = e1.number_input("Static field / mT", value=350.0)
    rf_max = e2.number_input("RF max / MHz", value=30.0)
    rf_lw = e3.number_input("RF linewidth / MHz", value=0.2, format="%.3f")
    if not system.nuclei:
        st.warning("Add at least one nucleus in the sidebar to simulate ENDOR.")
    else:
        rf = np.linspace(0.1, rf_max, 2000)
        with st.spinner("Simulating ENDOR..."):
            espec = osp.endor_spectrum(system, endor_field, rf, mw_freq, rf_lw, n_orient, rf_max_MHz=float(rf_max))
        st.plotly_chart(line_figure(rf, {"ENDOR": espec}, "RF frequency / MHz", "ENDOR intensity (a.u.)",
                                    f"ENDOR at {endor_field} mT"), use_container_width=True)
        # show Larmor markers
        st.caption("Tip: peaks at |ν_n ± A/2| (weak coupling) or |A/2 ± ν_n| (strong coupling).")
        st.download_button("⬇ Download ENDOR (CSV)",
                           pd.DataFrame({"RF_MHz": rf, "Intensity": espec}).to_csv(index=False).encode(),
                           "openspin_endor.csv", "text/csv")

# ── ESEEM ─────────────────────────────────────────────────────────────────────
with tabs[3]:
    with st.expander("📖 ESEEM background", expanded=False):
        st.markdown(science.ESEEM)
    s1, s2, s3, s4 = st.columns(4)
    eseem_field = s1.number_input("Field / mT", value=350.0, key="eseem_field")
    tau_max = s2.number_input("Max delay / µs", value=4.0)
    seq = s3.selectbox("Sequence", ["2-pulse (vs τ)", "3-pulse (vs T)"])
    tau_fixed = s4.number_input("τ (3-pulse) / µs", value=0.2, format="%.3f")
    if not system.nuclei:
        st.warning("Add a nucleus with **anisotropic** hyperfine (Ax≠Az) — ESEEM needs the dipolar term.")
    else:
        t = np.linspace(0.0, tau_max, 2048)
        with st.spinner("Simulating ESEEM..."):
            if seq.startswith("2"):
                echo = osp.two_pulse_eseem(system, t, eseem_field, min(n_orient, 800))
            else:
                echo = osp.three_pulse_eseem(system, t, tau_fixed, eseem_field, min(n_orient, 800))
        freq, fspec = osp.eseem_fft(t, echo)
        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(line_figure(t, {"echo": echo}, "delay / µs", "echo amplitude",
                                     "ESEEM time domain"), use_container_width=True)
        mask = freq <= 30.0
        cc2.plotly_chart(line_figure(freq[mask], {"|FFT|": fspec[mask]}, "frequency / MHz", "amplitude",
                                     "ESEEM frequency domain"), use_container_width=True)
        st.download_button("⬇ Download ESEEM (CSV)",
                           pd.DataFrame({"time_us": t, "echo": echo}).to_csv(index=False).encode(),
                           "openspin_eseem.csv", "text/csv")

# ── Magnetometry ──────────────────────────────────────────────────────────────
with tabs[4]:
    with st.expander("📖 Magnetometry background", expanded=False):
        st.markdown(science.MAGNETOMETRY)
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**χT vs T**")
        t_max = st.number_input("T max / K", value=300.0)
        chi_field = st.number_input("Measuring field / T", value=0.5, format="%.2f")
        T = np.linspace(2.0, t_max, 80)
        with st.spinner("Computing χT(T)..."):
            chiT = osp.chiT_vs_T(system, T, chi_field, n_orientations=min(n_orient, 200))
        st.plotly_chart(line_figure(T, {"χT": chiT}, "Temperature / K", "χT / cm³ K mol⁻¹",
                                    "Magnetic susceptibility"), use_container_width=True)
        st.caption(f"High-T limit ≈ Curie constant 0.12505·g²·S(S+1) = {0.12505*((gx+gy+gz)/3)**2*s_spin*(s_spin+1):.3f} cm³ K mol⁻¹")
    with mcol2:
        st.markdown("**M vs B**")
        b_max = st.number_input("B max / T", value=7.0)
        m_temp = st.number_input("Temperature / K", value=2.0, format="%.1f")
        B = np.linspace(0.0, b_max, 60)
        with st.spinner("Computing M(B)..."):
            M = osp.magnetisation_muB(system, B, m_temp, n_orientations=min(n_orient, 200))
        st.plotly_chart(line_figure(B, {"M": M}, "Field / T", "M / µ_B",
                                    "Magnetisation"), use_container_width=True)
        st.caption(f"Saturation ≈ g·S = {((gx+gy+gz)/3)*s_spin:.2f} µ_B")

# ── Theory ────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(science.VALIDATION)
    st.divider()
    st.markdown(science.INTRO)
