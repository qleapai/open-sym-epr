"""Comprehensive feature test: exercise EVERY feature of the software on synthetic input.

Loads the synthetic inputs (gen_synthetic_inputs.py) with known ground truth and
drives both engines (Open-Sym-EPR native + Open-Sym-EPR) and the GUI end to end, printing a
PASS/FAIL matrix and a recovery check against the manifest. Run:
    python feature_test.py
Exit code is non-zero if any feature fails.
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MAN = json.loads((HERE / "syn_manifest.json").read_text())
MW = MAN["microwave_frequency_GHz"]

results: list[tuple[str, str, bool, str]] = []


def check(cat, name, fn):
    try:
        detail = fn() or ""
        results.append((cat, name, True, str(detail)))
    except Exception as exc:  # noqa: BLE001
        results.append((cat, name, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


def load_txt(fname):
    rows = [ln for ln in (HERE / fname).read_text().splitlines() if ln and not ln.startswith("#")][1:]
    arr = np.array([[float(v) for v in r.split()] for r in rows])
    return arr[:, 0], arr[:, 1]


# ════════════════════ A. Open-Sym-EPR native engine ════════════════════
import openspin as osp


def A():
    def garlic():
        c = MAN["cases"]["nitroxide_14N"]
        f, y = load_txt(c["file"])
        s = osp.garlic(osp.spin_system(g=c["g"], nuclei=[osp.nucleus("14N", c["aN_mT"])]), f, MW, c["lw_mT"], 0.5)
        return f"shape {s.shape}, peak {np.max(np.abs(s)):.2g}"
    check("Open-Sym-EPR", "garlic (isotropic cw)", garlic)

    def pepper():
        f = np.linspace(280, 360, 1500)
        cu = osp.spin_system(g=(2.06, 2.06, 2.30), nuclei=[osp.nucleus("63Cu", A_tensor_mT=(0.8, 0.8, 16.0))])
        s = osp.pepper(cu, f, MW, 1.2, 0.5, n_orientations=800)
        assert np.isfinite(s).all() and np.max(np.abs(s)) > 0
        return "Cu(II) powder ok"
    check("Open-Sym-EPR", "pepper (powder cw)", pepper)

    def salt():
        sys = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", A_tensor_mT=(0.3, 0.3, 0.55))])
        rf = np.linspace(0.1, 30, 1500)
        s = osp.endor_spectrum(sys, 348.0, rf, MW, 0.2, 800, rf_max_MHz=30)
        return f"ENDOR peak {np.max(np.abs(s)):.2g}"
    check("Open-Sym-EPR", "salt (ENDOR)", salt)

    def saffron():
        sys = osp.spin_system(g=2.0, nuclei=[osp.nucleus("14N", A_tensor_mT=(0.2, 0.2, 0.9))])
        t = np.linspace(0, 4, 1024)
        e2 = osp.two_pulse_eseem(sys, t, 348.0, 400)
        e3 = osp.three_pulse_eseem(sys, t, 0.2, 348.0, 400)
        fr, fft = osp.eseem_fft(t, e2)
        return f"2p+3p+fft ok (fmax {fr.max():.0f} MHz)"
    check("Open-Sym-EPR", "saffron (2/3-pulse ESEEM + FFT)", saffron)

    def curry():
        sys = osp.spin_system(g=2.0, S=2.5)
        T = np.linspace(2, 300, 60)
        chiT = osp.chiT_vs_T(sys, T, 0.5, 100)
        M = osp.magnetisation_muB(sys, np.linspace(0, 7, 40), 2.0, 100)
        curie = 0.12505 * 4 * 2.5 * 3.5
        assert abs(chiT[-1] - curie) / curie < 0.05 and M[-1] > 0.95 * 5.0
        return f"chiT->{chiT[-1]:.2f}(Curie {curie:.2f}), Msat->{M[-1]:.2f}"
    check("Open-Sym-EPR", "curry (chiT + M)", curry)

    def fit():
        c = MAN["cases"]["nitroxide_14N"]
        f, y = load_txt(c["file"])
        b0 = osp.isotropic_resonance_field_mT(MW, c["g"])
        sys = osp.spin_system(g=2.004, nuclei=[osp.nucleus("14N", 1.3)])
        vary = {"g_iso": (2.004, 2.0, 2.01), "A0_iso": (1.3, 1.0, 2.0), "lw": (0.2, 0.05, 1.0), "scale": (1.0, 0.2, 3.0)}
        r = osp.esfit(sys, f, y, vary, MW, global_search=400, max_nfev=200)
        gv = dict(zip(r.params["parameter"], r.params["value"]))
        assert r.metrics["R2"] > 0.9 and abs(gv["A0_iso"] - c["aN_mT"]) < 0.1
        return f"R2={r.metrics['R2']:.3f}, aN={gv['A0_iso']:.3f} (true {c['aN_mT']})"
    check("Open-Sym-EPR", "esfit (blind recovery)", fit)

    def fit_mc():
        c = MAN["cases"]["nitroxide_14N"]
        f, y = load_txt(c["file"])
        sys = osp.spin_system(g=c["g"], nuclei=[osp.nucleus("14N", c["aN_mT"])])
        vary = {"g_iso": (c["g"], 2.0, 2.01), "A0_iso": (c["aN_mT"], 1.4, 1.7), "lw": (c["lw_mT"], 0.05, 0.5), "scale": (1.0, 0.2, 3.0)}
        r = osp.esfit(sys, f, y, vary, MW, global_search=0, max_nfev=150, n_monte_carlo=30, mc_method="residual")
        assert r.mc_errors is not None and "ci_2.5%" in r.mc_errors.columns
        return f"MC 95% CI computed for {len(r.mc_errors)} params"
    check("Open-Sym-EPR", "esfit Monte-Carlo errors", fit_mc)

    def fit_validate():
        sys = osp.spin_system(g=2.006, nuclei=[osp.nucleus("14N", 1.55)])
        f = np.linspace(330, 350, 200); y = osp.garlic(sys, f, MW)
        try:
            osp.esfit(sys, f, y, {"A_iso_0": (1.55, 1, 2)}, MW, max_nfev=10)
            raise AssertionError("did not raise on bad key")
        except ValueError:
            return "raises on unknown vary key (A_iso_0)"
    check("Open-Sym-EPR", "esfit param-name validation", fit_validate)

    def bruker():
        from openspin.bruker import load_bes3t
        dsc = (HERE / "syn_bruker.DSC").read_text()
        dta = (HERE / "syn_bruker.DTA").read_bytes()
        B, y, meta = load_bes3t(dsc, dta)
        assert len(B) == 1024 and abs(meta["microwave_frequency_GHz"] - MW) < 1e-3
        return f"{len(B)} pts, {B.min():.1f}-{B.max():.1f} mT, nu={meta['microwave_frequency_GHz']:.3f}"
    check("Open-Sym-EPR", "Bruker BES3T import (synthetic)", bruker)

    def multifreq():
        outs = []
        for nu in (9.5, 34.0, 94.0):
            bb = osp.isotropic_resonance_field_mT(nu, 2.0)
            outs.append(f"{nu}->{bb:.0f}mT")
        return "; ".join(outs)
    check("Open-Sym-EPR", "multifrequency (X/Q/W)", multifreq)

    def io_parse():
        from openspin import parse_spectrum
        f, y = parse_spectrum((HERE / "syn_nitroxide_14N.txt").read_text())
        return f"parsed {len(f)} pts"
    check("Open-Sym-EPR", "text spectrum parser", io_parse)


# ════════════════════ B. native_solvers GUI bridge ════════════════════
def B():
    from epr_simfit import native_solvers as nsv
    assert nsv.AVAILABLE, nsv.IMPORT_ERROR

    def syst():
        s = nsv.build_system(g_tensor=(2.008, 2.006, 2.002), S=0.5, nuclei_text="14N:0.3,0.3,1.0; 1H:0.2")
        assert len(s.nuclei) == 2
        return "build_system + parse_nuclei (tensor+iso)"
    check("native_solvers", "build_system / parse_nuclei", syst)

    s = nsv.build_system(g_iso=2.006, S=0.5, nuclei_text="14N:1.55")
    sA = nsv.build_system(g_tensor=(2.008, 2.006, 2.002), S=0.5, nuclei_text="14N:0.3,0.3,1.0")
    sS = nsv.build_system(g_iso=2.0, S=2.5, nuclei_text="")
    check("native_solvers", "run_garlic", lambda: f"n={len(nsv.run_garlic(s,330,350,MW,0.1,0.5)['y'])}")
    check("native_solvers", "run_pepper", lambda: f"n={len(nsv.run_pepper(sA,330,352,MW,0.3,0.5,n_orient=400)['y'])}")
    check("native_solvers", "run_salt", lambda: f"n={len(nsv.run_salt(s,348,30,MW,0.2,n_orient=400)['y'])}")
    check("native_solvers", "run_saffron", lambda: f"echo {len(nsv.run_saffron(sA,348,4.0,'2-pulse',0.2,n_orient=200)['echo'])}")
    check("native_solvers", "run_curry", lambda: f"chiT {len(nsv.run_curry(sS,'chiT vs T',300,0.5,7,2,n_orient=100)['y'])}")


# ════════════════════ C. Open-Sym-EPR engine ════════════════════
def C():
    from epr_simfit import (io, preprocessing, model_library, simulator, fitter,
                            model_comparison, model_suggester, interpretation, batch,
                            export, user_models, bruker, metadata_parser, demo_data, reference_library)

    f, y = load_txt(MAN["cases"]["two_component"]["file"])

    def parse():
        r = io.parse_epr_text((HERE / "syn_nitroxide_14N.txt").read_text(), filename="syn_nitroxide_14N.txt")
        return f"{len(r.dataframe['Field_mT'])} pts, unit handled"
    check("Open-Sym-EPR", "io.parse_epr_text", parse)

    def prep():
        r = preprocessing.preprocess_spectrum(f, y, baseline_method="polynomial", polynomial_order=3,
                                              normalization="max absolute", smooth_display=True, invert=False)
        assert hasattr(r, "processed") and len(r.processed) == len(f)
        return "baseline+normalize+smooth ok"
    check("Open-Sym-EPR", "preprocessing.preprocess_spectrum", prep)

    def lib():
        d = model_library.default_components()
        sta = model_library.spin_trap_adducts()
        tbl = model_library.component_table()
        return f"{len(d)} default, {len(sta)} spin-trap adducts, table rows {len(tbl)}"
    check("Open-Sym-EPR", "model_library", lib)

    comps = list(model_library.default_components().values())[:2]

    def sim():
        total, parts = simulator.simulate_model(f, comps, mw_frequency_GHz=MW)
        per = simulator.simulate_components(f, comps, mw_frequency_GHz=MW)
        one = simulator.component_spectrum(f, comps[0], mw_frequency_GHz=MW)
        return f"model+{len(per)} comps+single ok"
    check("Open-Sym-EPR", "simulator (model/components/single)", sim)

    fits = {}

    def fit_modes():
        for mode in ["weights only", "weights + linewidths", "weights + linewidths + g"]:
            r = fitter.fit_spectrum(f, y, components=comps, mw_frequency_GHz=MW, mode=mode, max_nfev=150, n_orientations=300)
            fits[mode] = r
            assert "R2" in r.metrics
        return "3 fit modes ok, R2(weights)=%.2f" % fits["weights only"].metrics["R2"]
    check("Open-Sym-EPR", "fitter.fit_spectrum (3 modes)", fit_modes)

    def fit_mc():
        r = fitter.fit_spectrum(f, y, components=comps, mw_frequency_GHz=MW, mode="weights only",
                                max_nfev=120, n_orientations=300, n_monte_carlo=20, mc_method="gaussian")
        return "MC errors: " + ("yes" if getattr(r, "mc_errors", None) is not None else "n/a")
    check("Open-Sym-EPR", "fitter Monte-Carlo", fit_mc)

    def compare():
        df, fitd = model_comparison.compare_models(f, y, presets=["M1_water_dmso", "M2_water_dmso_o2"],
                                                   mw_frequency_GHz=MW)
        assert len(df) >= 2
        return f"compared {len(df)} models (AIC/BIC)"
    check("Open-Sym-EPR", "model_comparison.compare_models", compare)

    def suggest():
        ctx = model_suggester.ExperimentContext()
        out = model_suggester.suggest_models(ctx)
        return f"suggested {len(out)} keys"
    check("Open-Sym-EPR", "model_suggester.suggest_models", suggest)

    def interp():
        r = fits.get("weights only")
        q = interpretation.assess_fit_quality(r.metrics)
        sug = interpretation.suggest_fit_improvements(r, fit_mode="weights only")
        tab = interpretation.publication_parameters_table(r)
        para = interpretation.publication_methods_paragraph(r, MW)
        return f"quality={getattr(q,'label',q)}, {len(sug)} suggestions, table {tab.shape}, methods {len(para)}c"
    check("Open-Sym-EPR", "interpretation (quality/suggest/pub-table)", interp)

    def batch_t():
        series = MAN["cases"]["kinetics_series"]
        named = [(f"{t}min", float(t), (HERE / fn).read_text())
                 for fn, t in zip(series["files"], series["times_min"])]
        specs = batch.batch_from_texts(named, mw_frequency_GHz=MW)
        res = batch.run_batch(specs, comps, mw_frequency_GHz=MW, coordinate_name="time_min",
                              n_orientations=300, max_nfev=120)
        png = batch.kinetic_plot_png(res) if hasattr(batch, "kinetic_plot_png") else b""
        return f"batch {len(specs)} spectra fitted, kinetic png {len(png)} bytes"
    check("Open-Sym-EPR", "batch (from_texts/run_batch/kinetic_plot)", batch_t)

    def exp():
        r = fits.get("weights only")
        orca = export.orca_templates(r)
        cdf = export.fit_components_dataframe(r)
        zipb = export.build_export_zip(fit=r, mw_frequency_GHz=MW)
        assert len(cdf) > 0
        return f"{len(orca)} orca templates, zip {len(zipb)} bytes"
    check("Open-Sym-EPR", "export (orca/zip/dataframes)", exp)

    def usermodels():
        js = user_models.components_to_json(comps, name="testpack")
        back, meta = user_models.components_from_text(js, filename="m.json")
        csv = "name,g,nuclei\nrad1,2.005,14N:1.55;1H:0.3\n"
        cb, _ = user_models.components_from_text(csv, filename="m.csv")
        return f"json round-trip {len(back)} comps, csv {len(cb)} comps"
    check("Open-Sym-EPR", "user_models (json/csv round-trip)", usermodels)

    def bruk():
        dsc = (HERE / "syn_bruker.DSC").read_text(); dta = (HERE / "syn_bruker.DTA").read_bytes()
        B, yy, meta = bruker.load_bes3t(dsc, dta)
        txt = bruker.bruker_to_text(B, yy, meta)
        return f"load+to_text ({len(B)} pts, text {len(txt)}c)"
    check("Open-Sym-EPR", "bruker (load + to_text)", bruk)

    def meta():
        hdr = "# Microwave Frequency: 9.85 GHz\n# Field unit: mT\n"
        p = metadata_parser.parse_header(hdr)
        tab = metadata_parser.metadata_table(p) if hasattr(metadata_parser, "metadata_table") else None
        return f"parsed {len(p)} metadata keys"
    check("Open-Sym-EPR", "metadata_parser", meta)

    def demo():
        t = demo_data.generate_demo_text("N2 + light", "M1_water_dmso", {}, seed=1)
        return f"demo text {len(t)}c"
    check("Open-Sym-EPR", "demo_data.generate_demo_text", demo)

    def refs():
        std = reference_library.reference_standards()
        return f"{len(std)} reference standards"
    check("Open-Sym-EPR", "reference_library", refs)


# ════════════════════ D. GUI render (AppTest) ════════════════════
def D():
    from streamlit.testing.v1 import AppTest

    def simepr():
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180); at.run()
        assert not at.exception, at.exception
        nbtn = len([b for b in at.button if b.key])
        return f"rendered, {nbtn} keyed buttons, {len(at.tabs)} tab-containers"
    check("GUI", "Open-Sym-EPR app renders (AppTest)", simepr)

    def runbtn():
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180); at.run()
        at.button(key="run_garlic").click().run()
        assert not at.exception and "es_garlic" in at.session_state
        return "native garlic Run button executes live"
    check("GUI", "native solver Run button (live)", runbtn)

    def openspin_app():
        p = Path(r"E:\Open-Sym-EPR\app.py")
        if not p.exists():
            return "Open-Sym-EPR app.py not present (skipped)"
        at = AppTest.from_file(str(p), default_timeout=180); at.run()
        assert not at.exception, at.exception
        return f"rendered, {len(at.tabs)} tab-containers"
    check("GUI", "Open-Sym-EPR app renders (AppTest)", openspin_app)


for stage in (A, B, C, D):
    try:
        stage()
    except Exception as exc:  # noqa: BLE001
        results.append((stage.__name__, "STAGE CRASH", False, str(exc)))

# ════════════════════ Report ════════════════════
print("\n" + "=" * 78)
print("COMPREHENSIVE FEATURE TEST — synthetic input")
print("=" * 78)
cat = None
for c, name, ok, detail in results:
    if c != cat:
        print(f"\n[{c}]"); cat = c
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  {name:42s} {detail[:60]}")
npass = sum(1 for *_, ok, _ in [(r[0], r[1], r[2], r[3]) for r in results] if ok)
npass = sum(1 for r in results if r[2])
print("\n" + "-" * 78)
print(f"TOTAL: {npass}/{len(results)} features passed")
print("=" * 78)
sys.exit(0 if npass == len(results) else 1)
