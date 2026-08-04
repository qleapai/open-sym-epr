"""Validation tests for Open-Sym-EPR native solvers."""

import numpy as np
import pytest

import openspin as osp
from openspin.constants import MU_B_OVER_H_GHZ_PER_T, NMAGN_MHZ_PER_T
from openspin.spin_hamiltonian import NUCLEAR_G_FACTORS


# ── cw-EPR ────────────────────────────────────────────────────────────────────

def test_garlic_isotropic_center_field():
    sys = osp.spin_system(g=2.00)
    field = np.linspace(330, 350, 4000)
    spec = osp.garlic(sys, field, 9.5, linewidth_mT=0.1)
    expected = 9.5 * 1000 / (2.00 * MU_B_OVER_H_GHZ_PER_T)
    # derivative zero-crossing near the centre = max-slope point
    centre = field[np.argmin(np.abs(spec - 0)[np.argmax(spec):np.argmin(spec)]) + np.argmax(spec)]
    assert abs(centre - expected) < 0.2


def test_garlic_14N_triplet_three_lines():
    sys = osp.spin_system(g=2.006, nuclei=[osp.nucleus("14N", 1.5)])
    b0 = osp.isotropic_resonance_field_mT(9.5, 2.006)
    field = np.linspace(b0 - 6, b0 + 6, 6000)
    spec = osp.garlic(sys, field, 9.5, linewidth_mT=0.05)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(spec, height=0.2)
    assert len(pk) == 3  # 14N triplet: three positive derivative lobes


def test_pepper_axial_g_nonzero_normalised():
    sys = osp.spin_system(g=(2.00, 2.00, 2.30))
    field = np.linspace(290, 345, 1500)
    spec = osp.pepper(sys, field, 9.5, linewidth_mT=0.6, n_orientations=2000)
    assert np.all(np.isfinite(spec))
    assert abs(np.max(np.abs(spec)) - 1.0) < 1e-6


def test_cw_auto_routes_correctly():
    iso = osp.spin_system(g=2.006, nuclei=[osp.nucleus("14N", 1.5)])
    aniso = osp.spin_system(g=(2.09, 2.06, 2.02))
    field = np.linspace(300, 360, 600)
    _, u1 = osp.cw_auto(iso, field, 9.5, n_orientations=200)
    _, u2 = osp.cw_auto(aniso, field, 9.5, n_orientations=200)
    assert u1 == "garlic"
    assert u2 == "pepper"


# ── ENDOR ─────────────────────────────────────────────────────────────────────

def test_endor_weak_coupling_lines():
    gnH = NUCLEAR_G_FACTORS["1H"]
    field = 350.0
    nuH = gnH * NMAGN_MHZ_PER_T * field / 1000.0
    A = 6.0
    sys = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", A / (2.0 * MU_B_OVER_H_GHZ_PER_T))])
    rf = np.linspace(1, 30, 2000)
    spec = osp.endor_spectrum(sys, field, rf, linewidth_MHz=0.3, n_orientations=150)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(spec, height=0.2)
    freqs = np.sort(rf[pk])
    assert len(freqs) == 2
    assert abs(freqs[0] - (nuH - A / 2)) < 0.5
    assert abs(freqs[1] - (nuH + A / 2)) < 0.5


# ── ESEEM ─────────────────────────────────────────────────────────────────────

def test_eseem_no_modulation_for_isotropic():
    # purely isotropic hyperfine -> pseudo-secular term = 0 -> no modulation
    sys = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", 0.5)])
    t = np.linspace(0, 4, 1024)
    V = osp.two_pulse_eseem(sys, t, 350.0, n_orientations=200)
    assert np.ptp(V) < 1e-6


def test_eseem_modulation_for_anisotropic():
    sys = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", A_tensor_mT=(-0.05, -0.05, 0.1))])
    t = np.linspace(0, 4, 2048)
    V = osp.two_pulse_eseem(sys, t, 350.0, n_orientations=300)
    assert np.ptp(V) > 1e-4  # measurable modulation


# ── Magnetometry ──────────────────────────────────────────────────────────────

def test_curie_law_s_half():
    sys = osp.spin_system(S=0.5, g=2.0)
    chiT = osp.chiT_vs_T(sys, np.array([300.0]), 0.1, n_orientations=50)
    assert abs(float(chiT[0]) - 0.375) < 0.005


def test_curie_law_s_five_halves():
    sys = osp.spin_system(S=2.5, g=2.0)
    chiT = osp.chiT_vs_T(sys, np.array([300.0]), 0.1, n_orientations=50)
    expected = 0.12505 * 4 * 2.5 * 3.5
    assert abs(float(chiT[0]) - expected) < 0.05


def test_magnetisation_saturation_s_half():
    sys = osp.spin_system(S=0.5, g=2.0)
    M = osp.magnetisation_muB(sys, np.array([7.0]), 1.0, n_orientations=50)
    assert abs(float(M[0]) - 1.0) < 0.05  # saturates to g*S = 1 muB


def test_builder_rejects_unknown_isotope():
    with pytest.raises(ValueError):
        osp.nucleus("99Z", 1.0)


# ── Fitting ───────────────────────────────────────────────────────────────────

def test_esfit_recovers_parameters_from_close_start():
    true = osp.spin_system(g=2.0060, nuclei=[osp.nucleus("14N", 1.55)])
    b0 = osp.isotropic_resonance_field_mT(9.5, 2.006)
    field = np.linspace(b0 - 6, b0 + 6, 1500)
    exp = osp.garlic(true, field, 9.5, linewidth_mT=0.12)
    exp = exp + np.random.default_rng(0).normal(0, 0.02, exp.size)
    start = osp.spin_system(g=2.0058, nuclei=[osp.nucleus("14N", 1.50)])
    vary = {"g_iso": (2.0058, 2.0, 2.01), "A0_iso": (1.5, 1.0, 2.0),
            "lw": (0.13, 0.05, 1.0), "scale": (1.0, 0.5, 2.0)}
    fit = osp.esfit(start, field, exp, vary, 9.5, max_nfev=200)
    assert fit.metrics["R2"] > 0.95
    vals = dict(zip(fit.params["parameter"], fit.params["value"]))
    assert abs(vals["g_iso"] - 2.0060) < 5e-4
    assert abs(vals["A0_iso"] - 1.55) < 0.05


def test_esfit_global_search_robust_from_poor_start():
    true = osp.spin_system(g=2.0060, nuclei=[osp.nucleus("14N", 1.55)])
    b0 = osp.isotropic_resonance_field_mT(9.5, 2.006)
    field = np.linspace(b0 - 6, b0 + 6, 1200)
    exp = osp.garlic(true, field, 9.5, linewidth_mT=0.12)
    exp = exp + np.random.default_rng(1).normal(0, 0.02, exp.size)
    start = osp.spin_system(g=2.0040, nuclei=[osp.nucleus("14N", 1.20)])
    vary = {"g_iso": (2.004, 2.0, 2.01), "A0_iso": (1.2, 1.0, 2.0),
            "lw": (0.2, 0.05, 1.0), "scale": (1.0, 0.5, 2.0)}
    fit = osp.esfit(start, field, exp, vary, 9.5, max_nfev=200, global_search=400)
    assert fit.metrics["R2"] > 0.95
    assert "std_error" in fit.params.columns


def test_esfit_reports_uncertainties():
    sys = osp.spin_system(g=2.006, nuclei=[osp.nucleus("14N", 1.55)])
    b0 = osp.isotropic_resonance_field_mT(9.5, 2.006)
    field = np.linspace(b0 - 6, b0 + 6, 1000)
    exp = osp.garlic(sys, field, 9.5, 0.12) + np.random.default_rng(2).normal(0, 0.01, 1000)
    vary = {"g_iso": (2.006, 2.0, 2.01), "scale": (1.0, 0.5, 2.0)}
    fit = osp.esfit(sys, field, exp, vary, 9.5, max_nfev=100)
    se = fit.params.set_index("parameter")["std_error"]
    assert np.isfinite(se["g_iso"])


def test_parse_spectrum_gauss_to_mT():
    text = "# header\n3400 0.1\n3401 0.5\n3402 -0.3\n"
    field, inten = osp.parse_spectrum(text)
    assert abs(field[0] - 340.0) < 1e-6  # Gauss -> mT
    assert len(inten) == 3


def test_esfit_monte_carlo_errors():
    sys = osp.spin_system(g=2.006, nuclei=[osp.nucleus("14N", 1.55)])
    b0 = osp.isotropic_resonance_field_mT(9.5, 2.006)
    field = np.linspace(b0 - 6, b0 + 6, 800)
    exp = osp.garlic(sys, field, 9.5, 0.12) + np.random.default_rng(3).normal(0, 0.03, 800)
    vary = {"g_iso": (2.006, 2.0, 2.01), "A0_iso": (1.55, 1.0, 2.0),
            "lw": (0.12, 0.05, 1.0), "scale": (1.0, 0.5, 2.0)}
    fit = osp.esfit(sys, field, exp, vary, 9.5, max_nfev=120, n_monte_carlo=25, mc_method="gaussian")
    assert fit.mc_errors is not None
    assert {"mc_std", "ci_2.5%", "ci_97.5%"}.issubset(fit.mc_errors.columns)
    assert (fit.mc_errors["mc_std"] >= 0).all()
    # CI must bracket the value
    for _, r in fit.mc_errors.iterrows():
        assert r["ci_2.5%"] <= r["value"] + 1e-6
        assert r["ci_97.5%"] >= r["value"] - 1e-6
