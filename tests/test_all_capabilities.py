"""Comprehensive capability tests: every solver and feature against independent
analytical predictions. Mirrors the validation reported in the manuscript."""

import numpy as np
import pytest

import openspin as osp
from openspin.spin_operators import spin_operators
from openspin.spin_hamiltonian import (SpinSystem, NuclearSpec, BMAGN_MHZ_PER_T,
                                        NMAGN_MHZ_PER_T, NUCLEAR_G_FACTORS,
                                        build_field_independent, build_zeeman_direction)
from openspin.constants import MU_B_OVER_H_GHZ_PER_T
from openspin.powder import resonance_fields_one_orientation
from scipy.signal import find_peaks

NU = 9.5
Z = np.array([0, 0, 1.0]); X = np.array([1.0, 0, 0])
def Bf(g, f=NU): return f * 1000 / (g * MU_B_OVER_H_GHZ_PER_T)
def AMHz(A, g=2.0): return A * g * MU_B_OVER_H_GHZ_PER_T


# ── spin algebra ──
@pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_commutator(S):
    o = spin_operators(S)
    assert np.allclose(o["x"] @ o["y"] - o["y"] @ o["x"], 1j * o["z"], atol=1e-10)


@pytest.mark.parametrize("S", [0.5, 1.0, 2.5])
def test_s_squared(S):
    o = spin_operators(S)
    S2 = o["x"] @ o["x"] + o["y"] @ o["y"] + o["z"] @ o["z"]
    assert abs(float(np.real(S2[0, 0])) - S * (S + 1)) < 1e-9


def test_hamiltonian_hermitian():
    s = SpinSystem(S=2.5, g_principal=(2.0, 2.05, 2.1), D_MHz=300, E_MHz=50,
                   nuclei=[NuclearSpec(I=2.5, A_principal_MHz=(200, 220, 250))])
    H = build_field_independent(s) + 0.35 * build_zeeman_direction(s, Z)
    assert np.max(np.abs(H - H.conj().T)) < 1e-9


# ── garlic / pepper line counts and positions ──
@pytest.mark.parametrize("iso,I,A", [("1H", 0.5, 1.0), ("14N", 1.0, 1.5), ("63Cu", 1.5, 5.0),
                                     ("55Mn", 2.5, 8.0), ("51V", 3.5, 6.0)])
def test_multiplet_count_and_spacing(iso, I, A):
    s = SpinSystem(S=0.5, g_principal=(2, 2, 2), nuclei=[NuclearSpec(I=I, A_principal_MHz=(AMHz(A),) * 3)])
    b, _ = resonance_fields_one_orientation(s, Z, NU * 1000, Bf(2) - (2 * I * A + 4), Bf(2) + (2 * I * A + 4))
    lines = np.unique(np.round(np.sort(b), 4))
    assert len(lines) == int(2 * I + 1)
    assert abs(float(np.mean(np.diff(lines))) - A) < 0.02


def test_axial_powder_fields():
    s = SpinSystem(S=0.5, g_principal=(2.00, 2.00, 2.30))
    bz, _ = resonance_fields_one_orientation(s, Z, NU * 1000, 280, 360)
    bx, _ = resonance_fields_one_orientation(s, X, NU * 1000, 280, 360)
    assert abs(float(bz[0]) - Bf(2.30)) < 1e-3
    assert abs(float(bx[0]) - Bf(2.00)) < 1e-3


def test_large_linewidth_no_crash():
    """Regression: large linewidth relative to a narrow field window must not crash."""
    s = osp.spin_system(g=2.0, nuclei=[osp.nucleus("14N", 1.5)])
    field = np.linspace(338, 341, 400)  # narrow window
    spec = osp.pepper(s, field, NU, linewidth_mT=1.5, n_orientations=200)  # huge lw
    assert spec.shape == field.shape and np.all(np.isfinite(spec))


# ── ENDOR weak and strong coupling ──
@pytest.mark.parametrize("A,regime", [(6.0, "weak"), (40.0, "strong")])
def test_endor_regimes(A, regime):
    nuH = NUCLEAR_G_FACTORS["1H"] * NMAGN_MHZ_PER_T * 350.0 / 1000
    s = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", A / (2 * MU_B_OVER_H_GHZ_PER_T))])
    rf = np.linspace(0.5, 60, 5000)
    spec = osp.endor_spectrum(s, 350.0, rf, linewidth_MHz=0.25, n_orientations=150, rf_max_MHz=70)
    fr = np.sort(rf[find_peaks(spec, height=0.3)[0]])
    if regime == "weak":
        assert abs(fr[0] - (nuH - A / 2)) < 0.06 and abs(fr[-1] - (nuH + A / 2)) < 0.06
    else:
        assert abs(fr[0] - (A / 2 - nuH)) < 0.12 and abs(fr[-1] - (A / 2 + nuH)) < 0.12


# ── ESEEM ──
def test_eseem_isotropic_null():
    s = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", 0.5)])
    t = np.linspace(0, 4, 1024)
    assert np.ptp(osp.two_pulse_eseem(s, t, 350.0, n_orientations=200)) < 1e-5


def test_eseem_anisotropic_modulation():
    s = osp.spin_system(g=2.0, nuclei=[osp.nucleus("1H", A_tensor_mT=(-0.1, -0.1, 0.2))])
    t = np.linspace(0, 4, 2048)
    assert np.ptp(osp.two_pulse_eseem(s, t, 350.0, n_orientations=400)) > 1e-4


# ── magnetometry ──
@pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_curie_and_saturation(S):
    chiT = osp.chiT_vs_T(osp.spin_system(S=S, g=2.0), np.array([300.0]), 0.1, 80)
    assert abs(float(chiT[0]) - 0.12505 * 4 * S * (S + 1)) < 5e-3
    M = osp.magnetisation_muB(osp.spin_system(S=S, g=2.0), np.array([7.0]), 0.5, 80)
    assert abs(float(M[0]) - 2 * S) < 0.03


def test_brillouin_s_half():
    g, T = 2.0, 4.0
    B = np.array([0.5, 1.0, 2.0, 4.0])
    Mc = osp.magnetisation_muB(osp.spin_system(S=0.5, g=g), B, T, 60)
    x = g * (BMAGN_MHZ_PER_T * 1e6 * 6.62607015e-27) * B / (2 * 1.380649e-16 * T)
    assert np.max(np.abs(Mc - np.tanh(x))) < 0.02


# ── multifrequency ──
@pytest.mark.parametrize("f", [9.5, 34.0, 94.0])
def test_multifrequency(f):
    b, _ = resonance_fields_one_orientation(SpinSystem(S=0.5, g_principal=(2, 2, 2)), Z, f * 1000, Bf(2, f) - 10, Bf(2, f) + 10)
    assert abs(float(b[0]) - Bf(2, f)) < 1e-3


# ── fitting + uncertainty coverage ──
def test_fit_recovery_with_matched_conversion():
    G0, A0 = 2.0, 1.55
    true = osp.spin_system(g=G0, nuclei=[osp.nucleus("14N", A0, g_for_conversion=G0)])
    b0 = osp.isotropic_resonance_field_mT(NU, G0)
    field = np.linspace(b0 - 6, b0 + 6, 1200)
    exp = osp.garlic(true, field, NU, 0.12) + np.random.default_rng(7).normal(0, 0.02, 1200)
    vary = {"g_iso": (1.998, 1.99, 2.01), "A0_iso": (1.2, 1.0, 2.0), "lw": (0.2, 0.05, 1.0), "scale": (1.0, 0.5, 2.0)}
    fit = osp.esfit(true, field, exp, vary, NU, max_nfev=200, global_search=2000)
    v = dict(zip(fit.params["parameter"], fit.params["value"]))
    assert abs(v["g_iso"] - G0) < 5e-4
    assert abs(v["A0_iso"] - A0) < 0.05
    assert fit.metrics["R2"] > 0.95
