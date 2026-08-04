"""Tests for the native (MATLAB-free) native GUI solver bridge.

These guard the live native solver panels: every solver must run
through ``epr_simfit.native_solvers`` and return physically sane arrays, with no
MATLAB anywhere in the path.
"""

import numpy as np
import pytest

from epr_simfit import native_solvers as nsv

pytestmark = pytest.mark.skipif(not nsv.AVAILABLE,
                                reason=f"Open-Sym-EPR engine not importable: {nsv.IMPORT_ERROR}")


def test_catalog_maps_five_solvers():
    cat = nsv.native_catalog()
    assert set(cat) == {"garlic", "pepper", "salt", "saffron", "curry"}
    # every entry names the native engine it runs on
    assert all(c.engine.startswith("openspin") for c in cat.values())


def test_parse_nuclei_isotropic_and_tensor():
    nuclei = nsv.parse_nuclei("14N:1.55; 1H:0.30")
    assert len(nuclei) == 2
    tens = nsv.parse_nuclei("14N:0.3,0.3,1.0")
    assert len(tens) == 1
    Ax, Ay, Az = tens[0].A_principal_MHz
    assert Az > Ax  # anisotropy preserved (z larger than x)


def test_build_system_iso_and_tensor():
    s_iso = nsv.build_system(g_iso=2.006, S=0.5, nuclei_text="14N:1.55")
    assert abs(s_iso.g_principal[0] - 2.006) < 1e-9
    s_ten = nsv.build_system(g_tensor=(2.30, 2.07, 2.05), S=0.5, nuclei_text="")
    assert s_ten.g_principal[0] > s_ten.g_principal[2]


def test_garlic_runs_and_normalises():
    s = nsv.build_system(g_iso=2.006, S=0.5, nuclei_text="14N:1.55")
    r = nsv.run_garlic(s, 330, 350, 9.5, 0.1, 0.5)
    assert r["y"].shape == r["x"].shape
    assert np.isfinite(r["y"]).all()
    assert np.max(np.abs(r["y"])) > 0


def test_pepper_powder_runs():
    s = nsv.build_system(g_tensor=(2.008, 2.006, 2.002), S=0.5, nuclei_text="14N:0.17,0.17,1.0")
    r = nsv.run_pepper(s, 330, 352, 9.5, 0.3, 0.5, n_orient=600)
    assert np.isfinite(r["y"]).all() and np.max(np.abs(r["y"])) > 0


def test_salt_endor_runs():
    s = nsv.build_system(g_iso=2.0, S=0.5, nuclei_text="1H:0.30")
    r = nsv.run_salt(s, 348.0, 30.0, 9.5, 0.2, n_orient=400)
    assert r["x"][0] >= 0 and np.max(np.abs(r["y"])) > 0


def test_saffron_eseem_time_and_frequency():
    s = nsv.build_system(g_iso=2.0, S=0.5, nuclei_text="14N:0.3,0.3,1.0")
    r = nsv.run_saffron(s, 348.0, 4.0, "2-pulse", 0.2, n_orient=200)
    assert r["echo"].shape[0] > 0
    assert r["freq"].max() <= 30.0 + 1e-6


def test_curry_chiT_high_T_approaches_curie():
    g, S = 2.0, 2.5
    s = nsv.build_system(g_iso=g, S=S, nuclei_text="")
    r = nsv.run_curry(s, "chiT vs T", 300, 0.5, 7, 2, n_orient=100)
    curie = 0.12505 * g**2 * S * (S + 1)
    # high-temperature value should be within a few percent of the Curie constant
    assert abs(r["y"][-1] - curie) / curie < 0.05


def test_curry_magnetisation_saturates_to_gS():
    g, S = 2.0, 2.5
    s = nsv.build_system(g_iso=g, S=S, nuclei_text="")
    r = nsv.run_curry(s, "M vs B", 300, 0.5, 7, 2, n_orient=100)
    # at 7 T / 2 K the magnetisation should be near the g*S saturation value
    assert r["y"][-1] > 0.95 * g * S
