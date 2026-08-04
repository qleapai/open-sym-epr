"""Tests for the PBN photocatalysis model preset and project save/load."""
from epr_simfit import model_library as ml
from epr_simfit import project
from epr_simfit.user_models import components_to_json, components_from_json


def test_pbn_photocatalysis_model_fractions():
    comps = ml.pbn_photocatalysis_model()
    assert len(comps) == 6
    assert abs(sum(c.weight for c in comps) - 1.0) < 1e-6
    frac = {c.component_id: c.weight for c in comps}
    assert abs(frac["pbn_ch3_dmso"] - 0.526) < 1e-9
    assert abs(frac["pbn_oh"] - 0.163) < 1e-9


def test_pbn_preset_loadable():
    comps = ml.components_for_preset("M3_pbn_photocat_decomposition")
    assert len(comps) == 6
    assert "M3_pbn_photocat_decomposition" in ml.MODEL_DESCRIPTIONS


def test_project_round_trip():
    comps = ml.pbn_photocatalysis_model()
    cj = components_to_json(comps, name="proj")
    blob = project.save_project(microwave_frequency_GHz=9.816,
                                spectrum_text="Field_mT\tI\n344.5\t0.01\n344.6\t-0.02\n",
                                filename="d.asc", components_json=cj, fit_summary={"R2": 0.97})
    obj = project.load_project(blob)
    assert obj["schema"] == "Open-Sym-EPR.project"
    assert obj["microwave_frequency_GHz"] == 9.816
    back, _ = components_from_json(obj["components_json"])
    assert len(back) == 6
    assert "GHz" in project.project_summary(obj)


def test_comprehensive_spin_trap_library():
    new = ml.comprehensive_spin_trap_adducts()
    assert len(new) >= 18
    # covers PBN, PBN/DMSO, DMPO, POBN, DEPMPO
    cats = " ".join(c.category for c in new.values())
    for trap in ("PBN", "DMSO", "DMPO", "POBN", "DEPMPO"):
        assert trap in cats
    # every component is in the full library and every preset resolves
    lib = ml.default_components()
    for cid in new:
        assert cid in lib
    for p in ("ST_pbn_all_adducts", "ST_pbn_dmso_radicals", "ST_dmpo_all_adducts",
              "ST_pobn_adducts", "ST_depmpo_ros"):
        assert len(ml.components_for_preset(p)) >= 2
        assert p in ml.MODEL_DESCRIPTIONS


def test_depmpo_has_phosphorus():
    dep = ml.default_components()["depmpo_ooh"]
    assert any(n.isotope == "31P" for n in dep.nuclei)


def test_project_rejects_foreign_file():
    import pytest
    with pytest.raises(ValueError):
        project.load_project(b'{"schema": "something_else"}')
