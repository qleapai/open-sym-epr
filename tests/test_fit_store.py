"""Tests for the global fit registry."""
import numpy as np

from epr_simfit import model_library as ml
from epr_simfit import fit_store


def _entry_args():
    B = np.linspace(344.0, 356.0, 200)
    comps = [ml.default_components()["pbn_oh"].clone()]
    return dict(field_mT=B, experimental=np.sin(B), fit_total=np.sin(B) * 0.9,
                components=comps, weights={"pbn_oh": 1.0}, mw_frequency_GHz=9.82,
                R2=0.97, source="Fit", n_parameters=3)


def test_add_and_get_roundtrip():
    state = {}
    key = fit_store.add(state, "myfit", **_entry_args())
    assert key == "myfit"
    e = fit_store.get(state, "myfit")
    assert e["R2"] == 0.97 and e["source"] == "Fit"
    assert fit_store.names(state) == ["myfit"]


def test_duplicate_names_are_disambiguated():
    state = {}
    a = fit_store.add(state, "run", **_entry_args())
    b = fit_store.add(state, "run", **_entry_args())
    assert a == "run" and b == "run (2)"
    assert len(fit_store.names(state)) == 2


def test_delete():
    state = {}
    fit_store.add(state, "run", **_entry_args())
    fit_store.delete(state, "run")
    assert fit_store.names(state) == []


def test_overlay_traces_interpolate_and_mask_out_of_range():
    state = {}
    fit_store.add(state, "run", **_entry_args())          # covers 344..356
    target = np.linspace(340.0, 360.0, 100)               # wider than the saved fit
    tr = fit_store.overlay_traces(state, ["run"], target)
    y = tr["[saved] run"]
    assert np.isnan(y[0]) and np.isnan(y[-1])             # outside range -> NaN
    assert np.isfinite(y[len(y) // 2])                    # inside range -> value


def test_export_csv_has_header_and_rows():
    state = {}
    fit_store.add(state, "run", **_entry_args())
    csv = fit_store.export_csv(state, "run")
    assert csv.splitlines()[0] == "Field_mT,experimental,fit,residual"
    assert len(csv.splitlines()) == 201                   # header + 200 points
