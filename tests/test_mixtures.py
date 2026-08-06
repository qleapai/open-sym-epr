"""Tests for the spin-adduct mixture (ratio) capability."""
import numpy as np

from epr_simfit import model_library as ml
from epr_simfit import mixtures


def _pbn_pair():
    lib = ml.default_components()
    return [lib["pbn_oh"].clone(), lib["pbn_ch3_dmso"].clone()]


def test_normalise_ratios_sums_to_one():
    n = mixtures.normalise_ratios({"a": 3.0, "b": 1.0, "c": 0.0})
    assert abs(sum(n.values()) - 1.0) < 1e-12
    assert abs(n["a"] - 0.75) < 1e-12
    # negative inputs are clamped, never negative fractions
    assert all(v >= 0.0 for v in mixtures.normalise_ratios({"a": -5.0, "b": 2.0}).values())


def test_simulate_mixture_composite_is_weighted_sum():
    field = np.linspace(344.0, 356.0, 800)
    comps = _pbn_pair()
    ratios = {"pbn_oh": 70.0, "pbn_ch3_dmso": 30.0}
    total, weighted = mixtures.simulate_mixture(field, comps, ratios=ratios,
                                                mw_frequency_GHz=9.82, n_orientations=1)
    assert total.shape == field.shape
    stacked = sum(weighted.values())
    assert np.allclose(total, stacked, atol=1e-8)


def test_fit_mixture_recovers_known_ratio():
    """Synthetic 60:40 PBN-OH:PBN-CH3 mixture -> recovered ratios close to truth."""
    field = np.linspace(344.0, 356.0, 1000)
    comps = _pbn_pair()
    truth = {"pbn_oh": 60.0, "pbn_ch3_dmso": 40.0}
    total, _ = mixtures.simulate_mixture(field, comps, ratios=truth,
                                         mw_frequency_GHz=9.82, n_orientations=1)
    total = total / (np.max(np.abs(total)) or 1.0)
    res = mixtures.fit_mixture_ratios(field, total, _pbn_pair(), mw_frequency_GHz=9.82,
                                      mode="weights only", max_nfev=300, n_orientations=1,
                                      seeded=False)
    assert res["R2"] > 0.95
    assert abs(res["percent"]["pbn_oh"] - 60.0) < 8.0
    assert abs(res["percent"]["pbn_ch3_dmso"] - 40.0) < 8.0


def test_hyperfine_mode_reports_avalues():
    field = np.linspace(344.0, 356.0, 900)
    comps = _pbn_pair()
    total, _ = mixtures.simulate_mixture(field, comps, mw_frequency_GHz=9.82, n_orientations=1)
    total = total / (np.max(np.abs(total)) or 1.0)
    res = mixtures.fit_mixture_ratios(field, total, _pbn_pair(), mw_frequency_GHz=9.82,
                                      mode="weights + g + hyperfine", max_nfev=300,
                                      n_orientations=1)
    assert res["hyperfine_G"]                       # a-values reported
    for avals in res["hyperfine_G"].values():
        assert all(0.0 < a < 30.0 for a in avals)   # physical Gauss range
