"""Tests for the PBN spin-trap adduct models in Open-Sym-EPR."""
import numpy as np
import openspin as osp
from openspin import spin_trap_models as stm


def test_pbn_photocatalysis_fractions_sum_to_one():
    models = stm.pbn_photocatalysis_adducts()
    assert len(models) == 6
    assert abs(sum(m.fraction for m in models) - 1.0) < 1e-6


def test_raw_and_systems_consistent():
    raw = stm.pbn_photocatalysis_raw()
    models = stm.pbn_photocatalysis_adducts()
    assert len(raw) == len(models) == 6
    # PBN-OH is the well-established one; check g and that it builds a valid system
    oh = next(m for m in raw if m["key"] == "pbn_oh")
    assert abs(oh["g"] - 2.0055) < 1e-4
    assert oh["fraction"] == 0.163


def test_composite_simulates():
    field = np.linspace(330, 350, 1000)
    total = np.zeros_like(field)
    for m in stm.pbn_photocatalysis_adducts():
        total += m.fraction * osp.garlic(m.system, field, 9.5, 0.1, 0.5)
    assert np.isfinite(total).all() and np.max(np.abs(total)) > 0
