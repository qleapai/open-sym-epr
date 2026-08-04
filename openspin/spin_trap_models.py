"""PBN/DMPO spin-trap adduct models as native Open-Sym-EPR spin systems.

Isotropic hyperfine values (mT) for solution cw-EPR. PBN-OH is a well-established
assignment; the N-centred and O2-derived entries are CANDIDATE assignments that
require ¹⁵N-labelling and blank/DMSO controls before any mechanistic claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from .spin_hamiltonian import SpinSystem
from . import spin_system as _mk_system, nucleus as _mk_nucleus


@dataclass
class AdductModel:
    key: str
    name: str
    fraction: float          # example model fraction (weight)
    system: SpinSystem
    note: str


# Raw definitions: key, display name, g, isotropic hyperfine [(isotope, A/mT)], fraction, note.
_PBN_PHOTOCAT_RAW = [
    ("pbn_ch3", "PBN-CH3 (DMSO-derived methyl)", 2.0056, [("14N", 1.55), ("1H", 0.28)], 0.526,
     "DMSO-derived methyl-radical PBN adduct; mandatory control when PBN is used in DMSO."),
    ("pbn_h", "PBN-H (hydrogen-atom adduct)", 2.0056, [("14N", 1.50), ("1H", 0.45)], 0.191,
     "Hydrogen-atom PBN adduct."),
    ("pbn_oh", "PBN-OH (hydroxyl / ROS)", 2.0055, [("14N", 1.53), ("1H", 0.27)], 0.163,
     "PBN hydroxyl-radical adduct (aN ≈ 15.3 G, aβH ≈ 2.7 G); well-established ROS assignment."),
    ("pbn_broad", "Broad PBN/surface-derived component", 2.0040, [], 0.072,
     "Broad, unresolved surface/defect-derived background (large linewidth)."),
    ("pbn_ncentred_candidate", "Unassigned N-centred candidate", 2.0055, [("14N", 1.45), ("1H", 0.25), ("14N", 0.18)], 0.038,
     "CANDIDATE N-centred adduct (second nitrogen). Requires 15N-labelling / blank controls."),
    ("pbn_o2_candidate", "Residual O2-derived candidate (PBN-OOH)", 2.0055, [("14N", 1.49), ("1H", 0.43)], 0.010,
     "CANDIDATE superoxide/hydroperoxyl PBN adduct; confirm with O2/Ar controls."),
]


def _sys(g, nuclei_mT):
    return _mk_system(g=g, nuclei=[_mk_nucleus(iso, a) for iso, a in nuclei_mT])


def pbn_photocatalysis_raw() -> list[dict]:
    """Raw, GUI-friendly adduct definitions (g and hyperfine in mT) with fractions."""
    return [{"key": k, "name": n, "g": g, "nuclei_mT": nuc, "fraction": f, "note": note}
            for (k, n, g, nuc, f, note) in _PBN_PHOTOCAT_RAW]


# Comprehensive, generalized spin-trapping library: trap, key, name, g, [(isotope, A/mT)], note.
# Literature isotropic approximations (solvent/pH dependent); refine by fitting.
_ALL_SPIN_TRAP_RAW = [
    # PBN adducts
    ("PBN", "pbn_oh", "PBN-OH (hydroxyl)", 2.0057, [("14N", 1.53), ("1H", 0.27)], "well-established ROS assignment"),
    ("PBN", "pbn_ooh", "PBN-OOH / O2•- (superoxide)", 2.0057, [("14N", 1.48), ("1H", 0.26)], "unstable superoxide adduct"),
    ("PBN", "pbn_carbon", "PBN-R (carbon-centred)", 2.0057, [("14N", 1.58), ("1H", 0.35)], "generic carbon radical"),
    ("PBN", "pbn_h", "PBN-H (hydrogen atom)", 2.0057, [("14N", 1.60), ("1H", 0.80)], "large aβH"),
    ("PBN", "pbn_alkoxyl", "PBN-OR (alkoxyl)", 2.0057, [("14N", 1.40), ("1H", 0.20)], "low aN"),
    ("PBN", "pbn_methoxyl", "PBN-OCH3 (methoxyl)", 2.0057, [("14N", 1.38), ("1H", 0.19)], ""),
    ("PBN", "pbn_peroxyl", "PBN-OOR (peroxyl)", 2.0057, [("14N", 1.36), ("1H", 0.18)], ""),
    ("PBN", "pbn_thiyl", "PBN-SR (thiyl)", 2.0058, [("14N", 1.48), ("1H", 0.33)], ""),
    ("PBN", "pbn_co2", "PBN-CO2•- (carbon dioxide anion)", 2.0057, [("14N", 1.58), ("1H", 0.44)], "CO2RR context"),
    ("PBN", "pbn_ox", "PBNOx (di-tert-alkyl nitroxide, degradation)", 2.0058, [("14N", 1.55)], "trap breakdown; clean triplet, no β-H"),
    # PBN + DMSO-derived radicals
    ("PBN/DMSO", "pbn_dmso_ch3", "PBN-CH3 (•CH3 from DMSO)", 2.0056, [("14N", 1.59), ("1H", 0.35)], "DMSO methyl; mandatory control"),
    ("PBN/DMSO", "pbn_dmso_sulfinylmethyl", "PBN-CH2S(O)CH3", 2.0056, [("14N", 1.56), ("1H", 0.33)], "DMSO-derived"),
    ("PBN/DMSO", "pbn_dmso_methylsulfinyl", "PBN-S(O)CH3 (methylsulfinyl)", 2.0057, [("14N", 1.46), ("1H", 0.21)], "DMSO-derived"),
    # DMPO adducts
    ("DMPO", "dmpo_oh", "DMPO-OH (1:2:2:1 quartet)", 2.0057, [("14N", 1.49), ("1H", 1.49)], "aN ≈ aH"),
    ("DMPO", "dmpo_ooh", "DMPO-OOH (superoxide)", 2.0058, [("14N", 1.43), ("1H", 1.17), ("1H", 0.13)], "γ-H distinguishes from OH"),
    ("DMPO", "dmpo_h", "DMPO-H (hydrogen atom)", 2.0057, [("14N", 1.66), ("1H", 2.25), ("1H", 2.25)], ""),
    ("DMPO", "dmpo_ch3", "DMPO-CH3 (methyl)", 2.0056, [("14N", 1.45), ("1H", 2.07)], "DMSO methyl control"),
    ("DMPO", "dmpo_carbon", "DMPO-R (carbon-centred)", 2.0057, [("14N", 1.60), ("1H", 2.25)], ""),
    ("DMPO", "dmpo_co2", "DMPO-CO2•-", 2.0057, [("14N", 1.58), ("1H", 1.87)], ""),
    ("DMPO", "dmpo_thiyl", "DMPO-SR (thiyl)", 2.0058, [("14N", 1.53), ("1H", 1.65)], ""),
    # POBN / DEPMPO
    ("POBN", "pobn_oh", "POBN-OH", 2.0057, [("14N", 1.53), ("1H", 0.17)], "water-soluble PBN analogue"),
    ("POBN", "pobn_carbon", "POBN-R (carbon)", 2.0057, [("14N", 1.55), ("1H", 0.25)], ""),
    ("DEPMPO", "depmpo_ooh", "DEPMPO-OOH (superoxide, 31P)", 2.0058, [("31P", 4.94), ("14N", 1.33), ("1H", 1.14)], "persistent; 31P distinguishes O2- from OH"),
    ("DEPMPO", "depmpo_oh", "DEPMPO-OH (hydroxyl, 31P)", 2.0058, [("31P", 4.73), ("14N", 1.40), ("1H", 1.32)], ""),
]


def all_spin_trap_raw() -> list[dict]:
    """Generalized spin-trapping library (PBN, PBN/DMSO, DMPO, POBN, DEPMPO) as raw
    GUI-friendly definitions (g and hyperfine in mT). Literature approximations."""
    return [{"trap": t, "key": k, "name": n, "g": g, "nuclei_mT": nuc, "note": note}
            for (t, k, n, g, nuc, note) in _ALL_SPIN_TRAP_RAW]


def all_spin_trap_systems() -> dict[str, SpinSystem]:
    """{display name: SpinSystem} for the full generalized spin-trap library."""
    return {m["name"]: _sys(m["g"], m["nuclei_mT"]) for m in all_spin_trap_raw()}


def pbn_photocatalysis_adducts() -> list[AdductModel]:
    """The example PBN water/DMSO photocatalysis decomposition (component + fraction).

    Fractions: PBN-CH3 52.6%, PBN-H 19.1%, PBN-OH 16.3%, broad surface-derived 7.2%,
    unassigned N-centred candidate 3.8%, residual O2-derived candidate 1.0%.
    """
    return [AdductModel(k, n, f, _sys(g, nuc), note) for (k, n, g, nuc, f, note) in _PBN_PHOTOCAT_RAW]


def pbn_photocatalysis_dict() -> dict[str, SpinSystem]:
    """Convenience: {display name: SpinSystem} for the PBN photocatalysis adducts."""
    return {m.name: m.system for m in pbn_photocatalysis_adducts()}
