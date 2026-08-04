"""Built-in PBN spin-trapping component library and model presets."""

from __future__ import annotations

from .spin_models import Nucleus, SpinComponent


def default_components() -> dict[str, SpinComponent]:
    comps = {
        "organic_radical_singlet": SpinComponent(
            component_id="organic_radical_singlet",
            display_name="Generic organic radical singlet",
            radical_assignment="organic radical",
            category="general organic",
            g=2.0023,
            g_bounds=(1.9980, 2.0080),
            nuclei=[],
            linewidth_mT=0.18,
            linewidth_bounds=(0.03, 2.00),
            eta=0.45,
            weight=0.5,
            interpretation="Single-line carbon/organic radical approximation.",
        ),
        "carbon_centered_h": SpinComponent(
            component_id="carbon_centered_h",
            display_name="Carbon-centered radical with beta-H",
            radical_assignment="C-centered radical",
            category="general organic",
            g=2.0026,
            g_bounds=(1.9990, 2.0080),
            nuclei=[Nucleus(isotope="1H", A_mT=1.80, label="beta H", bounds=(0.05, 4.00))],
            linewidth_mT=0.12,
            linewidth_bounds=(0.03, 1.00),
            eta=0.45,
            weight=0.3,
            interpretation="Generic beta-proton-coupled carbon radical.",
        ),
        "semiquinone_radical": SpinComponent(
            component_id="semiquinone_radical",
            display_name="Semiquinone/phenoxyl radical",
            radical_assignment="semiquinone or phenoxyl",
            category="organic/ROS",
            g=2.0046,
            g_bounds=(2.0020, 2.0075),
            nuclei=[Nucleus(isotope="1H", A_mT=0.22, label="ring H", bounds=(0.00, 0.80))],
            linewidth_mT=0.10,
            linewidth_bounds=(0.03, 0.80),
            eta=0.40,
            weight=0.3,
            interpretation="Generic oxygenated aromatic radical model.",
        ),
        "nitroxide_14n": SpinComponent(
            component_id="nitroxide_14n",
            display_name="Generic nitroxide 14N triplet",
            radical_assignment="nitroxide",
            category="nitroxide/spin label",
            g=2.0060,
            g_bounds=(2.0020, 2.0090),
            nuclei=[Nucleus(isotope="14N", A_mT=1.55, label="nitroxide N", bounds=(1.20, 2.00))],
            linewidth_mT=0.10,
            linewidth_bounds=(0.03, 0.80),
            eta=0.45,
            weight=0.5,
            interpretation="General isotropic nitroxide triplet model.",
        ),
        "nitroxide_15n": SpinComponent(
            component_id="nitroxide_15n",
            display_name="Generic 15N nitroxide doublet",
            radical_assignment="15N nitroxide",
            category="nitroxide/spin label",
            g=2.0060,
            g_bounds=(2.0020, 2.0090),
            nuclei=[Nucleus(isotope="15N", A_mT=2.15, label="nitroxide 15N", bounds=(1.60, 2.80))],
            linewidth_mT=0.10,
            linewidth_bounds=(0.03, 0.80),
            eta=0.45,
            weight=0.4,
            interpretation="General isotropic 15N nitroxide doublet model.",
        ),
        "triphenylmethyl_trityl": SpinComponent(
            component_id="triphenylmethyl_trityl",
            display_name="Trityl/TAM narrow radical",
            radical_assignment="trityl/TAM",
            category="spin probe",
            g=2.0030,
            g_bounds=(2.0000, 2.0065),
            nuclei=[],
            linewidth_mT=0.035,
            linewidth_bounds=(0.01, 0.30),
            eta=0.35,
            weight=0.3,
            interpretation="Narrow single-line trityl/TAM-like spin probe approximation.",
        ),
        "cu2_isotropic": SpinComponent(
            component_id="cu2_isotropic",
            display_name="Cu(II) isotropic hyperfine",
            radical_assignment="Cu(II)",
            category="transition metal",
            g=2.0800,
            g_bounds=(2.0000, 2.3500),
            nuclei=[Nucleus(isotope="63Cu", A_mT=8.50, label="63Cu", bounds=(1.00, 20.00))],
            linewidth_mT=0.80,
            linewidth_bounds=(0.10, 8.00),
            eta=0.65,
            weight=0.3,
            interpretation="Approximate isotropic Cu(II) quartet; anisotropic spectra require specialized analysis.",
            warning="High-field isotropic approximation; do not over-interpret anisotropic g/A tensors.",
        ),
        "mn2_sixline": SpinComponent(
            component_id="mn2_sixline",
            display_name="Mn(II) six-line marker",
            radical_assignment="Mn(II)",
            category="transition metal",
            g=2.0000,
            g_bounds=(1.9900, 2.0200),
            nuclei=[Nucleus(isotope="55Mn", A_mT=9.40, label="55Mn", bounds=(5.00, 12.50))],
            linewidth_mT=0.35,
            linewidth_bounds=(0.05, 3.00),
            eta=0.55,
            weight=0.2,
            interpretation="Common Mn(II) sextet approximation.",
        ),
        "vo2_vanadyl": SpinComponent(
            component_id="vo2_vanadyl",
            display_name="VO2+/vanadyl 51V pattern",
            radical_assignment="VO2+ / V(IV)",
            category="transition metal",
            g=1.9800,
            g_bounds=(1.9200, 2.0200),
            nuclei=[Nucleus(isotope="51V", A_mT=10.50, label="51V", bounds=(5.00, 18.00))],
            linewidth_mT=0.45,
            linewidth_bounds=(0.05, 4.00),
            eta=0.55,
            weight=0.2,
            interpretation="Approximate isotropic vanadyl/V(IV) eight-line pattern.",
            warning="Powder vanadyl spectra are often anisotropic; use this as a screening model.",
        ),
        "defect_broad_general": SpinComponent(
            component_id="defect_broad_general",
            display_name="Generic broad defect/trapped-electron background",
            radical_assignment="defect / trapped electron",
            category="solid defect",
            g=2.0030,
            g_bounds=(1.9500, 2.1500),
            nuclei=[],
            linewidth_mT=2.00,
            linewidth_bounds=(0.20, 20.00),
            eta=0.70,
            weight=0.3,
            interpretation="Broad solid-state defect, trapped-electron, or unresolved radical envelope.",
        ),
        "pbn_oh": SpinComponent(
            component_id="pbn_oh",
            display_name="PBN-OH / water oxidation radical",
            radical_assignment="PBN-OH",
            category="water/ROS",
            g=2.0055,
            nuclei=[
                Nucleus(isotope="14N", A_mT=1.50, label="PBN N", bounds=(1.35, 1.65)),
                Nucleus(isotope="1H", A_mT=0.30, label="beta H", bounds=(0.15, 0.45)),
            ],
            linewidth_mT=0.08,
            linewidth_bounds=(0.03, 0.25),
            interpretation="Hydroxyl/water oxidation radical; not N2RR proof.",
        ),
        "pbn_ch3_dmso": SpinComponent(
            component_id="pbn_ch3_dmso",
            display_name="PBN-CH3 / DMSO-derived carbon radical",
            radical_assignment="PBN-CH3",
            category="DMSO artifact",
            g=2.0056,
            nuclei=[
                Nucleus(isotope="14N", A_mT=1.55, label="PBN N", bounds=(1.40, 1.70)),
                Nucleus(isotope="1H", A_mT=0.28, label="beta H", bounds=(0.15, 0.45)),
            ],
            linewidth_mT=0.08,
            linewidth_bounds=(0.03, 0.25),
            interpretation="DMSO-derived carbon radical; mandatory because PBN is in DMSO.",
            warning="Always compare this before assigning N-associated candidates.",
        ),
        "pbn_ooh_o2minus": SpinComponent(
            component_id="pbn_ooh_o2minus",
            display_name="PBN-OOH / PBN-O2-",
            radical_assignment="PBN-OOH/O2-",
            category="oxygen contamination",
            g=2.0055,
            nuclei=[
                Nucleus(isotope="14N", A_mT=1.49, label="PBN N", bounds=(1.35, 1.65)),
                Nucleus(isotope="1H", A_mT=0.43, label="beta H", bounds=(0.25, 0.60)),
            ],
            linewidth_mT=0.10,
            linewidth_bounds=(0.04, 0.30),
            interpretation="Oxygen contamination / superoxide / hydroperoxyl.",
        ),
    }
    comps.update(anisotropic_components())
    comps.update(spin_trap_adducts())
    comps.update(comprehensive_spin_trap_adducts())
    comps.update(advanced_chemistry_components())
    return comps


def spin_trap_adducts() -> dict[str, SpinComponent]:
    """PBN/DMPO spin-trap adducts including N2RR (nitrogen reduction) candidates.

    Hyperfine values are isotropic approximations (mT) for solution cw-EPR.
    N2RR-derived nitrogen radical adducts are *candidate* assignments and MUST be
    validated against isotope labelling (14N/15N), DMSO/solvent controls, and
    blank experiments before any mechanistic claim.
    """
    return {
        # ── DMPO adducts (classic ROS spin trap) ──
        "dmpo_oh": SpinComponent(
            component_id="dmpo_oh",
            display_name="DMPO-OH (1:2:2:1 quartet)",
            radical_assignment="DMPO-OH (hydroxyl adduct)",
            category="spin trap / ROS",
            g=2.0057,
            nuclei=[Nucleus(isotope="14N", A_mT=1.49, label="14N", bounds=(1.40, 1.55)),
                    Nucleus(isotope="1H", A_mT=1.49, label="beta H", bounds=(1.40, 1.55))],
            linewidth_mT=0.08, linewidth_bounds=(0.03, 0.30), eta=0.5, weight=0.3,
            interpretation="DMPO hydroxyl-radical adduct; characteristic 1:2:2:1 quartet (aN ≈ aH ≈ 1.49 mT).",
        ),
        "dmpo_ooh": SpinComponent(
            component_id="dmpo_ooh",
            display_name="DMPO-OOH (superoxide adduct)",
            radical_assignment="DMPO-OOH / DMPO-O2-",
            category="spin trap / ROS",
            g=2.0058,
            nuclei=[Nucleus(isotope="14N", A_mT=1.43, label="14N", bounds=(1.30, 1.50)),
                    Nucleus(isotope="1H", A_mT=1.17, label="beta H", bounds=(1.00, 1.30)),
                    Nucleus(isotope="1H", A_mT=0.13, label="gamma H", bounds=(0.00, 0.30))],
            linewidth_mT=0.09, linewidth_bounds=(0.04, 0.30), eta=0.5, weight=0.3,
            interpretation="DMPO superoxide/hydroperoxyl adduct; distinct from DMPO-OH by aH and gamma-H coupling.",
        ),
        "dmpo_ch3": SpinComponent(
            component_id="dmpo_ch3",
            display_name="DMPO-CH3 (carbon adduct, DMSO)",
            radical_assignment="DMPO-CH3",
            category="spin trap / artifact",
            g=2.0056,
            nuclei=[Nucleus(isotope="14N", A_mT=1.45, label="14N", bounds=(1.35, 1.55)),
                    Nucleus(isotope="1H", A_mT=2.07, label="beta H", bounds=(1.80, 2.30))],
            linewidth_mT=0.08, linewidth_bounds=(0.03, 0.30), eta=0.5, weight=0.3,
            interpretation="DMSO-derived methyl-radical DMPO adduct; mandatory control when DMPO is used in DMSO.",
            warning="Compare before assigning other carbon/nitrogen adducts.",
        ),
        # ── PBN N2RR (nitrogen reduction) candidate adducts ──
        "pbn_nnh_candidate": SpinComponent(
            component_id="pbn_nnh_candidate",
            display_name="PBN-NNH candidate (early N2RR)",
            radical_assignment="PBN-NNH candidate",
            category="N2RR candidate",
            g=2.0055,
            nuclei=[Nucleus(isotope="14N", A_mT=1.45, label="PBN N", bounds=(1.30, 1.65)),
                    Nucleus(isotope="1H", A_mT=0.25, label="beta H", bounds=(0.10, 0.45)),
                    Nucleus(isotope="14N", A_mT=0.18, label="proximal N", bounds=(0.02, 0.35)),
                    Nucleus(isotope="14N", A_mT=0.06, label="distal N", bounds=(0.00, 0.18)),
                    Nucleus(isotope="1H", A_mT=0.08, label="N-H", bounds=(0.00, 0.20))],
            linewidth_mT=0.09, linewidth_bounds=(0.04, 0.30), eta=0.5, weight=0.2,
            interpretation="Candidate early N2 activation/hydrogenation (diazenyl-like) radical adduct.",
            warning="N2RR candidate: confirm with 15N2 isotope labelling and Ar/N2 controls before any claim.",
        ),
        "pbn_nh2_candidate": SpinComponent(
            component_id="pbn_nh2_candidate",
            display_name="PBN-NH2 candidate (late N2RR)",
            radical_assignment="PBN-NH2 candidate",
            category="N2RR candidate",
            g=2.0055,
            nuclei=[Nucleus(isotope="14N", A_mT=1.45, label="PBN N", bounds=(1.30, 1.65)),
                    Nucleus(isotope="1H", A_mT=0.25, label="beta H", bounds=(0.10, 0.45)),
                    Nucleus(isotope="14N", A_mT=0.25, label="NH2 N", bounds=(0.05, 0.45)),
                    Nucleus(isotope="1H", A_mT=0.10, label="NH H1", bounds=(0.00, 0.25)),
                    Nucleus(isotope="1H", A_mT=0.10, label="NH H2", bounds=(0.00, 0.25))],
            linewidth_mT=0.09, linewidth_bounds=(0.04, 0.30), eta=0.5, weight=0.2,
            interpretation="Candidate late NHx/amino N2RR radical adduct.",
            warning="N2RR candidate: confirm with 15N labelling and DMSO controls before any claim.",
        ),
        "pbn_nhnh_candidate": SpinComponent(
            component_id="pbn_nhnh_candidate",
            display_name="PBN-NHNH / diazene-like candidate",
            radical_assignment="PBN-NHNH candidate",
            category="N2RR candidate",
            g=2.0055,
            nuclei=[Nucleus(isotope="14N", A_mT=1.45, label="PBN N", bounds=(1.30, 1.65)),
                    Nucleus(isotope="1H", A_mT=0.25, label="beta H", bounds=(0.10, 0.45)),
                    Nucleus(isotope="14N", A_mT=0.12, label="N1", bounds=(0.00, 0.30)),
                    Nucleus(isotope="14N", A_mT=0.12, label="N2", bounds=(0.00, 0.30)),
                    Nucleus(isotope="1H", A_mT=0.07, label="NH H1", bounds=(0.00, 0.20)),
                    Nucleus(isotope="1H", A_mT=0.07, label="NH H2", bounds=(0.00, 0.20))],
            linewidth_mT=0.09, linewidth_bounds=(0.04, 0.30), eta=0.5, weight=0.2,
            interpretation="Candidate diazene/hydrazyl-like N2RR radical adduct (high parameter count).",
            warning="N2RR candidate with high overfitting risk: isotope and blank controls are essential.",
        ),
        "pbn_n2h3_candidate": SpinComponent(
            component_id="pbn_n2h3_candidate",
            display_name="PBN-N2H3 hydrazyl candidate",
            radical_assignment="PBN-N2H3 candidate",
            category="N2RR candidate",
            g=2.0055,
            nuclei=[Nucleus(isotope="14N", A_mT=1.44, label="PBN N", bounds=(1.30, 1.65)),
                    Nucleus(isotope="1H", A_mT=0.24, label="beta H", bounds=(0.10, 0.45)),
                    Nucleus(isotope="14N", A_mT=0.30, label="hydrazyl N", bounds=(0.05, 0.50)),
                    Nucleus(isotope="1H", A_mT=0.12, label="N-H", bounds=(0.00, 0.30))],
            linewidth_mT=0.09, linewidth_bounds=(0.04, 0.30), eta=0.5, weight=0.2,
            interpretation="Candidate hydrazine-derived (N2H3) radical adduct from late N2RR intermediates.",
            warning="N2RR candidate: requires 15N isotope confirmation and hydrazine spike controls.",
        ),
    }


def comprehensive_spin_trap_adducts() -> dict[str, SpinComponent]:
    """Generalized spin-trapping library: common nitrone spin traps (PBN, DMPO, POBN,
    DEPMPO) across the usual radical classes, plus the PBN/DMSO-derived radical set.

    Hyperfine values are isotropic literature approximations (converted to mT) after
    Buettner (Free Radic. Biol. Med. 1987), Janzen, and Villamena & Zweier; they are
    solvent- and pH-dependent and are provided as starting points to be refined by
    fitting. Nitrogen-centred and "candidate" entries MUST be validated with 15N/2H
    isotope labelling and spin-trap/solvent blank controls before any assignment.
    """
    # (id, display, trap, radical class, g, [(isotope, A_Gauss), ...], lw_mT, category, note, warning)
    D = [
        # ── PBN adducts (triplet of doublets: aN + aβH) ──
        ("pbn_alkoxyl_ro", "PBN-OR (alkoxyl RO•)", "PBN", "alkoxyl", 2.0057, [("14N", 14.0), ("1H", 2.0)], 0.10,
         "spin trap / PBN", "PBN alkoxyl-radical adduct (low aN ~14 G).", ""),
        ("pbn_methoxyl", "PBN-OCH3 (methoxyl)", "PBN", "methoxyl", 2.0057, [("14N", 13.8), ("1H", 1.9)], 0.10,
         "spin trap / PBN", "PBN methoxyl-radical adduct.", ""),
        ("pbn_peroxyl_roo", "PBN-OOR (peroxyl ROO•)", "PBN", "peroxyl", 2.0057, [("14N", 13.6), ("1H", 1.8)], 0.11,
         "spin trap / PBN", "PBN peroxyl-radical adduct (low aN, small aβH).", ""),
        ("pbn_thiyl_rs", "PBN-SR (thiyl RS•)", "PBN", "thiyl", 2.0058, [("14N", 14.8), ("1H", 3.3)], 0.10,
         "spin trap / PBN", "PBN thiyl-radical adduct.", ""),
        ("pbn_carbon_general", "PBN-R (carbon-centred)", "PBN", "carbon", 2.0057, [("14N", 15.8), ("1H", 3.5)], 0.10,
         "spin trap / PBN", "Generic PBN carbon-centred radical adduct.", ""),
        ("pbn_co2_radical", "PBN-CO2•- (carbon dioxide anion)", "PBN", "CO2 radical anion", 2.0057, [("14N", 15.8), ("1H", 4.4)], 0.10,
         "spin trap / PBN", "PBN carbon-dioxide-radical-anion adduct (CO2RR context).", ""),
        ("pbn_h_atom_adduct", "PBN-H (hydrogen atom)", "PBN", "H atom", 2.0057, [("14N", 16.0), ("1H", 8.0)], 0.11,
         "spin trap / PBN", "PBN hydrogen-atom adduct (large aβH).", ""),
        ("pbn_ox_degradation", "PBNOx (di-tert-alkyl nitroxide, degradation)", "PBN", "trap breakdown", 2.0058, [("14N", 15.5)], 0.09,
         "spin trap / artifact", "PBN oxidation/decomposition nitroxide; clean triplet, no β-H. Common background.",
         "Trap-derived artifact, not a trapped reactive radical."),
        # ── PBN + DMSO-derived radical set (DMSO is the usual PBN stock solvent) ──
        ("pbn_dmso_methyl", "PBN-CH3 (•CH3 from DMSO)", "PBN/DMSO", "methyl", 2.0056, [("14N", 15.9), ("1H", 3.5)], 0.09,
         "spin trap / DMSO", "DMSO-derived methyl radical trapped by PBN; mandatory PBN/DMSO control.",
         "Solvent-derived: run PBN/DMSO blank before assigning reaction radicals."),
        ("pbn_dmso_sulfinylmethyl", "PBN-CH2S(O)CH3 (methanesulfinylmethyl)", "PBN/DMSO", "sulfinylmethyl", 2.0056, [("14N", 15.6), ("1H", 3.3)], 0.10,
         "spin trap / DMSO", "DMSO-derived •CH2S(O)CH3 radical trapped by PBN.", ""),
        ("pbn_dmso_methylsulfinyl", "PBN-S(O)CH3 (methylsulfinyl)", "PBN/DMSO", "methylsulfinyl", 2.0057, [("14N", 14.6), ("1H", 2.1)], 0.10,
         "spin trap / DMSO", "DMSO-derived methylsulfinyl radical trapped by PBN.", ""),
        # ── DMPO adducts ──
        ("dmpo_h_atom", "DMPO-H (hydrogen atom)", "DMPO", "H atom", 2.0057, [("14N", 16.6), ("1H", 22.5), ("1H", 22.5)], 0.10,
         "spin trap / DMPO", "DMPO hydrogen-atom adduct (aN and two large 1H couplings).", ""),
        ("dmpo_co2_radical", "DMPO-CO2•- (carbon dioxide anion)", "DMPO", "CO2 radical anion", 2.0057, [("14N", 15.8), ("1H", 18.7)], 0.10,
         "spin trap / DMPO", "DMPO carbon-dioxide-radical-anion adduct.", ""),
        ("dmpo_thiyl_rs", "DMPO-SR (thiyl RS•)", "DMPO", "thiyl", 2.0058, [("14N", 15.3), ("1H", 16.5)], 0.10,
         "spin trap / DMPO", "DMPO thiyl-radical adduct.", ""),
        ("dmpo_alkoxyl_ro", "DMPO-OR (alkoxyl RO•)", "DMPO", "alkoxyl", 2.0057, [("14N", 14.6), ("1H", 7.8)], 0.11,
         "spin trap / DMPO", "DMPO alkoxyl-radical adduct.", ""),
        ("dmpo_carbon_general", "DMPO-R (carbon-centred)", "DMPO", "carbon", 2.0057, [("14N", 16.0), ("1H", 22.5)], 0.10,
         "spin trap / DMPO", "Generic DMPO carbon-centred radical adduct (aN ≈ 16, aβH ≈ 22 G).", ""),
        # ── POBN adducts (water-soluble PBN analogue) ──
        ("pobn_carbon", "POBN-R (carbon-centred)", "POBN", "carbon", 2.0057, [("14N", 15.5), ("1H", 2.5)], 0.10,
         "spin trap / POBN", "POBN carbon-centred radical adduct; water-soluble PBN analogue.", ""),
        ("pobn_oh", "POBN-OH (hydroxyl)", "POBN", "hydroxyl", 2.0057, [("14N", 15.3), ("1H", 1.7)], 0.10,
         "spin trap / POBN", "POBN hydroxyl-radical adduct.", ""),
        # ── DEPMPO adducts (31P-bearing, distinguishes O2•- from •OH) ──
        ("depmpo_ooh", "DEPMPO-OOH (superoxide, 31P)", "DEPMPO", "superoxide", 2.0058, [("31P", 49.4), ("14N", 13.3), ("1H", 11.4)], 0.12,
         "spin trap / DEPMPO", "DEPMPO superoxide adduct; persistent, 31P coupling distinguishes it from •OH.", ""),
        ("depmpo_oh", "DEPMPO-OH (hydroxyl, 31P)", "DEPMPO", "hydroxyl", 2.0058, [("31P", 47.3), ("14N", 14.0), ("1H", 13.2)], 0.12,
         "spin trap / DEPMPO", "DEPMPO hydroxyl-radical adduct with characteristic 31P coupling.", ""),
    ]
    out: dict[str, SpinComponent] = {}
    for cid, name, trap, radical, g, hf, lw, cat, note, warn in D:
        nuclei = []
        for i, (iso, a_g) in enumerate(hf):
            label = ("14N" if i == 0 and iso == "14N" else
                     ("31P" if iso == "31P" else ("beta H" if iso == "1H" else iso)))
            a_mT = a_g / 10.0
            nuclei.append(Nucleus(isotope=iso, A_mT=a_mT, label=label,
                                  bounds=(max(a_mT - 0.30, 0.0), a_mT + 0.30)))
        out[cid] = SpinComponent(
            component_id=cid, display_name=name, radical_assignment=f"{trap} {radical} adduct",
            category=cat, g=g, g_bounds=(2.0030, 2.0075), nuclei=nuclei,
            linewidth_mT=lw, linewidth_bounds=(0.03, 0.60), eta=0.5, weight=0.2,
            interpretation=note, warning=warn)
    return out


def advanced_chemistry_components() -> dict[str, SpinComponent]:
    """General chemistry and electro/photocatalysis screening components.

    These are deliberately conservative candidate models.  They help test
    whether a spectral feature is compatible with a chemically plausible class,
    but they do not prove a mechanism without reference spectra, isotope labels,
    controls, and independent chemistry.
    """
    return {
        "tempo_standard": SpinComponent(
            component_id="tempo_standard",
            display_name="TEMPO / nitroxide reference",
            radical_assignment="TEMPO-like nitroxide",
            category="reference standard",
            g=2.0060,
            g_bounds=(2.0030, 2.0095),
            nuclei=[Nucleus(isotope="14N", A_mT=1.60, label="nitroxide 14N", bounds=(1.35, 1.85))],
            linewidth_mT=0.09,
            linewidth_bounds=(0.02, 0.60),
            eta=0.45,
            weight=0.4,
            interpretation="TEMPO-like nitroxide reference triplet for calibration and benchmarking.",
        ),
        "dpph_standard": SpinComponent(
            component_id="dpph_standard",
            display_name="DPPH reference singlet",
            radical_assignment="DPPH",
            category="reference standard",
            g=2.0036,
            g_bounds=(2.0020, 2.0055),
            nuclei=[],
            linewidth_mT=0.12,
            linewidth_bounds=(0.02, 1.00),
            eta=0.40,
            weight=0.4,
            interpretation="DPPH-like narrow reference line; useful for g calibration.",
        ),
        "carbonate_radical": SpinComponent(
            component_id="carbonate_radical",
            display_name="Carbonate radical candidate",
            radical_assignment="CO3- / carbonate radical candidate",
            category="CO2RR / carbonate chemistry",
            g=2.0110,
            g_bounds=(2.0050, 2.0180),
            nuclei=[],
            linewidth_mT=0.18,
            linewidth_bounds=(0.03, 1.50),
            eta=0.45,
            weight=0.25,
            interpretation="Candidate carbonate-derived radical; relevant in bicarbonate/CO2 electrolyte systems.",
            warning="Confirm with bicarbonate/carbonate controls and 13C-labelled carbonate/CO2 where possible.",
        ),
        "co2_radical_anion_candidate": SpinComponent(
            component_id="co2_radical_anion_candidate",
            display_name="CO2 radical anion candidate",
            radical_assignment="CO2- candidate",
            category="CO2RR candidate",
            g=2.0008,
            g_bounds=(1.9960, 2.0060),
            nuclei=[],
            linewidth_mT=0.20,
            linewidth_bounds=(0.03, 2.00),
            eta=0.45,
            weight=0.25,
            interpretation="Candidate CO2 radical anion / adsorbed CO2 activation signal.",
            warning="CO2RR candidate: validate with Ar/N2 controls, no-CO2 controls, and 13CO2 isotope experiments.",
        ),
        "cooh_ads_candidate": SpinComponent(
            component_id="cooh_ads_candidate",
            display_name="Adsorbed COOH candidate",
            radical_assignment="*COOH candidate",
            category="CO2RR candidate",
            g=2.0045,
            g_bounds=(1.9990, 2.0100),
            nuclei=[Nucleus(isotope="1H", A_mT=0.35, label="COOH H", bounds=(0.00, 1.20))],
            linewidth_mT=0.16,
            linewidth_bounds=(0.03, 1.50),
            eta=0.45,
            weight=0.25,
            interpretation="Candidate adsorbed carboxyl/COOH-type radical intermediate for CO2RR screening.",
            warning="Mechanistic candidate only; compare with pH, electrolyte, CO2/Ar, and 13CO2 controls.",
        ),
        "formate_radical_candidate": SpinComponent(
            component_id="formate_radical_candidate",
            display_name="Formate/carboxylate radical candidate",
            radical_assignment="HCOO / carboxylate radical candidate",
            category="CO2RR candidate",
            g=2.0030,
            g_bounds=(1.9980, 2.0090),
            nuclei=[Nucleus(isotope="1H", A_mT=0.70, label="formyl H", bounds=(0.05, 2.50))],
            linewidth_mT=0.14,
            linewidth_bounds=(0.03, 1.50),
            eta=0.45,
            weight=0.25,
            interpretation="Candidate formate/carboxylate radical contribution.",
            warning="Use with isotopic and product controls; fit alone does not distinguish CO2RR products.",
        ),
        "pbn_co2_minus_candidate": SpinComponent(
            component_id="pbn_co2_minus_candidate",
            display_name="PBN-CO2- candidate",
            radical_assignment="PBN-CO2- / carboxyl radical adduct candidate",
            category="CO2RR spin trap candidate",
            g=2.0055,
            g_bounds=(2.0020, 2.0090),
            nuclei=[
                Nucleus(isotope="14N", A_mT=1.50, label="PBN N", bounds=(1.30, 1.70)),
                Nucleus(isotope="1H", A_mT=0.25, label="beta H", bounds=(0.05, 0.60)),
                Nucleus(isotope="13C", A_mT=0.12, label="13C optional", bounds=(0.00, 0.50)),
            ],
            linewidth_mT=0.09,
            linewidth_bounds=(0.03, 0.40),
            eta=0.50,
            weight=0.20,
            interpretation="Candidate PBN-trapped CO2/carboxyl-derived radical; includes optional weak 13C coupling.",
            warning="CO2RR spin-trap candidate: require 13CO2 controls and solvent-artifact checks.",
        ),
        "surface_electron_trap": SpinComponent(
            component_id="surface_electron_trap",
            display_name="Surface trapped electron / vacancy",
            radical_assignment="surface trapped electron",
            category="photocatalysis / solid defect",
            g=2.0025,
            g_bounds=(1.9500, 2.0800),
            nuclei=[],
            linewidth_mT=1.20,
            linewidth_bounds=(0.10, 15.00),
            eta=0.65,
            weight=0.30,
            interpretation="Broad or semi-broad trapped-electron/oxygen-vacancy/surface-defect contribution.",
        ),
        "surface_hole_trap": SpinComponent(
            component_id="surface_hole_trap",
            display_name="Surface trapped hole / O- candidate",
            radical_assignment="O- / trapped hole candidate",
            category="photocatalysis / OER",
            g=2.0150,
            g_bounds=(2.0050, 2.0600),
            nuclei=[],
            linewidth_mT=1.00,
            linewidth_bounds=(0.10, 12.00),
            eta=0.60,
            weight=0.25,
            interpretation="Candidate trapped-hole/oxygen-radical contribution for oxide photocatalysts.",
            warning="Broad solid-state signals are non-unique; validate with light/dark, atmosphere, and temperature controls.",
        ),
        "h_atom_spin_trap_candidate": SpinComponent(
            component_id="h_atom_spin_trap_candidate",
            display_name="H-atom spin-trap candidate",
            radical_assignment="H atom / HER candidate adduct",
            category="HER / hydrogen radical candidate",
            g=2.0056,
            g_bounds=(2.0020, 2.0090),
            nuclei=[
                Nucleus(isotope="14N", A_mT=1.50, label="spin-trap N", bounds=(1.20, 1.80)),
                Nucleus(isotope="1H", A_mT=0.45, label="H adduct", bounds=(0.05, 1.20)),
            ],
            linewidth_mT=0.09,
            linewidth_bounds=(0.03, 0.40),
            eta=0.50,
            weight=0.20,
            interpretation="Generic spin-trapped hydrogen-atom/HER candidate pattern.",
            warning="HER/H-atom candidate: use D2O/H2O isotope controls and no-catalyst blanks.",
        ),
        "superoxide_general": SpinComponent(
            component_id="superoxide_general",
            display_name="Superoxide / O2- general candidate",
            radical_assignment="O2- / superoxide candidate",
            category="ORR / ROS",
            g=2.0100,
            g_bounds=(2.0020, 2.0300),
            nuclei=[],
            linewidth_mT=0.35,
            linewidth_bounds=(0.04, 3.00),
            eta=0.50,
            weight=0.25,
            interpretation="General superoxide-like component for ORR/ROS/oxygen-contamination screening.",
            warning="Oxygen-derived signals are sensitive to matrix and temperature; compare O2, air, and inert controls.",
        ),
    }


def anisotropic_components() -> dict[str, SpinComponent]:
    """Tensor-resolved / high-spin component models simulated by the powder engine.

    These use anisotropic g and A tensors, electron spin S, and zero-field
    splitting (D, E).  They are solved by full spin-Hamiltonian diagonalisation
    with orientation (powder) averaging, valid for frozen-solution, powder, and
    solid-state cw-EPR spectra.
    """
    return {
        "nitroxide_aniso": SpinComponent(
            component_id="nitroxide_aniso",
            display_name="Nitroxide (rigid-limit, anisotropic g/A)",
            radical_assignment="nitroxide spin label (powder)",
            category="anisotropic radical",
            g=2.0059, spin_S=0.5,
            g_tensor=(2.00906, 2.00617, 2.00220),
            nuclei=[Nucleus(isotope="14N", A_mT=1.10, label="14N",
                            A_tensor_mT=(0.62, 0.62, 3.62))],
            linewidth_mT=0.20, linewidth_bounds=(0.05, 2.00),
            eta=0.40, weight=0.4,
            interpretation="Rigid-limit nitroxide with rhombic g and axial 14N hyperfine; frozen-solution powder pattern.",
            warning="Powder/rigid-limit model; for fast-tumbling solution use the isotropic nitroxide instead.",
        ),
        "cu2_axial": SpinComponent(
            component_id="cu2_axial",
            display_name="Cu(II) axial (g-parallel/perp, 63Cu A-parallel)",
            radical_assignment="Cu(II) (axial powder)",
            category="anisotropic metal",
            g=2.12, spin_S=0.5,
            g_tensor=(2.060, 2.060, 2.270),
            nuclei=[Nucleus(isotope="63Cu", A_mT=1.5, label="63Cu",
                            A_tensor_mT=(1.0, 1.0, 16.5))],
            linewidth_mT=1.20, linewidth_bounds=(0.20, 8.00),
            eta=0.50, weight=0.4,
            interpretation="Axial Cu(II) with g-parallel > g-perp and resolved parallel 63Cu hyperfine; classic square-planar/tetragonal powder pattern.",
            warning="Cu(II) tensors vary with coordination; refine against standards.",
        ),
        "fe3_highspin": SpinComponent(
            component_id="fe3_highspin",
            display_name="Fe(III) high-spin (S=5/2, g≈2 + ZFS)",
            radical_assignment="high-spin Fe(III)",
            category="high-spin metal",
            g=2.00, spin_S=2.5,
            g_tensor=(2.00, 2.00, 2.00),
            D_MHz=3000.0, E_MHz=300.0,
            nuclei=[],
            linewidth_mT=3.00, linewidth_bounds=(0.50, 20.00),
            eta=0.60, weight=0.3,
            interpretation="High-spin Fe(III) (S=5/2) with zero-field splitting; effective g≈2, 4.3 features depend on E/D.",
            warning="High-spin ZFS systems are orientation-rich; increase powder orientations for converged patterns.",
        ),
        "mn2_aqueous": SpinComponent(
            component_id="mn2_aqueous",
            display_name="Mn(II) aqueous (S=5/2, 55Mn sextet + ZFS)",
            radical_assignment="Mn(II) (S=5/2)",
            category="high-spin metal",
            g=2.00, spin_S=2.5,
            g_tensor=(2.00, 2.00, 2.00),
            D_MHz=250.0, E_MHz=50.0,
            nuclei=[Nucleus(isotope="55Mn", A_mT=9.5, label="55Mn",
                            A_tensor_mT=(9.5, 9.5, 9.5))],
            linewidth_mT=0.80, linewidth_bounds=(0.10, 5.00),
            eta=0.55, weight=0.3,
            interpretation="Aqueous/solid Mn(II) six-line pattern from the central -1/2<->+1/2 transition with weak ZFS broadening of outer lines.",
            warning="Full Mn(II) fine structure needs many orientations; central sextet dominates at X-band.",
        ),
        "triplet_organic": SpinComponent(
            component_id="triplet_organic",
            display_name="Organic triplet / biradical (S=1, ZFS D,E)",
            radical_assignment="triplet state (S=1)",
            category="high-spin organic",
            g=2.003, spin_S=1.0,
            g_tensor=(2.003, 2.003, 2.003),
            D_MHz=1000.0, E_MHz=100.0,
            nuclei=[],
            linewidth_mT=1.00, linewidth_bounds=(0.20, 10.00),
            eta=0.50, weight=0.3,
            interpretation="Photoexcited/ground-state triplet or strongly-coupled biradical with dipolar zero-field splitting (Pake-like powder pattern).",
            warning="D and E set the line spacing; half-field (delta-Ms=2) transitions are not included in this X-band window.",
        ),
        "vo2_axial": SpinComponent(
            component_id="vo2_axial",
            display_name="VO2+ vanadyl axial (anisotropic g/51V)",
            radical_assignment="VO2+ / V(IV) (axial powder)",
            category="anisotropic metal",
            g=1.97, spin_S=0.5,
            g_tensor=(1.984, 1.984, 1.943),
            nuclei=[Nucleus(isotope="51V", A_mT=6.0, label="51V",
                            A_tensor_mT=(6.3, 6.3, 17.0))],
            linewidth_mT=0.80, linewidth_bounds=(0.10, 6.00),
            eta=0.50, weight=0.3,
            interpretation="Axial vanadyl with g-perp > g-parallel and large parallel 51V hyperfine; eight-line anisotropic powder pattern.",
            warning="Vanadyl tensors are coordination-sensitive; refine against authentic complexes.",
        ),
    }


MODEL_PRESETS: dict[str, list[str]] = {
    "R1_reference_standards": ["dpph_standard", "tempo_standard", "mn2_sixline", "cu2_isotropic", "vo2_vanadyl"],
    "G0_single_line": ["organic_radical_singlet"],
    "G1_organic_radicals": ["organic_radical_singlet", "carbon_centered_h", "semiquinone_radical"],
    "G2_nitroxide_spin_label": ["nitroxide_14n", "nitroxide_15n", "triphenylmethyl_trityl"],
    "G3_ros_spin_trap_general": ["organic_radical_singlet", "nitroxide_14n", "semiquinone_radical"],
    "G4_transition_metal_screen": ["cu2_isotropic", "mn2_sixline", "vo2_vanadyl"],
    "G5_defect_plus_radical": ["defect_broad_general", "organic_radical_singlet", "semiquinone_radical"],
    "P1_photocatalysis_surface": ["surface_electron_trap", "surface_hole_trap", "superoxide_general", "semiquinone_radical"],
    "E1_electrocatalysis_general": ["surface_electron_trap", "superoxide_general", "h_atom_spin_trap_candidate", "organic_radical_singlet"],
    "O1_oer_orr_ros": ["superoxide_general", "dmpo_oh", "dmpo_ooh", "surface_hole_trap"],
    "H1_her_hydrogen": ["h_atom_spin_trap_candidate", "surface_electron_trap", "organic_radical_singlet"],
    "C1_co2rr_screen": ["co2_radical_anion_candidate", "cooh_ads_candidate", "formate_radical_candidate", "carbonate_radical"],
    "C2_co2rr_spintrap": ["pbn_ch3_dmso", "pbn_ooh_o2minus", "pbn_co2_minus_candidate", "carbonate_radical"],
    "M1_water_dmso": ["pbn_oh", "pbn_ch3_dmso"],
    "M2_water_dmso_o2": ["pbn_oh", "pbn_ch3_dmso", "pbn_ooh_o2minus"],
    "M3_pbn_photocat_decomposition": ["pbn_ch3_dmso", "h_atom_spin_trap_candidate", "pbn_oh",
                                      "defect_broad_general", "pbn_nnh_candidate", "pbn_ooh_o2minus"],
    "ST_pbn_all_adducts": ["pbn_oh", "pbn_ooh_o2minus", "pbn_carbon_general", "pbn_h_atom_adduct",
                           "pbn_alkoxyl_ro", "pbn_methoxyl", "pbn_peroxyl_roo", "pbn_thiyl_rs",
                           "pbn_co2_radical", "pbn_ox_degradation"],
    "ST_pbn_dmso_radicals": ["pbn_dmso_methyl", "pbn_dmso_sulfinylmethyl", "pbn_dmso_methylsulfinyl",
                             "pbn_oh", "pbn_ox_degradation"],
    "ST_dmpo_all_adducts": ["dmpo_oh", "dmpo_ooh", "dmpo_h_atom", "dmpo_ch3", "dmpo_carbon_general",
                            "dmpo_co2_radical", "dmpo_thiyl_rs", "dmpo_alkoxyl_ro"],
    "ST_pobn_adducts": ["pobn_oh", "pobn_carbon"],
    "ST_depmpo_ros": ["depmpo_ooh", "depmpo_oh"],
    "M3_ros_spin_trap_full": ["defect_broad_general", "pbn_oh", "pbn_ch3_dmso", "pbn_ooh_o2minus"],
    # ── Anisotropic / high-spin (powder) presets ──
    "A1_nitroxide_rigid": ["nitroxide_aniso"],
    "A2_cu_axial": ["cu2_axial"],
    "A3_vanadyl_axial": ["vo2_axial"],
    "A4_highspin_fe3": ["fe3_highspin"],
    "A5_mn2_highspin": ["mn2_aqueous"],
    "A6_organic_triplet": ["triplet_organic"],
    # ── DMPO ROS spin-trap presets ──
    "D1_dmpo_oh": ["dmpo_oh"],
    "D2_dmpo_ros": ["dmpo_oh", "dmpo_ooh"],
    "D3_dmpo_full": ["dmpo_oh", "dmpo_ooh", "dmpo_ch3"],
    # ── PBN N2RR (nitrogen reduction) candidate presets ──
    "N1_n2rr_dmso_corrected": ["pbn_oh", "pbn_ch3_dmso", "pbn_nnh_candidate", "pbn_nh2_candidate"],
    "N2_n2rr_extended": ["pbn_oh", "pbn_ch3_dmso", "pbn_ooh_o2minus", "pbn_nnh_candidate",
                          "pbn_nh2_candidate", "pbn_nhnh_candidate"],
    "N3_n2rr_full": ["pbn_oh", "pbn_ch3_dmso", "pbn_ooh_o2minus", "pbn_nnh_candidate",
                     "pbn_nh2_candidate", "pbn_nhnh_candidate", "pbn_n2h3_candidate"],
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "R1_reference_standards": "Reference-style components for DPPH, TEMPO/nitroxide, Mn(II), Cu(II), and vanadyl benchmarking.",
    "G0_single_line": "General one-component radical or unresolved singlet.",
    "G1_organic_radicals": "Carbon-centered, semiquinone/phenoxyl, and generic organic radical screening.",
    "G2_nitroxide_spin_label": "Common 14N/15N nitroxide and trityl/TAM-like spin-probe models.",
    "G3_ros_spin_trap_general": "General ROS/spin-trap screening without N2RR-specific assignments.",
    "G4_transition_metal_screen": "Approximate isotropic Cu(II), Mn(II), and V(IV)/vanadyl screening.",
    "G5_defect_plus_radical": "Broad defect envelope plus narrow radical components.",
    "P1_photocatalysis_surface": "Photocatalysis screen for surface trapped electrons/holes, superoxide, and organic/ROS radicals.",
    "E1_electrocatalysis_general": "General electrocatalysis screen for surface traps, ROS, H-atom/HER, and organic radicals.",
    "O1_oer_orr_ros": "OER/ORR/ROS screen including superoxide and common DMPO spin-trap adducts.",
    "H1_her_hydrogen": "HER/H-atom candidate screen with trapped-electron and generic radical controls.",
    "C1_co2rr_screen": "CO2RR candidate screen: CO2-, *COOH, formate/carboxylate, and carbonate radical contributions.",
    "C2_co2rr_spintrap": "CO2RR spin-trap screen with PBN/DMSO/O2 controls plus PBN-CO2 candidate.",
    "M1_water_dmso": "PBN water/DMSO model with mandatory PBN-CH3 when PBN is introduced in DMSO.",
    "M3_pbn_photocat_decomposition": "PBN water/DMSO photocatalysis decomposition (example fractions): PBN-CH3 52.6%, PBN-H 19.1%, PBN-OH 16.3%, broad PBN/surface-derived 7.2%, unassigned N-centred candidate 3.8%, residual O2-derived candidate 1.0%. The N-centred and O2-derived entries are CANDIDATE assignments requiring 15N-labelling and blank/DMSO controls before any mechanistic claim.",
    "ST_pbn_all_adducts": "Comprehensive PBN spin-adduct set across radical classes (OH, OOH/O2-, carbon, H-atom, alkoxyl, methoxyl, peroxyl, thiyl, CO2-, and the PBNOx degradation product).",
    "ST_pbn_dmso_radicals": "PBN + DMSO-derived radical set: methyl (CH3), methanesulfinylmethyl, and methylsulfinyl radicals from DMSO, plus PBN-OH and PBNOx controls. Use when PBN is delivered in a DMSO stock.",
    "ST_dmpo_all_adducts": "Comprehensive DMPO spin-adduct set (OH quartet, OOH/superoxide, H-atom, methyl, carbon, CO2-, thiyl, alkoxyl).",
    "ST_pobn_adducts": "POBN (water-soluble PBN analogue) hydroxyl and carbon-centred adducts.",
    "ST_depmpo_ros": "DEPMPO ROS adducts (superoxide and hydroxyl) with the diagnostic 31P coupling that distinguishes O2- from OH.",
    "M2_water_dmso_o2": "PBN water/DMSO/O2 model including PBN-OOH/O2-derived adducts.",
    "M3_ros_spin_trap_full": "Full ROS spin-trap model with broad background, hydroxyl, DMSO, and O2 components.",
    "A1_nitroxide_rigid": "Rigid-limit nitroxide powder pattern (rhombic g, axial 14N) for frozen solutions.",
    "A2_cu_axial": "Axial Cu(II) powder pattern with resolved parallel 63Cu hyperfine.",
    "A3_vanadyl_axial": "Axial VO2+/V(IV) vanadyl eight-line anisotropic powder pattern.",
    "A4_highspin_fe3": "High-spin Fe(III) (S=5/2) with zero-field splitting.",
    "A5_mn2_highspin": "High-spin Mn(II) (S=5/2) six-line central transition with ZFS.",
    "A6_organic_triplet": "Organic triplet / biradical (S=1) dipolar zero-field-split powder pattern.",
    "D1_dmpo_oh": "DMPO hydroxyl-radical adduct (1:2:2:1 quartet).",
    "D2_dmpo_ros": "DMPO ROS screen: hydroxyl + superoxide/hydroperoxyl adducts.",
    "D3_dmpo_full": "DMPO full ROS + DMSO methyl-adduct control.",
    "N1_n2rr_dmso_corrected": "DMSO-corrected PBN N2RR model with NNH and NH2 candidate adducts.",
    "N2_n2rr_extended": "Extended PBN N2RR model adding O2 and diazene-like NHNH candidate.",
    "N3_n2rr_full": "Full PBN N2RR candidate set including hydrazyl N2H3 (highest overfitting risk).",
}


def components_for_preset(preset: str) -> list[SpinComponent]:
    library = default_components()
    return [library[cid].clone() for cid in MODEL_PRESETS[preset]]


# Example PBN water/DMSO photocatalysis decomposition: component id -> model fraction.
# Fractions are an illustrative fitted decomposition. PBN-OH is a well-established
# assignment; the N-centred and O2-derived entries are CANDIDATE assignments that
# require 15N-labelling and blank/DMSO controls before any mechanistic claim.
PBN_PHOTOCAT_FRACTIONS: dict[str, float] = {
    "pbn_ch3_dmso": 0.526,                # PBN-CH3 (DMSO-derived methyl)
    "h_atom_spin_trap_candidate": 0.191,  # PBN-H (hydrogen-atom adduct)
    "pbn_oh": 0.163,                      # PBN-OH (hydroxyl / ROS)
    "defect_broad_general": 0.072,        # broad PBN/surface-derived component
    "pbn_nnh_candidate": 0.038,           # unassigned N-centred candidate
    "pbn_ooh_o2minus": 0.010,             # residual O2-derived candidate
}


def pbn_photocatalysis_model() -> list[SpinComponent]:
    """Return the example PBN water/DMSO photocatalysis component decomposition with the
    documented model fractions preset as component weights (see PBN_PHOTOCAT_FRACTIONS)."""
    library = default_components()
    out: list[SpinComponent] = []
    for cid, frac in PBN_PHOTOCAT_FRACTIONS.items():
        c = library[cid].clone()
        c.weight = float(frac)
        out.append(c)
    return out


def component_table() -> list[dict]:
    return [component.to_table_row() for component in default_components().values()]
