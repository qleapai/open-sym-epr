"""Chemical model suggestion logic for PBN/DMSO/N2RR EPR experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DMSO_WARNING = (
    "PBN was introduced in DMSO. PBN-CH3/DMSO-derived radical must be considered "
    "before assigning any N2RR intermediate."
)


@dataclass
class ExperimentContext:
    analysis_mode: str = "PBN/N2RR spin trapping"
    sample_class: str = "unknown/general"
    condition: str = "N2 + light"
    catalyst: str = "Pd-Ov-Bi2MoO6"
    solvent: str = "water + DMSO stock"
    pbn_used: bool = True
    isotope_labelled_nh3: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def condition_l(self) -> str:
        return self.condition.lower()

    @property
    def analysis_mode_l(self) -> str:
        return self.analysis_mode.lower()

    @property
    def sample_class_l(self) -> str:
        return self.sample_class.lower()

    @property
    def catalyst_l(self) -> str:
        return self.catalyst.lower()

    @property
    def solvent_l(self) -> str:
        return self.solvent.lower()

    @property
    def has_dmso(self) -> bool:
        keys = " ".join(self.metadata.get("header_keywords", []))
        return "dmso" in self.solvent_l or "dmso" in keys.lower()

    @property
    def has_water(self) -> bool:
        keys = " ".join(self.metadata.get("header_keywords", []))
        return "water" in self.solvent_l or "water" in keys.lower()

    @property
    def is_n2_light_pd_ov(self) -> bool:
        return "n2" in self.condition_l and "light" in self.condition_l and "pd" in self.catalyst_l and "ov" in self.catalyst_l


def suggest_models(context: ExperimentContext) -> dict[str, Any]:
    suggestions: list[str] = []
    warnings: list[str] = []
    explanations: dict[str, str] = {}

    is_general = "general" in context.analysis_mode_l
    if is_general:
        sample = context.sample_class_l
        cond_cat = f"{context.condition_l} {context.catalyst_l} {context.solvent_l}"
        if "n2rr" in sample or "nitrogen reduction" in sample or "n2 reduction" in cond_cat:
            suggestions = ["N1_n2rr_dmso_corrected", "N2_n2rr_extended", "M2_water_dmso_o2", "G5_defect_plus_radical"]
            warnings.append("N2RR candidate components require 15N2 isotope controls, Ar/N2 comparisons, blanks, and solvent-artifact checks.")
        elif "co2rr" in sample or "co2 reduction" in sample or "co2" in cond_cat:
            suggestions = ["C1_co2rr_screen", "C2_co2rr_spintrap", "R1_reference_standards", "G5_defect_plus_radical"]
            warnings.append("CO2RR candidate components require no-CO2 controls, Ar/N2 controls, 13CO2 labelling where possible, and product analysis.")
        elif "electro" in sample:
            suggestions = ["E1_electrocatalysis_general", "O1_oer_orr_ros", "H1_her_hydrogen", "C1_co2rr_screen"]
            warnings.append("Electrocatalysis assignments should be checked against potential-dependent controls and electrolyte/blank spectra.")
        elif "photo" in sample:
            suggestions = ["P1_photocatalysis_surface", "G5_defect_plus_radical", "O1_oer_orr_ros", "G1_organic_radicals"]
            warnings.append("Photocatalysis assignments should compare light/dark, gas, solvent, and catalyst/material controls.")
        elif "reference" in sample or "standard" in sample or "calibration" in sample:
            suggestions = ["R1_reference_standards", "G0_single_line", "G2_nitroxide_spin_label"]
        elif "her" in sample or "hydrogen" in sample:
            suggestions = ["H1_her_hydrogen", "E1_electrocatalysis_general", "G5_defect_plus_radical"]
            warnings.append("HER/H-atom assignments should be tested with H2O/D2O and no-catalyst controls.")
        elif "oer" in sample or "orr" in sample:
            suggestions = ["O1_oer_orr_ros", "P1_photocatalysis_surface", "G3_ros_spin_trap_general"]
        elif "nitroxide" in sample or "spin label" in sample:
            suggestions = ["G2_nitroxide_spin_label", "G0_single_line"]
        elif "transition" in sample or "metal" in sample or "cu" in sample or "mn" in sample or "vanadyl" in sample:
            suggestions = ["G4_transition_metal_screen", "A2_cu_axial", "A3_vanadyl_axial", "A5_mn2_highspin", "G5_defect_plus_radical"]
            warnings.append("Transition-metal presets are isotropic screening models; anisotropic powder spectra require specialized tensor analysis.")
        elif "defect" in sample or "solid" in sample:
            suggestions = ["G5_defect_plus_radical", "G0_single_line"]
        elif "ros" in sample or "spin trap" in sample:
            suggestions = ["G3_ros_spin_trap_general", "D2_dmpo_ros", "O1_oer_orr_ros", "G1_organic_radicals"]
        elif "organic" in sample:
            suggestions = ["G1_organic_radicals", "G0_single_line"]
        else:
            suggestions = ["G0_single_line", "G1_organic_radicals", "G2_nitroxide_spin_label", "R1_reference_standards", "G5_defect_plus_radical"]
        if context.pbn_used and context.has_dmso:
            warnings.append(DMSO_WARNING)
            suggestions.append("M1_water_dmso")
        explanations.update(_default_explanations())
        return {
            "suggested_models": list(dict.fromkeys(suggestions)),
            "recommended_model": suggestions[0] if suggestions else "G0_single_line",
            "primary_comparison": ("G0_single_line", suggestions[0]) if suggestions and suggestions[0] != "G0_single_line" else None,
            "warnings": warnings,
            "explanations": {name: explanations.get(name, "") for name in list(dict.fromkeys(suggestions))},
        }

    if not context.pbn_used:
        return {
            "suggested_models": ["G5_defect_plus_radical"],
            "recommended_model": "G5_defect_plus_radical",
            "primary_comparison": None,
            "warnings": ["No spin trap was selected; use general radical/defect models unless a specific trap is present."],
            "explanations": {"G5_defect_plus_radical": "Broad defect/background plus generic radical components."},
        }

    if context.has_dmso:
        warnings.append(DMSO_WARNING)

    cond = context.condition_l
    if "dark" in cond:
        suggestions = ["G5_defect_plus_radical", "M1_water_dmso"]
        warnings.append("Strong spin adduct intensity in dark can indicate non-photocatalytic artifact chemistry.")
    elif "ar" in cond:
        suggestions = ["M1_water_dmso", "M2_water_dmso_o2"]
        warnings.append("N2RR candidate components should not be required in Ar controls.")
    elif "air" in cond or "o2" in cond:
        suggestions = ["M2_water_dmso_o2"]
        warnings.append("O2-derived radical chemistry may dominate; inspect PBN-OOH/O2- fraction.")
    elif context.is_n2_light_pd_ov:
        suggestions = ["M1_water_dmso", "M2_water_dmso_o2", "N1_n2rr_dmso_corrected", "N2_n2rr_extended"]
    elif "n2" in cond and "light" in cond:
        suggestions = ["M1_water_dmso", "M2_water_dmso_o2", "N1_n2rr_dmso_corrected", "N2_n2rr_extended"]
    else:
        suggestions = ["M1_water_dmso", "M2_water_dmso_o2"]

    explanations.update(_default_explanations())
    primary = ("M1_water_dmso", "N1_n2rr_dmso_corrected") if "N1_n2rr_dmso_corrected" in suggestions else None
    if any(name in context.catalyst_l for name in ["bi2moo6", "bmo", "pd", "ov"]):
        warnings.append("For catalyst series, batch analysis should check the expected trend: BMO < Ov-BMO / Pd-BMO < Pd-Ov-BMO.")
    return {
        "suggested_models": suggestions,
        "recommended_model": suggestions[-1] if suggestions else "M1_water_dmso",
        "primary_comparison": primary,
        "warnings": warnings,
        "explanations": {name: explanations[name] for name in suggestions},
    }


def _default_explanations() -> dict[str, str]:
    return {
        "G0_single_line": "Tests a generic one-line radical or unresolved singlet.",
        "R1_reference_standards": "Reference-style standards: DPPH, TEMPO/nitroxide, Mn(II), Cu(II), and vanadyl.",
        "G1_organic_radicals": "Screens carbon-centered and semiquinone/phenoxyl-like organic radicals.",
        "G2_nitroxide_spin_label": "Screens common nitroxide/spin-label patterns, including 14N and 15N nitroxides.",
        "G3_ros_spin_trap_general": "Screens general ROS/spin-trap patterns without N2RR-specific assignments.",
        "G4_transition_metal_screen": "Screens approximate isotropic Cu(II), Mn(II), and vanadyl/V(IV) patterns.",
        "G5_defect_plus_radical": "Combines broad solid-state defect/background signals with narrower radical lines.",
        "P1_photocatalysis_surface": "Screens trapped electrons/holes, superoxide, and organic/ROS radicals in photocatalysis.",
        "E1_electrocatalysis_general": "Screens surface traps, ROS, H-atom/HER, and generic radical components in electrochemistry.",
        "O1_oer_orr_ros": "Screens OER/ORR/ROS components including superoxide and DMPO adducts.",
        "H1_her_hydrogen": "Screens H-atom/HER candidate adducts with surface/electron-trap controls.",
        "C1_co2rr_screen": "Screens CO2RR candidate components: CO2-, *COOH, formate/carboxylate, and carbonate.",
        "C2_co2rr_spintrap": "Screens PBN/DMSO/O2 controls plus candidate PBN-CO2 spin-trap components.",
        "D2_dmpo_ros": "Screens DMPO-OH and DMPO-OOH ROS spin-trap adducts.",
        "A2_cu_axial": "Axial Cu(II) anisotropic powder model.",
        "A3_vanadyl_axial": "Axial vanadyl/V(IV) anisotropic powder model.",
        "A5_mn2_highspin": "High-spin Mn(II) powder model.",
        "N1_n2rr_dmso_corrected": "DMSO-corrected PBN N2RR candidate screen.",
        "N2_n2rr_extended": "Extended N2RR candidate screen with O2 and diazene-like alternatives.",
        "M1_water_dmso": "Includes water/ROS and mandatory DMSO-derived PBN-CH3 adducts.",
        "M2_water_dmso_o2": "Adds PBN-OOH/O2- when air or oxygen contamination may contribute.",
        "M3_ros_spin_trap_full": "Full ROS spin-trap model with broad background, hydroxyl, DMSO, and O2 components.",
        "N1_n2rr_dmso_corrected": "Adds N2RR candidate PBN adducts after DMSO/water alternatives.",
        "N2_n2rr_extended": "Higher-complexity N2RR candidate model; useful only with strong controls.",
        "N3_n2rr_full": "Full N2RR candidate set including hydrazyl-like alternatives; highest overfitting risk.",
    }
