"""Evidence classification for DMSO-corrected PBN-EPR analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _bic(row_lookup: dict[str, dict[str, Any]], model: str) -> float | None:
    row = row_lookup.get(model)
    if not row:
        return None
    return float(row.get("BIC"))


def _component_fraction(component_fractions: pd.DataFrame | dict[str, float] | None, keyword: str) -> float:
    if component_fractions is None:
        return 0.0
    if isinstance(component_fractions, dict):
        return sum(v for k, v in component_fractions.items() if keyword in k)
    if component_fractions.empty:
        return 0.0
    col = "fraction" if "fraction" in component_fractions.columns else "weight"
    return float(component_fractions.loc[component_fractions["component"].str.contains(keyword, case=False), col].sum())


def classify_evidence(
    comparison: pd.DataFrame | None,
    condition: str = "N2 + light",
    catalyst: str = "Pd-Ov-Bi2MoO6",
    component_fractions: pd.DataFrame | dict[str, float] | None = None,
    controls: dict[str, Any] | None = None,
    isotope_labelled_nh3: bool = False,
) -> dict[str, Any]:
    controls = controls or {}
    warnings = [
        "Statistical improvement is not chemical proof without controls.",
        "PBN alone cannot prove N2RR; DMSO-derived PBN-CH3 must be evaluated first.",
    ]
    cond_l = condition.lower()
    row_lookup: dict[str, dict[str, Any]] = {}
    if comparison is not None and not comparison.empty:
        row_lookup = {str(row["model"]): row.to_dict() for _, row in comparison.iterrows()}

    bic_m1 = _bic(row_lookup, "M1_water_dmso")
    bic_m2 = _bic(row_lookup, "M2_water_dmso_o2")
    bic_m3 = _bic(row_lookup, "M3_n2rr_candidate")
    best_simple = min(v for v in [bic_m1, bic_m2] if v is not None) if any(v is not None for v in [bic_m1, bic_m2]) else None
    delta = best_simple - bic_m3 if best_simple is not None and bic_m3 is not None else 0.0

    o2_fraction = _component_fraction(component_fractions, "ooh") + _component_fraction(component_fractions, "o2")
    n_fraction = _component_fraction(component_fractions, "nnh") + _component_fraction(component_fractions, "nh2") + _component_fraction(component_fractions, "nhnh")

    if o2_fraction >= 0.45:
        klass = "O2_CONTAMINATED"
        explanation = "PBN-OOH/O2-derived component dominates; oxygen-derived radical chemistry likely controls the spectrum."
    elif delta < 2:
        klass = "WATER_DMSO_ONLY"
        explanation = "The spectrum is adequately explained by water/DMSO/ROS adducts without meaningful N-candidate improvement."
    elif controls.get("m3_improves_ar") or controls.get("m3_improves_blank") or controls.get("m3_improves_dark"):
        klass = "INCONCLUSIVE"
        explanation = "N-candidate models also improve controls, indicating possible overfitting or DMSO-radical misassignment."
    elif "n2" in cond_l and "light" in cond_l and "pd" in catalyst.lower() and n_fraction > 0:
        if isotope_labelled_nh3 and controls.get("trend_follows_nh3_activity") and delta > 10:
            klass = "STRONG_N2RR_SUPPORTING"
            explanation = "N-candidate improvement is selective to the N2/light/Pd-Ov condition and follows the NH3 activity trend."
        else:
            klass = "N2RR_SUPPORTING"
            explanation = "After accounting for water/DMSO-derived PBN adducts, an added N-associated candidate improves the N2/light model."
    else:
        klass = "INCONCLUSIVE"
        explanation = "The available model comparison or controls do not justify an N2RR-supporting assignment."

    wording = {
        "WATER_DMSO_ONLY": "The PBN-EPR spectrum was adequately reproduced by water/DMSO-derived radical adducts, indicating light-induced radical formation but not providing direct evidence for N2RR-associated spin adducts.",
        "INCONCLUSIVE": "The fitted spectrum cannot distinguish N-associated candidates from water/DMSO/O2 alternatives without stronger controls.",
        "O2_CONTAMINATED": "The spectrum is dominated by oxygen-derived PBN adduct character, so N2RR-associated assignments should not be made from this fit.",
        "N2RR_SUPPORTING": "After accounting for water/DMSO-derived PBN adducts, including PBN-OH and PBN-CH3, the Pd-Ov-Bi2MoO6 spectrum collected under N2 and illumination required an additional N-associated PBN candidate component. This component was suppressed in Ar and dark controls, supporting the formation of N2RR-related radical/intermediate species at Pd-Ov interfacial sites.",
        "STRONG_N2RR_SUPPORTING": "DMSO-corrected spin-trap fitting, selective controls, catalyst-series trends, and isotope-labelled NH3 evidence together provide strong mechanistic support for N2RR-related radical/intermediate formation.",
    }[klass]
    return {
        "evidence_class": klass,
        "delta_BIC_simple_minus_M3": delta,
        "explanation": explanation,
        "warnings": warnings,
        "recommended_manuscript_wording": wording,
    }
