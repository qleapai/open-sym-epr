"""Synthetic demo data generation for the GUI and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_library import components_for_preset
from .simulator import simulate_model
from .utils import ensure_path, normalize_maxabs


DEMO_CONDITIONS = {
    "demo_generic_organic_radical.asc": ("generic_organic_radical", "G1_organic_radicals", {"organic_radical_singlet": 0.55, "carbon_centered_h": 0.30, "semiquinone_radical": 0.15}),
    "demo_generic_nitroxide_spin_label.asc": ("generic_nitroxide_spin_label", "G2_nitroxide_spin_label", {"nitroxide_14n": 0.80, "triphenylmethyl_trityl": 0.06}),
    "demo_generic_cu_mn_screen.asc": ("generic_cu_mn_screen", "G4_transition_metal_screen", {"cu2_isotropic": 0.55, "mn2_sixline": 0.20}),
    "demo_PBN_DMSO_water_light_blank.asc": ("PBN_DMSO_water_light_blank", "M1_water_dmso", {"pbn_oh": 0.30, "pbn_ch3_dmso": 0.55}),
}


def _header(condition: str) -> str:
    sample = condition.replace("_", " ")
    if condition.startswith("generic_organic"):
        return "\n".join(
            [
                "# Sample: generic organic radical mixture",
                "# Condition: room temperature solution cw-EPR",
                "# Solvent: user-defined solvent",
                "# Microwave Frequency: 9.85 GHz",
                "# Field unit: G",
                "# XPTS: 1800",
                "# Date: Open-Sym-EPR demo",
                "Field_G    Intensity",
            ]
        )
    if condition.startswith("generic_nitroxide"):
        return "\n".join(
            [
                "# Sample: generic nitroxide spin label",
                "# Condition: room temperature solution cw-EPR",
                "# Solvent: buffer or organic solvent",
                "# Microwave Frequency: 9.85 GHz",
                "# Field unit: G",
                "# XPTS: 1800",
                "# Date: Open-Sym-EPR demo",
                "Field_G    Intensity",
            ]
        )
    if condition.startswith("generic_cu"):
        return "\n".join(
            [
                "# Sample: generic transition-metal screening mixture",
                "# Condition: frozen solution or solid-state cw-EPR screening",
                "# Solvent: frozen glass / solid matrix",
                "# Microwave Frequency: 9.85 GHz",
                "# Field unit: G",
                "# XPTS: 1800",
                "# Date: Open-Sym-EPR demo",
                "Field_G    Intensity",
            ]
        )
    gas = "N2" if "N2" in condition else "Ar" if "Ar" in condition else "air/O2" if "air" in condition else "blank"
    light = "dark" if "dark" in condition else "light"
    return "\n".join(
        [
            "# Sample: Pd-Ov-Bi2MoO6 PBN spin trapping" if "PdOvBMO" in condition else f"# Sample: {sample}",
            f"# Condition: {gas} {light}",
            "# Solvent: water + DMSO PBN stock",
            "# Microwave Frequency: 9.85 GHz",
            "# Field unit: G",
            "# XPTS: 1800",
            "# Date: demo",
            "Field_G    Intensity",
        ]
    )


def generate_demo_text(condition: str, preset: str, weights: dict[str, float], seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    field_mT = np.linspace(345.0, 358.0, 1800)
    components = components_for_preset(preset)
    total, _ = simulate_model(field_mT, components, weights=weights, mw_frequency_GHz=9.85, baseline0=0.0, baseline1=0.0008)
    noise = rng.normal(0.0, 0.012, size=total.size)
    y = normalize_maxabs(total + noise)
    field_G = field_mT * 10.0
    rows = [f"{b:.6f}    {val:.9f}" for b, val in zip(field_G, y)]
    return _header(condition) + "\n" + "\n".join(rows) + "\n"


def generate_examples(out: str | Path) -> list[Path]:
    out_dir = ensure_path(out)
    written: list[Path] = []
    for i, (filename, (condition, preset, weights)) in enumerate(DEMO_CONDITIONS.items(), 1):
        path = out_dir / filename
        path.write_text(generate_demo_text(condition, preset, weights, seed=i), encoding="utf-8")
        written.append(path)
    conditions_csv = out_dir / "conditions_template.csv"
    conditions_csv.write_text(
        "filename,condition\n"
        + "\n".join(f"{name},{spec[0]}" for name, spec in DEMO_CONDITIONS.items())
        + "\n",
        encoding="utf-8",
    )
    readme = out_dir / "README_examples.md"
    readme.write_text(
        "# Open-Sym-EPR Demo EPR Data\n\nSynthetic cw-EPR ASC files for testing Open-Sym-EPR. "
        "The generic examples include organic radical, nitroxide/spin-label, and transition-metal screening spectra. "
        "Specialist PBN/DMSO examples are also included as optional spin-trapping demonstrations.\n",
        encoding="utf-8",
    )
    return written
