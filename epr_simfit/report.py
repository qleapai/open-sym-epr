"""Interpretation report generation."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from .about import DETAILED_INFO
from .model_suggester import DMSO_WARNING, ExperimentContext


DMSO_REPORT_WARNING = (
    "DMSO was included in the selected context. DMSO-derived radical or solvent-artifact "
    "components should be considered before assigning chemically specific spin adducts."
)


def generate_report_text(
    metadata: dict[str, Any],
    context: ExperimentContext,
    preprocessing_settings: dict[str, Any],
    selected_model: str,
    parameter_table: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    evidence_summary: dict[str, Any],
) -> str:
    lines = [
        "Open-Sym-EPR interpretation report",
        "=" * 40,
        DETAILED_INFO,
        "",
        "1. File metadata",
        str(metadata),
        "",
        "2. Parsed experiment details",
        f"Analysis mode: {context.analysis_mode}",
        f"Sample class: {context.sample_class}",
        f"Condition: {context.condition}",
        f"Catalyst/material: {context.catalyst}",
        f"Solvent: {context.solvent}",
        f"Spin-trap / DMSO-specific mode: {context.pbn_used}",
        "",
        "3. Solvent/artifact note",
        DMSO_REPORT_WARNING if context.has_dmso else "No DMSO-specific warning was triggered by the selected solvent/context.",
        "",
        "4. Preprocessing choices",
        str(preprocessing_settings),
        "",
        "5. Selected model",
        selected_model,
        "",
        "6. Fitted parameters",
        parameter_table.to_string(index=False) if parameter_table is not None and not parameter_table.empty else "No fit table available.",
        "",
        "7. Model comparison",
        comparison.to_string(index=False) if comparison is not None and not comparison.empty else "No comparison table available.",
        "",
        "8. Evidence class",
        evidence_summary.get("evidence_class", "INCONCLUSIVE"),
        "",
        "9. Explanation",
        evidence_summary.get("explanation", ""),
        "",
        "10. Limitations",
        "Open-Sym-EPR uses high-field isotropic screening models. Fit quality is not standalone chemical proof, and anisotropic tensor-resolved spectra require specialist EPR treatment.",
        "",
        "11. Recommended next controls",
        "Use relevant blanks, standards, solvent/matrix controls, isotope or substitution controls, repeated measurements, and independent chemical/product analysis where applicable.",
        "",
        "12. Suggested manuscript wording",
        evidence_summary.get("recommended_manuscript_wording", ""),
    ]
    return "\n".join(lines)


def generate_report_html(report_text: str) -> str:
    escaped = html.escape(report_text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Open-Sym-EPR report</title>"
        "<style>body{font-family:Arial,sans-serif;line-height:1.45;max-width:980px;margin:32px auto;color:#1f2933;}"
        "pre{white-space:pre-wrap;background:#f7f8fa;border:1px solid #d9dde3;padding:18px;border-radius:8px;}</style>"
        "</head><body><pre>"
        + escaped
        + "</pre></body></html>"
    )
