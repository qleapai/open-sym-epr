"""Export helpers for fitted spectra and interpretation artifacts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .about import ABOUT_TEXT, DETAILED_INFO, SHORT_CREDIT
from .fitter import FitResult


def fit_components_dataframe(fit: FitResult) -> pd.DataFrame:
    data = {
        "Field_mT": fit.field_mT,
        "Experimental": fit.experimental,
        "Fit_total": fit.fit_total,
        "Residual": fit.residual,
    }
    for cid, curve in fit.component_curves.items():
        data[f"component_{cid}"] = fit.weights.get(cid, 1.0) * curve
    return pd.DataFrame(data)


def fit_metrics_dataframe(fit: FitResult) -> pd.DataFrame:
    rows = [{"metric": key, "value": value} for key, value in fit.metrics.items()]
    rows.extend(
        [
            {"metric": "baseline0", "value": fit.baseline0},
            {"metric": "baseline1", "value": fit.baseline1},
            {"metric": "n_parameters", "value": fit.n_parameters},
            {"metric": "fit_success", "value": fit.success},
            {"metric": "fit_message", "value": fit.message},
        ]
    )
    return pd.DataFrame(rows)


def fit_overlay_dataframe(fit: FitResult) -> pd.DataFrame:
    return pd.DataFrame({"Field_mT": fit.field_mT, "Experimental": fit.experimental, "Fit_total": fit.fit_total})


def residual_dataframe(fit: FitResult) -> pd.DataFrame:
    return pd.DataFrame({"Field_mT": fit.field_mT, "Residual": fit.residual})


def component_decomposition_dataframe(fit: FitResult) -> pd.DataFrame:
    data = {"Field_mT": fit.field_mT}
    for cid, curve in fit.component_curves.items():
        data[cid] = fit.weights.get(cid, 1.0) * curve
    return pd.DataFrame(data)


def orca_templates(fit: FitResult) -> dict[str, str]:
    templates: dict[str, str] = {}
    for component in fit.components:
        templates[f"ORCA_templates/{component.component_id}.inp"] = "\n".join(
            [
                "# Auto-generated Open-Sym-EPR ORCA EPR template",
                "# Provide an optimized structure before running.",
                "! UKS B3LYP EPRNMR def2-TZVP TightSCF",
                "%eprnmr",
                "  GTensor true",
                "  Nuclei = all N {aiso, adip, aorb}",
                "end",
                "* xyz 0 2",
                "# coordinates here",
                "*",
            ]
        )
    return templates


def _save_current_figure(fmt: str = "png", dpi: int = 600) -> bytes:
    buffer = io.BytesIO()
    plt.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close()
    return buffer.getvalue()


def _line_figure(x, ys: dict[str, object], title: str, ylabel: str = "Intensity / a.u.", fmt: str = "png") -> bytes:
    plt.figure(figsize=(7.0, 4.2))
    for name, y in ys.items():
        plt.plot(x, y, linewidth=1.1, label=name)
    plt.xlabel("Magnetic field / mT")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return _save_current_figure(fmt)


def _bar_figure(labels, values, title: str, ylabel: str, fmt: str = "png") -> bytes:
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return _save_current_figure(fmt)


def build_export_zip(
    fit: FitResult | None = None,
    processed: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    evidence_summary: dict | None = None,
    report_txt: str | None = None,
    report_html: str | None = None,
    config: dict | None = None,
    mw_frequency_GHz: float = 9.85,
) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zf:
        zf.writestr("about/developer_info.txt", DETAILED_INFO)
        zf.writestr("about/about.txt", ABOUT_TEXT)
        zf.writestr("about/short_credit.txt", SHORT_CREDIT)
        if processed is not None:
            zf.writestr("processed_data.csv", processed.to_csv(index=False))
            if {"Field_mT", "Intensity_raw", "Intensity_processed"}.issubset(processed.columns):
                x = processed["Field_mT"]
                zf.writestr("plot_data_csv/processed_spectrum_plot.csv", processed.to_csv(index=False))
                zf.writestr("plot_data_csv/raw_spectrum_plot.csv", processed[["Field_mT", "Intensity_raw"]].to_csv(index=False))
                for fmt in ["png", "svg", "pdf"]:
                    zf.writestr(
                        f"figures/processed_spectrum.{fmt}",
                        _line_figure(x, {"raw": processed["Intensity_raw"], "processed": processed["Intensity_processed"]}, "Processed EPR spectrum", fmt=fmt),
                    )
        if fit is not None:
            zf.writestr("fit_components.csv", fit_components_dataframe(fit).to_csv(index=False))
            zf.writestr("fit_metrics.csv", fit_metrics_dataframe(fit).to_csv(index=False))
            zf.writestr("fit_parameters.csv", fit.parameters.to_csv(index=False))
            zf.writestr("component_fractions.csv", fit.component_fractions.to_csv(index=False))
            zf.writestr("plot_data_csv/fit_overlay_plot.csv", fit_overlay_dataframe(fit).to_csv(index=False))
            zf.writestr("plot_data_csv/residual_plot.csv", residual_dataframe(fit).to_csv(index=False))
            zf.writestr("plot_data_csv/component_decomposition_plot.csv", component_decomposition_dataframe(fit).to_csv(index=False))
            zf.writestr("plot_data_csv/all_fit_curves_plot.csv", fit_components_dataframe(fit).to_csv(index=False))
            for fmt in ["png", "svg", "pdf"]:
                zf.writestr(
                    f"figures/fit_overlay.{fmt}",
                    _line_figure(fit.field_mT, {"experimental": fit.experimental, "fit": fit.fit_total}, "Fit overlay", fmt=fmt),
                )
                zf.writestr(
                    f"figures/residual.{fmt}",
                    _line_figure(fit.field_mT, {"residual": fit.residual}, "Residual", ylabel="Experimental - fit", fmt=fmt),
                )
                weighted = {cid: fit.weights.get(cid, 1.0) * curve for cid, curve in fit.component_curves.items()}
                zf.writestr(
                    f"figures/component_decomposition.{fmt}",
                    _line_figure(fit.field_mT, weighted, "Component decomposition", fmt=fmt),
                )
            for name, content in orca_templates(fit).items():
                zf.writestr(name, content)
        if comparison is not None:
            zf.writestr("model_comparison.csv", comparison.to_csv(index=False))
            zf.writestr("fit_metrics_all_models.csv", comparison.to_csv(index=False))
            if not comparison.empty and "BIC" in comparison:
                zf.writestr("plot_data_csv/model_comparison_bic_plot.csv", comparison[["model", "BIC"]].to_csv(index=False))
                available = [col for col in ["model", "RSS", "RMSE", "normalized RMSE", "R2", "AIC", "BIC", "Delta_AIC", "Delta_BIC"] if col in comparison.columns]
                zf.writestr("plot_data_csv/model_comparison_all_metrics_plot.csv", comparison[available].to_csv(index=False))
                for fmt in ["png", "svg", "pdf"]:
                    zf.writestr(
                        f"figures/model_comparison.{fmt}",
                        _bar_figure(comparison["model"], comparison["BIC"], "Model comparison", "BIC", fmt=fmt),
                    )
        if evidence_summary is not None:
            zf.writestr("evidence_summary.json", json.dumps(evidence_summary, indent=2))
        if report_txt is not None:
            zf.writestr("interpretation_report.txt", report_txt)
        if report_html is not None:
            zf.writestr("interpretation_report.html", report_html)
        if config is not None:
            zf.writestr("config.json", json.dumps(config, indent=2, default=str))
    return buffer.getvalue()


def write_text_exports(out_dir: str | Path, files: dict[str, str]) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = p / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
