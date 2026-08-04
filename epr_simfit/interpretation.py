"""Fit interpretation, publication-ready parameter tables, intermediate assignment, and improvement suggestions."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .fitter import FitResult


# ── Fit improvement suggestions ───────────────────────────────────────────────

@dataclass
class FitSuggestion:
    priority: str   # "High" / "Medium" / "Low" / "Info"
    category: str   # "component" / "bounds" / "mode" / "alignment" / "masking"
    icon: str
    title: str
    reason: str
    action: str


def suggest_fit_improvements(fit: "FitResult", fit_mode: str = "weights only") -> list[FitSuggestion]:
    """Return prioritised suggestions for improving the current fit."""
    suggestions: list[FitSuggestion] = []
    r2    = fit.metrics.get("R2", 0.0)
    nrmse = fit.metrics.get("normalized RMSE", 1.0)
    total_w = sum(max(0.0, w) for w in fit.weights.values())

    # 1 ── Poor R²: suggest adding components
    if r2 < 0.970:
        suggestions.append(FitSuggestion(
            priority="High", category="component", icon="🔴",
            title="Add more spectral components",
            reason=f"R² = {r2:.4f} — {(1-r2)*100:.1f}% of spectral variance is unexplained by the current model.",
            action="Inspect the residual for peaks above the noise floor. "
                   "Add the matching component in the **Extend model** section below or in the Model builder tab.",
        ))

    # 2 ── Moderate R² + weights-only mode: suggest linewidth optimisation
    if 0.900 <= r2 < 0.990 and "linewidth" not in fit_mode.lower():
        suggestions.append(FitSuggestion(
            priority="Medium", category="mode", icon="🟡",
            title="Try 'weights + linewidths' fit mode",
            reason=f"R² = {r2:.4f} and norm. RMSE = {nrmse:.4f}. Linewidths may not be at their optimum.",
            action="Change fit mode to **weights + linewidths** and rerun. Only upgrade to **+ g** if g-values are genuinely uncertain.",
        ))

    # 3 ── Parameter at bound
    for _, row in fit.parameters.iterrows():
        if row.get("hit_bound", False) and row.get("parameter") in ("linewidth_mT", "g"):
            lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
            suggestions.append(FitSuggestion(
                priority="Medium", category="bounds", icon="🟡",
                title=f"Parameter at bound: {row['component']} → {row['parameter']}",
                reason=f"Fitted value {float(row['value']):.4g} touches bound [{lo:.3g}, {hi:.3g}]. "
                       "The optimiser cannot move further — the result may be constrained.",
                action="Widen the parameter bounds in the Model builder, or verify that this component is correctly identified.",
            ))

    # 4 ── Low-weight component (noise fitting risk)
    for cid, w in fit.weights.items():
        frac = (w / total_w * 100.0) if total_w > 0 else 0.0
        if 0.0 <= frac < 2.0:
            suggestions.append(FitSuggestion(
                priority="Low", category="component", icon="🔵",
                title=f"Consider removing low-weight component: '{cid}'",
                reason=f"Spectral fraction {frac:.1f}% — contribution is negligible and may be noise-fitting.",
                action="Remove this component in the Model builder and refit. "
                       "Check if BIC improves (lower BIC after removal confirms it was unnecessary).",
            ))

    # 5 ── Large residual relative to signal
    if fit.residual.size:
        res_pp  = float(np.ptp(fit.residual))
        exp_pp  = float(np.ptp(fit.experimental))
        res_frac = res_pp / exp_pp if exp_pp > 0 else 0.0
        if res_frac > 0.15 and r2 > 0.85:
            suggestions.append(FitSuggestion(
                priority="Medium", category="component", icon="🟡",
                title="Significant residual features detected",
                reason=f"Residual peak-to-peak = {res_frac*100:.0f}% of experimental range — likely unmodelled signal.",
                action="Raise the **Noise floor %** slider to identify significant residual peaks. "
                       "Use **Extend model** below to add a matching component and refit.",
            ))

    # 6 ── Everything looks good
    if r2 >= 0.990 and nrmse <= 0.025 and not any(s.priority == "High" for s in suggestions):
        suggestions.append(FitSuggestion(
            priority="Info", category="quality", icon="✅",
            title="Fit quality is excellent",
            reason=f"R² = {r2:.4f}, norm. RMSE = {nrmse:.4f}. No critical issues detected.",
            action="Proceed to publication parameters and export. Remember to validate assignments chemically.",
        ))

    return suggestions


# ── Fit quality assessment ────────────────────────────────────────────────────

@dataclass
class FitQuality:
    label: str          # "Excellent" / "Good" / "Moderate" / "Poor"
    color: str          # Streamlit status color token
    icon: str
    r2_note: str
    nrmse_note: str
    overall: str


def assess_fit_quality(metrics: dict) -> FitQuality:
    """Return a structured quality assessment from fit metrics."""
    r2 = float(metrics.get("R2", 0.0))
    nrmse = float(metrics.get("normalized RMSE", 1.0))

    if r2 >= 0.990 and nrmse <= 0.025:
        return FitQuality(
            label="Excellent",
            color="success",
            icon="✅",
            r2_note=f"R² = {r2:.4f} — the model explains {r2*100:.2f}% of spectral variance (threshold for excellent: ≥ 0.990).",
            nrmse_note=f"Norm. RMSE = {nrmse:.4f} — residual amplitude is {nrmse*100:.1f}% of the peak-to-peak range (threshold: ≤ 0.025).",
            overall=(
                "The isotropic simulation describes the experimental spectrum with high fidelity. "
                "Spectral assignments may be reported with high confidence, "
                "subject to independent chemical validation."
            ),
        )
    if r2 >= 0.970 and nrmse <= 0.050:
        return FitQuality(
            label="Good",
            color="info",
            icon="ℹ️",
            r2_note=f"R² = {r2:.4f} — the model explains {r2*100:.2f}% of spectral variance (threshold for good: ≥ 0.970).",
            nrmse_note=f"Norm. RMSE = {nrmse:.4f} — residual amplitude is {nrmse*100:.1f}% of the peak-to-peak range (threshold: ≤ 0.050).",
            overall=(
                "The fit is good. Minor residuals may indicate small baseline imperfections, "
                "a missing minor component, or slight g/linewidth anisotropy not captured by the isotropic model."
            ),
        )
    if r2 >= 0.900 and nrmse <= 0.100:
        return FitQuality(
            label="Moderate",
            color="warning",
            icon="⚠️",
            r2_note=f"R² = {r2:.4f} — the model explains {r2*100:.2f}% of spectral variance (threshold for moderate: ≥ 0.900).",
            nrmse_note=f"Norm. RMSE = {nrmse:.4f} — residual amplitude is {nrmse*100:.1f}% of the peak-to-peak range (threshold: ≤ 0.100).",
            overall=(
                "The fit is moderate. Consider: (1) additional components, "
                "(2) improved baseline correction, (3) field re-alignment, "
                "(4) anisotropic powder simulation for solid/frozen-glass samples."
            ),
        )
    return FitQuality(
        label="Poor",
        color="error",
        icon="❌",
        r2_note=f"R² = {r2:.4f} — only {r2*100:.1f}% of spectral variance is explained by the model.",
        nrmse_note=f"Norm. RMSE = {nrmse:.4f} — residual amplitude is {nrmse*100:.0f}% of the peak-to-peak range.",
        overall=(
            "The fit is poor. The model is likely incomplete or incorrect. "
            "Check: wrong preset/components, poor preprocessing, large field offset, "
            "or a spectrum that requires anisotropic or powder-average treatment."
        ),
    )


# ── Intermediate / radical assignment ────────────────────────────────────────

@dataclass
class DetectedIntermediate:
    rank: int
    component_id: str
    name: str
    assignment: str
    category: str
    g: float
    linewidth_mT: float
    nuclei_str: str
    weight_abs: float
    fraction_pct: float
    confidence: str
    interpretation: str
    warning: str


def suggest_intermediates(fit: "FitResult", threshold_pct: float = 3.0) -> list[DetectedIntermediate]:
    """Return significant components ranked by spectral weight."""
    total = sum(max(0.0, w) for w in fit.weights.values())
    results: list[DetectedIntermediate] = []
    comp_by_id = {c.component_id: c for c in fit.components}
    for cid, w in fit.weights.items():
        frac = (w / total * 100.0) if total > 0 else 0.0
        if frac < threshold_pct:
            continue
        comp = comp_by_id.get(cid)
        if comp is None:
            continue
        nuclei_str = "; ".join(f"{n.isotope}: A = {n.A_mT:.3f} mT" for n in comp.nuclei) or "none"
        if frac >= 25:
            confidence = "High (dominant component)"
        elif frac >= 10:
            confidence = "Moderate (major component)"
        elif frac >= 5:
            confidence = "Tentative (minor component)"
        else:
            confidence = "Low (trace component)"
        results.append(DetectedIntermediate(
            rank=0,
            component_id=cid,
            name=comp.display_name,
            assignment=comp.radical_assignment,
            category=comp.category,
            g=comp.g,
            linewidth_mT=comp.linewidth_mT,
            nuclei_str=nuclei_str,
            weight_abs=float(w),
            fraction_pct=float(frac),
            confidence=confidence,
            interpretation=comp.interpretation,
            warning=comp.warning,
        ))
    results.sort(key=lambda x: -x.fraction_pct)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def intermediates_dataframe(intermediates: list[DetectedIntermediate]) -> pd.DataFrame:
    rows = []
    for d in intermediates:
        rows.append({
            "Rank": d.rank,
            "Component": d.name,
            "Assignment": d.assignment,
            "Category": d.category,
            "g-value": f"{d.g:.5f}",
            "ΔBpp (mT)": f"{d.linewidth_mT:.4f}",
            "Hyperfine A (mT)": d.nuclei_str,
            "Fraction (%)": f"{d.fraction_pct:.1f}",
            "Confidence": d.confidence,
        })
    return pd.DataFrame(rows)


# ── Publication-ready parameters ──────────────────────────────────────────────

def publication_parameters_table(fit: "FitResult") -> pd.DataFrame:
    """Return a publication-formatted parameter table (one row per component)."""
    total = sum(max(0.0, w) for w in fit.weights.values())
    # Merge fitted values from fit.parameters back into component objects
    fitted_vals: dict[tuple[str, str], float] = {}
    for _, row in fit.parameters.iterrows():
        fitted_vals[(str(row["component"]), str(row["parameter"]))] = float(row["value"])

    rows = []
    for rank, comp in enumerate(fit.components, 1):
        cid = comp.component_id
        w = fit.weights.get(cid, 0.0)
        frac = (w / total * 100.0) if total > 0 else 0.0
        g_fit = fitted_vals.get((cid, "g"), comp.g)
        lw_fit = fitted_vals.get((cid, "linewidth_mT"), comp.linewidth_mT)
        nuclei_parts = [f"{n.isotope}: {n.A_mT:.3f}" for n in comp.nuclei]
        nuclei_str = "; ".join(nuclei_parts) if nuclei_parts else "—"
        rows.append({
            "#": rank,
            "Component": comp.display_name,
            "Assignment": comp.radical_assignment,
            "g-value": round(g_fit, 5),
            "ΔBpp (mT)": round(lw_fit, 4),
            "η (L/G)": round(comp.eta, 2),
            "Hyperfine A (mT)": nuclei_str,
            "Weight (a.u.)": round(w, 4),
            "Fraction (%)": round(frac, 1),
            "Interpretation": comp.interpretation,
        })
    return pd.DataFrame(rows)


def publication_methods_paragraph(fit: "FitResult", mw_freq_GHz: float, field_shift_mT: float = 0.0) -> str:
    """Generate a ready-to-paste methods/supplementary paragraph."""
    metrics = fit.metrics
    r2 = metrics.get("R2", 0)
    nrmse = metrics.get("normalized RMSE", 0)
    aic = metrics.get("AIC", 0)
    bic = metrics.get("BIC", 0)
    n_comp = len(fit.components)
    total = sum(max(0.0, w) for w in fit.weights.values())

    comp_sentences = []
    for comp in fit.components:
        cid = comp.component_id
        w = fit.weights.get(cid, 0.0)
        frac = (w / total * 100.0) if total > 0 else 0.0
        nuclei_parts = [f"{n.isotope} hyperfine coupling A_iso = {n.A_mT:.3f} mT" for n in comp.nuclei]
        hf_str = (", with " + "; ".join(nuclei_parts)) if nuclei_parts else ""
        lineshape = (
            "Lorentzian" if comp.eta >= 0.9 else
            "Gaussian" if comp.eta <= 0.1 else
            f"pseudo-Voigt (η = {comp.eta:.2f})"
        )
        comp_sentences.append(
            f"Component {fit.components.index(comp)+1} ({comp.display_name}, {comp.radical_assignment}) "
            f"was modelled with g = {comp.g:.5f}, peak-to-peak linewidth ΔBpp = {comp.linewidth_mT:.4f} mT, "
            f"{lineshape} lineshape{hf_str}, contributing {frac:.1f}% of the total spectral intensity."
        )
    shift_note = (
        f" A field-axis calibration offset of {field_shift_mT:+.2f} mT was applied." if abs(field_shift_mT) > 0.01 else ""
    )
    para = textwrap.dedent(f"""
        EPR spectra were simulated using Open-Sym-EPR (isotropic high-field cw-EPR model, X-band,
        microwave frequency ν = {mw_freq_GHz:.4f} GHz).{shift_note}
        The experimental spectrum was decomposed into {n_comp} component{'s' if n_comp != 1 else ''}
        using bounded least-squares optimisation (scipy.optimize.least_squares).
        {' '.join(comp_sentences)}
        The overall fit quality was R² = {r2:.4f}, normalised RMSE = {nrmse:.4f},
        AIC = {aic:.1f}, BIC = {bic:.1f}, with {fit.n_parameters} free parameters.
        All spectral assignments are candidate identifications that require independent
        validation (isotope labelling, concentration series, chemical controls, or
        comparison with authentic standards).
    """).strip()
    return para


# ── Scientific background text (per section) ─────────────────────────────────

SCIENCE_IMPORT = """
**Continuous-wave (cw) EPR fundamentals**

EPR spectra are acquired as the **first derivative** of the microwave absorption with respect
to the applied magnetic field (dχ″/dB₀) using phase-sensitive lock-in detection. The x-axis
is the static field B₀ (mT or Gauss), and the y-axis is in arbitrary (instrumental) units.

**Resonance condition:**  B₀ = hν / (g μ_B) ≈ ν (GHz) / (g × 13.9962) × 1000 mT

At X-band (~9.5 GHz) a free electron (g ≈ 2.0023) resonates at ~339 mT.
Deviations from g_e indicate spin–orbit coupling (transition metals) or heteroatom character
(nitroxide, semiquinone).

**Data format:** Open-Sym-EPR accepts Bruker ASC / plain-text / CSV files with numeric field and
intensity columns. Field units are detected automatically (median value heuristic: > 1000 = Gauss).
"""

SCIENCE_PREPROCESS = """
**Why preprocessing matters**

1. **Baseline correction** — non-resonant microwave absorption by the sample cavity produces
   a slowly varying background superimposed on the EPR signal. Subtraction of a polynomial
   fit through the spectral wings (regions with no resonance signal) removes this artefact.
   *Linear edge* is suitable for most solution spectra; *polynomial edge* (order 2–4) handles
   curved solid-state backgrounds.

2. **Normalization** — scales intensities to a common reference so that components can be
   compared and fitted on equal footing.  *Max-absolute* is recommended for fitting;
   *area* normalization is appropriate when absolute spin concentrations are compared.

3. **Field alignment** — spectrometer field axes carry calibration uncertainties (typically
   0.01–1 mT).  A global field offset is corrected by shifting the field axis until the
   experimental dominant peak overlaps the simulated resonance field.  The shift is recorded
   and reported in the methods paragraph.
"""

SCIENCE_MODELBUILDER = """
**cw-EPR simulation — two engines, automatically selected**

Open-Sym-EPR routes each component to the engine matching its physics:

**1. Fast isotropic engine** — for freely tumbling radicals in solution (g and A scalar).
Resonance field per hyperfine line:

> B₀ (mT) = ν (GHz) / (g × 13.99624) × 1000

A nucleus of spin I splits the resonance into 2I + 1 equally spaced lines (spacing = A).

**2. General anisotropic engine** — full **spin-Hamiltonian diagonalisation** with **powder
averaging**, automatically used when the component has an anisotropic g-tensor (gx≠gy≠gz),
anisotropic hyperfine A-tensor, electron spin **S > ½**, or **zero-field splitting** (D, E).
The Hamiltonian (in frequency units) is:

> H/h = (μ_B/h) **B**·**g**·**S**  +  Σ_k **S**·**A**_k·**I**_k  +  D(S_z² − S(S+1)/3) + E(S_x² − S_y²)

For each molecular orientation, H(B) is diagonalised and EPR-allowed transitions are located
where an eigenstate gap equals hν; resonance fields are refined by bisection and weighted by the
orientation-dependent transition probability. Sticks from a near-uniform spherical grid
(Fibonacci) are accumulated and broadened. This is valid for **frozen solutions, powders,
rigid-limit nitroxides, transition-metal ions, and high-spin systems with zero-field splitting**.

**Multifrequency:** because the engine diagonalises the field-dependent Hamiltonian, anisotropic
patterns correctly scale with microwave frequency (X-, Q-, W-band).

**Supported nuclei:** ¹H, ²H, ¹³C, ¹⁴N, ¹⁵N, ¹⁹F, ³¹P, ²⁷Al, ⁵¹V, ⁵⁵Mn, ⁶³Cu, ⁶⁵Cu.

**Lineshape:** pseudo-Voigt broadening, S(B) = η·Lorentzian + (1−η)·Gaussian.

**Multi-component:** the total spectrum is the weighted sum of all component spectra; weights
represent relative spin concentrations (not absolute).
"""

SCIENCE_FIT = """
**Fitting algorithm and statistics**

Open-Sym-EPR minimises the **residual sum of squares** (RSS = Σᵢ (yᵢ_exp − yᵢ_model)²) using
bounded Levenberg–Marquardt / trust-region least squares (scipy.optimize.least_squares).
Physical bounds on all parameters (g-values, linewidths, weights ≥ 0) prevent unphysical solutions.

**Fit modes:**
| Mode | Free parameters | Use when |
|------|----------------|----------|
| Weights only | Component amplitudes | Model g and ΔBpp are well-known |
| + Linewidths | + ΔBpp per component | Linewidths are uncertain or concentration-dependent |
| + g | + g per component | g-values are uncertain (risk of overfitting — use carefully) |

**Quality metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | 1 − RSS/TSS | ≥ 0.990 excellent; ≥ 0.970 good; ≥ 0.900 moderate |
| Norm. RMSE | RMSE / (max − min) | < 0.025 excellent; < 0.050 good; < 0.10 moderate |
| AIC | n·ln(RSS/n) + 2k | Penalises parameters lightly; better for discovery |
| BIC | n·ln(RSS/n) + k·ln(n) | Penalises parameters heavily; better for model selection |

**ΔBIC interpretation (Kass & Raftery, 1995):** < 2 no evidence; 2–6 weak; 6–10 moderate; > 10 strong evidence for the lower-BIC model.
"""
