"""Bounded least-squares fitting of Open-Sym-EPR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import EPS
from .model_library import components_for_preset
from .simulator import simulate_model
from .spin_models import SpinComponent


@dataclass
class FitResult:
    preset: str
    components: list[SpinComponent]
    field_mT: np.ndarray
    experimental: np.ndarray
    fit_total: np.ndarray
    component_curves: dict[str, np.ndarray]
    residual: np.ndarray
    weights: dict[str, float]
    baseline0: float
    baseline1: float
    parameters: pd.DataFrame
    metrics: dict[str, float]
    success: bool
    message: str
    n_parameters: int
    mc_errors: pd.DataFrame | None = None   # Monte-Carlo / bootstrap uncertainties

    @property
    def weight_errors(self) -> dict[str, float]:
        """Standard error of each component weight, keyed by component id."""
        errors: dict[str, float] = {}
        if self.parameters is not None and "std_error" in self.parameters.columns:
            for _, row in self.parameters.iterrows():
                if row.get("parameter") == "weight":
                    errors[str(row.get("component"))] = float(row.get("std_error", float("nan")))
        return errors

    @property
    def component_fractions(self) -> pd.DataFrame:
        total = sum(max(0.0, w) for w in self.weights.values())
        errs = self.weight_errors
        rows = []
        for cid, weight in self.weights.items():
            we = errs.get(cid, float("nan"))
            # Fraction uncertainty (first-order): treat weight errors as independent.
            frac = weight / total if total > 0 else 0.0
            frac_err = (we / total) if (total > 0 and np.isfinite(we)) else float("nan")
            rows.append(
                {
                    "component": cid,
                    "weight": weight,
                    "weight_std_error": we,
                    "fraction": frac,
                    "fraction_std_error": frac_err,
                }
            )
        return pd.DataFrame(rows)


def _parameter_std_errors(result, n_data: int) -> np.ndarray:
    """Standard errors of the fitted parameters from the least-squares Jacobian.

    Covariance C = sigma^2 (J^T J)^{-1} with sigma^2 = RSS / (N - k).  Returns
    sqrt(diag(C)); entries are NaN where the covariance is undefined (e.g. a
    parameter that did not influence the residual, or N <= k).
    """
    try:
        jac = np.asarray(result.jac, dtype=float)
        k = jac.shape[1]
        dof = max(n_data - k, 1)
        rss = float(2.0 * result.cost)  # scipy cost = 0.5 * sum(residual^2)
        sigma2 = rss / dof
        # Pseudo-inverse of J^T J via SVD for numerical stability.
        _, s, VT = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * (s[0] if s.size else 0.0)
        s_inv2 = np.array([1.0 / (sv * sv) if sv > threshold else 0.0 for sv in s])
        cov = (VT.T * s_inv2) @ VT * sigma2
        var = np.diag(cov)
        return np.sqrt(np.clip(var, 0.0, None))
    except Exception:  # noqa: BLE001
        return np.full(len(getattr(result, "x", [])), np.nan)


def fit_metrics(y: np.ndarray, yhat: np.ndarray, k: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    residual = y - yhat
    rss = float(np.sum(residual**2))
    n = max(len(y), 1)
    rmse = float(np.sqrt(rss / n))
    denom = float(np.max(y) - np.min(y)) if len(y) else 0.0
    nrmse = rmse / denom if denom > EPS else rmse
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > EPS else 0.0
    safe_rss = max(rss, EPS)
    aic = float(n * np.log(safe_rss / n) + 2 * k)
    bic = float(n * np.log(safe_rss / n) + k * np.log(n))
    return {"RSS": rss, "RMSE": rmse, "normalized RMSE": nrmse, "R2": r2, "AIC": aic, "BIC": bic}


def _pack_initial(components: list[SpinComponent], mode: str, baseline_order: int) -> tuple[list[float], list[float], list[float], list[tuple[str, str, Any]]]:
    x0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    spec: list[tuple[str, str, Any]] = []
    for component in components:
        x0.append(max(component.weight, 0.0))
        lo.append(0.0)
        hi.append(np.inf)
        spec.append((component.component_id, "weight", None))
    x0.append(0.0)
    lo.append(-np.inf)
    hi.append(np.inf)
    spec.append(("baseline", "constant", None))
    if baseline_order >= 1:
        x0.append(0.0)
        lo.append(-np.inf)
        hi.append(np.inf)
        spec.append(("baseline", "linear", None))

    mode_l = mode.lower()
    if "linewidth" in mode_l:
        for component in components:
            x0.append(component.linewidth_mT)
            lo.append(component.linewidth_bounds[0])
            hi.append(component.linewidth_bounds[1])
            spec.append((component.component_id, "linewidth_mT", None))
    if "+ g" in mode_l or mode_l.endswith("g"):
        for component in components:
            x0.append(component.g)
            lo.append(component.g_bounds[0])
            hi.append(component.g_bounds[1])
            spec.append((component.component_id, "g", None))
    return x0, lo, hi, spec


def _apply_params(components: list[SpinComponent], x: np.ndarray, spec: list[tuple[str, str, Any]]) -> tuple[dict[str, float], float, float]:
    weights: dict[str, float] = {}
    baseline0 = 0.0
    baseline1 = 0.0
    by_id = {component.component_id: component for component in components}
    for value, (cid, param, _) in zip(x, spec):
        if param == "weight":
            weights[cid] = float(value)
        elif param == "constant":
            baseline0 = float(value)
        elif param == "linear":
            baseline1 = float(value)
        elif param == "linewidth_mT":
            by_id[cid].linewidth_mT = float(value)
        elif param == "g":
            by_id[cid].g = float(value)
    return weights, baseline0, baseline1


def fit_spectrum(
    field_mT,
    intensity,
    preset: str = "M1_water_dmso",
    components: list[SpinComponent] | None = None,
    mw_frequency_GHz: float = 9.85,
    mode: str = "weights only",
    baseline_order: int = 0,
    max_nfev: int = 400,
    n_orientations: int = 600,
    n_monte_carlo: int = 0,
    mc_method: str = "gaussian",
) -> FitResult:
    field = np.asarray(field_mT, dtype=float)
    y = np.asarray(intensity, dtype=float)
    components = [c.clone() for c in (components or components_for_preset(preset))]
    x0, lo, hi, spec = _pack_initial(components, mode, baseline_order)

    # Anisotropic components are expensive: use a reduced orientation grid during
    # the iterative fit, then a final pass at the requested resolution.
    any_aniso = any(c.is_anisotropic() for c in components)
    fit_orient = min(n_orientations, 400) if any_aniso else n_orientations

    def model_at(x: np.ndarray) -> np.ndarray:
        local_components = [c.clone() for c in components]
        weights, b0, b1 = _apply_params(local_components, x, spec)
        yhat_x, _ = simulate_model(field, local_components, weights, mw_frequency_GHz, b0, b1, n_orientations=fit_orient)
        return yhat_x

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return model_at(x) - y

    result = least_squares(residual_fn, np.asarray(x0), bounds=(np.asarray(lo), np.asarray(hi)), max_nfev=max_nfev)
    weights, baseline0, baseline1 = _apply_params(components, result.x, spec)
    yhat, curves = simulate_model(field, components, weights, mw_frequency_GHz, baseline0, baseline1, n_orientations=n_orientations)
    residual = y - yhat

    # ── Parameter standard errors from the covariance matrix ──────────────────
    # cov = sigma^2 (J^T J)^-1, sigma^2 = RSS / (N - k); SE_i = sqrt(cov_ii).
    std_errors = _parameter_std_errors(result, y.size)

    params = []
    for value, lower, upper, se, (cid, param, _) in zip(result.x, lo, hi, std_errors, spec):
        hit_bound = (np.isfinite(lower) and abs(value - lower) < 1e-5) or (np.isfinite(upper) and abs(value - upper) < 1e-5)
        params.append(
            {
                "component": cid,
                "parameter": param,
                "value": float(value),
                "std_error": float(se) if np.isfinite(se) else float("nan"),
                "lower_bound": lower,
                "upper_bound": upper,
                "fitted_or_fixed": "fitted",
                "hit_bound": bool(hit_bound),
            }
        )
    metrics = fit_metrics(y, yhat, len(result.x))

    # ── Optional bootstrap / Monte-Carlo uncertainties ────────────────────────
    # Refit synthetic spectra (best fit + noise) to obtain standard deviations and
    # confidence intervals that capture nonlinearity and parameter correlations,
    # unlike the linearised Jacobian-covariance errors above.
    mc_errors = None
    if n_monte_carlo and n_monte_carlo > 0:
        rng = np.random.default_rng(0)
        sigma = float(np.std(residual))
        x_best = result.x
        samples = np.empty((n_monte_carlo, len(x_best)), dtype=float)
        for r in range(n_monte_carlo):
            if mc_method == "residual":
                noise = rng.choice(residual, size=residual.size, replace=True)
            else:
                noise = rng.normal(0.0, sigma, size=residual.size)
            y_synth = yhat + noise
            try:
                rr = least_squares(lambda x, _t=y_synth: model_at(x) - _t,
                                   x_best, bounds=(np.asarray(lo), np.asarray(hi)),
                                   max_nfev=max(max_nfev // 2, 80))
                samples[r] = rr.x
            except Exception:  # noqa: BLE001
                samples[r] = x_best
        mc_rows = []
        for j, (cid, param, _) in enumerate(spec):
            col = samples[:, j]
            mc_rows.append({
                "component": cid, "parameter": param, "value": float(x_best[j]),
                "mc_std": float(np.std(col)),
                "ci_2.5%": float(np.percentile(col, 2.5)),
                "ci_97.5%": float(np.percentile(col, 97.5)),
                "linear_std_error": float(std_errors[j]) if np.isfinite(std_errors[j]) else float("nan"),
            })
        mc_errors = pd.DataFrame(mc_rows)

    return FitResult(
        preset=preset,
        components=components,
        field_mT=field,
        experimental=y,
        fit_total=yhat,
        component_curves=curves,
        residual=residual,
        weights=weights,
        baseline0=baseline0,
        baseline1=baseline1,
        parameters=pd.DataFrame(params),
        metrics=metrics,
        success=bool(result.success),
        message=str(result.message),
        n_parameters=len(result.x),
        mc_errors=mc_errors,
    )
