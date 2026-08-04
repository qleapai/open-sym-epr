"""Least-squares fitting of spin-Hamiltonian models to experimental data.

Native-Python least-squares fitter (``esfit``) for cw-EPR: the user chooses
which spin-system parameters to vary (g, hyperfine A, zero-field splitting D/E,
linewidth, amplitude scale, field-calibration shift); Open-Sym-EPR minimises the
residual to the experimental spectrum with bounded least squares, and reports
fitted values with standard errors, residuals, and goodness-of-fit metrics
(R^2, RMSE, AIC, BIC).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import A_mT_to_MHz, MU_B_OVER_H_GHZ_PER_T
from .cw import cw_auto
from .spin_hamiltonian import SpinSystem


@dataclass
class FitResult:
    system: SpinSystem
    field_mT: np.ndarray
    experimental: np.ndarray
    fit: np.ndarray
    residual: np.ndarray
    params: pd.DataFrame          # name, value, std_error, lower, upper
    metrics: dict
    scale: float
    field_shift_mT: float
    engine: str
    success: bool
    message: str
    n_parameters: int
    mc_errors: pd.DataFrame | None = None   # Monte-Carlo error estimates (if requested)
    mc_samples: np.ndarray | None = None     # raw refit samples (n_realisations, n_params)


# Parameter names the GUI can offer:
#   g_iso, gx, gy, gz, lw, shift, scale, D, E, A<i>_iso, A<i>_x, A<i>_y, A<i>_z
def default_vary(system: SpinSystem) -> dict[str, tuple[float, float, float]]:
    """Suggest a sensible starting set of (initial, lo, hi) for common parameters."""
    g0 = sum(system.g_principal) / 3.0
    vary: dict[str, tuple[float, float, float]] = {
        "g_iso": (g0, g0 - 0.02, g0 + 0.02),
        "lw": (0.3, 0.02, 5.0),
        "shift": (0.0, -3.0, 3.0),
        "scale": (1.0, 0.1, 5.0),
    }
    for i, nuc in enumerate(system.nuclei):
        a_iso_mT = float(np.mean(nuc.A_principal_MHz)) / (g0 * MU_B_OVER_H_GHZ_PER_T)
        vary[f"A{i}_iso"] = (a_iso_mT, max(a_iso_mT - 1.0, 0.0), a_iso_mT + 1.0)
    return vary


def _apply(base: SpinSystem, names: list[str], x: np.ndarray):
    """Apply a parameter vector to a copy of the base system; return knobs."""
    system = deepcopy(base)
    gx, gy, gz = system.g_principal
    lw, shift, scale = 0.3, 0.0, 1.0
    vals = dict(zip(names, x))

    if "g_iso" in vals:
        gx = gy = gz = vals["g_iso"]
    gx = vals.get("gx", gx); gy = vals.get("gy", gy); gz = vals.get("gz", gz)
    system.g_principal = (gx, gy, gz)
    g_avg = (gx + gy + gz) / 3.0

    if "D" in vals:
        system.D_MHz = vals["D"]
    if "E" in vals:
        system.E_MHz = vals["E"]

    for i, nuc in enumerate(system.nuclei):
        ax, ay, az = nuc.A_principal_MHz
        if f"A{i}_iso" in vals:
            a = A_mT_to_MHz(vals[f"A{i}_iso"], g_avg)
            ax = ay = az = a
        if f"A{i}_x" in vals:
            ax = A_mT_to_MHz(vals[f"A{i}_x"], g_avg)
        if f"A{i}_y" in vals:
            ay = A_mT_to_MHz(vals[f"A{i}_y"], g_avg)
        if f"A{i}_z" in vals:
            az = A_mT_to_MHz(vals[f"A{i}_z"], g_avg)
        nuc.A_principal_MHz = (ax, ay, az)

    lw = vals.get("lw", lw)
    shift = vals.get("shift", shift)
    scale = vals.get("scale", scale)
    return system, lw, shift, scale


def _std_errors(result, n_data: int) -> np.ndarray:
    try:
        jac = np.asarray(result.jac, dtype=float)
        k = jac.shape[1]
        dof = max(n_data - k, 1)
        sigma2 = float(2.0 * result.cost) / dof
        _, s, VT = np.linalg.svd(jac, full_matrices=False)
        thr = np.finfo(float).eps * max(jac.shape) * (s[0] if s.size else 0.0)
        inv = np.array([1.0 / (v * v) if v > thr else 0.0 for v in s])
        cov = (VT.T * inv) @ VT * sigma2
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:  # noqa: BLE001
        return np.full(len(getattr(result, "x", [])), np.nan)


def esfit(
    system: SpinSystem,
    field_mT: np.ndarray,
    experimental: np.ndarray,
    vary: dict[str, tuple[float, float, float]],
    mw_freq_GHz: float = 9.5,
    eta: float = 0.5,
    n_orientations: int = 400,
    max_nfev: int = 200,
    global_search: int = 0,
    seed: int = 0,
    n_monte_carlo: int = 0,
    mc_method: str = "gaussian",
) -> FitResult:
    """Fit selected spin-system parameters to an experimental cw-EPR spectrum.

    ``global_search`` > 0 runs that many random pre-search samples (Latin-style)
    over the bounds and starts local refinement from the best, which makes the
    fit robust to poor initial guesses -- essential for derivative cw-EPR spectra
    whose cost landscape has no gradient when lines do not yet overlap.

    ``n_monte_carlo`` > 0 estimates parameter uncertainties by **bootstrap /
    Monte-Carlo**: that many synthetic spectra are generated from the best fit
    plus noise and refitted; the spread of refits gives standard errors and
    confidence intervals that capture nonlinearity and parameter correlations,
    unlike the linearised (asymptotic) Jacobian-covariance errors.
    ``mc_method``: "gaussian" (add Gaussian noise of the residual std) or
    "residual" (resample the fit residuals with replacement, distribution-free).
    """
    field = np.asarray(field_mT, dtype=float)
    y = np.asarray(experimental, dtype=float)
    names = list(vary.keys())

    # Validate parameter names up front. esfit applies only the knobs it knows;
    # an unrecognised key (e.g. a typo or wrong nucleus-index convention) would
    # otherwise be silently optimised as a no-op dummy, corrupting the fit while
    # still reporting a value. Fail loudly instead.
    _valid = {"g_iso", "gx", "gy", "gz", "lw", "shift", "scale", "D", "E"}
    for _i in range(len(system.nuclei)):
        _valid |= {f"A{_i}_iso", f"A{_i}_x", f"A{_i}_y", f"A{_i}_z"}
    _unknown = [n for n in names if n not in _valid]
    if _unknown:
        raise ValueError(
            f"esfit: unrecognised vary parameter(s) {_unknown}. "
            f"Per-nucleus hyperfine uses the 'A<index>_iso' convention "
            f"(e.g. 'A0_iso', 'A1_iso'); valid keys here: {sorted(_valid)}."
        )

    x0 = np.array([vary[n][0] for n in names], dtype=float)
    lo = np.array([vary[n][1] for n in names], dtype=float)
    hi = np.array([vary[n][2] for n in names], dtype=float)

    def simulate(x):
        sysx, lw, shift, scale = _apply(system, names, x)
        spec, engine = cw_auto(sysx, field, mw_freq_GHz, lw, eta, n_orientations)
        spec_shifted = np.interp(field, field + shift, spec)
        return scale * spec_shifted, engine

    def residual(x):
        sim, _ = simulate(x)
        return sim - y

    # Global pre-search: sample the bounded box and keep the best start.
    # Scoring is done on the *integrated absorption* envelope (the cumulative
    # integral of the derivative spectrum), which is smooth and positive, so a
    # rough alignment still produces a usable gradient -- this makes the fit
    # robust to poor initial guesses, the key difficulty for derivative cw-EPR.
    def _absorption_envelope(d: np.ndarray) -> np.ndarray:
        a = np.cumsum(d - np.mean(d))
        a = a - np.linspace(a[0], a[-1], a.size)  # remove linear drift
        s = np.max(np.abs(a))
        return a / s if s > 0 else a

    if global_search and global_search > 0:
        rng = np.random.default_rng(seed)
        y_env = _absorption_envelope(y)

        def env_rss(cand):
            sim, _ = simulate(cand)
            return float(np.sum((_absorption_envelope(sim) - y_env) ** 2))

        best_rss, best_x = env_rss(x0), x0
        samples = lo + rng.random((global_search, len(names))) * (hi - lo)
        for cand in samples:
            rss = env_rss(cand)
            if rss < best_rss:
                best_rss, best_x = rss, cand
        x0 = best_x

    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev)
    fit_spec, engine = simulate(result.x)
    res = y - fit_spec
    se = _std_errors(result, y.size)

    rows = []
    final = dict(zip(names, result.x))
    for n, v, e, l, h in zip(names, result.x, se, lo, hi):
        rows.append({"parameter": n, "value": float(v),
                     "std_error": float(e) if np.isfinite(e) else float("nan"),
                     "lower": float(l), "upper": float(h)})
    params = pd.DataFrame(rows)

    rss = float(np.sum(res ** 2))
    nobs = y.size
    rmse = float(np.sqrt(rss / nobs))
    denom = float(np.max(y) - np.min(y)) or 1.0
    tss = float(np.sum((y - np.mean(y)) ** 2)) or 1e-12
    k = len(names)
    metrics = {
        "RSS": rss, "RMSE": rmse, "normalized RMSE": rmse / denom,
        "R2": 1.0 - rss / tss,
        "AIC": nobs * np.log(max(rss, 1e-12) / nobs) + 2 * k,
        "BIC": nobs * np.log(max(rss, 1e-12) / nobs) + k * np.log(nobs),
    }

    fitted_system, _, shift, scale = _apply(system, names, result.x)

    # ── Optional bootstrap / Monte-Carlo uncertainties ────────────────────────
    mc_errors = None
    mc_samples = None
    if n_monte_carlo and n_monte_carlo > 0:
        rng = np.random.default_rng(seed + 1)
        sigma = float(np.std(res))
        x_best = result.x.copy()
        samples = np.empty((n_monte_carlo, len(names)), dtype=float)
        for r in range(n_monte_carlo):
            if mc_method == "residual":
                noise = rng.choice(res, size=res.size, replace=True)
            else:  # gaussian
                noise = rng.normal(0.0, sigma, size=res.size)
            y_synth = fit_spec + noise

            def residual_mc(x, _t=y_synth):
                sim, _ = simulate(x)
                return sim - _t

            try:
                rr = least_squares(residual_mc, x_best, bounds=(lo, hi), max_nfev=max(max_nfev // 2, 60))
                samples[r] = rr.x
            except Exception:  # noqa: BLE001
                samples[r] = x_best
        mc_samples = samples
        mc_rows = []
        for j, n in enumerate(names):
            col = samples[:, j]
            mc_rows.append({
                "parameter": n,
                "value": float(x_best[j]),
                "mc_std": float(np.std(col)),
                "ci_2.5%": float(np.percentile(col, 2.5)),
                "ci_97.5%": float(np.percentile(col, 97.5)),
                "linear_std_error": float(se[j]) if np.isfinite(se[j]) else float("nan"),
            })
        mc_errors = pd.DataFrame(mc_rows)

    return FitResult(
        system=fitted_system, field_mT=field, experimental=y, fit=fit_spec, residual=res,
        params=params, metrics=metrics, scale=float(scale), field_shift_mT=float(shift),
        engine=engine, success=bool(result.success), message=str(result.message), n_parameters=k,
        mc_errors=mc_errors, mc_samples=mc_samples,
    )
