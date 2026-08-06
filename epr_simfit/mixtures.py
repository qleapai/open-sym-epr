"""Spin-adduct mixtures with adjustable ratios (Open-Sym-EPR Pro).

Compose any set of spin-trap adduct (or other) components at user-defined ratios,
simulate the composite spectrum and its stacked contributions, and fit a mixture to
an experimental spectrum to recover the component **ratios** (normalised to 100 %).
Built on the validated forward model and least-squares fitter, so recovered ratios
carry the same R^2 and uncertainty machinery as any other fit.
"""
from __future__ import annotations

import numpy as np

from .simulator import simulate_model
from .fitter import fit_spectrum

# Centre-field relation g = PLANCK_GHz_mT * nu_GHz / B_mT   (h / muB, unit-scaled to GHz & mT)
PLANCK_GHz_mT = 6.62607015e-34 * 1e12 / 9.2740100783e-24  # ~71.447


def normalise_ratios(ratios: dict[str, float]) -> dict[str, float]:
    """Normalise a {component_id: ratio} mapping to fractions summing to 1."""
    total = float(sum(max(0.0, v) for v in ratios.values())) or 1.0
    return {k: max(0.0, v) / total for k, v in ratios.items()}


def simulate_mixture(field_mT, components, ratios: dict[str, float] | None = None,
                     mw_frequency_GHz: float = 9.85, n_orientations: int = 1000,
                     normalize: bool = True):
    """Simulate a weighted mixture of components at given ratios.

    Returns (composite, per_component) where per_component maps id -> weighted curve.
    """
    if ratios is None:
        ratios = {c.component_id: c.weight for c in components}
    weights = normalise_ratios(ratios) if normalize else dict(ratios)
    total, parts = simulate_model(field_mT, components, weights=weights,
                                  mw_frequency_GHz=mw_frequency_GHz, n_orientations=n_orientations)
    weighted = {cid: weights.get(cid, 0.0) * parts.get(cid, np.zeros_like(field_mT)) for cid in weights}
    return total, weighted


def _seed_grid(components, field_mT, intensity, mw_frequency_GHz, n_orientations):
    """Position-seeded starts: shift every component's g so the modelled centre lands
    on the experimental centre-of-mass, and scan a small window around it. Narrow-line
    cw spectra have many local minima; seeding the centre makes the fit reproducible
    (the same idea esfit uses with multiple starting points)."""
    field = np.asarray(field_mT, float)
    y = np.asarray(intensity, float)
    absy = np.abs(y)
    b_exp = float(np.sum(field * absy) / (np.sum(absy) or 1.0))     # experimental centroid
    seeds = []
    for db in (-0.6, -0.3, 0.0, 0.3, 0.6):                          # mT window around centroid
        variant = []
        for c in components:
            cc = c.clone()
            # g that places this component's centre at (b_exp + db)
            b0 = b_exp + db
            if b0 > 0:
                cc.g = float(PLANCK_GHz_mT * mw_frequency_GHz / b0)
                lo, hi = cc.g * 0.985, cc.g * 1.015
                cc.g_bounds = (min(lo, cc.g_bounds[0]), max(hi, cc.g_bounds[1]))
            variant.append(cc)
        seeds.append(variant)
    return seeds


def fit_mixture_ratios(field_mT, intensity, components, mw_frequency_GHz: float = 9.85,
                       baseline_order: int = 0, max_nfev: int = 400, n_orientations: int = 600,
                       mode: str = "weights only", seeded: bool = True):
    """Fit a mixture of the given components to a spectrum and recover the ratios.

    ``mode`` selects what is refined alongside the component weights (ratios):
    ``"weights only"`` fixes every spin-Hamiltonian parameter; add ``"linewidths"``,
    ``"g"``, and/or ``"hyperfine"`` (esfit-style a-value refinement) for a full fit
    that can match an experimental triplet whose splitting differs from the library
    defaults. ``seeded`` runs several position-seeded starts and keeps the best R^2,
    which makes narrow-line fits reproducible. Returns the FitResult, recovered
    fractions (summing to 1), percentages, the fit R^2, and the refined per-component
    hyperfine (a-values, in Gauss).
    """
    def _run(comps):
        return fit_spectrum(field_mT, intensity, components=comps,
                            mw_frequency_GHz=mw_frequency_GHz, mode=mode,
                            baseline_order=baseline_order, max_nfev=max_nfev,
                            n_orientations=n_orientations)

    if seeded and "g" in mode.lower():
        fit = None
        for variant in _seed_grid(components, field_mT, intensity, mw_frequency_GHz, n_orientations):
            cand = _run(variant)
            if fit is None or cand.metrics.get("R2", -np.inf) > fit.metrics.get("R2", -np.inf):
                fit = cand
    else:
        fit = _run([c.clone() for c in components])
    weights = {cid: float(max(0.0, w)) for cid, w in fit.weights.items()}
    fractions = normalise_ratios(weights)
    hyperfine_G = {}
    if fit.parameters is not None and "parameter" in getattr(fit.parameters, "columns", []):
        for _, row in fit.parameters[fit.parameters["parameter"] == "A_mT"].iterrows():
            hyperfine_G.setdefault(row["component"], []).append(round(float(row["value"]) * 10.0, 2))
    return {
        "fit": fit,
        "fractions": fractions,
        "percent": {cid: 100.0 * f for cid, f in fractions.items()},
        "R2": float(fit.metrics.get("R2", float("nan"))),
        "hyperfine_G": hyperfine_G,
    }


def mixture_table(components, fractions: dict[str, float]):
    """Human-readable rows: component name, assignment, and model fraction (%)."""
    rows = []
    for c in components:
        f = fractions.get(c.component_id, 0.0)
        rows.append({"component": c.display_name, "assignment": c.radical_assignment,
                     "fraction_%": round(100.0 * f, 1)})
    rows.sort(key=lambda r: -r["fraction_%"])
    return rows
