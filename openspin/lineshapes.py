"""Derivative cw-EPR line shapes."""

from __future__ import annotations

import numpy as np


def _norm(y: np.ndarray) -> np.ndarray:
    scale = np.nanmax(np.abs(y)) if y.size else 0.0
    if not np.isfinite(scale) or scale == 0:
        return y
    return y / scale


def derivative_lorentzian(x, center: float, linewidth: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    gamma = max(float(linewidth), 1e-6) / 2.0
    z = x - center
    y = -2.0 * z * gamma**2 / (z**2 + gamma**2) ** 2
    return _norm(y)


def derivative_gaussian(x, center: float, linewidth: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = max(float(linewidth), 1e-6) / 2.354820045
    z = x - center
    y = -(z / sigma**2) * np.exp(-0.5 * (z / sigma) ** 2)
    return _norm(y)


def derivative_pseudo_voigt(x, center: float, linewidth: float, eta: float = 0.5) -> np.ndarray:
    eta = min(max(float(eta), 0.0), 1.0)
    return _norm(eta * derivative_lorentzian(x, center, linewidth) + (1.0 - eta) * derivative_gaussian(x, center, linewidth))


def derivative_lineshape(kind: str, x, center: float, linewidth: float, eta: float = 0.5) -> np.ndarray:
    kind = (kind or "pseudo-Voigt").lower()
    if "lorentz" in kind:
        return derivative_lorentzian(x, center, linewidth)
    if "gauss" in kind:
        return derivative_gaussian(x, center, linewidth)
    return derivative_pseudo_voigt(x, center, linewidth, eta)
