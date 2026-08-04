"""Continuous-wave EPR simulation: ``garlic`` (isotropic) and ``pepper`` (powder).

Native-Python cw-EPR solvers:

- ``garlic`` -- fast isotropic / fast-motion solution spectra (scalar g, A).
- ``pepper`` -- solid-state powder / frozen-solution / single-crystal spectra
  with full anisotropic g- and A-tensors, high electron spin, and zero-field
  splitting, by spin-Hamiltonian diagonalisation and orientation averaging.

The two share the validated diagonalisation engine; ``garlic`` additionally has
an analytical fast path for the common isotropic case.
"""

from __future__ import annotations

import numpy as np

from .constants import A_mT_to_MHz, MU_B_OVER_H_GHZ_PER_T
from .lineshapes import derivative_lineshape
from .powder import powder_spectrum
from .spin_hamiltonian import SpinSystem


def isotropic_resonance_field_mT(mw_freq_GHz: float, g: float) -> float:
    """Centre resonance field (mT) for an isotropic g at frequency nu."""
    return float(mw_freq_GHz) / (float(g) * MU_B_OVER_H_GHZ_PER_T) * 1000.0


def _isotropic_hyperfine_pattern(system: SpinSystem) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic hyperfine line shifts (mT) and relative intensities.

    Uses the average of each nucleus' principal hyperfine values, converted from
    MHz back to mT with the average g.
    """
    g_avg = sum(system.g_principal) / 3.0
    shifts = np.array([0.0])
    intens = np.array([1.0])
    for nuc in system.nuclei:
        A_iso_MHz = float(np.mean(nuc.A_principal_MHz))
        A_iso_mT = A_iso_MHz / (g_avg * MU_B_OVER_H_GHZ_PER_T)
        count = int(round(2 * nuc.I + 1))
        m_vals = np.linspace(-nuc.I, nuc.I, count)
        nuc_shifts = m_vals * A_iso_mT
        shifts = (shifts[:, None] + nuc_shifts[None, :]).ravel()
        intens = (intens[:, None] * np.ones_like(nuc_shifts)[None, :]).ravel()
    intens = intens / intens.sum()
    return shifts, intens


def garlic(
    system: SpinSystem,
    field_mT: np.ndarray,
    mw_freq_GHz: float = 9.5,
    linewidth_mT: float = 0.1,
    eta: float = 0.5,
    lineshape: str = "pseudo-Voigt",
) -> np.ndarray:
    """Isotropic / fast-motion cw-EPR spectrum (first derivative, unit-normalised)."""
    field = np.asarray(field_mT, dtype=float)
    g_avg = sum(system.g_principal) / 3.0
    b0 = isotropic_resonance_field_mT(mw_freq_GHz, g_avg)
    shifts, intens = _isotropic_hyperfine_pattern(system)
    y = np.zeros_like(field)
    for s, w in zip(shifts, intens):
        y += w * derivative_lineshape(lineshape, field, b0 + s, linewidth_mT, eta)
    scale = np.nanmax(np.abs(y))
    return y / scale if scale > 0 else y


def pepper(
    system: SpinSystem,
    field_mT: np.ndarray,
    mw_freq_GHz: float = 9.5,
    linewidth_mT: float = 0.5,
    eta: float = 0.5,
    n_orientations: int = 2000,
    derivative: bool = True,
) -> np.ndarray:
    """Solid-state powder cw-EPR spectrum by diagonalisation + orientation averaging."""
    return powder_spectrum(
        np.asarray(field_mT, dtype=float), system, mw_freq_GHz,
        linewidth_mT=linewidth_mT, eta=eta, n_orientations=n_orientations,
        derivative=derivative,
    )


def cw_auto(
    system: SpinSystem,
    field_mT: np.ndarray,
    mw_freq_GHz: float = 9.5,
    linewidth_mT: float = 0.3,
    eta: float = 0.5,
    n_orientations: int = 2000,
) -> tuple[np.ndarray, str]:
    """Pick garlic (isotropic) or pepper (anisotropic/high-spin) automatically."""
    aniso = system.S > 0.5 or system.D_MHz or system.E_MHz
    if not aniso:
        gx, gy, gz = system.g_principal
        if abs(gx - gy) > 1e-6 or abs(gy - gz) > 1e-6:
            aniso = True
    if not aniso:
        for nuc in system.nuclei:
            ax, ay, az = nuc.A_principal_MHz
            if abs(ax - ay) > 1e-6 or abs(ay - az) > 1e-6:
                aniso = True
                break
    if aniso:
        return pepper(system, field_mT, mw_freq_GHz, linewidth_mT, eta, n_orientations), "pepper"
    return garlic(system, field_mT, mw_freq_GHz, linewidth_mT, eta), "garlic"
