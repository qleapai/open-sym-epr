"""ENDOR: ``salt`` -- electron-nuclear double resonance spectra.

Native-Python ENDOR solver (``salt``).  At a fixed magnetic field set
within the EPR line, an RF field drives nuclear transitions.  For each molecular
orientation the spin Hamiltonian H(B0) is diagonalised and the ENDOR lines are
the frequency differences between eigenstates connected by a nuclear-spin
operator, weighted by the RF transition probability.  Powder (orientation)
averaging gives the ENDOR powder pattern.

For a simple S=1/2, I=1/2 system the two ENDOR lines appear near |nu_n +/- A/2|
(weak coupling) or |A/2 +/- nu_n| (strong coupling), where nu_n is the nuclear
Larmor frequency -- the textbook ENDOR result, reproduced here from full
diagonalisation so it generalises to anisotropic A, quadrupole-free high-I, and
multi-nucleus systems.
"""

from __future__ import annotations

import numpy as np

from .powder import _perpendicular_basis, fibonacci_hemisphere
from .spin_hamiltonian import (
    SpinSystem,
    _nuclear_ops,
    build_field_independent,
    build_nuclear_zeeman_direction,
    build_zeeman_direction,
)


def _nuclear_direction_operator(ix, iy, iz, direction):
    l, m, n = direction
    return l * ix + m * iy + n * iz


def endor_spectrum(
    system: SpinSystem,
    field_mT: float,
    rf_MHz: np.ndarray,
    mw_freq_GHz: float = 9.5,           # informational; ENDOR is at fixed field
    linewidth_MHz: float = 0.1,
    n_orientations: int = 1500,
    rf_max_MHz: float = 200.0,
) -> np.ndarray:
    """Powder ENDOR spectrum (intensity vs RF frequency, MHz).

    Parameters
    ----------
    field_mT : the static field at which ENDOR is recorded.
    rf_MHz   : RF frequency axis (MHz).
    rf_max_MHz : ignore transitions above this (separates nuclear from electron).
    """
    rf = np.asarray(rf_MHz, dtype=float)
    if rf.size < 2:
        return np.zeros_like(rf)
    H0 = build_field_independent(system)
    B_T = float(field_mT) / 1000.0
    dirs = fibonacci_hemisphere(n_orientations)

    spec = np.zeros_like(rf)
    step = float(rf[1] - rf[0])
    n_nuc = len(system.nuclei)
    if n_nuc == 0:
        return spec

    for d in dirs:
        Hz = build_zeeman_direction(system, d) + build_nuclear_zeeman_direction(system, d)
        w, V = np.linalg.eigh(H0 + B_T * Hz)
        u, v = _perpendicular_basis(d)
        # accumulate over all nuclei
        for k in range(n_nuc):
            ix, iy, iz = _nuclear_ops(system, k)
            Iu = _nuclear_direction_operator(ix, iy, iz, u)
            Iv = _nuclear_direction_operator(ix, iy, iz, v)
            Mu = V.conj().T @ Iu @ V
            Mv = V.conj().T @ Iv @ V
            prob = np.abs(Mu) ** 2 + np.abs(Mv) ** 2
            n = w.size
            for i in range(n - 1):
                for j in range(i + 1, n):
                    freq = abs(w[j] - w[i])
                    if freq > rf_max_MHz or freq < step:
                        continue
                    p = prob[i, j]
                    if p < 1e-10:
                        continue
                    pos = (freq - rf[0]) / step
                    lo = int(np.floor(pos))
                    frac = pos - lo
                    if 0 <= lo < rf.size:
                        spec[lo] += p * (1 - frac)
                    if 0 <= lo + 1 < rf.size:
                        spec[lo + 1] += p * frac
    # broaden with a Gaussian
    sigma = max(linewidth_MHz, step) / 2.3548
    half = int(max(4 * sigma / step, 3))
    x = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    spec = np.convolve(spec, kernel, mode="same")
    scale = np.nanmax(np.abs(spec))
    return spec / scale if scale > 0 else spec


def larmor_frequency_MHz(isotope_gn: float, field_mT: float) -> float:
    """Nuclear Larmor frequency (MHz) = (g_n mu_N / h) * B."""
    from .constants import NMAGN_MHZ_PER_T
    return abs(isotope_gn) * NMAGN_MHZ_PER_T * field_mT / 1000.0
