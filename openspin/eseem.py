"""ESEEM: ``saffron`` (lite) -- electron spin-echo envelope modulation.

Native-Python ESEEM solver (``saffron``) for the most
common case: an S=1/2 electron coupled to an I=1/2 nucleus with an anisotropic
hyperfine tensor.  The echo envelope is modulated at the nuclear frequencies of
the two electron manifolds, from the standard two- and three-pulse ESEEM
formulae (Schweiger & Jeschke, *Principles of Pulse EPR*, 2001).

Per orientation, with field direction n in the hyperfine principal frame:

    a   = A . n                         (hyperfine field vector)
    A_s = n . a                         (secular term)
    B_p = sqrt(|a|^2 - A_s^2)            (pseudo-secular term)
    nu_a = sqrt((A_s/2 + nu_I)^2 + (B_p/2)^2)
    nu_b = sqrt((A_s/2 - nu_I)^2 + (B_p/2)^2)
    k    = (nu_I * B_p / (nu_a*nu_b))^2   (modulation depth)

Two-pulse echo vs tau and three-pulse echo vs T (fixed tau) are summed over a
powder grid.  Multi-nucleus modulation multiplies; HYSCORE/DEER are roadmap.
"""

from __future__ import annotations

import numpy as np

from .constants import NMAGN_MHZ_PER_T
from .powder import fibonacci_hemisphere
from .spin_hamiltonian import NUCLEAR_G_FACTORS, SpinSystem, euler_to_matrix


def _nuclear_freqs(A_tensor_MHz, gn, field_mT, direction):
    nu_I = abs(gn) * NMAGN_MHZ_PER_T * field_mT / 1000.0
    a = A_tensor_MHz @ direction
    A_s = float(direction @ a)
    B_p = float(np.sqrt(max(np.dot(a, a) - A_s * A_s, 0.0)))
    nu_a = np.sqrt((A_s / 2 + nu_I) ** 2 + (B_p / 2) ** 2)
    nu_b = np.sqrt((A_s / 2 - nu_I) ** 2 + (B_p / 2) ** 2)
    denom = nu_a * nu_b
    k = (nu_I * B_p / denom) ** 2 if denom > 1e-9 else 0.0
    return nu_a, nu_b, float(np.clip(k, 0.0, 1.0))


def _A_tensor(nuc) -> np.ndarray:
    A = np.diag(np.asarray(nuc.A_principal_MHz, dtype=float))
    a, b, c = np.radians(nuc.euler_deg)
    if a or b or c:
        R = euler_to_matrix(a, b, c)
        A = R @ A @ R.T
    return A


def two_pulse_eseem(
    system: SpinSystem,
    tau_us: np.ndarray,
    field_mT: float = 350.0,
    n_orientations: int = 800,
) -> np.ndarray:
    """Two-pulse ESEEM echo envelope V(tau), powder-averaged, normalised to 1 at tau=0."""
    tau = np.asarray(tau_us, dtype=float)
    dirs = fibonacci_hemisphere(n_orientations)
    nuclei = [(n, n.gn or NUCLEAR_G_FACTORS.get(_iso(n.label), 0.0)) for n in system.nuclei]
    nuclei = [(n, gn, _A_tensor(n)) for (n, gn) in nuclei if gn != 0.0]
    if not nuclei:
        return np.ones_like(tau)

    V = np.zeros_like(tau)
    for d in dirs:
        mod = np.ones_like(tau)
        for nuc, gn, A in nuclei:
            nu_a, nu_b, k = _nuclear_freqs(A, gn, field_mT, d)
            wa, wb = 2 * np.pi * nu_a * tau, 2 * np.pi * nu_b * tau
            wp, wm = wa + wb, wa - wb
            v = 1.0 - (k / 4.0) * (2 - 2 * np.cos(wa) - 2 * np.cos(wb) + np.cos(wp) + np.cos(wm))
            mod = mod * v
        V += mod
    V /= len(dirs)
    return V


def three_pulse_eseem(
    system: SpinSystem,
    T_us: np.ndarray,
    tau_us: float = 0.2,
    field_mT: float = 350.0,
    n_orientations: int = 800,
) -> np.ndarray:
    """Three-pulse (stimulated-echo) ESEEM V(T) at fixed tau, powder-averaged."""
    T = np.asarray(T_us, dtype=float)
    dirs = fibonacci_hemisphere(n_orientations)
    nuclei = [(n, n.gn or NUCLEAR_G_FACTORS.get(_iso(n.label), 0.0)) for n in system.nuclei]
    nuclei = [(n, gn, _A_tensor(n)) for (n, gn) in nuclei if gn != 0.0]
    if not nuclei:
        return np.ones_like(T)

    V = np.zeros_like(T)
    for d in dirs:
        mod = np.ones_like(T)
        for nuc, gn, A in nuclei:
            nu_a, nu_b, k = _nuclear_freqs(A, gn, field_mT, d)
            wa, wb = 2 * np.pi * nu_a, 2 * np.pi * nu_b
            ca_t, cb_t = np.cos(wa * tau_us), np.cos(wb * tau_us)
            term_a = (1 - cb_t) * (1 - np.cos(wa * (tau_us + T)))
            term_b = (1 - ca_t) * (1 - np.cos(wb * (tau_us + T)))
            v = 1.0 - (k / 4.0) * (term_a + term_b)
            mod = mod * v
        V += mod
    V /= len(dirs)
    return V


def eseem_fft(time_us: np.ndarray, echo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude FFT of an ESEEM trace -> (frequency MHz, amplitude)."""
    y = echo - np.mean(echo)
    n = len(y)
    dt = float(time_us[1] - time_us[0])
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freq = np.fft.rfftfreq(n, d=dt)  # MHz (since time in us)
    if spec.max() > 0:
        spec = spec / spec.max()
    return freq, spec


def _iso(label: str) -> str:
    import re
    m = re.match(r"\s*(\d+[A-Za-z]+)", str(label))
    return m.group(1) if m else str(label).strip()
