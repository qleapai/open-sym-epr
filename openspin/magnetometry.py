"""Magnetometry: ``curry`` -- magnetic susceptibility and magnetisation.

Native-Python magnetometry solver (``curry``).  From the spin-Hamiltonian
eigenvalues E_i(B) and the magnetic moments mu_i = -dE_i/dB, the molar magnetic
susceptibility and the magnetisation are obtained from the Boltzmann partition
function:

    Z      = sum_i exp(-E_i / kT)
    <mu>   = (1/Z) sum_i mu_i exp(-E_i / kT)
    M(B,T) = N_A <mu>                                   (molar magnetisation)
    chi    = N_A <mu> / B   (low-field)   ->   chi*T product

Everything is evaluated in CGS-emu so that chi*T comes out in cm^3 K mol^-1 and
M in Bohr magnetons, the conventional magnetochemistry units.  Powder averaging
makes the result valid for anisotropic g and zero-field-split high-spin systems.

Validation: for an isolated spin the high-temperature limit reproduces the Curie
law, chi*T = 0.12505 * g^2 * S(S+1) cm^3 K mol^-1 (0.375 for S=1/2, g=2).
"""

from __future__ import annotations

import numpy as np

from .powder import fibonacci_hemisphere
from .spin_hamiltonian import SpinSystem, build_field_independent, build_zeeman_direction

# CGS constants
NA = 6.02214076e23          # 1/mol
KB_ERG_PER_K = 1.380649e-16  # erg/K
# Convert an energy derivative dE/dB from MHz/T to a magnetic moment in erg/G.
MHZ_PER_T_TO_ERG_PER_G = 6.62607015e-21 / 1.0e4  # h[erg s]*1e6 / (1e4 G/T)
# Convert MHz to erg (h * 1e6).
MHZ_TO_ERG = 6.62607015e-21
MU_B_ERG_PER_G = 9.2740100783e-21  # Bohr magneton in CGS


def _orientation_energies_and_moments(
    H0: np.ndarray, Hz: np.ndarray, B_T: float, delta_T: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenenergies (MHz) at field B and magnetic moments mu_i = -dE_i/dB (MHz/T)."""
    e0 = np.linalg.eigvalsh(H0 + B_T * Hz)
    ep = np.linalg.eigvalsh(H0 + (B_T + delta_T) * Hz)
    em = np.linalg.eigvalsh(H0 + (B_T - delta_T) * Hz)
    moments = -(ep - em) / (2.0 * delta_T)  # MHz/T
    return e0, moments


def magnetisation_muB(
    system: SpinSystem,
    field_T: np.ndarray,
    temperature_K: float,
    n_orientations: int = 200,
) -> np.ndarray:
    """M(B) at fixed T in Bohr magnetons per formula unit (per molecule)."""
    field_T = np.asarray(field_T, dtype=float)
    H0 = build_field_independent(system)
    dirs = fibonacci_hemisphere(n_orientations)
    beta = 1.0 / (KB_ERG_PER_K * float(temperature_K))
    M = np.zeros_like(field_T)
    for d in dirs:
        Hz = build_zeeman_direction(system, d)
        for k, B in enumerate(field_T):
            e, mu = _orientation_energies_and_moments(H0, Hz, float(B))
            e_erg = e * MHZ_TO_ERG
            e_erg -= e_erg.min()
            w = np.exp(-beta * e_erg)
            mu_erg = mu * MHZ_PER_T_TO_ERG_PER_G
            M[k] += float(np.sum(w * mu_erg) / w.sum())
    M /= len(dirs)
    return M / MU_B_ERG_PER_G  # per-molecule moment in mu_B


def chiT_vs_T(
    system: SpinSystem,
    temperature_K: np.ndarray,
    measuring_field_T: float = 0.5,
    n_orientations: int = 200,
) -> np.ndarray:
    """chi*T product (cm^3 K mol^-1) vs temperature at a small measuring field."""
    temps = np.asarray(temperature_K, dtype=float)
    H0 = build_field_independent(system)
    dirs = fibonacci_hemisphere(n_orientations)
    B = float(measuring_field_T)
    B_G = B * 1.0e4  # Tesla -> Gauss

    chiT = np.zeros_like(temps)
    for d in dirs:
        Hz = build_zeeman_direction(system, d)
        e, mu = _orientation_energies_and_moments(H0, Hz, B)
        e_erg = (e * MHZ_TO_ERG)
        e_erg -= e_erg.min()
        mu_erg = mu * MHZ_PER_T_TO_ERG_PER_G  # erg/G
        for k, T in enumerate(temps):
            beta = 1.0 / (KB_ERG_PER_K * T)
            w = np.exp(-beta * e_erg)
            Mmol = NA * float(np.sum(w * mu_erg) / w.sum())  # emu/mol (erg/G/mol)
            chi = Mmol / B_G  # cm^3/mol
            chiT[k] += chi * T
    chiT /= len(dirs)
    return chiT
