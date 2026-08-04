"""Physical constants (SI / spectroscopy units) used across Open-Sym-EPR."""

from __future__ import annotations

# Bohr magneton over Planck constant, MHz per Tesla.
BMAGN_MHZ_PER_T = 13996.24180856
# Same quantity in GHz per Tesla (convenient for field <-> frequency).
MU_B_OVER_H_GHZ_PER_T = 13.99624180856

# Nuclear magneton over Planck constant, MHz per Tesla.
NMAGN_MHZ_PER_T = 7.622593285

# Boltzmann constant in MHz per Kelvin (k_B / h), for thermodynamics.
KB_MHZ_PER_K = 20836.619123

# Free-electron g-factor.
G_FREE = 2.00231930436

# Numerical floor.
EPS = 1e-12


def A_mT_to_MHz(A_mT: float, g: float = 2.0) -> float:
    """Convert a hyperfine coupling from field units (mT) to frequency units (MHz)."""
    return float(A_mT) * float(g) * MU_B_OVER_H_GHZ_PER_T


def mhz_to_mT(freq_MHz: float, g: float = 2.0) -> float:
    """Convert a frequency (MHz) to the equivalent field (mT) for given g."""
    return float(freq_MHz) / (float(g) * MU_B_OVER_H_GHZ_PER_T)
