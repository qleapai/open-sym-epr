"""Open-Sym-EPR — native-Python EPR/ENDOR/ESEEM/magnetometry simulation.

An open re-implementation of native solvers (garlic, pepper, salt,
saffron, curry) by direct spin-Hamiltonian diagonalisation. No MATLAB required.
"""

from __future__ import annotations

from .constants import A_mT_to_MHz, mhz_to_mT
from .spin_hamiltonian import NUCLEAR_G_FACTORS, NuclearSpec, SpinSystem
from .cw import garlic, pepper, cw_auto, isotropic_resonance_field_mT
from .endor import endor_spectrum, larmor_frequency_MHz
from .eseem import two_pulse_eseem, three_pulse_eseem, eseem_fft
from .magnetometry import chiT_vs_T, magnetisation_muB
from .fitting import esfit, default_vary, FitResult
from .io import parse_spectrum, normalise, to_dataframe
from .bruker import load_bes3t, load_bes3t_files, bruker_to_text

__version__ = "0.1.0"


def nucleus(isotope: str, A_mT: float | tuple = 0.0, *,
            A_tensor_mT: tuple[float, float, float] | None = None,
            euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
            g_for_conversion: float = 2.0) -> NuclearSpec:
    """Build a :class:`NuclearSpec` from an isotope and hyperfine in **mT**.

    Either give a scalar ``A_mT`` (isotropic) or ``A_tensor_mT=(Ax,Ay,Az)``.
    The nuclear spin and g-factor are looked up from the isotope.
    """
    from .spin_operators import spin_dim  # noqa: F401
    from .spin_hamiltonian import NUCLEAR_G_FACTORS

    I = _nuclear_spin(isotope)
    gn = NUCLEAR_G_FACTORS.get(isotope, 0.0)
    if A_tensor_mT is not None:
        ax, ay, az = (A_mT_to_MHz(a, g_for_conversion) for a in A_tensor_mT)
    else:
        a = A_mT_to_MHz(float(A_mT), g_for_conversion)
        ax = ay = az = a
    return NuclearSpec(I=I, A_principal_MHz=(ax, ay, az), euler_deg=euler_deg,
                       label=isotope, gn=gn)


def spin_system(g=2.0023, nuclei=None, S: float = 0.5,
                D_MHz: float = 0.0, E_MHz: float = 0.0) -> SpinSystem:
    """Convenience builder for a :class:`SpinSystem`.

    ``g`` may be a scalar or a (gx, gy, gz) tuple.
    """
    if isinstance(g, (int, float)):
        g_princ = (float(g), float(g), float(g))
    else:
        g_princ = tuple(float(x) for x in g)  # type: ignore[assignment]
    return SpinSystem(S=S, g_principal=g_princ, nuclei=list(nuclei or []),
                      D_MHz=D_MHz, E_MHz=E_MHz)


_NUCLEAR_SPINS = {
    "1H": 0.5, "2H": 1.0, "13C": 0.5, "14N": 1.0, "15N": 0.5, "19F": 0.5,
    "31P": 0.5, "27Al": 2.5, "51V": 3.5, "55Mn": 2.5, "63Cu": 1.5, "65Cu": 1.5,
}


def _nuclear_spin(isotope: str) -> float:
    if isotope not in _NUCLEAR_SPINS:
        raise ValueError(f"Unknown isotope '{isotope}'. Known: {sorted(_NUCLEAR_SPINS)}")
    return _NUCLEAR_SPINS[isotope]
