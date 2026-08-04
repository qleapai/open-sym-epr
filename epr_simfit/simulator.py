"""cw-EPR simulation: fast isotropic path + general anisotropic (powder) engine."""

from __future__ import annotations

import numpy as np

from .constants import DEFAULT_MW_FREQUENCY_GHZ, MU_B_OVER_H_GHZ_PER_T
from .lineshapes import derivative_lineshape
from .model_library import components_for_preset
from .spin_models import SpinComponent, generate_hyperfine_lines
from .utils import normalize_maxabs

# Default orientation count for powder averaging (balance of speed vs. accuracy).
DEFAULT_POWDER_ORIENTATIONS = 1000


def resonance_field_mT(mw_frequency_GHz: float, g: float) -> float:
    return float(mw_frequency_GHz) / (float(g) * MU_B_OVER_H_GHZ_PER_T) * 1000.0


def _A_mT_to_MHz(A_mT: float, g: float) -> float:
    """Convert a hyperfine coupling from field units (mT) to frequency units (MHz)."""
    return float(A_mT) * float(g) * (MU_B_OVER_H_GHZ_PER_T)  # GHz/T * mT -> MHz


def system_from_component(component: SpinComponent):
    """Build an anisotropic :class:`SpinSystem` from a :class:`SpinComponent`."""
    from .spin_hamiltonian import NuclearSpec, SpinSystem

    g_princ = component.g_principal()
    g_avg = sum(g_princ) / 3.0
    nuclei = []
    for nuc in component.nuclei:
        ax, ay, az = nuc.A_principal_mT()
        nuclei.append(NuclearSpec(
            I=nuc.spin,
            A_principal_MHz=(_A_mT_to_MHz(ax, g_avg), _A_mT_to_MHz(ay, g_avg), _A_mT_to_MHz(az, g_avg)),
            euler_deg=nuc.A_euler_deg,
            label=nuc.label or nuc.isotope,
        ))
    return SpinSystem(
        S=component.spin_S,
        g_principal=g_princ,
        nuclei=nuclei,
        D_MHz=component.D_MHz,
        E_MHz=component.E_MHz,
    )


def component_spectrum(
    field_mT,
    component: SpinComponent,
    mw_frequency_GHz: float = DEFAULT_MW_FREQUENCY_GHZ,
    lineshape: str = "pseudo-Voigt",
    n_orientations: int = DEFAULT_POWDER_ORIENTATIONS,
) -> np.ndarray:
    """Single-component spectrum, routed to the engine matching its physics."""
    field = np.asarray(field_mT, dtype=float)

    if component.is_anisotropic():
        from .powder import powder_spectrum
        system = system_from_component(component)
        return powder_spectrum(
            field, system, mw_frequency_GHz,
            linewidth_mT=component.linewidth_mT, eta=component.eta,
            n_orientations=n_orientations, derivative=True,
        )

    # Fast isotropic path
    b0 = resonance_field_mT(mw_frequency_GHz, component.g)
    shifts, intensities = generate_hyperfine_lines(component.nuclei)
    y = np.zeros_like(field)
    for shift, intensity in zip(shifts, intensities):
        y += intensity * derivative_lineshape(lineshape, field, b0 + shift, component.linewidth_mT, component.eta)
    return normalize_maxabs(y)


def simulate_components(
    field_mT,
    components: list[SpinComponent],
    mw_frequency_GHz: float = DEFAULT_MW_FREQUENCY_GHZ,
    lineshape: str = "pseudo-Voigt",
    n_orientations: int = DEFAULT_POWDER_ORIENTATIONS,
) -> dict[str, np.ndarray]:
    return {
        component.component_id: component_spectrum(
            field_mT, component, mw_frequency_GHz, lineshape, n_orientations
        )
        for component in components
    }


def simulate_model(
    field_mT,
    components: list[SpinComponent],
    weights: dict[str, float] | None = None,
    mw_frequency_GHz: float = DEFAULT_MW_FREQUENCY_GHZ,
    baseline0: float = 0.0,
    baseline1: float = 0.0,
    lineshape: str = "pseudo-Voigt",
    n_orientations: int = DEFAULT_POWDER_ORIENTATIONS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    field = np.asarray(field_mT, dtype=float)
    comps = simulate_components(field, components, mw_frequency_GHz, lineshape, n_orientations)
    center = float(np.mean(field)) if field.size else 0.0
    total = np.full_like(field, baseline0) + baseline1 * (field - center)
    weights = weights or {component.component_id: component.weight for component in components}
    for component in components:
        total += float(weights.get(component.component_id, component.weight)) * comps[component.component_id]
    return total, comps


def simulate_preset(field_mT, preset: str, mw_frequency_GHz: float = DEFAULT_MW_FREQUENCY_GHZ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = components_for_preset(preset)
    return simulate_model(field_mT, components, mw_frequency_GHz=mw_frequency_GHz)
