"""Spin component data models and hyperfine splitting generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


NUCLEAR_SPINS = {
    "14N": 1.0,
    "15N": 0.5,
    "1H": 0.5,
    "2H": 1.0,
    "13C": 0.5,
    "19F": 0.5,
    "31P": 0.5,
    "63Cu": 1.5,
    "65Cu": 1.5,
    "55Mn": 2.5,
    "51V": 3.5,
    "27Al": 2.5,
}


class Nucleus(BaseModel):
    isotope: str
    A_mT: float
    label: str = ""
    bounds: tuple[float, float] | None = None
    # Optional anisotropic hyperfine principal values (mT). When set, these
    # override the isotropic A_mT in the anisotropic engine.
    A_tensor_mT: tuple[float, float, float] | None = None
    A_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def spin(self) -> float:
        if self.isotope not in NUCLEAR_SPINS:
            raise ValueError(f"Unsupported nucleus isotope: {self.isotope}")
        return NUCLEAR_SPINS[self.isotope]

    def A_principal_mT(self) -> tuple[float, float, float]:
        if self.A_tensor_mT is not None:
            return tuple(float(a) for a in self.A_tensor_mT)  # type: ignore[return-value]
        return (float(self.A_mT), float(self.A_mT), float(self.A_mT))


class SpinComponent(BaseModel):
    component_id: str
    display_name: str
    radical_assignment: str
    category: str
    g: float = 2.0055
    g_bounds: tuple[float, float] = (2.002, 2.008)
    nuclei: list[Nucleus] = Field(default_factory=list)
    linewidth_mT: float = 0.08
    linewidth_bounds: tuple[float, float] = (0.03, 0.5)
    eta: float = 0.5
    eta_bounds: tuple[float, float] = (0.0, 1.0)
    weight: float = 1.0
    interpretation: str = ""
    warning: str = ""
    # ── Anisotropic / high-spin extensions (all optional, isotropic by default) ──
    spin_S: float = 0.5
    g_tensor: tuple[float, float, float] | None = None  # (gx, gy, gz)
    D_MHz: float = 0.0                                   # axial zero-field splitting
    E_MHz: float = 0.0                                   # rhombic zero-field splitting
    mode: str = "auto"   # "auto" | "isotropic" | "powder"

    def is_anisotropic(self) -> bool:
        """True when tensor anisotropy, high spin, or ZFS makes the full engine required."""
        if self.mode == "powder":
            return True
        if self.mode == "isotropic":
            return False
        if self.spin_S > 0.5:
            return True
        if self.D_MHz != 0.0 or self.E_MHz != 0.0:
            return True
        if self.g_tensor is not None:
            gx, gy, gz = self.g_tensor
            if abs(gx - gy) > 1e-6 or abs(gy - gz) > 1e-6:
                return True
        for n in self.nuclei:
            if n.A_tensor_mT is not None:
                ax, ay, az = n.A_tensor_mT
                if abs(ax - ay) > 1e-9 or abs(ay - az) > 1e-9:
                    return True
        return False

    def g_principal(self) -> tuple[float, float, float]:
        if self.g_tensor is not None:
            return tuple(float(g) for g in self.g_tensor)  # type: ignore[return-value]
        return (float(self.g), float(self.g), float(self.g))

    def clone(self) -> "SpinComponent":
        return deepcopy(self)

    def to_table_row(self) -> dict[str, Any]:
        nuclei = "; ".join(f"{n.label or n.isotope} {n.isotope} A={n.A_mT:.3g} mT" for n in self.nuclei) or "none"
        return {
            "component ID": self.component_id,
            "name": self.display_name,
            "radical assignment": self.radical_assignment,
            "category": self.category,
            "g": self.g,
            "nuclei": nuclei,
            "linewidth mT": self.linewidth_mT,
            "eta": self.eta,
            "weight": self.weight,
            "interpretation": self.interpretation,
            "warning": self.warning,
        }


def _m_values(spin: float) -> np.ndarray:
    count = int(round(2 * spin + 1))
    return np.linspace(-spin, spin, count)


def generate_hyperfine_lines(nuclei: list[Nucleus], merge_tolerance: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Generate isotropic hyperfine field shifts and relative intensities."""
    shifts = np.array([0.0])
    intensities = np.array([1.0])
    for nucleus in nuclei:
        m_vals = _m_values(nucleus.spin)
        nuc_shifts = m_vals * float(nucleus.A_mT)
        nuc_intensity = np.ones_like(nuc_shifts, dtype=float)
        shifts = (shifts[:, None] + nuc_shifts[None, :]).ravel()
        intensities = (intensities[:, None] * nuc_intensity[None, :]).ravel()

    if shifts.size == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(shifts)
    shifts = shifts[order]
    intensities = intensities[order]
    merged_shifts: list[float] = []
    merged_intensities: list[float] = []
    for shift, intensity in zip(shifts, intensities):
        if merged_shifts and abs(shift - merged_shifts[-1]) <= merge_tolerance:
            total = merged_intensities[-1] + intensity
            merged_shifts[-1] = (merged_shifts[-1] * merged_intensities[-1] + shift * intensity) / total
            merged_intensities[-1] = total
        else:
            merged_shifts.append(float(shift))
            merged_intensities.append(float(intensity))
    out_i = np.asarray(merged_intensities, dtype=float)
    out_i = out_i / out_i.sum()
    return np.asarray(merged_shifts, dtype=float), out_i
