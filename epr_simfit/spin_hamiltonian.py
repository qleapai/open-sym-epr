"""General spin Hamiltonian construction for cw-EPR simulation.

The Hamiltonian is built in **frequency units (MHz)** in the coupled product
basis of one electron spin S and an arbitrary number of nuclei I_k:

    H/h = (mu_B/h) * B * [ l gx Sx + m gy Sy + n gz Sz ]      (electron Zeeman)
        + sum_k  S . A_k . I_k                                  (hyperfine)
        + D (Sz^2 - S(S+1)/3) + E (Sx^2 - Sy^2)                 (zero-field splitting)
        + sum_<ij> J_ij  S_i . S_j                              (isotropic exchange, optional)

Tensors g and A are specified by their principal values.  An optional set of
ZYZ Euler angles rotates each hyperfine tensor relative to the g principal
frame, allowing non-collinear g/A systems.  This covers the great majority of
cw-EPR systems: organic radicals, nitroxides, transition-metal ions (including
high-spin S > 1/2 with zero-field splitting), and triplet/biradical states.

References
----------
Stoll, Schweiger, J. Magn. Reson. 178 (2006) 42.
Abragam, Bleaney, *Electron Paramagnetic Resonance of Transition Ions* (1970).
Rieger, *Electron Spin Resonance: Analysis and Interpretation* (2007).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spin_operators import embed, spin_dim, spin_operators

# Bohr magneton over Planck constant, MHz per Tesla.
BMAGN_MHZ_PER_T = 13996.24180856


@dataclass
class NuclearSpec:
    """One nucleus: spin I and hyperfine principal values (MHz)."""
    I: float
    A_principal_MHz: tuple[float, float, float]
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # ZYZ, A frame rel. to g frame
    label: str = ""


@dataclass
class SpinSystem:
    """A general electron-spin system with anisotropic interactions."""
    S: float = 0.5
    g_principal: tuple[float, float, float] = (2.0023, 2.0023, 2.0023)
    nuclei: list[NuclearSpec] = field(default_factory=list)
    D_MHz: float = 0.0
    E_MHz: float = 0.0
    g_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # reserved, g frame = lab ref

    @property
    def dims(self) -> list[int]:
        return [spin_dim(self.S)] + [spin_dim(n.I) for n in self.nuclei]

    @property
    def hilbert_dim(self) -> int:
        d = 1
        for x in self.dims:
            d *= x
        return d


def euler_to_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZYZ active rotation matrix from Euler angles in radians."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    return Rz1 @ Ry @ Rz2


def _electron_ops(system: SpinSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ops = spin_operators(system.S)
    dims = system.dims
    sx = embed(ops["x"], 0, dims)
    sy = embed(ops["y"], 0, dims)
    sz = embed(ops["z"], 0, dims)
    return sx, sy, sz


def _nuclear_ops(system: SpinSystem, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nuc = system.nuclei[k]
    ops = spin_operators(nuc.I)
    dims = system.dims
    pos = k + 1  # electron occupies position 0
    ix = embed(ops["x"], pos, dims)
    iy = embed(ops["y"], pos, dims)
    iz = embed(ops["z"], pos, dims)
    return ix, iy, iz


def build_field_independent(system: SpinSystem) -> np.ndarray:
    """Field-independent part: hyperfine + zero-field splitting (MHz)."""
    n = system.hilbert_dim
    H = np.zeros((n, n), dtype=complex)
    sx, sy, sz = _electron_ops(system)
    S = [sx, sy, sz]

    # Hyperfine: S . A . I (with optional A-tensor rotation into the g frame)
    for k, nuc in enumerate(system.nuclei):
        ix, iy, iz = _nuclear_ops(system, k)
        I = [ix, iy, iz]
        A_diag = np.diag(np.asarray(nuc.A_principal_MHz, dtype=float))
        a, b, c = np.radians(nuc.euler_deg)
        if a or b or c:
            R = euler_to_matrix(a, b, c)
            A = R @ A_diag @ R.T
        else:
            A = A_diag
        for p in range(3):
            for q in range(3):
                if A[p, q] != 0.0:
                    H = H + A[p, q] * (S[p] @ I[q])

    # Zero-field splitting (only meaningful for S > 1/2)
    if system.S > 0.5 and (system.D_MHz != 0.0 or system.E_MHz != 0.0):
        SS = system.S * (system.S + 1.0)
        ident = np.eye(n, dtype=complex)
        H = H + system.D_MHz * (sz @ sz - (SS / 3.0) * ident)
        H = H + system.E_MHz * (sx @ sx - sy @ sy)

    return H


def build_zeeman_direction(system: SpinSystem, direction: np.ndarray) -> np.ndarray:
    """Field-linear electron-Zeeman operator for a unit field direction (MHz per Tesla).

    H_zeeman(B) = B[T] * build_zeeman_direction(direction)
    """
    l, m, n = direction
    sx, sy, sz = _electron_ops(system)
    gx, gy, gz = system.g_principal
    return BMAGN_MHZ_PER_T * (l * gx * sx + m * gy * sy + n * gz * sz)


def transition_operator(system: SpinSystem, direction: np.ndarray) -> np.ndarray:
    """Electron spin component along a microwave (B1) direction, for transition moments."""
    l, m, n = direction
    sx, sy, sz = _electron_ops(system)
    # The microwave couples through the electron magnetic moment (g-weighted).
    gx, gy, gz = system.g_principal
    return l * gx * sx + m * gy * sy + n * gz * sz
