"""Spin angular-momentum operators for arbitrary spin quantum numbers.

All operators are returned as dense complex NumPy matrices in the |S, m> basis
ordered from m = +S down to m = -S.  Matrices for multi-spin systems are built
with Kronecker (tensor) products.

References
----------
Sakurai & Napolitano, *Modern Quantum Mechanics* (angular momentum algebra).
Stoll & Schweiger, *J. Magn. Reson.* 178 (2006) 42 (EasySpin spin operators).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def spin_dim(spin: float) -> int:
    """Multiplicity 2S + 1 for a spin quantum number S."""
    return int(round(2.0 * float(spin) + 1.0))


@lru_cache(maxsize=64)
def _spin_matrices(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Sx, Sy, Sz) for a single spin S in the m = +S..-S basis.

    Uses the ladder-operator construction:
        S+ |S,m> = sqrt(S(S+1) - m(m+1)) |S,m+1>
        Sx = (S+ + S-) / 2,  Sy = (S+ - S-) / 2i
        Sz |S,m> = m |S,m>
    """
    s = float(spin)
    dim = spin_dim(s)
    # m values ordered descending: +S, S-1, ..., -S
    m = np.array([s - k for k in range(dim)], dtype=float)

    sz = np.diag(m).astype(complex)

    # Raising operator S+ connects |m> to |m+1>; in descending ordering the
    # element (i, i+1) raises m by +1 (row i has m_i, column i+1 has m_i - 1).
    sp = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        mi = m[i + 1]  # the lower m that gets raised into row i
        sp[i, i + 1] = np.sqrt(s * (s + 1.0) - mi * (mi + 1.0))
    sm = sp.conj().T

    sx = 0.5 * (sp + sm)
    sy = (sp - sm) / (2.0j)
    return sx, sy, sz


def spin_operators(spin: float) -> dict[str, np.ndarray]:
    """Return a dict with Sx, Sy, Sz, Sp, Sm, S2, and identity for spin S."""
    sx, sy, sz = _spin_matrices(spin)
    dim = sx.shape[0]
    return {
        "x": sx,
        "y": sy,
        "z": sz,
        "p": sx + 1j * sy,
        "m": sx - 1j * sy,
        "id": np.eye(dim, dtype=complex),
        "dim": dim,
    }


def embed(op: np.ndarray, position: int, dims: list[int]) -> np.ndarray:
    """Embed a single-particle operator into a multi-spin product space.

    Parameters
    ----------
    op : operator acting on the subspace at ``position``.
    position : index of the subspace the operator acts on.
    dims : list of subspace dimensions (one per spin), in order.
    """
    mats: list[np.ndarray] = []
    for i, d in enumerate(dims):
        mats.append(op if i == position else np.eye(d, dtype=complex))
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def vector_operators(spin: float, position: int, dims: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Sx, Sy, Sz) for the spin at ``position`` embedded in the full space."""
    ops = spin_operators(spin)
    sx = embed(ops["x"], position, dims)
    sy = embed(ops["y"], position, dims)
    sz = embed(ops["z"], position, dims)
    return sx, sy, sz
