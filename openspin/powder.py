"""Powder averaging and resonance-field calculation by matrix diagonalization.

For each molecular orientation relative to the static field B0, the spin
Hamiltonian H(B) = H0 + B * Hz is diagonalised on a coarse field grid.  Allowed
EPR transitions are located where the gap between two eigenstates equals the
microwave quantum h*nu; the exact resonance field is refined by bisection and
the transition probability is evaluated from the eigenvectors.  Sticks from all
orientations are accumulated and convolved with the line broadening to give the
powder (or single-crystal) spectrum.

This is the rigorous field-domain approach used by general EPR simulators and is
valid for anisotropic g and A tensors, high-spin systems with zero-field
splitting, and arbitrary microwave frequency (multifrequency EPR).
"""

from __future__ import annotations

import numpy as np

from .spin_hamiltonian import (
    BMAGN_MHZ_PER_T,
    SpinSystem,
    _electron_ops,
    build_field_independent,
    build_zeeman_direction,
    transition_operator,
)


class _PrecomputedSystem:
    """Cache the constant operators of a spin system for fast powder loops.

    H0 (field-independent), and the embedded electron operators Sx, Sy, Sz, are
    built once.  Per-orientation quantities are then cheap linear combinations,
    avoiding the costly Kronecker products inside the orientation loop.
    """

    def __init__(self, system: SpinSystem):
        self.H0 = build_field_independent(system)
        self.sx, self.sy, self.sz = _electron_ops(system)
        self.gx, self.gy, self.gz = system.g_principal
        self.n = self.H0.shape[0]

    def zeeman(self, direction: np.ndarray) -> np.ndarray:
        l, m, n = direction
        return BMAGN_MHZ_PER_T * (l * self.gx * self.sx + m * self.gy * self.sy + n * self.gz * self.sz)

    def transition(self, d: np.ndarray) -> np.ndarray:
        l, m, n = d
        return l * self.gx * self.sx + m * self.gy * self.sy + n * self.gz * self.sz


def fibonacci_hemisphere(n_points: int) -> np.ndarray:
    """Return ``n_points`` near-uniform unit vectors on the upper hemisphere.

    Equal-area sampling makes the orientation weights uniform (the sin(theta)
    factor of the spherical integral is absorbed into the point density), which
    is exactly what is required for an unweighted powder average.  EPR spectra
    are centrosymmetric in field direction, so a hemisphere is sufficient.
    """
    n = max(int(n_points), 1)
    i = np.arange(n) + 0.5
    # z in (0, 1] for the upper hemisphere
    z = 1.0 - i / n
    z = np.clip(z, 1e-9, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i  # golden-angle azimuth
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``direction``."""
    d = direction / (np.linalg.norm(direction) + 1e-15)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, d) * d
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(d, u)
    return u, v


def _resonance_fields_pre(
    pre: _PrecomputedSystem,
    direction: np.ndarray,
    mw_freq_MHz: float,
    field_min_mT: float,
    field_max_mT: float,
    coarse_points: int = 48,
    bisection_iters: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Resonance fields/intensities for one orientation using cached operators."""
    H0 = pre.H0
    Hz = pre.zeeman(direction)
    n = pre.n

    b_grid_mT = np.linspace(field_min_mT, field_max_mT, coarse_points)
    H_stack = H0[None, :, :] + (b_grid_mT[:, None, None] / 1000.0) * Hz[None, :, :]
    evals = np.linalg.eigvalsh(H_stack)  # (Nb, n), ascending

    # Collect every (i, j) bracket where the level gap crosses the microwave quantum.
    blo_list: list[float] = []
    bhi_list: list[float] = []
    ii_list: list[int] = []
    jj_list: list[int] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            gap = evals[:, j] - evals[:, i] - mw_freq_MHz
            cross = np.where(gap[:-1] * gap[1:] < 0)[0]
            for a in cross:
                blo_list.append(b_grid_mT[a])
                bhi_list.append(b_grid_mT[a + 1])
                ii_list.append(i)
                jj_list.append(j)

    if not blo_list:
        return np.empty(0), np.empty(0)

    blo = np.asarray(blo_list)
    bhi = np.asarray(bhi_list)
    ii = np.asarray(ii_list)
    jj = np.asarray(jj_list)
    rows = np.arange(blo.size)

    def gap_at(bmT: np.ndarray) -> np.ndarray:
        Hs = H0[None, :, :] + (bmT[:, None, None] / 1000.0) * Hz[None, :, :]
        ev = np.linalg.eigvalsh(Hs)
        return ev[rows, jj] - ev[rows, ii] - mw_freq_MHz

    g_lo = gap_at(blo)
    for _ in range(bisection_iters):
        bmid = 0.5 * (blo + bhi)
        g_mid = gap_at(bmid)
        left = g_lo * g_mid <= 0
        bhi = np.where(left, bmid, bhi)
        blo = np.where(left, blo, bmid)
        g_lo = np.where(left, g_lo, g_mid)
    b_res = 0.5 * (blo + bhi)

    # Transition probabilities at the resonance fields (batched eigh).
    u, v = _perpendicular_basis(direction)
    Tu = pre.transition(u)
    Tv = pre.transition(v)
    Hr = H0[None, :, :] + (b_res[:, None, None] / 1000.0) * Hz[None, :, :]
    _, V = np.linalg.eigh(Hr)  # (Nb, n, n)
    vi = V[rows, :, ii]  # (Nb, n)
    vj = V[rows, :, jj]
    TuVj = np.einsum("pq,bq->bp", Tu, vj)
    TvVj = np.einsum("pq,bq->bp", Tv, vj)
    mu = np.abs(np.einsum("bp,bp->b", vi.conj(), TuVj)) ** 2 \
        + np.abs(np.einsum("bp,bp->b", vi.conj(), TvVj)) ** 2

    keep = mu > 1e-12
    return b_res[keep], mu[keep]


def resonance_fields_one_orientation(
    system: SpinSystem,
    direction: np.ndarray,
    mw_freq_MHz: float,
    field_min_mT: float,
    field_max_mT: float,
    coarse_points: int = 48,
    bisection_iters: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Public single-orientation resonance finder (builds operators each call)."""
    pre = _PrecomputedSystem(system)
    return _resonance_fields_pre(
        pre, direction, mw_freq_MHz, field_min_mT, field_max_mT, coarse_points, bisection_iters
    )


def absorption_kernel(
    field_axis_mT: np.ndarray,
    width_mT: float,
    eta: float,
) -> np.ndarray:
    """Symmetric absorption broadening kernel (pseudo-Voigt) centred at 0.

    The kernel half-width is capped so that the kernel is never longer than the
    field axis; otherwise a large linewidth relative to a narrow field window
    would make ``numpy.convolve(..., mode="same")`` return a longer array.
    """
    step = float(field_axis_mT[1] - field_axis_mT[0])
    n_field = len(field_axis_mT)
    half = max(8.0 * width_mT, 6.0 * step)
    # cap so total kernel length (2*half/step + 1) stays below the field length
    max_half = max(((n_field - 1) // 2 - 1), 1) * step
    half = min(half, max_half)
    x = np.arange(-half, half + step, step)
    sigma = max(width_mT, 1e-4) / 2.0
    gamma = max(width_mT, 1e-4) / 2.0
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    lorentz = gamma**2 / (x**2 + gamma**2)
    eta = min(max(float(eta), 0.0), 1.0)
    k = eta * lorentz + (1.0 - eta) * gauss
    s = k.sum()
    return k / s if s > 0 else k


def powder_spectrum(
    field_mT: np.ndarray,
    system: SpinSystem,
    mw_freq_GHz: float,
    linewidth_mT: float = 0.5,
    eta: float = 0.5,
    n_orientations: int = 1500,
    derivative: bool = True,
) -> np.ndarray:
    """Compute a powder (orientation-averaged) cw-EPR spectrum.

    Returns the first-derivative spectrum (or absorption if derivative=False),
    normalised to unit maximum absolute amplitude on the supplied field axis.
    """
    field = np.asarray(field_mT, dtype=float)
    if field.size < 2:
        return np.zeros_like(field)
    fmin, fmax = float(field.min()), float(field.max())
    mw_MHz = float(mw_freq_GHz) * 1000.0

    dirs = fibonacci_hemisphere(n_orientations)
    # widen the resonance search slightly beyond the display window
    pad = 0.05 * (fmax - fmin) + 5.0 * linewidth_mT
    absorption = np.zeros_like(field)
    step = float(field[1] - field[0])

    pre = _PrecomputedSystem(system)  # build constant operators once
    for d in dirs:
        bres, inten = _resonance_fields_pre(
            pre, d, mw_MHz, fmin - pad, fmax + pad
        )
        if bres.size == 0:
            continue
        # linear deposition of each stick into the two nearest bins
        pos = (bres - field[0]) / step
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        for b_lo, f, w in zip(lo, frac, inten):
            if 0 <= b_lo < field.size:
                absorption[b_lo] += w * (1.0 - f)
            if 0 <= b_lo + 1 < field.size:
                absorption[b_lo + 1] += w * f

    # convolve with absorption broadening (length-preserving)
    kernel = absorption_kernel(field, linewidth_mT, eta)
    conv = np.convolve(absorption, kernel, mode="same")
    if conv.size != field.size:  # safety: keep the central, field-aligned window
        start = (conv.size - field.size) // 2
        conv = conv[start:start + field.size]
    absorption = conv

    if derivative:
        spec = np.gradient(absorption, field)
    else:
        spec = absorption

    scale = np.nanmax(np.abs(spec))
    if np.isfinite(scale) and scale > 0:
        spec = spec / scale
    return spec
