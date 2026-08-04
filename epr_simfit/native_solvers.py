"""Native-Python native solvers, wired for the Open-Sym-EPR GUI.

This
module exposes the **Open-Sym-EPR** engine — an open, native-Python re-implementation
of the same spin-Hamiltonian physics (validated against analytical limits) — so
that every native experiment runs **live inside Open-Sym-EPR with no MATLAB**.

Each function below takes plain GUI parameters, builds an Open-Sym-EPR ``SpinSystem``
by direct spin-Hamiltonian construction, runs the solver, and returns arrays
ready to plot. The five native solvers:

    garlic  -> isotropic / fast-motion cw-EPR
    pepper  -> solid-state powder cw-EPR (full g/A/D tensors, high spin)
    salt    -> ENDOR
    saffron -> ESEEM (2-pulse and 3-pulse, time + frequency domain)
    curry   -> magnetometry (chi*T vs T, M vs B)

If Open-Sym-EPR is not importable the module sets ``AVAILABLE = False`` and
``IMPORT_ERROR`` carries the reason, so the GUI can degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import openspin as osp
    AVAILABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    osp = None  # type: ignore[assignment]
    AVAILABLE = False
    IMPORT_ERROR = str(exc)


# ── Catalog (solver -> native panel metadata) ───────────────────

@dataclass
class NativeSolver:
    name: str            # solver name
    title: str           # human title
    experiment: str      # experiment class
    engine: str          # Open-Sym-EPR function(s) used
    needs_nucleus: bool  # whether a nucleus is required
    needs_aniso: bool    # whether anisotropic hyperfine is required


def native_catalog() -> dict[str, NativeSolver]:
    return {
        "garlic": NativeSolver("garlic", "Isotropic / fast-motion cw-EPR", "cw-EPR (solution)",
                               "openspin.garlic", False, False),
        "pepper": NativeSolver("pepper", "Solid-state powder cw-EPR", "cw-EPR (solid)",
                               "openspin.pepper", False, False),
        "salt": NativeSolver("salt", "ENDOR", "ENDOR",
                             "openspin.endor_spectrum", True, False),
        "saffron": NativeSolver("saffron", "Pulse EPR / ESEEM", "pulse EPR",
                                "openspin.two_pulse_eseem / three_pulse_eseem", True, True),
        "curry": NativeSolver("curry", "Magnetometry (chi*T, M)", "SQUID / magnetometry",
                              "openspin.chiT_vs_T / magnetisation_muB", False, False),
    }


# ── Spin-system builder from GUI parameters ───────────────────────────────────

def parse_nuclei(text: str):
    """Parse ``14N:1.55; 1H:0.30`` (isotropic mT) or ``14N:0.3,0.3,1.0`` (tensor mT).

    Returns a list of Open-Sym-EPR ``NuclearSpec`` objects.
    """
    if not AVAILABLE:
        raise RuntimeError(IMPORT_ERROR or "Open-Sym-EPR not available")
    nuclei = []
    for item in str(text or "").replace(";", "\n").splitlines():
        item = item.strip()
        if not item or ":" not in item:
            continue
        iso, rhs = item.split(":", 1)
        iso = iso.strip()
        vals = [v.strip() for v in rhs.replace(";", ",").split(",") if v.strip()]
        if len(vals) >= 3:
            tensor = (float(vals[0]), float(vals[1]), float(vals[2]))
            nuclei.append(osp.nucleus(iso, A_tensor_mT=tensor))
        elif len(vals) == 1:
            nuclei.append(osp.nucleus(iso, float(vals[0])))
    return nuclei


def build_system(g_iso: float | None = None,
                 g_tensor: tuple[float, float, float] | None = None,
                 S: float = 0.5, D_MHz: float = 0.0, E_MHz: float = 0.0,
                 nuclei_text: str = ""):
    """Build an Open-Sym-EPR ``SpinSystem`` from GUI form fields."""
    if not AVAILABLE:
        raise RuntimeError(IMPORT_ERROR or "Open-Sym-EPR not available")
    g = g_tensor if g_tensor is not None else (g_iso if g_iso is not None else 2.0023)
    return osp.spin_system(g=g, nuclei=parse_nuclei(nuclei_text), S=float(S),
                           D_MHz=float(D_MHz), E_MHz=float(E_MHz))


# ── Solver wrappers (each returns a dict the GUI can plot/download) ────────────

def run_garlic(system, fmin, fmax, mw_freq, lw_mT, eta, n_points=2000):
    field = np.linspace(fmin, fmax, int(n_points))
    spec = osp.garlic(system, field, mw_freq, lw_mT, eta)
    return {"x": field, "y": spec, "xlabel": "Magnetic field / mT",
            "ylabel": "dχ″/dB (a.u.)", "kind": "spectrum"}


def run_pepper(system, fmin, fmax, mw_freq, lw_mT, eta, n_orient=2000, n_points=2000):
    field = np.linspace(fmin, fmax, int(n_points))
    spec = osp.pepper(system, field, mw_freq, lw_mT, eta, n_orientations=int(n_orient))
    return {"x": field, "y": spec, "xlabel": "Magnetic field / mT",
            "ylabel": "dχ″/dB (a.u.)", "kind": "spectrum"}


def run_salt(system, field_mT, rf_max, mw_freq, rf_lw, n_orient=1500, n_points=2000):
    rf = np.linspace(0.1, rf_max, int(n_points))
    spec = osp.endor_spectrum(system, field_mT, rf, mw_freq, rf_lw,
                              n_orientations=int(n_orient), rf_max_MHz=float(rf_max))
    return {"x": rf, "y": spec, "xlabel": "RF frequency / MHz",
            "ylabel": "ENDOR intensity (a.u.)", "kind": "spectrum"}


def run_saffron(system, field_mT, tau_max, sequence, tau_fixed, n_orient=800,
                fft_max_MHz=30.0, n_points=2048):
    t = np.linspace(0.0, tau_max, int(n_points))
    if str(sequence).startswith("2"):
        echo = osp.two_pulse_eseem(system, t, field_mT, int(n_orient))
    else:
        echo = osp.three_pulse_eseem(system, t, float(tau_fixed), field_mT, int(n_orient))
    freq, fspec = osp.eseem_fft(t, echo)
    mask = freq <= float(fft_max_MHz)
    return {"t": t, "echo": echo, "freq": freq[mask], "fft": fspec[mask],
            "kind": "eseem"}


def run_curry(system, mode, t_max, chi_field_T, b_max, m_temp, n_orient=200, n_points=80):
    if mode.startswith("χT") or mode.lower().startswith("chi"):
        T = np.linspace(2.0, t_max, int(n_points))
        chiT = osp.chiT_vs_T(system, T, chi_field_T, n_orientations=int(n_orient))
        return {"x": T, "y": chiT, "xlabel": "Temperature / K",
                "ylabel": "χT / cm³ K mol⁻¹", "kind": "spectrum"}
    B = np.linspace(0.0, b_max, int(n_points))
    M = osp.magnetisation_muB(system, B, m_temp, n_orientations=int(n_orient))
    return {"x": B, "y": M, "xlabel": "Field / T", "ylabel": "M / µ_B",
            "kind": "spectrum"}
