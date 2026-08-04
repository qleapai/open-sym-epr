"""Reference EPR standards with literature spin-Hamiltonian parameters.

These standards (DPPH, TEMPO, Mn(II), Cu(II), vanadyl, PBN adducts) let users
validate Open-Sym-EPR against well-characterised systems and benchmark fits in
parallel with EasySpin.  Parameters are typical literature values for X-band;
exact values depend on solvent, temperature, and coordination.

Each entry provides a Open-Sym-EPR :class:`SpinComponent`, a recommended field window,
and a parallel EasySpin script string, so the same model can be run in both
programs and compared.

Indicative references
---------------------
DPPH      : g ~ 2.0036, narrow single line (g-marker standard).
TEMPO     : nitroxide, a(14N) ~ 1.6 mT (solution); rigid limit g = [2.0099 2.0061 2.0024].
Mn(II)    : g ~ 2.00, a(55Mn) ~ 9.0 mT, six-line.
Cu(II)    : axial, g|| ~ 2.25, g_perp ~ 2.06, A||(63Cu) ~ 16 mT.
VO2+      : axial, g|| ~ 1.94, g_perp ~ 1.98, A||(51V) ~ 18 mT.
PBN-OH    : a(14N) ~ 1.5 mT, a(H_beta) ~ 0.3 mT.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spin_models import Nucleus, SpinComponent


@dataclass
class ReferenceStandard:
    key: str
    name: str
    component: SpinComponent
    field_window_mT: tuple[float, float]
    mw_freq_GHz: float
    note: str


def reference_standards() -> dict[str, ReferenceStandard]:
    """Return the dictionary of built-in reference standards."""
    refs: dict[str, ReferenceStandard] = {}

    refs["dpph"] = ReferenceStandard(
        key="dpph", name="DPPH (g-marker)",
        component=SpinComponent(
            component_id="ref_dpph", display_name="DPPH standard",
            radical_assignment="DPPH (2,2-diphenyl-1-picrylhydrazyl)",
            category="reference standard", g=2.0036, linewidth_mT=0.20, eta=0.7, weight=1.0,
            interpretation="g-marker standard; narrow nearly-isotropic single line.",
        ),
        field_window_mT=(346.0, 352.0), mw_freq_GHz=9.80,
        note="Common field/g-factor calibration standard, g = 2.0036.",
    )

    refs["tempo_solution"] = ReferenceStandard(
        key="tempo_solution", name="TEMPO (solution, 14N triplet)",
        component=SpinComponent(
            component_id="ref_tempo_sol", display_name="TEMPO (fast tumbling)",
            radical_assignment="TEMPO nitroxide (solution)",
            category="reference standard", g=2.0060,
            nuclei=[Nucleus(isotope="14N", A_mT=1.62, label="14N")],
            linewidth_mT=0.10, eta=0.5, weight=1.0,
            interpretation="Fast-tumbling TEMPO three-line spectrum (a(14N) ~ 1.6 mT).",
        ),
        field_window_mT=(344.0, 354.0), mw_freq_GHz=9.80,
        note="Isotropic nitroxide standard; three equal-intensity lines.",
    )

    refs["tempo_rigid"] = ReferenceStandard(
        key="tempo_rigid", name="TEMPO (rigid limit, anisotropic)",
        component=SpinComponent(
            component_id="ref_tempo_rigid", display_name="TEMPO (rigid limit)",
            radical_assignment="TEMPO nitroxide (frozen)",
            category="reference standard", g=2.0061, spin_S=0.5,
            g_tensor=(2.0099, 2.0061, 2.0024),
            nuclei=[Nucleus(isotope="14N", A_mT=0.6, label="14N", A_tensor_mT=(0.6, 0.6, 3.6))],
            linewidth_mT=0.18, eta=0.4, weight=1.0,
            interpretation="Rigid-limit nitroxide powder pattern; g = [2.0099 2.0061 2.0024], Azz ~ 3.6 mT.",
        ),
        field_window_mT=(342.0, 354.0), mw_freq_GHz=9.80,
        note="Frozen-solution nitroxide; anisotropic g and Azz resolved.",
    )

    refs["mn2"] = ReferenceStandard(
        key="mn2", name="Mn(II) (six-line)",
        component=SpinComponent(
            component_id="ref_mn2", display_name="Mn(II) aqueous standard",
            radical_assignment="Mn(II) (S=5/2)",
            category="reference standard", g=2.0010, spin_S=2.5,
            g_tensor=(2.0010, 2.0010, 2.0010), D_MHz=250.0, E_MHz=40.0,
            nuclei=[Nucleus(isotope="55Mn", A_mT=9.0, label="55Mn", A_tensor_mT=(9.0, 9.0, 9.0))],
            linewidth_mT=0.7, eta=0.55, weight=1.0,
            interpretation="Mn(II) six-line central transition; a(55Mn) ~ 9.0 mT.",
        ),
        field_window_mT=(300.0, 400.0), mw_freq_GHz=9.50,
        note="Mn(II) intensity standard; six lines spaced ~9 mT.",
    )

    refs["cu2"] = ReferenceStandard(
        key="cu2", name="Cu(II) (axial)",
        component=SpinComponent(
            component_id="ref_cu2", display_name="Cu(II) axial standard",
            radical_assignment="Cu(II) (axial)",
            category="reference standard", g=2.12, spin_S=0.5,
            g_tensor=(2.060, 2.060, 2.250),
            nuclei=[Nucleus(isotope="63Cu", A_mT=1.5, label="63Cu", A_tensor_mT=(1.5, 1.5, 16.0))],
            linewidth_mT=1.2, eta=0.5, weight=1.0,
            interpretation="Axial Cu(II); g|| ~ 2.25 with parallel 63Cu hyperfine ~16 mT.",
        ),
        field_window_mT=(280.0, 350.0), mw_freq_GHz=9.50,
        note="Square-planar/tetragonal Cu(II) powder pattern.",
    )

    refs["vanadyl"] = ReferenceStandard(
        key="vanadyl", name="VO2+ vanadyl (axial)",
        component=SpinComponent(
            component_id="ref_vo2", display_name="VO2+ vanadyl standard",
            radical_assignment="VO2+ / V(IV) (axial)",
            category="reference standard", g=1.97, spin_S=0.5,
            g_tensor=(1.980, 1.980, 1.940),
            nuclei=[Nucleus(isotope="51V", A_mT=7.0, label="51V", A_tensor_mT=(7.0, 7.0, 18.0))],
            linewidth_mT=0.8, eta=0.5, weight=1.0,
            interpretation="Axial vanadyl; eight-line 51V pattern, A|| ~ 18 mT.",
        ),
        field_window_mT=(300.0, 385.0), mw_freq_GHz=9.50,
        note="Vanadyl porphyrin/oxo-vanadium standard.",
    )

    refs["pbn_oh_ref"] = ReferenceStandard(
        key="pbn_oh_ref", name="PBN-OH adduct",
        component=SpinComponent(
            component_id="ref_pbn_oh", display_name="PBN-OH adduct standard",
            radical_assignment="PBN-OH",
            category="reference standard", g=2.0057,
            nuclei=[Nucleus(isotope="14N", A_mT=1.50, label="14N"),
                    Nucleus(isotope="1H", A_mT=0.30, label="beta H")],
            linewidth_mT=0.08, eta=0.5, weight=1.0,
            interpretation="PBN hydroxyl adduct; a(14N) ~ 1.5 mT, a(H) ~ 0.3 mT.",
        ),
        field_window_mT=(345.0, 353.0), mw_freq_GHz=9.80,
        note="Representative PBN spin-trap adduct.",
    )

    return refs


def easyspin_reference_script(ref: ReferenceStandard) -> str:
    """Generate a parallel EasySpin script for a reference standard."""
    from .export import easyspin_component_block

    header = [
        f"% EasySpin reference script for {ref.name}",
        f"% {ref.note}",
        "clear; clf;",
        f"Exp.mwFreq = {ref.mw_freq_GHz:.6g};   % GHz",
        f"Exp.Range  = [{ref.field_window_mT[0]:.6g} {ref.field_window_mT[1]:.6g}];  % mT",
        "Exp.Harmonic = 1;",
        "",
    ]
    body = easyspin_component_block(ref.component, index=1)
    solver = "pepper" if ref.component.is_anisotropic() else "garlic"
    footer = ["", f"[B,spc] = {solver}(Sys1,Exp);", "plot(B,spc); xlabel('Field (mT)'); ylabel('dchi''''/dB');"]
    return "\n".join(header + body + footer)
