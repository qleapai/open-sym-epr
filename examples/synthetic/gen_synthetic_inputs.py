"""Generate synthetic EPR inputs with known ground-truth parameters.

Produces GUI-loadable two-column text spectra and a synthetic Bruker BES3T
(.DSC/.DTA) pair, each built from the Open-Sym-EPR forward model with realistic
noise, plus a manifest of the true parameters so that fitting/feature tests can
check recovery. Run:  python gen_synthetic_inputs.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import openspin as osp

OUT = Path(__file__).resolve().parent
MW = 9.85  # GHz


def b0(g):
    return osp.isotropic_resonance_field_mT(MW, g)


def save_txt(name, field, inten, header):
    lines = [f"# {header}", f"# Microwave Frequency: {MW} GHz", "# Field unit: mT",
             "Field_mT\tIntensity"]
    for x, y in zip(field, inten):
        lines.append(f"{x:.5f}\t{y:.7e}")
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")


def noisy(y, frac, seed):
    rng = np.random.default_rng(seed)
    return y / (np.max(np.abs(y)) or 1) + rng.normal(0, frac, y.size)


manifest = {"microwave_frequency_GHz": MW, "cases": {}}

# 1. Nitroxide 14N triplet (garlic)
g, aN, lw = 2.0060, 1.55, 0.12
c = b0(g)
f1 = np.linspace(c - 6, c + 6, 1024)
s1 = osp.garlic(osp.spin_system(g=g, nuclei=[osp.nucleus("14N", aN)]), f1, MW, lw, 0.5)
save_txt("syn_nitroxide_14N.txt", f1, noisy(s1, 0.02, 1), "Synthetic 14N nitroxide (garlic)")
manifest["cases"]["nitroxide_14N"] = {"file": "syn_nitroxide_14N.txt", "engine": "garlic",
                                      "g": g, "aN_mT": aN, "lw_mT": lw, "noise": 0.02}

# 2. Organic radical: 2 equivalent 1H (garlic)
g2, aH, lw2 = 2.0045, 0.45, 0.08
c2 = b0(g2)
f2 = np.linspace(c2 - 4, c2 + 4, 1024)
s2 = osp.garlic(osp.spin_system(g=g2, nuclei=[osp.nucleus("1H", aH), osp.nucleus("1H", aH)]), f2, MW, lw2, 0.5)
save_txt("syn_organic_radical.txt", f2, noisy(s2, 0.025, 2), "Synthetic organic radical, 2x1H (garlic)")
manifest["cases"]["organic_radical"] = {"file": "syn_organic_radical.txt", "engine": "garlic",
                                        "g": g2, "aH_mT": aH, "lw_mT": lw2, "noise": 0.025}

# 3. Cu(II) axial powder (pepper)
gpar, gperp, ACu = 2.30, 2.06, 16.0
f3 = np.linspace(280, 360, 2048)
cu = osp.spin_system(g=(gperp, gperp, gpar),
                     nuclei=[osp.nucleus("63Cu", A_tensor_mT=(0.8, 0.8, ACu))])
s3 = osp.pepper(cu, f3, MW, 1.2, 0.5, n_orientations=2000)
save_txt("syn_cu_powder.txt", f3, noisy(s3, 0.02, 3), "Synthetic Cu(II) axial powder (pepper)")
manifest["cases"]["cu_powder"] = {"file": "syn_cu_powder.txt", "engine": "pepper",
                                  "g_perp": gperp, "g_par": gpar, "ACu_par_mT": ACu, "noise": 0.02}

# 4. Two-component: sharp nitroxide + broad radical (for model comparison / multi-component)
f4 = np.linspace(c - 8, c + 8, 1024)
sharp = osp.garlic(osp.spin_system(g=2.0060, nuclei=[osp.nucleus("14N", 1.55)]), f4, MW, 0.10, 0.5)
broad = osp.garlic(osp.spin_system(g=2.0035, nuclei=[]), f4, MW, 1.4, 0.5)
mix = 0.7 * sharp / np.max(np.abs(sharp)) + 0.3 * broad / np.max(np.abs(broad))
save_txt("syn_two_component.txt", f4, noisy(mix, 0.02, 4), "Synthetic 2-component (sharp nitroxide + broad radical)")
manifest["cases"]["two_component"] = {"file": "syn_two_component.txt", "engine": "garlic",
                                      "components": ["nitroxide g=2.0060 aN=1.55", "broad g=2.0035"], "noise": 0.02}

# 5. Synthetic Bruker BES3T pair (nitroxide), big-endian float64
xpts, xmin_G, xwid_G = 1024, (c - 6) * 10.0, 12.0 * 10.0
fG = xmin_G + np.arange(xpts) * (xwid_G / (xpts - 1))
fmT = fG / 10.0
sB = osp.garlic(osp.spin_system(g=g, nuclei=[osp.nucleus("14N", aN)]), fmT, MW, lw, 0.5)
sB = noisy(sB, 0.015, 5)
(OUT / "syn_bruker.DTA").write_bytes(sB.astype(">f8").tobytes())
dsc = "\n".join([
    "#DESC\t1.2 * DESCRIPTOR INFORMATION", "DSRC\tEXP", "BSEQ\tBIG", "IKKF\tREAL",
    "XTYP\tIDX", "YTYP\tNODATA", "ZTYP\tNODATA", "IRFMT\tD",
    f"XPTS\t{xpts}", f"XMIN\t{xmin_G:.6f}", f"XWID\t{xwid_G:.6f}",
    "TITL\t'syn_bruker Synthetic_14N_nitroxide'", "XNAM\t'Field'", "XUNI\t'G'",
    "IRNAM\t'Intensity'", "EXPT\tCW", f"MWFQ\t{MW*1e9:.6e}", "*",
])
(OUT / "syn_bruker.DSC").write_text(dsc, encoding="utf-8")
manifest["cases"]["bruker_nitroxide"] = {"files": ["syn_bruker.DSC", "syn_bruker.DTA"],
                                         "engine": "garlic", "g": g, "aN_mT": aN, "lw_mT": lw,
                                         "format": "BES3T big-endian float64"}

# 6. Time-series for batch/kinetics (nitroxide adduct growing then decaying)
times = [0, 5, 10, 20, 30]
amps = [0.1, 0.6, 1.0, 0.7, 0.4]
for t, a in zip(times, amps):
    st = a * s1 / np.max(np.abs(s1))
    save_txt(f"syn_kinetics_{t:02d}min.txt", f1, st + np.random.default_rng(100 + t).normal(0, 0.02, st.size),
             f"Synthetic kinetics t={t} min (nitroxide adduct)")
manifest["cases"]["kinetics_series"] = {"files": [f"syn_kinetics_{t:02d}min.txt" for t in times],
                                        "times_min": times, "rel_amplitude": amps}

(OUT / "syn_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote {len(list(OUT.glob('syn_*')))} synthetic input files + manifest to {OUT}")
for k, v in manifest["cases"].items():
    print(f"  {k}: {v.get('file', v.get('files'))}")
