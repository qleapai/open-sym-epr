# Open-Sym-EPR

**A free, open-source, native-Python engine and browser app for the simulation and fitting of
continuous-wave EPR, ENDOR, ESEEM, and magnetometry — no MATLAB, no proprietary dependencies.**

[![tests](https://github.com/USER/open-sym-epr/actions/workflows/tests.yml/badge.svg)](https://github.com/USER/open-sym-epr/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Open-Sym-EPR solves the general electron–nuclear spin Hamiltonian by direct matrix
diagonalisation and exposes it through a zero-installation browser interface and a scriptable
Python API. It is an **independent, original implementation**; it contains no third-party
simulation code.

---

## Capabilities

- **Five solvers from one spin-Hamiltonian core** (cw solution, cw powder, ENDOR, ESEEM,
  magnetometry) — anisotropic *g*/*A* tensors, arbitrary electron spin, zero-field splitting,
  nuclear Zeeman.
- **Least-squares fitting** with a global pre-search **and** bootstrap/Monte-Carlo uncertainties,
  plus AIC/BIC model comparison.
- **Machine-learning-accelerated fitting** — a physics-trained neural initialiser (`ML-assisted`)
  and an optimisation-free `ML-driven` mode, both validated against the forward model with an
  out-of-distribution guard.
- **Comprehensive spin-trapping model library** — PBN, PBN/DMSO-derived radicals, DMPO, POBN,
  and ³¹P-bearing DEPMPO adducts across all common radical classes.
- **Spin-adduct mixtures at adjustable ratios** — compose any set of adducts at **manually
  entered ratios**, simulate the composite plus stacked contributions, and **automatically
  recover** the ratios from an experimental spectrum by position-seeded least-squares with
  optional esfit-style hyperfine refinement (`Adduct mixture` tab).
- **Native Bruker BES3T (.DTA/.DSC) import**, batch/kinetics, preprocessing, and
  publication-ready export (CSV, figures, reports, ORCA templates).
- **Project save / resume** — persist and reopen a full working session.
- Validated against **fifty analytical limits** and a **73-test capability battery**.

## Quick start

```bash
git clone https://github.com/USER/open-sym-epr.git
cd open-sym-epr
python -m venv .venv
.venv/Scripts/pip install -e .        # Linux/macOS: .venv/bin/pip install -e .
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Use online (Streamlit Community Cloud)

Fork this repo, go to [share.streamlit.io](https://share.streamlit.io), select your fork, and set
the main file to `streamlit_app.py`. `requirements.txt` and `.streamlit/config.toml` are already
configured.

## Python API

```python
import numpy as np, openspin as osp

sys = osp.spin_system(g=2.0060, nuclei=[osp.nucleus("14N", 1.55)])   # nitroxide
field = np.linspace(330, 350, 2000)
spec = osp.garlic(sys, field, mw_freq_GHz=9.5, linewidth_mT=0.1)     # cw solution

fit = osp.esfit(sys, field, experimental,
                {"g_iso": (2.006, 2.0, 2.01), "A0_iso": (1.55, 1.0, 2.0),
                 "lw": (0.1, 0.02, 0.5), "scale": (1.0, 0.2, 3.0)}, 9.5)
print(fit.metrics["R2"])
```

Spin-trapping model library:

```python
from epr_simfit import model_library as ml
ml.pbn_photocatalysis_model()                     # example PBN/DMSO decomposition (with fractions)
ml.components_for_preset("ST_dmpo_all_adducts")   # all DMPO adducts
from openspin import spin_trap_models as stm
stm.all_spin_trap_systems()                       # PBN / PBN-DMSO / DMPO / POBN / DEPMPO
```

## Repository layout

```
open-sym-epr/
├── streamlit_app.py     # app entry point
├── app_simepr.py        # main simulation + fitting GUI
├── app_openspin.py      # engine-focused GUI (solvers)
├── openspin/            # native spin-Hamiltonian engine, solvers, ML fitting, spin-trap models
├── epr_simfit/          # application layer (model library, fitter, batch, export, project)
├── tests/               # automated test suite
├── examples/            # synthetic inputs + end-to-end feature test
└── paper/               # manuscript, Supporting Information, figures
```

> Internal package names `openspin` (engine) and `epr_simfit` (application layer) are the
> implementation modules of Open-Sym-EPR.

## Testing

```bash
pip install -e .[dev]
pytest tests/
```

## Scope and honesty

Open-Sym-EPR solves the spin-Hamiltonian model exactly (to numerical precision) for the systems
within its scope. Slow-motional (SLE) lineshapes, HYSCORE/DEER, MD-trajectory simulation, and
strain distributions are outside the current scope. Spin-trapping hyperfine constants are
literature approximations to be refined against your data; candidate / N-centred adducts require
isotope (¹⁵N/²H) and blank/solvent controls before assignment. The engine is an independent
reimplementation of well-established spin-Hamiltonian physics and credits the prior art it builds
on in the accompanying paper.

## Citation

See [`CITATION.cff`](CITATION.cff) and the accompanying paper in `paper/`.

## License

MIT — see [`LICENSE`](LICENSE).
