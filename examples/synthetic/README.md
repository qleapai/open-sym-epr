# Synthetic inputs & comprehensive feature test

Synthetic EPR spectra with **known ground-truth parameters**, used both as
GUI-loadable example inputs and as the basis for an end-to-end feature test that
exercises every capability of the software.

## Regenerate the inputs
```
python gen_synthetic_inputs.py
```
Produces (all built from the Open-Sym-EPR forward model + realistic noise; ground
truth in `syn_manifest.json`):

| File | Engine | Ground truth |
|------|--------|--------------|
| `syn_nitroxide_14N.txt` | garlic | g=2.0060, a(14N)=1.55 mT, ΔBpp=0.12 mT |
| `syn_organic_radical.txt` | garlic | g=2.0045, 2×a(1H)=0.45 mT |
| `syn_cu_powder.txt` | pepper | g⊥=2.06, g∥=2.30, A∥(63Cu)=16 mT |
| `syn_two_component.txt` | garlic | sharp nitroxide + broad radical |
| `syn_bruker.DSC/.DTA` | garlic | BES3T big-endian, g=2.0060, a(14N)=1.55 mT |
| `syn_kinetics_00..30min.txt` | garlic | adduct rise→decay time series |

All text files are two-column (`Field_mT`, `Intensity`) and load directly in the
Open-Sym-EPR **Import** tab or the Open-Sym-EPR **Fit** tab; the `.DSC/.DTA` pair loads via
the Bruker uploader.

## Run the full feature test
```
python feature_test.py
```
Exercises **36 features** across four groups and prints a PASS/FAIL matrix
(exit code non-zero on any failure):

- **Open-Sym-EPR engine** — garlic, pepper, salt, saffron (2/3-pulse + FFT), curry
  (χT, M), esfit blind recovery, Monte-Carlo errors, parameter-name validation,
  Bruker import, multifrequency, text parser.
- **native_solvers bridge** — build_system/parse_nuclei and live run of all five
  native solvers (no MATLAB).
- **Open-Sym-EPR engine** — io, preprocessing, model library, simulator, fitter (3
  modes + Monte-Carlo), model comparison (AIC/BIC), model suggester,
  interpretation, batch/kinetics, export (EasySpin/ORCA/zip), user-model
  round-trip, Bruker, metadata, demo-data, reference standards.
- **GUI** — headless `AppTest` render of both apps and a live native-solver Run.

The recovery checks compare fitted parameters against `syn_manifest.json`, so the
test verifies both that each feature *runs* and that the physics is *correct*.
