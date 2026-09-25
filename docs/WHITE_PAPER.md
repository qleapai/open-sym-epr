# Open-Sym-EPR White Paper

**Open-Sym-EPR: A Public General-Purpose GUI for High-Field cw-EPR Simulation, Fitting, and Transparent Export**

**Developer:** Md Sakib Hasan Khan, PhD Student, Research School of Chemistry, The Australian National University; Assistant Professor, Department of Electrical and Electronic Engineering (EEE), Khulna University of Engineering & Technology (KUET), Bangladesh.

## Abstract

Open-Sym-EPR is a public-distributable graphical software tool for high-field continuous-wave electron paramagnetic resonance (cw-EPR) data import, preprocessing, isotropic spectral simulation, mixture fitting, model comparison, and publication-ready export. The software is intended to make routine cw-EPR interpretation more transparent by exporting raw and processed spectra, fitted parameters, all fitted metrics, model-comparison tables, fitted component curves, residuals, and plot datasets in CSV format.

## Scientific Scope

Open-Sym-EPR models cw-EPR spectra using high-field isotropic resonance positions and derivative line shapes. The resonance field is calculated from

```text
B0 = mwFreq / (g * mu_B_over_h)
```

and isotropic hyperfine splitting is generated from supported nuclei by enumerating their magnetic sublevels. Components are combined as weighted derivative spectra with optional constant or linear baseline terms. Fit optimization uses bounded least squares.

## Appropriate Use

Open-Sym-EPR is appropriate for:

- importing ASCII, ASC, TXT, DAT, and CSV cw-EPR spectra;
- baseline correction, cropping, normalization, and display smoothing;
- screening common isotropic radical, spin-probe, defect, and transition-metal patterns;
- fitting mixtures with bounded weights, linewidths, and g values;
- comparing candidate models by RSS, RMSE, R2, AIC, and BIC;
- exporting reproducible fit tables, plot data, reports, and ORCA EPR templates.

## Limitations

Open-Sym-EPR fit quality is not standalone chemical proof. Assignments should be checked against standards, controls, isotope substitution, known chemistry, independent product analysis, and appropriate literature values.

The present version is not a full anisotropic EPR tensor simulator. It does not replace specialist methods for:

- powder g/A anisotropy;
- multi-frequency tensor refinement;
- orientation selection;
- exchange coupling;
- saturation or relaxation analysis;
- pulsed EPR;
- spin-Hamiltonian refinement beyond isotropic screening.

## Development Roadmap

Several developments would strengthen Open-Sym-EPR as a research software platform. First, a DOI-linked release archive should be created for the exact manuscript version. Second, experimental reference spectra should be included for TEMPO, DPPH, Mn(II), Cu(II), vanadyl and representative PBN adducts, with Open-Sym-EPR fits. Third, uncertainty estimates should be added to exported fit tables. Fourth, a batch-processing mode could apply a fixed model to time-series or catalyst-series spectra and export kinetic plots. Fifth, tests should cover the powder engine more extensively, including convergence with orientation count and comparison against analytical reference cases.

The current public version already takes initial steps toward these goals by exporting first-pass least-squares standard errors for fitted parameters, supporting uploaded user model libraries, and including general-purpose powder/anisotropic components. The remaining roadmap items should be versioned, benchmarked, and archived with a DOI-linked release before being cited as validated reference functionality.

## Uploaded Custom Models

Open-Sym-EPR accepts user-uploaded custom model libraries in JSON, YAML, or CSV format. Custom models can define arbitrary component names, assignments, categories, g values, linewidths, weights, nuclei, hyperfine couplings, g-tensor principal values, electron spin S, zero-field splitting parameters D and E, and whether the component should use the isotropic or powder engine.

Custom model upload is intended to make Open-Sym-EPR extensible across chemistry areas such as N2 reduction reaction (N2RR), CO2 reduction reaction (CO2RR), HER, OER, ORR, photocatalysis, organic radical chemistry, spin-labelling, transition-metal spectroscopy, and solid-state defect spectroscopy. Mechanistic component names are candidate assignments and must be validated experimentally.

## Recommended Citation Text

If Open-Sym-EPR is used in a publication, users may cite it as:

> Khan, M. S. H. Open-Sym-EPR: A Public General-Purpose GUI for High-Field cw-EPR Simulation, Fitting, and Transparent Export, version 0.1.0, 2026.

## Suggested Methods Wording

cw-EPR spectra were imported, preprocessed, and fitted using Open-Sym-EPR, a high-field isotropic cw-EPR simulation and fitting GUI. Candidate spectral components were simulated from g values, isotropic hyperfine constants, derivative line shapes, and bounded linewidths. Mixture weights and selected spectral parameters were optimized by least-squares fitting. Model comparisons were evaluated using RSS, RMSE, R2, AIC, and BIC. Chemical assignments were treated as candidate interpretations and evaluated against controls and known chemical constraints.
