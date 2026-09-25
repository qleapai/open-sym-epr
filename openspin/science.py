"""Scientific background and user guidance text for the Open-Sym-EPR GUI.

Each block is GitHub-flavoured Markdown intended for an expandable help panel so
users understand the physics behind every simulation, the meaning of each input,
and how to interpret the output.
"""

INTRO = r"""
### What Open-Sym-EPR computes

Open-Sym-EPR simulates magnetic-resonance observables from a **spin Hamiltonian**
solved by matrix diagonalisation, entirely in native Python (no MATLAB). It
provides native solvers:

| Solver | Observable | Open-Sym-EPR module |
|--------|-----------|-----------------|
| **garlic** | isotropic / solution cw-EPR | `cw.garlic` |
| **pepper** | solid-state powder cw-EPR | `cw.pepper` |
| **salt** | ENDOR (nuclear frequencies) | `endor` |
| **saffron** (lite) | ESEEM (echo modulation) | `eseem` |
| **curry** | magnetometry (χT, M) | `magnetometry` |

All share the general electron–nuclear spin Hamiltonian:

> **H/h = (μ_B/h) B·g·S + Σ_k S·A_k·I_k − Σ_k g_n,k μ_N B·I_k + D(S_z²−S(S+1)/3) + E(S_x²−S_y²)**

with anisotropic **g**- and **A**-tensors, arbitrary electron spin **S**, nuclear
Zeeman, and zero-field splitting **D, E**.
"""

CW = r"""
### Continuous-wave EPR — `garlic` and `pepper`

cw-EPR records the **first derivative** of microwave absorption versus field.

**Resonance condition:**  hν = g μ_B B₀  →  B₀(mT) = ν(GHz) / (g × 13.99624) × 1000.
At X-band (9.5 GHz) a free electron (g ≈ 2.0023) resonates near 339 mT.

**`garlic` (isotropic / solution).** Fast-tumbling molecules average anisotropy,
giving sharp lines split by isotropic hyperfine couplings: a nucleus of spin *I*
splits the line into *2I+1* equally spaced lines (spacing = A).

**`pepper` (solid / powder).** In rigid samples the g- and A-tensor anisotropy is
not averaged; every molecular orientation resonates at a slightly different field,
producing a **powder pattern**. Open-Sym-EPR diagonalises H(B) at each orientation on
a near-uniform spherical grid, finds the microwave-allowed transitions, weights
them by the transition probability, and sums — the rigorous field-domain method.

**Inputs:** g (scalar or gx,gy,gz), nuclei (isotope + A in mT or tensor), electron
spin S, D/E (for S>½), peak-to-peak linewidth ΔBpp, and η (Lorentz/Gauss mix).

**Tips:** increase *orientations* for smoother powder patterns; anisotropy and
hyperfine splitting both scale predictably with microwave frequency (multifrequency).
"""

ENDOR = r"""
### ENDOR — `salt`

ENDOR (Electron-Nuclear DOuble Resonance) measures **nuclear transition
frequencies** while sitting on an EPR line, dramatically improving resolution of
small hyperfine couplings that are buried in the cw-EPR linewidth.

At a fixed field B₀, the nuclear sublevels in the two electron manifolds
(m_S = ±½) have frequencies set by the competition between the **nuclear Larmor
frequency** ν_n = g_n μ_N B₀ and the **hyperfine coupling** A:

- **Weak coupling** (A < 2ν_n): two lines centred at **ν_n**, split by **A**:  ν = ν_n ± A/2.
- **Strong coupling** (A > 2ν_n): two lines centred at **A/2**, split by 2ν_n: ν = A/2 ± ν_n.

Open-Sym-EPR diagonalises the full Hamiltonian (including nuclear Zeeman) at each
orientation and reports every RF-driven nuclear transition, so it generalises to
anisotropic A and high nuclear spin.

**Inputs:** the static field (mT), the RF frequency window (MHz), and the spin
system. **Read-out:** peaks give |ν_n ± A/2|; the splitting or centring tells you
the coupling regime and hence the hyperfine magnitude.
"""

ESEEM = r"""
### ESEEM — `saffron` (lite)

ESEEM (Electron Spin-Echo Envelope Modulation) is a **pulse-EPR** experiment: the
spin echo amplitude is modulated as a function of the inter-pulse delay by nearby
nuclei. Its Fourier transform yields nuclear frequencies, like ENDOR but from the
time domain.

For an S=½, I=½ pair the modulation depends on the nuclear frequencies of the two
manifolds and the **pseudo-secular** (anisotropic) hyperfine term B:

> ν_α = √((A_s/2 + ν_I)² + (B/2)²),  ν_β = √((A_s/2 − ν_I)² + (B/2)²),  k = (ν_I B / ν_α ν_β)²

- **Two-pulse ESEEM** V(τ): modulated at ν_α, ν_β and their **sum/difference** combinations.
- **Three-pulse ESEEM** V(T): narrower lines at the fundamental frequencies ν_α, ν_β.

**Crucial physics:** ESEEM requires **anisotropic** hyperfine (B ≠ 0). A purely
isotropic coupling gives modulation depth k = 0 — no ESEEM. So ESEEM reports on
the *dipolar* part of the coupling and hence electron–nucleus distances.

**Inputs:** delay axis (µs), field (mT), the spin system (with an anisotropic A).
**Read-out:** the FFT peaks are the nuclear frequencies; the modulation depth
encodes the dipolar coupling.
"""

MAGNETOMETRY = r"""
### Magnetometry — `curry`

`curry` computes thermodynamic magnetic properties from the same spin
Hamiltonian, bridging EPR and SQUID magnetometry.

From the eigenenergies E_i(B) and moments μ_i = −∂E_i/∂B, the Boltzmann partition
function gives the magnetisation and susceptibility:

> Z = Σ exp(−E_i/kT),  ⟨μ⟩ = (1/Z) Σ μ_i e^(−E_i/kT),  M = N_A⟨μ⟩,  χ = N_A⟨μ⟩/B

- **χT vs T** (the magnetochemist's plot): a flat line at high T equals the
  Curie constant **C = 0.12505 g² S(S+1)** cm³ K mol⁻¹ (0.375 for S=½, g=2).
  Deviations at low T reveal **zero-field splitting** and exchange.
- **M vs B** (magnetisation): saturates at **g S** μ_B per molecule at high field
  / low temperature (1 μ_B for S=½, g=2).

Open-Sym-EPR powder-averages over orientations, so anisotropic g and ZFS are handled
correctly. Units are CGS-emu (χT in cm³ K mol⁻¹, M in μ_B), the convention in
molecular magnetism.

**Inputs:** temperature range (K), measuring field (T) for χT; field range (T) and
temperature (K) for M. **Read-out:** the high-T χT gives the spin/g; the low-T
drop gives D; the M(B) saturation gives the ground-state spin.
"""

FITTING = r"""
### Fitting experimental data

Open-Sym-EPR fits a spin-Hamiltonian model to your **experimental** spectrum by
bounded nonlinear least squares.

**Workflow**
1. **Upload** a two-column spectrum (field, intensity); Gauss axes auto-convert to mT.
2. **Choose which parameters to vary** — g (isotropic or gx/gy/gz), each hyperfine
   A, zero-field splitting D/E, linewidth, amplitude scale, and a field-calibration
   shift. Each gets an initial value and bounds.
3. **Run.** Open-Sym-EPR minimises the residual and returns fitted values with
   **standard errors**, the overlay, the residual, and goodness-of-fit
   (R², RMSE, AIC, BIC).

**Why the global pre-search matters.** Derivative cw-EPR spectra have a difficult
cost landscape: when the model lines do not overlap the data there is no gradient
to follow, so a purely local optimiser stalls. Open-Sym-EPR first scores many random
parameter sets on the **integrated absorption envelope** (smooth and positive),
starts the local refinement from the best, and so converges even from a poor
initial guess. Turn this on with *Global pre-search* when your starting parameters
are uncertain.

**Interpreting the fit.** R² → 1 and a structureless residual indicate a good fit;
peaks left in the residual suggest a missing component or anisotropy. The standard
errors quantify how well each parameter is determined — large errors mean the data
do not constrain that parameter (often correlated parameters).

**Uncertainties — two levels.**
1. **Linear (asymptotic) SE** — from the Jacobian covariance σ²(JᵀJ)⁻¹. Fast and
   standard (Bates & Watts, *Nonlinear Regression Analysis*, 1988), but it
   linearises the model and assumes Gaussian, uncorrelated parameters.
2. **Monte-Carlo / bootstrap** (tick *Monte-Carlo error bars*) — refits many
   synthetic spectra built from the best fit plus noise (Gaussian or resampled
   residuals). The spread of the refits gives standard deviations and 95%
   confidence intervals that **capture nonlinearity and parameter correlations**.
   This is the publication-grade error estimate; report the Monte-Carlo interval.
   When MC σ greatly exceeds the linear SE, the linearised error was over-optimistic.
"""

VALIDATION = r"""
### How Open-Sym-EPR is validated

Every solver is checked against analytical limits:

- **Resonance fields** match hν/(gμ_B) to < 0.01 mT (isotropic and axial g).
- **Hyperfine splitting** reproduces the 2I+1 multiplet spacing exactly.
- **Zero-field splitting** of an S=1 triplet gives the expected ±2D/(gμ_B) pattern.
- **ENDOR** reproduces the ν_n ± A/2 weak-coupling lines.
- **Magnetometry** reproduces the Curie law (χT = 0.375 for S=½, g=2) and the
  g·S μ_B magnetisation saturation.

Open-Sym-EPR is an independent, native-Python implementation that solves the
general electron–nuclear spin Hamiltonian directly by matrix diagonalisation; it
contains no third-party simulation code. Pulsed sequences beyond ESEEM (HYSCORE,
DEER), molecular-dynamics-based lineshapes, and slow-motion stochastic-Liouville
treatments are outside its present scope.
"""
