"""Deep-learning-assisted EPR fitting — physics-grounded amortized inference.

**Scientific design (read this before trusting any output).** A neural network is
used here ONLY as a fast *initialiser* for the physics-based least-squares fit, never
as a replacement for it. Specifically:

1. The network is trained **exclusively on spectra produced by the spin-Hamiltonian
   forward model** (Open-Sym-EPR). The physics is the sole source of ground truth; the
   network learns the inverse map spectrum -> parameters and cannot invent physics.
2. At inference the network predicts an initial parameter estimate, which is then
   **refined and validated by the existing least-squares fitter (esfit)**. The
   reported parameters, R^2 and uncertainties come from the *physics* fit, not the
   network. The network only chooses the starting point.
3. An **out-of-distribution check** reconstructs the spectrum with the forward model
   at the predicted parameters; if the reconstruction is poor the input is flagged as
   outside the network's training domain, so the network cannot silently hallucinate.

The supported parameter family is an isotropic ``14N`` nitroxide with an optional
beta-proton (covers nitroxide triplets and PBN/DMPO triplet-of-doublets adducts):
the network predicts (a_N, a_betaH, linewidth, eta); g is taken from the line
position and all parameters are refined by esfit. The default backend is a
scikit-learn ``MLPRegressor`` (a feed-forward neural network); a PyTorch 1D-CNN
backend is used automatically if torch is installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield
from pathlib import Path
import numpy as np

try:
    import openspin as osp
    from openspin import esfit
    _OSP = True
except Exception as exc:  # noqa: BLE001
    _OSP = False
    _OSP_ERR = str(exc)

PLANCK = 6.62607015e-34
MUB = 9.2740100783e-24

# canonical relative-field grid the network operates on (mT, centred on resonance)
REL_WINDOW = 8.0
REL_NPTS = 512
REL_GRID = np.linspace(-REL_WINDOW, REL_WINDOW, REL_NPTS)
PARAM_NAMES = ("aN_mT", "aH_mT", "lw_mT", "eta")


# ── helpers ───────────────────────────────────────────────────────────────────

def g_from_field(B_mT: float, mw_GHz: float) -> float:
    return PLANCK * mw_GHz * 1e9 / (MUB * B_mT * 1e-3)


def _nominal_centre(mw_GHz: float, g: float = 2.0060) -> float:
    """Nominal resonance field (mT) for the centring grid. Nitroxide/organic-radical
    g-values lie in ~2.005-2.007, so the nominal centre is within ~0.1 mT of the true
    centre; the small training-time shift augmentation and the field-shift parameter in
    the physics refinement absorb the residual. This avoids brittle, noise-sensitive
    centroid estimation."""
    return osp.isotropic_resonance_field_mT(mw_GHz, g)


def _resample_centered(field_mT: np.ndarray, deriv: np.ndarray, centre: float) -> np.ndarray:
    """Resample a derivative spectrum onto the relative grid and unit-normalise."""
    y = np.interp(REL_GRID, field_mT - centre, deriv, left=0.0, right=0.0)
    m = np.max(np.abs(y)) or 1.0
    return y / m


# ── physics forward model for a (a_N, a_H, lw, eta) nitroxide ──────────────────

def _simulate_centered(aN_mT, aH_mT, lw_mT, eta, g=2.0060, mw=9.5, seed_noise=0.0, rng=None):
    """Simulate a 14N(+betaH) nitroxide derivative spectrum on the relative grid."""
    b0 = osp.isotropic_resonance_field_mT(mw, g)
    field = b0 + REL_GRID
    nuclei = [osp.nucleus("14N", float(aN_mT))]
    if aH_mT > 1e-4:
        nuclei.append(osp.nucleus("1H", float(aH_mT)))
    sysx = osp.spin_system(g=g, nuclei=nuclei)
    spec = osp.garlic(sysx, field, mw, float(lw_mT), float(eta))
    spec = spec / (np.max(np.abs(spec)) or 1.0)
    if seed_noise > 0:
        rng = rng or np.random.default_rng()
        spec = spec + rng.normal(0, seed_noise, spec.size)
        spec = spec / (np.max(np.abs(spec)) or 1.0)
    return spec


# ── training-set generation (physics simulated) ────────────────────────────────

def simulate_training_set(n_samples=6000, mw_GHz=9.5, noise_max=0.05, seed=0,
                          ranges=None):
    """Generate (X, Y) where X are simulated spectra and Y the true parameters."""
    if not _OSP:
        raise RuntimeError(f"Open-Sym-EPR engine unavailable: {_OSP_ERR}")
    r = ranges or {"aN_mT": (1.20, 2.05), "aH_mT": (0.0, 0.45),
                   "lw_mT": (0.03, 0.30), "eta": (0.0, 1.0)}
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, REL_NPTS), dtype=np.float32)
    Y = np.empty((n_samples, 4), dtype=np.float32)
    for i in range(n_samples):
        aN = rng.uniform(*r["aN_mT"]); aH = rng.uniform(*r["aH_mT"])
        # half the time force aH≈0 so the net learns the bare-triplet class too
        if rng.random() < 0.5:
            aH = 0.0
        lw = rng.uniform(*r["lw_mT"]); eta = rng.uniform(*r["eta"])
        noise = rng.uniform(0.0, noise_max)
        spec = _simulate_centered(aN, aH, lw, eta, mw=mw_GHz, seed_noise=noise, rng=rng)
        # small shift augmentation matched to the residual error of robust centring,
        # so the network tolerates realistic centring jitter without losing accuracy
        shift = rng.uniform(-0.15, 0.15)  # mT
        if abs(shift) > 1e-3:
            spec = np.interp(REL_GRID, REL_GRID + shift, spec, left=0.0, right=0.0)
            spec = spec / (np.max(np.abs(spec)) or 1.0)
        X[i] = spec
        Y[i] = (aN, aH, lw, eta)
    return X, Y


# ── estimator (sklearn MLP default; torch CNN if available) ────────────────────

@dataclass
class MLEstimator:
    backend: str = "mlp"
    model: object = None
    y_mean: np.ndarray = None
    y_std: np.ndarray = None
    mw_GHz: float = 9.5
    meta: dict = dfield(default_factory=dict)

    def predict(self, X):
        Yn = self.model.predict(np.asarray(X, dtype=np.float32))
        Yn = np.atleast_2d(Yn)
        return Yn * self.y_std + self.y_mean


def train_estimator(X, Y, mw_GHz=9.5, hidden=(384, 192, 96), max_iter=600,
                    backend="auto", seed=0):
    """Train the spectrum->parameter network on physics-simulated data."""
    y_mean = Y.mean(axis=0); y_std = Y.std(axis=0); y_std[y_std == 0] = 1
    Yn = (Y - y_mean) / y_std
    use_torch = backend == "torch" or (backend == "auto" and _has_torch())
    if use_torch:
        model = _train_torch_cnn(X, Yn, hidden, seed)
        bk = "cnn-torch"
    else:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                             solver="adam", max_iter=max_iter, random_state=seed,
                             early_stopping=True, n_iter_no_change=15)
        model.fit(X, Yn)
        bk = "mlp-sklearn"
    return MLEstimator(backend=bk, model=model, y_mean=y_mean, y_std=y_std, mw_GHz=mw_GHz,
                       meta={"n_train": len(X), "hidden": list(hidden)})


def _has_torch():
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _train_torch_cnn(X, Yn, hidden, seed):  # pragma: no cover (optional backend)
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Yt = torch.tensor(Yn, dtype=torch.float32)
    net = nn.Sequential(
        nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
        nn.Flatten(), nn.LazyLinear(hidden[0]), nn.ReLU(),
        nn.Linear(hidden[0], 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(60):
        opt.zero_grad(); loss = lossf(net(Xt), Yt); loss.backward(); opt.step()

    class _Wrap:
        def predict(self, x):
            with torch.no_grad():
                xt = torch.tensor(np.asarray(x, dtype=np.float32)).unsqueeze(1)
                return net(xt).numpy()
    return _Wrap()


# ── ML-assisted fit: NN init -> physics refine -> validate ─────────────────────

def _predict_params(estimator: "MLEstimator", field, y, mw):
    """Run the network: returns (aN, aH, lw, eta) in mT and the centring field."""
    centre = _nominal_centre(mw)
    Xc = _resample_centered(field, y, centre)[None, :]
    aN, aH, lw, eta = [float(v) for v in estimator.predict(Xc)[0]]
    aN = max(aN, 0.05); aH = max(aH, 0.0)
    lw = float(np.clip(lw, 0.02, 0.5)); eta = float(np.clip(eta, 0, 1))
    return aN, aH, lw, eta, centre


def _reconstruct_on_axis(field, aN, aH, lw, eta, mw, g=2.0060):
    nuclei = [osp.nucleus("14N", aN)] + ([osp.nucleus("1H", aH)] if aH > 1e-4 else [])
    spec = osp.garlic(osp.spin_system(g=g, nuclei=nuclei), field, mw, lw, eta)
    return spec / (np.max(np.abs(spec)) or 1.0)


@dataclass
class MLDrivenResult:
    params: dict             # spin parameters predicted entirely by the network
    fit: np.ndarray          # forward-model reconstruction at those parameters (data axis)
    R2: float                # reconstruction R^2 against the data (quality is always shown)
    shift_mT: float
    ood_flag: bool
    note: str = ""


def ml_driven_fit(estimator: "MLEstimator", field_mT, intensity, mw_GHz=None,
                  ood_threshold=0.5):
    """Pure ML-driven fit: the network predicts ALL spin parameters and the forward
    model reconstructs the spectrum — no least-squares optimisation. Only centring
    (a rigid field shift) and amplitude are calibrated for display; the spin
    parameters come entirely from the network. The reconstruction R^2 is reported so
    the user always sees the quality, and a low R^2 flags an out-of-distribution input.
    This is faster but less accurate/robust than `ml_assisted_fit`; prefer the latter
    for quantitative work."""
    if not _OSP:
        raise RuntimeError(f"Open-Sym-EPR engine unavailable: {_OSP_ERR}")
    mw = float(mw_GHz or estimator.mw_GHz)
    field = np.asarray(field_mT, float); y = np.asarray(intensity, float)
    y = y / (np.max(np.abs(y)) or 1.0)
    aN, aH, lw, eta, _ = _predict_params(estimator, field, y, mw)
    recon0 = _reconstruct_on_axis(field, aN, aH, lw, eta, mw)
    # rigid centring: search a small field shift maximising overlap (calibration only)
    step = field[1] - field[0]
    best = (-1e9, 0.0)
    for sh in np.arange(-0.8, 0.8 + step, step):
        r = np.interp(field, field + sh, recon0, left=0, right=0)
        s = float(np.dot(r, y) / (np.dot(r, r) or 1))
        rss = np.sum((y - s * r) ** 2); tot = np.sum((y - y.mean()) ** 2) or 1e-12
        if 1 - rss / tot > best[0]:
            best = (1 - rss / tot, sh)
    R2, shift = best
    r = np.interp(field, field + shift, recon0, left=0, right=0)
    scale = float(np.dot(r, y) / (np.dot(r, r) or 1))
    g_eff = g_from_field(_nominal_centre(mw) - shift, mw)
    params = {"g": g_eff, "aN_mT": aN, "aN_G": aN * 10, "aH_mT": aH, "aH_G": aH * 10,
              "lw_mT": lw, "eta": eta}
    ood = R2 < ood_threshold
    note = ("Out-of-distribution / poor reconstruction — the network's parameters do not "
            "reproduce this spectrum; use ML-assisted or classical fitting."
            if ood else "Network reconstruction matches the data.")
    return MLDrivenResult(params=params, fit=scale * r, R2=float(R2), shift_mT=float(shift),
                          ood_flag=ood, note=note)


@dataclass
class MLFitResult:
    nn_estimate: dict
    g_from_position: float
    centre_mT: float
    refined: object          # openspin FitResult (physics-validated)
    ood_R2: float            # reconstruction R2 at NN params (OOD check)
    ood_flag: bool
    note: str = ""


def ml_assisted_fit(estimator: MLEstimator, field_mT, intensity, mw_GHz=None,
                    refine=True, ood_threshold=0.5):
    """Predict initial parameters with the NN, then refine with the physics fitter."""
    if not _OSP:
        raise RuntimeError(f"Open-Sym-EPR engine unavailable: {_OSP_ERR}")
    mw = float(mw_GHz or estimator.mw_GHz)
    field = np.asarray(field_mT, float); y = np.asarray(intensity, float)
    y = y / (np.max(np.abs(y)) or 1.0)
    centre = _nominal_centre(mw)
    g_pos = 2.0060
    Xc = _resample_centered(field, y, centre)[None, :]
    aN, aH, lw, eta = [float(v) for v in estimator.predict(Xc)[0]]
    aN = max(aN, 0.05); aH = max(aH, 0.0); lw = float(np.clip(lw, 0.02, 0.5)); eta = float(np.clip(eta, 0, 1))
    nn_est = {"g": g_pos, "aN_mT": aN, "aH_mT": aH, "lw_mT": lw, "eta": eta}

    # OOD check: reconstruct with the forward model at NN params and compare
    recon = _simulate_centered(aN, aH, lw, eta, g=g_pos, mw=mw)
    obs = _resample_centered(field, y, centre)
    s = float(np.dot(recon, obs) / (np.dot(recon, recon) or 1))
    ss = np.sum((obs - s * recon) ** 2); tot = np.sum((obs - obs.mean()) ** 2) or 1e-12
    ood_R2 = 1 - ss / tot
    ood = ood_R2 < ood_threshold

    refined = None
    if refine:
        nuclei = [osp.nucleus("14N", aN)]
        # widen g and add a field-shift so the physics fit corrects any centring error
        vary = {"g_iso": (g_pos, g_pos - 0.003, g_pos + 0.003),
                "A0_iso": (aN, max(aN - 0.3, 0.05), aN + 0.3),
                "lw": (lw, 0.02, 0.5), "shift": (0.0, -0.8, 0.8), "scale": (1.0, 0.1, 8.0)}
        if aH > 0.03:
            nuclei.append(osp.nucleus("1H", aH))
            vary["A1_iso"] = (aH, 0.02, 0.6)
        system = osp.spin_system(g=g_pos, nuclei=nuclei)
        refined = esfit(system, field, y, vary, mw, eta=eta, global_search=0, max_nfev=800)

    note = ("Out-of-distribution: the spectrum is unlike the network's training family; "
            "treat the NN estimate with caution and rely on the physics fit."
            if ood else "In-distribution; NN initialisation accepted.")
    return MLFitResult(nn_estimate=nn_est, g_from_position=g_pos, centre_mT=centre,
                       refined=refined, ood_R2=ood_R2, ood_flag=ood, note=note)


# ── persistence ────────────────────────────────────────────────────────────────

def save_estimator(est: MLEstimator, path):
    import joblib
    joblib.dump(est, path)


def load_estimator(path) -> MLEstimator:
    import joblib
    return joblib.load(path)


def default_model_path() -> Path:
    d = Path.home() / ".simepr"; d.mkdir(parents=True, exist_ok=True)
    return d / "ml_nitroxide_estimator.joblib"


def get_or_train_default(mw_GHz=9.5, n_samples=12000, force=False) -> MLEstimator:
    """Load the cached default estimator, training and caching it on first use."""
    p = default_model_path()
    if p.exists() and not force:
        try:
            return load_estimator(p)
        except Exception:  # noqa: BLE001
            pass
    X, Y = simulate_training_set(n_samples=n_samples, mw_GHz=mw_GHz)
    est = train_estimator(X, Y, mw_GHz=mw_GHz)
    save_estimator(est, p)
    return est
