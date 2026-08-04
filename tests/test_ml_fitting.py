"""Tests for the deep-learning-assisted (physics-grounded) fitting module."""
import numpy as np
import pytest

ml = pytest.importorskip("epr_simfit.ml_fitting")
pytestmark = pytest.mark.skipif(not ml._OSP, reason="Open-Sym-EPR engine unavailable")

try:
    import sklearn  # noqa: F401
    _SK = True
except Exception:  # noqa: BLE001
    _SK = False
pytestmark = [pytestmark, pytest.mark.skipif(not _SK, reason="scikit-learn not installed")]


@pytest.fixture(scope="module")
def estimator():
    X, Y = ml.simulate_training_set(n_samples=1500, mw_GHz=9.5, seed=1)
    return ml.train_estimator(X, Y, mw_GHz=9.5, max_iter=200, seed=1)


def test_training_set_shapes():
    X, Y = ml.simulate_training_set(n_samples=200, seed=0)
    assert X.shape == (200, ml.REL_NPTS)
    assert Y.shape == (200, 4)
    assert np.isfinite(X).all() and np.isfinite(Y).all()


def test_nn_recovers_parameters_on_holdout(estimator):
    Xt, Yt = ml.simulate_training_set(n_samples=200, mw_GHz=9.5, seed=42)
    pred = estimator.predict(Xt)
    # aN and lw should be recovered to a small fraction of their range
    aN_mae = np.mean(np.abs(pred[:, 0] - Yt[:, 0]))
    lw_mae = np.mean(np.abs(pred[:, 2] - Yt[:, 2]))
    assert aN_mae < 0.10      # mT
    assert lw_mae < 0.05      # mT


def test_ml_assisted_fit_refines_to_physics(estimator):
    mw = 9.5
    import openspin as osp
    b0 = osp.isotropic_resonance_field_mT(mw, 2.006)
    field = b0 + ml.REL_GRID
    spec = ml._simulate_centered(1.55, 0.0, 0.12, 0.5, mw=mw, seed_noise=0.02,
                                 rng=np.random.default_rng(7))
    res = ml.ml_assisted_fit(estimator, field, spec, mw_GHz=mw)
    assert not res.ood_flag
    assert res.refined is not None
    v = dict(zip(res.refined.params["parameter"], res.refined.params["value"]))
    assert res.refined.metrics["R2"] > 0.9
    assert abs(v["A0_iso"] - 1.55) < 0.1            # physics-validated a_N
    assert abs(v["g_iso"] - 2.006) < 2e-3


def test_ml_driven_fit_reconstructs(estimator):
    mw = 9.5
    import openspin as osp
    b0 = osp.isotropic_resonance_field_mT(mw, 2.006)
    field = b0 + ml.REL_GRID
    spec = ml._simulate_centered(1.55, 0.0, 0.12, 0.5, mw=mw, seed_noise=0.02,
                                 rng=np.random.default_rng(11))
    d = ml.ml_driven_fit(estimator, field, spec, mw_GHz=mw)
    # pure NN-driven reconstruction should be reasonable and recover a_N roughly
    assert d.R2 > 0.6
    assert abs(d.params["aN_mT"] - 1.55) < 0.15
    assert "aN_G" in d.params and not d.ood_flag


def test_out_of_distribution_flagged(estimator):
    mw = 9.5
    import openspin as osp
    field = osp.isotropic_resonance_field_mT(mw, 2.006) + ml.REL_GRID
    noise = np.random.default_rng(3).normal(0, 1, field.size)
    res = ml.ml_assisted_fit(estimator, field, noise, mw_GHz=mw, refine=False)
    assert res.ood_flag and res.ood_R2 < 0.5
