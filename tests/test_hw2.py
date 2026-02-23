import numpy as np
import pytest

from dstools_dtdy import mylm, coef, fitted_values, residuals, cv_mylm


def test_numpy_vs_cython_coefficients_agree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + rng.normal(size=80)

    fit_np = mylm(X, y, engine="numpy")
    fit_cy = mylm(X, y, engine="cython")

    np.testing.assert_allclose(coef(fit_np), coef(fit_cy), rtol=1e-8, atol=1e-8)


def test_residual_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)

    fit = mylm(X, y, engine="numpy")
    np.testing.assert_allclose(
        residuals(fit),
        y.reshape(-1) - fitted_values(fit),
        rtol=1e-12,
        atol=1e-12,
    )


def test_cv_returns_k_scores_and_means_close_same_split():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = rng.normal(size=60)

    mses_np, mean_np = cv_mylm(X, y, k=5, engine="numpy", shuffle=True, random_state=123)
    mses_cy, mean_cy = cv_mylm(X, y, k=5, engine="cython", shuffle=True, random_state=123)

    assert len(mses_np) == 5
    assert len(mses_cy) == 5

    # very small numerical differences are possible; this tolerance is tight but realistic here
    assert abs(mean_np - mean_cy) < 1e-8
