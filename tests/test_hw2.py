import numpy as np
import pytest

from dstools_dtdy import mylm, coef, fitted_values, residuals, cv_mylm


def test_shapes_with_intercept_default():
    rng = np.random.default_rng(1)
    n, p = 5, 2
    X = rng.normal(size=(n, p))
    y = X @ np.array([1, 2]) + 1 + rng.normal(size=n)

    fit = mylm(X, y)  # intercept included by default (engine defaults to numpy)

    assert coef(fit).shape == (p + 1,)
    assert fitted_values(fit).shape == (n,)
    assert residuals(fit).shape == (n,)


def test_shapes_without_intercept():
    rng = np.random.default_rng(2)
    n, p = 6, 3
    X = rng.normal(size=(n, p))
    beta = np.array([0.5, -1.0, 2.0])
    y = X @ beta + rng.normal(size=n)

    fit = mylm(X, y, add_intercept=False)

    assert coef(fit).shape == (p,)
    assert fitted_values(fit).shape == (n,)
    assert residuals(fit).shape == (n,)


def test_residual_identity_holds():
    rng = np.random.default_rng(3)
    n, p = 10, 2
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    fit = mylm(X, y)

    np.testing.assert_allclose(
        residuals(fit),
        y.reshape(-1) - fitted_values(fit),
        rtol=1e-12,
        atol=1e-12,
    )


def test_raises_on_mismatched_rows():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=4)  # wrong length

    with pytest.raises(ValueError):
        mylm(X, y)

def _cython_available() -> bool:
    """
    Returns True if the cython backend is importable/built.
    We detect it by trying to run one cython fit.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=5)
    try:
        _ = mylm(X, y, engine="cython")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cython_available(), reason="Cython engine not available (extension not built).")
def test_numpy_vs_cython_coefficients_agree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + rng.normal(size=80)

    fit_np = mylm(X, y, engine="numpy")
    fit_cy = mylm(X, y, engine="cython")

    np.testing.assert_allclose(coef(fit_np), coef(fit_cy), rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not _cython_available(), reason="Cython engine not available (extension not built).")
def test_cv_returns_k_scores_and_means_close_same_split():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = rng.normal(size=60)

    # Same split via same random_state
    mses_np, mean_np = cv_mylm(X, y, k=5, engine="numpy", shuffle=True, random_state=123)
    mses_cy, mean_cy = cv_mylm(X, y, k=5, engine="cython", shuffle=True, random_state=123)

    assert len(mses_np) == 5
    assert len(mses_cy) == 5

    # Tight tolerance because splits & computations should match extremely closely
    assert abs(mean_np - mean_cy) < 1e-8
