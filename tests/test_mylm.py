import numpy as np
import pytest
import dstools_dtdy.linear_model as lm


def test_shapes_with_intercept_default():
    rng = np.random.default_rng(1)
    n, p = 5, 2
    X = rng.normal(size=(n, p))
    y = X @ np.array([1, 2]) + 1 + rng.normal(size=n)

    fit = lm.mylm(X, y)  # intercept included by default

    assert lm.coef(fit).shape == (p + 1,)
    assert lm.fitted_values(fit).shape == (n,)
    assert lm.residuals(fit).shape == (n,)


def test_shapes_without_intercept():
    rng = np.random.default_rng(2)
    n, p = 6, 3
    X = rng.normal(size=(n, p))
    beta = np.array([0.5, -1.0, 2.0])
    y = X @ beta + rng.normal(size=n)

    fit = lm.mylm(X, y, add_intercept=False)

    assert lm.coef(fit).shape == (p,)
    assert lm.fitted_values(fit).shape == (n,)
    assert lm.residuals(fit).shape == (n,)


def test_residual_identity():
    rng = np.random.default_rng(3)
    n, p = 10, 2
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    fit = lm.mylm(X, y)

    np.testing.assert_allclose(
        lm.residuals(fit),
        fit["y"] - lm.fitted_values(fit),
        rtol=1e-12,
        atol=1e-12,
    )


def test_raises_on_mismatched_rows():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=4)

    with pytest.raises(ValueError):
        lm.mylm(X, y)
