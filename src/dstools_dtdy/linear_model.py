from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearModelFit:
    coefficients: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n, p).")
    return X


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if y.ndim != 1:
        raise ValueError("y must be shape (n,) or (n, 1).")
    return y


def _design_matrix(X: np.ndarray, add_intercept: bool) -> np.ndarray:
    if not add_intercept:
        return X
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def _fit_numpy(Xd: np.ndarray, y: np.ndarray) -> np.ndarray:
    # normal equation, but use solve (more stable than inv)
    return np.linalg.solve(Xd.T @ Xd, Xd.T @ y)


def mylm(X, y, add_intercept: bool = True, engine: str = "numpy") -> LinearModelFit:
    """
    Fit least squares using the normal equation.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
    y : array-like, shape (n,) or (n,1)
    add_intercept : bool, default True
    engine : {"numpy", "cython"}, default "numpy"

    Returns
    -------
    LinearModelFit with attributes:
      - coefficients
      - residuals
      - fitted_values
    """
    X = _as_2d(X)
    y = _as_1d(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows/observations.")

    Xd = _design_matrix(X, add_intercept)

    engine = engine.lower()
    if engine == "numpy":
        beta = _fit_numpy(Xd, y)
    elif engine == "cython":
        # compiled extension (see linear_model_cython.pyx)
        from .linear_model_cython import fit_normal_equation_cython
        beta = fit_normal_equation_cython(Xd, y)
    else:
        raise ValueError("engine must be 'numpy' or 'cython'.")

    fitted = Xd @ beta
    resid = y - fitted
    return LinearModelFit(coefficients=beta, residuals=resid, fitted_values=fitted)


def coef(fit: LinearModelFit) -> np.ndarray:
    return fit.coefficients


def fitted_values(fit: LinearModelFit) -> np.ndarray:
    return fit.fitted_values


def residuals(fit: LinearModelFit) -> np.ndarray:
    return fit.residuals


def predict(fit: LinearModelFit, X, add_intercept: bool = True) -> np.ndarray:
    X = _as_2d(X)
    Xd = _design_matrix(X, add_intercept)
    return Xd @ fit.coefficients


def cv_mylm(
    X,
    y,
    k: int = 5,
    add_intercept: bool = True,
    engine: str = "numpy",
    shuffle: bool = True,
    random_state=None,
):
    """
    k-fold CV for mylm. Returns (per_fold_mse, mean_mse).
    """
    X = _as_2d(X)
    y = _as_1d(y)

    n = X.shape[0]
    if k < 2 or k > n:
        raise ValueError("k must be between 2 and n (inclusive).")

    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    if shuffle:
        rng.shuffle(idx)

    folds = np.array_split(idx, k)

    mses = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        fit = mylm(X[train_idx], y[train_idx], add_intercept=add_intercept, engine=engine)
        y_pred = predict(fit, X[test_idx], add_intercept=add_intercept)

        mse = float(np.mean((y[test_idx] - y_pred) ** 2))
        mses.append(mse)

    mses = np.asarray(mses, dtype=float)
    return mses, float(mses.mean())
