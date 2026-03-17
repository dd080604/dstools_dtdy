import numpy as np
from .qr_wrapper import householder_qr
from .linear_model import LinearModelFit

def my_lm_QR(X, y, add_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if add_intercept:
        X = np.column_stack((np.ones(X.shape[0]), X))

    Q, R = householder_qr(X)

    p = X.shape[1]
    beta = np.linalg.solve(R[:p, :p], (Q.T @ y)[:p])
    fitted = X @ beta
    resid = y - fitted

    return LinearModelFit(
        coefficients=beta,
        residuals=resid,
        fitted_values=fitted,
    )
