import numpy as np

def mylm(X, y, add_intercept=True):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.ndim == 1:
      X = X.reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
      raise ValueError("X and y must have the same number of rows/observations.")

    if add_intercept:
      intercept = np.ones((X.shape[0], 1))
      Xd = np.concatenate((intercept, X), axis=1)
    else:
      Xd = X

    beta_hat = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y
    y_hat = Xd @ beta_hat
    resid = y - y_hat

    fit = {
        "beta_hat": beta_hat,
        "X_design": Xd, 
        "y": y,
        "fitted_values": y_hat,
        "residuals": resid,
    }
    return fit


def coef(fit):
    return fit["beta_hat"]

def fitted_values(fit):
    return fit["fitted_values"]

def residuals(fit):
    return fit["residuals"]
