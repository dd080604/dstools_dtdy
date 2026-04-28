"""
aenet.py
========
Adaptive Elastic-Net regression with cross-validation for dstools.

Public API
----------
aenet(X, y, lam2, ulam, K=5)
    Fit adaptive elastic-net on a fixed (lam2, lambda1-path).
    Uses cvridge (cvmin) internally to obtain adaptive weights.

cvaenet(X, y, lam1_grid=None, lam2_grid=None, K=5, nlam=20, seed=None)
    2-D cross-validation over a grid of (lambda1, lambda2) values.
    Returns cvmin/cvupper/cvlower surfaces, the best (lam1, lam2), and
    the full-data coefficients at that optimum.

Notes
-----
- lam2 = 0 is supported (reduces to adaptive lasso).
- lam1 = 0 is supported (reduces to ridge, but we still use the
  coordinate-descent solver; for pure ridge cvridge is faster).
- Both lam1 = 0 AND lam2 = 0 simultaneously is NOT supported.
- Adaptive weights are computed via cvridge (cvmin rule) once per
  (lam2, fold) combination so they are always consistent with the
  data seen during training.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================================ #
# IMPORTS
# ============================================================================ #

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lslassonet import lslassoNET, _standardize
from ridge_regression import cvridge, predict_ridge


# ============================================================================ #
# INTERNAL HELPER: wraps cvridge with the cvmin rule
# Returns (best_lambda, beta_ridge_at_best_lambda)
# ============================================================================ #

def _cvridge_min(X, y, lamb, K=5):
    """
    Run cvridge and apply the cvmin rule.
    Returns the best ridge lambda and the corresponding coefficient vector.
    """
    cv   = cvridge(X, y.copy(), lamb, K=K)   # y.copy() guards against in-place bug
    idx  = int(np.argmin(cv["cvm"]))
    return cv["lambda"][idx], cv["betas"][:, idx]



# PUBLIC: aenet

def aenet(X, y, lam2, ulam, K=5):
    """
    Adaptive Elastic-Net: fit on the full dataset for a given lam2 and a
    sequence of lam1 values.

    Adaptive weights are determined by running K-fold CV ridge (cvmin rule)
    with regularization parameter lam2, then setting w_j = 1/|beta_j^ridge|.

    Parameters
    ----------
    X     : array (n, p)    Predictor matrix.
    y     : array (n,)      Response vector.
    lam2  : float >= 0      L2 penalty.  Set to 0 for adaptive lasso.
    ulam  : array (nlam,)   Decreasing sequence of lam1 values.
    K     : int             Number of CV folds for ridge weight estimation.

    Returns
    -------
    dict with keys:
        "b0"        : array (nlam,)   Intercepts.
        "beta"      : array (p, nlam) Coefficient matrix.
        "lam1"      : array (nlam,)   Lambda1 values.
        "lam2"      : float           Lambda2 used.
        "pf"        : array (p,)      Adaptive weights.
        "lam2_ridge": float           Best ridge lambda from cvridge.
    """
    X    = np.asarray(X, dtype=float)
    y    = np.asarray(y, dtype=float)
    ulam = np.asarray(ulam, dtype=float)
    n, p = X.shape

    # ---- Adaptive weights via cvridge (cvmin) ----
    if lam2 == 0.0:
        # Pure adaptive lasso: use a small default ridge lambda for weights
        lamb_ridge = np.exp(np.linspace(np.log(max(n, p)), np.log(0.01), 50))
    else:
        lamb_ridge = np.exp(np.linspace(np.log(lam2 * n * 10), np.log(lam2 * n * 0.01), 50))

    best_lam2_ridge, beta_ridge = _cvridge_min(X, y, lamb_ridge, K=K)
    pf = 1.0 / np.maximum(np.abs(beta_ridge), 1e-8)   # guard against zeros

    # ---- Fit adaptive elastic-net path ----
    b0_path, beta_path, lam1_path = lslassoNET(
        X, y, lam2=lam2, ulam=ulam, pf=pf
    )

    return {
        "b0"         : b0_path,
        "beta"       : beta_path,
        "lam1"       : lam1_path,
        "lam2"       : lam2,
        "pf"         : pf,
        "lam2_ridge" : best_lam2_ridge,
    }


# ============================================================================ #
# PUBLIC: cvaenet
# ============================================================================ #

def cvaenet(X, y, lam1_grid=None, lam2_grid=None, K=5, nlam=20, seed=None):
    """
    2-D cross-validation for adaptive elastic-net over a grid of
    (lambda1, lambda2) values.

    For each lam2 in lam2_grid:
      1. Compute the lam1 sequence on the FULL data (fixes the grid axis).
      2. For each CV fold, re-estimate adaptive weights on the training data
         using cvridge (cvmin), then fit lslassoNET on that training fold
         evaluated at the shared lam1 grid.
      3. Collect test MSE across all folds.

    Parameters
    ----------
    X         : array (n, p)
    y         : array (n,)
    lam1_grid : array (nlam,) or None
        Explicit lam1 values to evaluate. If None, a log-spaced grid of
        length `nlam` is computed automatically per lam2 from the full data.
    lam2_grid : array or None
        Grid of lam2 values.  Default: 8 log-spaced values from 2.0 to 0.001.
    K         : int    Number of CV folds.
    nlam      : int    Length of auto-generated lam1 grid per lam2.
    seed      : int    Random seed for fold assignment.

    Returns
    -------
    dict with keys:
        "cvm"       : array (nlam, n_lam2)   Mean CV error surface.
        "cvupper"   : array (nlam, n_lam2)   cvm + 1 SE.
        "cvlower"   : array (nlam, n_lam2)   cvm - 1 SE.
        "lam1_grid" : list of arrays         lam1 grids, one per lam2.
        "lam2_grid" : array (n_lam2,)        lam2 values.
        "best_lam1" : float                  lam1 at cvmin.
        "best_lam2" : float                  lam2 at cvmin.
        "best_i"    : int                    lam1 index at cvmin.
        "best_j"    : int                    lam2 index at cvmin.
        "b0"        : float                  Intercept at cvmin (full data).
        "beta"      : array (p,)             Coefficients at cvmin (full data).
        "cv_errors" : array (nlam, n_lam2, K) Raw fold errors.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    if seed is not None:
        np.random.seed(seed)

    if lam2_grid is None:
        lam2_grid = np.exp(np.linspace(np.log(2.0), np.log(0.001), 8))
    lam2_grid = np.asarray(lam2_grid, dtype=float)
    n_lam2    = len(lam2_grid)

    # ---- Build a fixed lam1 grid for each lam2 (from full data) ----
    # This ensures the CV error surface has a consistent lam1 axis per lam2.
    ulam_per_lam2 = []
    pf_per_lam2   = []

    for lam2 in lam2_grid:
        if lam2 == 0.0:
            lamb_ridge = np.exp(np.linspace(np.log(max(n, p)), np.log(0.01), 50))
        else:
            lamb_ridge = np.exp(np.linspace(np.log(lam2*n*10), np.log(lam2*n*0.01), 50))

        _, beta_ridge = _cvridge_min(X, y, lamb_ridge, K=K)
        pf_full = 1.0 / np.maximum(np.abs(beta_ridge), 1e-8)

        if lam1_grid is not None:
            ulam_j = np.asarray(lam1_grid, dtype=float)
        else:
            Xs_full, _, _ = _standardize(X)
            lm     = np.max(np.abs(Xs_full.T @ (y - y.mean())) / (n * pf_full))
            ulam_j = np.exp(np.linspace(np.log(lm), np.log(lm * 0.001), nlam))

        ulam_per_lam2.append(ulam_j)
        pf_per_lam2.append(pf_full)

    actual_nlam = len(ulam_per_lam2[0])

    # ---- CV fold assignment (shared across all lam2) ----
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)

    # cv_errors[i, j, k] = test MSE at (lam1_i, lam2_j, fold_k)
    cv_errors = np.zeros((actual_nlam, n_lam2, K))

    for j, lam2 in enumerate(lam2_grid):
        ulam_j = ulam_per_lam2[j]

        for k in range(K):
            test_idx  = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
            Xtr, ytr  = X[train_idx], y[train_idx]
            Xte, yte  = X[test_idx],  y[test_idx]
            ntr       = len(train_idx)

            # Re-estimate adaptive weights on THIS training fold
            if lam2 == 0.0:
                lamb_ridge = np.exp(np.linspace(np.log(max(ntr, p)), np.log(0.01), 50))
            else:
                lamb_ridge = np.exp(np.linspace(np.log(lam2*ntr*10), np.log(lam2*ntr*0.01), 50))

            _, beta_ridge_k = _cvridge_min(Xtr, ytr, lamb_ridge, K=min(K, ntr))
            pf_k = 1.0 / np.maximum(np.abs(beta_ridge_k), 1e-8)

            # Fit on training fold at the shared lam1 grid
            b0k, betak, _ = lslassoNET(Xtr, ytr, lam2=lam2, ulam=ulam_j, pf=pf_k)

            # Predict on test fold: shape (nte, actual_nlam)
            preds = Xte @ betak + b0k   # broadcasting: (nte,p)@(p,nlam) + (nlam,)
            cv_errors[:, j, k] = np.mean((yte[:, None] - preds) ** 2, axis=0)

    # ---- Summarize CV surface ----
    cvm     = cv_errors.mean(axis=2)               # (nlam, n_lam2)
    cvse    = cv_errors.std(axis=2) / np.sqrt(K)   # (nlam, n_lam2)
    cvupper = cvm + cvse
    cvlower = cvm - cvse

    # ---- cvmin: pick (lam1, lam2) with smallest mean CV error ----
    best_i, best_j = np.unravel_index(np.argmin(cvm), cvm.shape)
    best_lam1      = float(ulam_per_lam2[best_j][best_i])
    best_lam2      = float(lam2_grid[best_j])

    # ---- Refit on full data at (best_lam1, best_lam2) ----
    ulam_best = ulam_per_lam2[best_j]
    pf_best   = pf_per_lam2[best_j]
    b0_full, beta_full, _ = lslassoNET(X, y, lam2=best_lam2, ulam=ulam_best, pf=pf_best)

    return {
        "cvm"       : cvm,
        "cvupper"   : cvupper,
        "cvlower"   : cvlower,
        "lam1_grid" : ulam_per_lam2,
        "lam2_grid" : lam2_grid,
        "best_lam1" : best_lam1,
        "best_lam2" : best_lam2,
        "best_i"    : int(best_i),
        "best_j"    : int(best_j),
        "b0"        : float(b0_full[best_i]),
        "beta"      : beta_full[:, best_i],
        "cv_errors" : cv_errors,
    }
