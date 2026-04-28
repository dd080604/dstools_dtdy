import numpy as np

def _standardize(X):
    n = X.shape[0]
    xmean = X.mean(axis=0)
    xnorm = np.sqrt(((X - xmean) ** 2).sum(axis=0) / n)
    xnorm[xnorm == 0.0] = 1.0   # guard against constant columns
    return (X - xmean) / xnorm, xmean, xnorm

def lslassoNET(X, y, lam2, ulam, pf=None, pf2=None,
               eps=1e-6, maxit=int(1e6), intr=1):
    """
    Adaptive Elastic-Net along a lambda1 path (warm start + active set).

    Parameters
    ----------
    X     : array (n, p)   Predictor matrix (no intercept column).
    y     : array (n,)     Response vector.
    lam2  : float          Lambda2 for the L2 penalty (fixed/given).
    ulam  : array (nlam,)  Decreasing sequence of lambda1 values.
    pf    : array (p,)     Adaptive L1 weights w_j  (default: all ones).
    pf2   : array (p,)     Relative L2 weights       (default: all ones).
    eps   : float          Convergence threshold (default 1e-6).
    maxit : int            Max coordinate passes     (default 1e6).
    intr  : int            1 = fit intercept, 0 = no intercept.

    Returns
    -------
    b0_path   : array (nlam,)    Intercept for each lambda1.
    beta_path : array (p, nlam)  Coefficients for each lambda1.
    ulam      : array (nlam,)    Lambda1 values used.
    """

    X    = np.asarray(X,    dtype=float)
    y    = np.asarray(y,    dtype=float)
    ulam = np.asarray(ulam, dtype=float)

    nobs, nvars = X.shape
    nlam        = len(ulam)

    if pf  is None: pf  = np.ones(nvars)
    if pf2 is None: pf2 = np.ones(nvars)
    pf  = np.maximum(0.0, np.asarray(pf,  dtype=float))
    pf2 = np.maximum(0.0, np.asarray(pf2, dtype=float))

    # Standardize X
    Xs, xmean, xnorm = _standardize(X)
    maj = 2.0 * np.ones(nvars)

    # Output arrays
    b0_path   = np.zeros(nlam)
    beta_path = np.zeros((nvars, nlam))

    # Working variables (warm-started across lambda loop)
    # b[0] = intercept,  b[1:nvars+1] = coefficients on standardized X
    b     = np.zeros(nvars + 1)
    r     = y.copy()      # current residuals
    npass = 0

    # LAMBDA1 LOOP  (warm start: b and r carry over from previous lambda)
    for l_idx in range(nlam):
        al = ulam[l_idx]          # current lambda1

        # OUTER CONVERGENCE LOOP
        while True:
            oldbeta = b.copy()   

            # MIDDLE LOOP
            while True:
                npass += 1
                dif    = 0.0

                # Full pass over ALL p variables (identifies active set)
                for k in range(nvars):
                    oldb_k    = b[k + 1]
                    u         = np.dot(r, Xs[:, k])
                    u         = maj[k] * oldb_k + u / nobs  

                    threshold = al * pf[k]
                    val       = abs(u) - threshold
                    if val > 0.0:
                        b[k + 1] = np.sign(u) * val / (maj[k] + pf2[k] * lam2)
                    else:
                        b[k + 1] = 0.0

                    d = b[k + 1] - oldb_k
                    if abs(d) > 0.0:
                        dif  = max(dif, d * d)
                        r   -= Xs[:, k] * d   # residual update

                # Intercept update
                if intr == 1:
                    d = r.sum() / nobs
                    if d != 0.0:
                        b[0] += d
                        r    -= d
                        dif   = max(dif, d * d)

                if dif < eps:
                    break
                if npass > maxit:
                    break

                # Active-set inner loop
                # Cycle only over currently nonzero coefficients until convergence.
                active = np.where(b[1:] != 0.0)[0]   

                while True:
                    npass     += 1
                    dif_inner  = 0.0

                    for k in active:
                        oldb_k    = b[k + 1]
                        u         = np.dot(r, Xs[:, k])
                        u         = maj[k] * oldb_k + u / nobs

                        threshold = al * pf[k]
                        val       = abs(u) - threshold
                        if val > 0.0:
                            b[k + 1] = np.sign(u) * val / (maj[k] + pf2[k] * lam2)
                        else:
                            b[k + 1] = 0.0

                        d = b[k + 1] - oldb_k
                        if abs(d) > 0.0:
                            dif_inner  = max(dif_inner, d * d)
                            r         -= Xs[:, k] * d

                    if intr == 1:
                        d = r.sum() / nobs
                        if d != 0.0:
                            b[0] += d
                            r    -= d
                            dif_inner = max(dif_inner, d * d)

                    if dif_inner < eps:
                        break
                    if npass > maxit:
                        break
            # end middle loop

            # Outer convergence check
            # Re-run the full-variable pass above until the outer iterate is stable.
            converged = True
            if (b[0] - oldbeta[0]) ** 2 >= eps:
                converged = False
            if converged:
                for k in range(nvars):
                    if (b[k + 1] - oldbeta[k + 1]) ** 2 >= eps:
                        converged = False
                        break
            if converged:
                break
        # end outer loop

        # Back-transform coefficients to original scale 
        beta_orig = b[1:] / xnorm
        b0_orig   = b[0] - np.dot(beta_orig, xmean)

        beta_path[:, l_idx] = beta_orig
        b0_path[l_idx]      = b0_orig

    return b0_path, beta_path, ulam
