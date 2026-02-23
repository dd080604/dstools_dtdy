# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

def fit_normal_equation_cython(np.ndarray[np.float64_t, ndim=2] X,
                               np.ndarray[np.float64_t, ndim=1] y):
    """
    Compute beta = (X'X)^-1 X'y.
    We compute X'X and X'y using typed loops (fast),
    then solve using NumPy (stable).
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t p = X.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[np.float64_t, ndim=2] XtX = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Xty = np.zeros(p, dtype=np.float64)

    for i in range(n):
        for j in range(p):
            Xty[j] += X[i, j] * y[i]
            for k in range(p):
                XtX[j, k] += X[i, j] * X[i, k]

    return np.linalg.solve(XtX, Xty)
