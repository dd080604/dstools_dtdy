import numpy as np
import ctypes

lib = ctypes.CDLL("./qr.so")

lib.householder_qr.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double),
    np.ctypeslib.ndpointer(dtype=np.double),
    np.ctypeslib.ndpointer(dtype=np.double)
]

def householder_qr(A):
    A = np.asarray(A, dtype=np.double)
    m, n = A.shape

    Q = np.zeros((m, m), dtype=np.double)
    R = np.zeros((m, n), dtype=np.double)

    lib.householder_qr(m, n, A, Q, R)

    return Q, R
