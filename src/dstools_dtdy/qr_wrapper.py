import os
import ctypes
import numpy as np

# Path to the shared library inside the package directory
_lib_path = os.path.join(os.path.dirname(__file__), "qr.so")
lib = ctypes.CDLL(_lib_path)

lib.householder_qr.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),
]

lib.householder_qr.restype = None


def householder_qr(A):
    A = np.asarray(A, dtype=np.double, order="C")
    m, n = A.shape

    Q = np.zeros((m, m), dtype=np.double, order="C")
    R = np.zeros((m, n), dtype=np.double, order="C")

    lib.householder_qr(m, n, A, Q, R)
    return Q, R
