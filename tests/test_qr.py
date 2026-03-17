import numpy as np
import pytest

from dstools_dtdy import mylm, coef, householder_qr, my_lm_QR

def test_qr_reconstructs_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 3))
    Q, R = householder_qr(A)
    np.testing.assert_allclose(Q @ R, A, rtol=1e-8, atol=1e-8)

def test_q_is_orthonormal():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 3))
    Q, R = householder_qr(A)
    I = np.eye(Q.shape[0])
    np.testing.assert_allclose(Q.T @ Q, I, rtol=1e-8, atol=1e-8)

def test_my_lm_qr_matches_mylm():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + rng.normal(size=50)

    fit_ne = mylm(X, y)
    fit_qr = my_lm_QR(X, y)

    np.testing.assert_allclose(coef(fit_qr), coef(fit_ne), rtol=1e-8, atol=1e-8)