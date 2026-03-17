import numpy as np

def my_lm_QR(X, y, add_intercept=True):

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if add_intercept:
        X = np.column_stack((np.ones(len(X)), X))

    Q, R = householder_qr(X)

    Qt_y = Q.T @ y

    beta = np.linalg.solve(R[:R.shape[1], :], Qt_y[:R.shape[1]])

    return beta
