import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def ridge(X, y, lamb, standardize=True):
  if standardize:
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    my = np.mean(y)
    y -= my
      
  u, d, vt = np.linalg.svd(X, full_matrices=True)
  m = sum(d > 0)
  d = d[range(m)]
  M = np.empty([p, m])
    
  for j in range(m):
    M[:, j] = u[:, j] @ y * vt[j, :]
      
  betas = np.empty([p, len(lamb)]); b0 = 0 * lamb
    
  for l in range(len(lamb)):
    c_lamb = d / (d * d + lamb[l])
    betas[:, l] = M @ c_lamb
    if standardize:
      betas[:, l] /= scaler.scale_
      b0[l] = my - betas[:, l] @ scaler.mean_
    else:
      b0[l] = sum(y - X @ betas[:, l])
  return {"betas":betas, "b0":b0}


def predict_ridge(obj, Xnew):
    if Xnew.ndim == 1:
        Xnew = Xnew.reshape(-1, 1)
    p = obj['betas'].shape[0]
    if Xnew.shape[1] != p:
        raise ValueError("wrong dim: Xnew")
    
    y = Xnew @ obj['betas']  
    y = y + obj['b0'] 
    return y


def coef_ridge(obj):
    return obj['betas']


def cvridge(X, y, lamb, K=None, standardize=True):
    n, p = X.shape
    if K is None:
        K = n
    num_lamb = len(lamb)

    if K == n:
        if standardize:
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            my = np.mean(y)
            ys = y - my
        else:
            Xs = X
            ys = y.copy()
 
        u, d, vt = np.linalg.svd(Xs, full_matrices=False)
 
        cverr = np.empty(num_lamb)
        cvse = np.empty(num_lamb)
 
        for l in range(num_lamb):
            weights = d ** 2 / (d ** 2 + lamb[l])
            h_diag = np.sum(u ** 2 * weights, axis=1)
            yhat = u @ (weights * (u.T @ ys))
            sq_errors = ((ys - yhat) / (1 - h_diag)) ** 2
            cverr[l] = np.mean(sq_errors)
            cvse[l] = np.std(sq_errors) / np.sqrt(n)
 
        obj = ridge(X, y, lamb, standardize=standardize)
 
        return {"lambda": lamb, "cvm": cverr, "cvse": cvse,
                "betas": coef_ridge(obj), "b0": obj["b0"]}
 
    # K-fold CV
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)
 
    fold_errors = np.zeros((K, num_lamb))
 
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
 
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
 
        obj = ridge(X_train, y_train, lamb, standardize=standardize)
        preds = predict_ridge(obj, X_test)
 
        fold_errors[k, :] = np.mean((y_test[:, None] - preds) ** 2, axis=0)
 
    cverr = np.mean(fold_errors, axis=0)
    cvse = np.std(fold_errors, axis=0) / np.sqrt(K)
 
    obj_full = ridge(X, y, lamb, standardize=standardize)
 
    return {"lambda": lamb, "cvm": cverr, "cvse": cvse,
            "betas": coef_ridge(obj_full), "b0": obj_full["b0"]}


def plot_cvridge(cvobj, xvar="lambda"):
    cvm = cvobj["cvm"]
    cvse = cvobj["cvse"]
    lamb = cvobj["lambda"]
    betas = cvobj["betas"]
 
    # L2 norms
    l2_norms = np.sqrt(np.sum(betas ** 2, axis=0))
 
    # lambda with smallest CV error
    idx_min = np.argmin(cvm)
 
    fig, ax = plt.subplots(figsize=(8, 6))
 
    if xvar == "lambda":
        xvals = np.log10(lamb)
        xlabel = r"$\log_{10}(\lambda)$"
    elif xvar == "norm":
        xvals = l2_norms
        xlabel = r"$\|\beta\|_2$"
    else:
        raise ValueError("xvar must be 'lambda' or 'norm'")
 
    ax.errorbar(xvals, cvm, yerr=cvse, fmt='o', color='red',
                ecolor='grey', elinewidth=1, capsize=3, markersize=4)
 
    ax.axvline(x=xvals[idx_min], linestyle='--', color='blue', alpha=0.7,
               label=r"$\lambda_{\min}$")
 
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Mean-Squared Error", fontsize=12)
    ax.set_title("Cross-Validation for Ridge Regression", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
