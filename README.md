# dstools_dtdy

Package for fitting a least-squares linear model using the closed-form normal equation:
$\hat{\beta} = (X^\top X)^{-1}X^\top y$

Newest update includes:
* Cython implementation of `mylm()`
* `cv_mylm()`: k-fold CV functionality that returns per-fold MSEs and mean MSE.

## General Example 
```
from dstools_dtdy import mylm, coef, fitted_values, residuals 
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(5, 2))
y = X @ np.array([1, 2]) + 1 + rng.normal(size=5)

fit = mylm(X, y, engine="numpy") # Or engine="cython"

print(coef(fit))
print(fitted_values(fit))
print(residuals(fit))
```
## K-fold CV Example

```
from dstools_dtdy import cv_mylm
import numpy

rng = np.random.default_rng(1)
X = rng.normal(size=(5, 2))
y = X @ np.array([1, 2]) + 1 + rng.normal(size=5)
cv_mylm(X, y, k=5, add_intercept=True, engine="numpy", shuffle=True, random_state=42) # Or engine="cython"

# Returns (per-fold MSEs, mean MSE)
```

## Cython Comparison


|    n |   p |   numpy_mean |   numpy_sd |   cython_mean |   cython_sd |   speedup |
|-----:|----:|-------------:|-----------:|--------------:|------------:|----------:|
|  200 |   5 |     6.8e-05  |    6.7e-05 |      0.000117 |     0.00017 |  0.577904 |
|  500 |  10 |     5.8e-05  |    1.9e-05 |      9.9e-05  |     1.3e-05 |  0.586292 |
| 1000 |  20 |     0.00016  |    8.2e-05 |      0.000575 |     1.3e-05 |  0.278836 |
| 2000 |  30 |     0.000409 |    3.9e-05 |      0.002436 |     8.7e-05 |  0.167776 |

Note: This is a class assignment; not intended for full scale production.
