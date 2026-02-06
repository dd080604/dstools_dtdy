# dstools_dtdy

Package for fitting a least-squares linear model using the closed-form normal equation:
$\hat{\beta} = (X^\top X)^{-1}X^\top y$

## Example 
```
from dstools_dtdy import mylm, coef, fitted_values, residuals 
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(5, 2))
y = X @ np.array([1, 2]) + 1 + rng.normal(size=5)

fit = mylm(X, y)

print(coef(fit))
print(fitted_values(fit))
print(residuals(fit))
```
