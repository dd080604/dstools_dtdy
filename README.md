# dstools_dtdy

Package for fitting a least-squares linear model using the closed-form normal equation:
$\hat{\beta} = (X^\top X)^{-1}X^\top y$

## Example 
```
import numpy as np
from dstools_dtdy import mylm, coef

rng = np.random.default_rng(1)
X = rng.normal(size=(5, 2))
y = X @ np.array([1, 2]) + 1 + rng.normal(size=5)

fit = mylm(X, y)          # intercept included by default
print(coef(fit))          # should print 3 coefficients (intercept + 2 slopes)
```
