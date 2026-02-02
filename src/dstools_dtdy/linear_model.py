import numpy as np

class MyModel:
    def __init__(self, X, y):
        self.X_raw = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)

        if self.X_raw.ndim == 1:
            self.X_raw = self.X_raw.reshape(-1, 1)

        self.beta_hat = None
        self.add_intercept = None
        self.X_design = None

    def mylm(self, add_intercept=True):
        X = self.X_raw
        y = self.y
        self.add_intercept = add_intercept

        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            Xd = np.concatenate((intercept, X), axis=1)
        else:
            Xd = X
        self.X_design = Xd

        self.beta_hat = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y
        return self.beta_hat

    def fitted_values(self):
        return self.X_design @ self.beta_hat

    def residuals(self):
        return self.y - self.fitted_values()

    def coef(self): 
        return self.beta_hat     
