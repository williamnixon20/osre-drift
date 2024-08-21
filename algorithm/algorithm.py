from sklearn.base import BaseEstimator
from sklearn.base import clone

class Algorithm:
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.is_fitted = False

    def fit(self, X, y):
        pass

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model.predict(X)

    def reset(self):
        self.model = clone(self.model)
        self.is_fitted = False
