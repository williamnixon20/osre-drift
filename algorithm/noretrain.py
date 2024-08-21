from osre.algorithm.algorithm import Algorithm

class NoRetrain(Algorithm):
    def fit(self, X, y):
        if not self.is_fitted:
            print("TOTAL TRAIN LENGTH", str(len(X)))
            self.model.fit(X, y)
            self.is_fitted = True
            return str(len(X))
        return 0
