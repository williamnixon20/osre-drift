
from collections import deque
import numpy as np
from sklearn.base import BaseEstimator
from osre.algorithm.algorithm import Algorithm

class AlwaysRetrain(Algorithm):
    def __init__(self, model: BaseEstimator, memory=1):
        super().__init__(model)
        self.memory = memory
        self.history = []

    def fit(self, X, y):
        self.history.append((X,y))

        if len(self.history) > self.memory:
            self.history.pop(0)

        total_X = np.concatenate([x for x, _ in self.history])
        total_y = np.concatenate([y for _, y in self.history])

        print("FITTING ALWAYS RETRAIN, Memory:", str(self.memory), "Total Train Length:", str(len(total_X)))

        self.model.fit(total_X, total_y)

        self.is_fitted = True

        return str(len(total_X))

    def reset(self):
        super().reset()
        self.history = []
