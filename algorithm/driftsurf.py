from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from osre.algorithm.algorithm import Algorithm


class DriftSurf(Algorithm):
    def __init__(self, model: BaseEstimator, delta=0.1, r=3, wl=2):
        super().__init__(model)
        self.reac_len = r
        self.delta = delta
        self.win_len = wl
        self.models = {"pred": None, "stab": None, "reac": None}
        self.train_data_dict = {"pred": [0], "stab": [0], "reac": []}
        self.acc_best = 0
        self.window_counter = 0
        self.acc_dict = None
        self.reac_ctr = None
        self.state = "stab"
        self.model_key = "pred"  # Model used for prediction
        self.data_df_dict = defaultdict(dict)
        self.train_keys = ["pred", "stab"]

    def reset(self):
        self.models = {"pred": None, "stab": None, "reac": None}
        self.train_data_dict = {"pred": [0], "stab": [0], "reac": [0]}
        self.acc_best = 0
        self.counter = 0
        self.acc_dict = None
        self.reac_ctr = None
        self.state = "stab"
        self.model_key = "pred"
        self.is_fitted = False
        self.data_df_dict = defaultdict(dict)
        self.train_keys = ["pred", "stab"]
        self.window_counter = 0
        self.is_fitted = False


    def predict(self, X):
        y_pred = self.models[self.model_key].predict(X)
        return y_pred

    def _score(self, model_key, X, y):
        y_pred = self.models[model_key].predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def initialize_model(self, key):
        self.models[key] = clone(self.model)

    def _train(self):
        num_train = 0
        for key in self.train_keys:
            if self.models[key] is None:
                self.initialize_model(key)

            X_train_global, y_train_global = [], []
            for iter_id in self.train_data_dict[key]:
                X_train, y_train = self.data_df_dict[iter_id]
                X_train_global.extend(X_train)
                y_train_global.extend(y_train)
                num_train += len(X_train)
            self.models[key].fit(X_train_global, y_train_global)
        return num_train

    def set_key_df(self, iter_id, data_df_dict):
        self.data_df_dict[iter_id] = data_df_dict

    def _append_train_data(self, model_key, iter_id):
        self.train_data_dict[model_key].append(iter_id)
        if len(self.train_data_dict[model_key]) > self.win_len:
            self.train_data_dict[model_key].pop(0)

    def _reset(self, key):
        self.models[key] = None
        self.train_data_dict[key] = []

    def fit(self, X, y):
        curr_iter = self.window_counter
        self.window_counter += 1
        self.set_key_df(curr_iter, (X, y))

        if curr_iter == 0:
            self._train()
            return

        acc_pred = self._score("pred", X, y)
        print(f"DS Iteration {curr_iter}, acc: {acc_pred}")

        if acc_pred > self.acc_best:
            self.acc_best = acc_pred

        if self.state == "stab":
            if len(self.train_data_dict["stab"]) == 0:
                acc_stab = 0
            else:
                acc_stab = self._score("stab", X, y)

            if (acc_pred < self.acc_best - self.delta) or (acc_pred < acc_stab - self.delta / 2):
                print("Entering reactive state")
                self.state = "reac"
                self._reset("reac")
                self.reac_ctr = 0
                self.acc_dict = {"pred": np.zeros(self.reac_len), "reac": np.zeros(self.reac_len)}
            else:
                self._append_train_data("pred", curr_iter)
                self._append_train_data("stab", curr_iter)
                self.train_keys = ["pred", "stab"]

        if self.state == "reac":
            if self.reac_ctr > 0:
                acc_reac = self._score("reac", X, y)
                print(f"acc_reac = {acc_reac}")
                self.acc_dict["pred"][self.reac_ctr - 1] = acc_pred
                self.acc_dict["reac"][self.reac_ctr - 1] = acc_reac

                if acc_reac > acc_pred:
                    self.model_key = "reac"
                else:
                    self.model_key = "pred"

            self._append_train_data("pred", curr_iter)
            self._append_train_data("reac", curr_iter)
            self.train_keys = ["pred", "reac"]
            self.reac_ctr += 1

            if self.reac_ctr == self.reac_len:
                self.state = "stab"
                self._reset("stab")
                if np.mean(self.acc_dict["pred"]) < np.mean(self.acc_dict["reac"]):
                    self.models["pred"] = self.models["reac"]
                    self.train_data_dict["pred"] = self.train_data_dict["reac"]
                    self.acc_best = np.amax(self.acc_dict["reac"])
                    self.model_key = "pred"
                self.acc_dict = None
                self.reac_ctr = None
        print(f"DriftSurf State: {self.state}, Best Accuracy: {self.acc_best}, Current Model: {self.model_key}")
        num_train = self._train()
        return num_train