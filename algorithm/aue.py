from osre.algorithm.algorithm import Algorithm
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Adapted from https://github.com/w4k2/stream-learn/tree/d3142a3b973e27141a0108f5fffabc7017222d31
class StreamingEnsemble(ClassifierMixin, BaseEstimator):
    """Abstract, base ensemble streaming class"""

    def __init__(self, base_estimator, n_estimators, weighted=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.weighted = weighted

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self.partial_fit(X, y)

    def partial_fit(self, X, y, classes=None):
        """Partial fitting"""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        self.green_light = True

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Check label consistency
        if len(np.unique(y)) != len(np.unique(self.classes_)):
            y[: len(np.unique(self.classes_))] = np.copy(self.classes_)

        # Check if it is possible to train new estimator
        if len(np.unique(y)) != len(self.classes_):
            self.green_light = False

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        # print('ESM')
        return np.nan_to_num(np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_]))

    def predict_proba(self, X):
        """Predict proba."""
        esm = self.ensemble_support_matrix(X)
        if self.weighted:
            esm *= np.array(self.weights_)[:, np.newaxis, np.newaxis]

        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.

        :rtype: array-like, shape (n_samples, )
        :returns: The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")
        proba = self.predict_proba(X)
        prediction = np.argmax(proba, axis=1)

        # Return prediction
        return self.classes_[prediction]

    def msei(self, clf, X, y):
        """MSEi score from original AWE algorithm."""
        pprobas = clf.predict_proba(X)
        probas = np.zeros(len(y))
        for label in self.classes_:
            probas[y == label] = pprobas[y == label, label]
        return np.sum(np.power(1 - probas, 2)) / len(y)

    def prior_proba(self, y):
        """Calculate prior probability for given labels"""
        return np.unique(y, return_counts=True)[1] / len(y)

    def mser(self, y):
        """MSEr score from original AWE algorithm."""
        prior_proba = self.prior_proba(y)
        return np.sum(prior_proba * np.power((1 - prior_proba), 2))

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.
        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (array-like, shape = [n_samples, n_features], array-like, shape = [n_samples, n_features])
        :returns: Tuple of minority and majority class samples
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns minority and majority data

        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (object, object)
        :returns: Tuple of minority and majority class names.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name


class AUE(StreamingEnsemble, Algorithm):
    """Accuracy Updated Ensemble"""

    def __init__(self, base_estimator=None, n_estimators=10, n_splits=3, epsilon=0.0001, epochs=20):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_splits = n_splits
        self.epsilon = epsilon
        self.epochs = epochs

    def reset(self):
        self.is_fitted = False
        self.ensemble_ = []
        self.weights_ = []

    def partial_fit(self, X, y, classes=None):
        y = np.array(y)
        X = np.array(X)
        super().partial_fit(X, y, classes)
        if not self.green_light:
            print("Green light not true! Cannot train new classifier")
            return self

        # Compute baseline
        mser = self.mser(y)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)

        # Calculate its scores
        scores = []
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            if len(np.unique(y[train])) != len(self.classes_):
                continue
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            msei = self.msei(fold_candidate, self.X_[test], self.y_[test])
            scores.append(msei)

        # Save scores
        candidate_msei = np.mean(scores)
        candidate_weight = 1 / (candidate_msei + self.epsilon)

        # Calculate weights of current ensemble
        self.weights_ = [1 / (self.msei(clf, self.X_, self.y_) + self.epsilon) for clf in self.ensemble_]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        # AUE update procedure
        comparator = 1 / mser
        counter = 0
        for i, clf in enumerate(self.ensemble_):
            if i == len(self.ensemble_) - 1:
                break
            ## If current weight is > comparator, update by refitting
            if self.weights_[i] > comparator:
                counter += 1
                for _ in range(self.epochs):
                  clf.partial_fit(X, y)
        print("Model has {} classifiers, with weights {}".format(len(self.ensemble_), str(self.weights_)))
        print("Retrained {} classifiers, comparator: {}".format(counter, comparator))
        return len(X)