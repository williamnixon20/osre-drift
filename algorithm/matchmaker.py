from collections import defaultdict
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from osre.algorithm.algorithm import Algorithm
import numpy as np

class Matchmaker(Algorithm):

    def __init__(self, base_estimator, memory=7):
        super().__init__(base_estimator)
        self.memory = memory
        self.models = []
        self.datasets= []
        ## HIGH score -> GOOD.
        ## Covariate score will be calculated inference time
        self.concept_score = []

        self.forest_dict = None
        self.base_estimator = base_estimator

    def reset(self):
        self.is_fitted = False
        self.concept_score = []
        self.models = []
        self.datasets = []
        self.forest_dict = None

    def fit(self,X,y):
        ## clone base estimator, fit on data, and add to models
        model = clone(self.base_estimator)
        model.fit(X, y)

        self.models.append(model)
        self.datasets.append((X,y))
        self.rank_concept(X,y)
        if len(self.models) > self.memory:
            # model eviction, get index of lowest concept score
            min_index = self.concept_score.index(min(self.concept_score))
            self.models.pop(min_index)
            self.datasets.pop(min_index)
            self.concept_score.pop(min_index)
        print("Current model zoo length:", str(len(self.models)))
        return self.build_tree()

    def predict(self, X):
        model = self.get_model(X)
        return model.predict(X)

    def rank_concept(
        self,
        X,
        y
    ):
        # ranking the models based on ACC
        self.concept_score = [0] * len(self.models)
        for idx, model in enumerate(self.models):
            prediction_result = model.predict(X)
            self.concept_score[idx] = accuracy_score(y, prediction_result)

    def build_tree(self):
        X = np.concatenate([X for X, y in self.datasets])
        y = np.concatenate([y for X, y in self.datasets])


        from sklearn.ensemble import RandomForestClassifier

        forest = RandomForestClassifier(n_estimators=5, max_depth=15, random_state=42)
        # fit on whole data to create the tree
        self.forest = forest.fit(X, y)
        covariate_forest_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        # then loop for each model's dataset.
        # how much did this model's data end up at every leaf?
        for model_idx, dataset in enumerate(self.datasets):
            X_local, y_local = dataset
            X_leaf_forest = forest.apply(X_local)

            for instance_idx, instance_leaves in enumerate(X_leaf_forest):
                for tree_idx, tree_leaf_idx in enumerate(instance_leaves):
                    ## First index is the tree ID,
                    ## second index is the leaf ID
                    ## third index is the model ID -> A leaf will be populated by multiple models
                    covariate_forest_dict[tree_idx][tree_leaf_idx][model_idx] += 1
        self.forest_dict = covariate_forest_dict

        # return number of data used to build tree
        return len(X)

    def calculate_borda_count(self, ranking):
        # Given an array of scores, will return an array containing ranking of each indices
        ## eg: [3,2,1] means that model 0 is the highest ranked (3 point), followed by 2 and 1.
        ## Will return [1,2,3] -> Means that model 0 is ranked 1, index 1 is ranked 2, index 2 is ranked 3
        n = len(ranking)
        borda_count = [0] * n

        sorted_indices = sorted(range(n), key=lambda i: ranking[i], reverse=True)

        for i, index in enumerate(sorted_indices):
            borda_count[index] += i + 1

        return borda_count

    def get_model_borda_count(self, covariate_score):
        concept_borda_count = self.calculate_borda_count(self.concept_score)

        covariate_borda_count = self.calculate_borda_count(covariate_score)

        combined_borda_count = [concept + covariate for concept, covariate in zip(concept_borda_count, covariate_borda_count)]
        return combined_borda_count.index(min(combined_borda_count))

    def rank_covariate(self, X):
        X_forest = self.forest.apply(X)
        model_weights = defaultdict(lambda: 0)

        for instance_idx, instance_forest in enumerate(X_forest):
            for tree_idx, tree_leaf_idx in enumerate(instance_forest):
                if tree_leaf_idx in self.forest_dict[tree_idx]:
                    datapoints_each_model = self.forest_dict[tree_idx][tree_leaf_idx]
                    for model_idx, datapoints in datapoints_each_model.items():
                        model_weights[model_idx] += datapoints

        model_rankings = []
        for i in range(len(self.concept_score)):
            if i in model_weights:
                model_rankings.append(model_weights[i])
            else:
                model_rankings.append(0)
        return model_rankings

    def get_model(self, X):
        # On inference time, concept will never change.
        # Covariate does.
        covariate_rank = self.rank_covariate(X)
        model_idx = self.get_model_borda_count(covariate_rank)
        model = self.models[model_idx]
        return model