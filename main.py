from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from collections import defaultdict
import numpy as np
from osre.algorithm.algorithm import Algorithm
from osre.algorithm.alwaysretrain import AlwaysRetrain
from osre.algorithm.aue import AUE
from osre.algorithm.driftsurf import DriftSurf
from osre.algorithm.matchmaker import Matchmaker
from osre.algorithm.noretrain import NoRetrain
from osre.datastream.covcon_datastream import CovConDataStream
from osre.datastream.sine_datastream import SineDataStream
from osre.datastream.circle_datastream import CircleDataStream
from osre.pipeline.pipeline import Pipeline

circle_ds = CircleDataStream()
covcon_ds = CovConDataStream()
sine_ds = SineDataStream()


algorithms = {
    'MLP-NoRetrain': NoRetrain(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01)),
    'MLP-RetrainWin': AlwaysRetrain(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01), memory = 7),
    'MLP-RetrainOne': AlwaysRetrain(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01), memory = 1),
    'MLP-Matchmaker': Matchmaker(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01)),
    'MLP-AUE': AUE(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01), n_estimators=7),
    'MLP-DriftSurf': DriftSurf(MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, learning_rate_init=0.01)),
}

pipeline_circle = Pipeline(dataset_name="CircleDataStream", data_stream=circle_ds, algorithms=algorithms, drift_detector=AlwaysDriftDetector(), metric_evaluator=MetricEvaluator())
results_circle = pipeline_circle.run()
pipeline_circle.plot_results()

pipeline_covcon = Pipeline(dataset_name="CovConDataStream", data_stream=covcon_ds, algorithms=algorithms, drift_detector=AlwaysDriftDetector(), metric_evaluator=MetricEvaluator())
results_covcon = pipeline_covcon.run()
pipeline_covcon.plot_results()

pipeline_sine= Pipeline(dataset_name="SineDataStream", data_stream=sine_ds, algorithms=algorithms, drift_detector=AlwaysDriftDetector(), metric_evaluator=MetricEvaluator())
results_sine = pipeline_sine.run()
pipeline_sine.plot_results()