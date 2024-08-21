import time

class Pipeline:
    def __init__(self, dataset_name, data_stream, algorithms, drift_detector, metric_evaluator, test_size=0.2):
        self.dataset_name = dataset_name
        self.data_stream = data_stream
        self.algorithms = algorithms
        self.drift_detector = drift_detector
        self.metric_evaluator = metric_evaluator
        self.test_size = test_size

    def run(self):
        for algo_name, algorithm in self.algorithms.items():
            self.drift_detector.reset()
            algorithm.reset()

            for idx, data_window in enumerate(self.data_stream):
                print(f"Running {algo_name} on window {idx}")
                X = data_window[self.data_stream.get_feature_names()].to_numpy()
                y = data_window[self.data_stream.get_target_names()].to_numpy().ravel()

                retrained = False
                num_train_data = -1
                inference_time = 0
                training_time = 0

                if idx == 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

                    start_time = time.time()
                    num_train_data = algorithm.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    print(f"Training time: {training_time}")

                    retrained = True

                    start_time = time.time()
                    y_pred = algorithm.predict(X_test)
                    inference_time = time.time() - start_time
                    print(f"Inference time: {inference_time}")

                    self.metric_evaluator.evaluate(idx, self.dataset_name, algo_name, y_test, y_pred, retrained, inference_time, training_time, num_train_data)
                    continue

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
                start_time = time.time()
                y_pred = algorithm.predict(X_test)
                inference_time = time.time() - start_time
                print(f"Accuracy: {accuracy_score(y_test, y_pred)}, Inference time: {inference_time} ")

                if self.drift_detector.check_drift(data_window):
                    start_time = time.time()
                    num_train_data = algorithm.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    print(f"Training time: {training_time}")
                    retrained = True

                self.metric_evaluator.evaluate(idx, self.dataset_name, algo_name, y_test, y_pred, retrained, inference_time, training_time, num_train_data)

        return self.metric_evaluator.get_results()

    def plot_results(self, metric=['accuracy', 'auc']):
        self.metric_evaluator.plot_results(metric)