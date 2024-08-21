import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataStream:
    def __init__(self, n_batches=100, batch_size=1000, seed=42):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seed = seed
        self.data = self.generate_data()
        self.current_batch = 0

    def generate_data(self):
        raise NotImplementedError("Must be implemented by the subclass")

    def get_feature_names(self):
        if not hasattr(self, 'features'):
            raise NotImplementedError("Must be implemented by the subclass")
        return self.features

    def get_target_names(self):
        if not hasattr(self, 'target_names'):
            raise NotImplementedError("Must be implemented by the subclass")
        return self.target_names

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            batch_data = self.get_window(self.current_batch)
            self.current_batch += 1
            return batch_data
        else:
            self.current_batch = 0
            raise StopIteration

    def get_window(self, window):
        return self.data[self.data['window'] == window]

    @staticmethod
    def plot_window(window_data, title):
      plt.figure(figsize=(8, 6))
      # plot features only, see self.features
      scatter = plt.scatter(window_data['x1'], window_data['x2'], c=window_data['label'], cmap='viridis', alpha=0.5)
      plt.colorbar(scatter, label='Label')
      plt.xlabel('x1')
      plt.ylabel('x2')
      plt.title(title)
      plt.show()

    def plot_windows(self, every = 5):
        for idx, window in enumerate(self):
            if idx % every == 0:
              self.plot_window(window, f'Data Stream {self.type} - Window {window["window"].iloc[0]}')