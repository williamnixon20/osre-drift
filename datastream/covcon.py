
# Covcon: We construct this 2-dimensional dataset to have covariate shift and concept drift. The decision boundary at
# each point is given by α ∗ sin(πx1) > x2. We use 10000 points (100 batches, 1000 points per batch). Covariate shift is
# introduced by changing the location of x1 and x2 (for batch t x1 and x2 are drawn from the Gaussian distribution with
# mean ((t + 1)%7)/10 and standard deviation 0.015). Concept drift is introduced by alternating the value of α between
# 0.8 and 1, every 25 batches and also changing the inequality from > to < after 50 batches.

from osre.datastream.datastream import DataStream
import pandas as pd
import numpy as np

class CovConDataStream(DataStream):
    type = 'CovCon'
    features = ['x1', 'x2']
    target_names = ['label']

    def generate_data(self):
        data = []
        alpha = 0.8
        for batch in range(self.n_batches):
            mean = ((batch + 1) % 7) / 10
            x1 = np.random.normal(mean, 0.15, self.batch_size)
            x2 = np.random.normal(mean, 0.15, self.batch_size)
            if batch < 50:
                labels = (alpha * np.sin(np.pi * x1) > x2).astype(int)
            else:
                labels = (alpha * np.sin(np.pi * x1) < x2).astype(int)
            batch_data = pd.DataFrame({'x1': x1, 'x2': x2, 'label': labels, 'window': batch})
            data.append(batch_data)

            # Concept drift
            if batch % 25 == 0 and batch != 0:
                alpha = 1.0 if alpha == 0.8 else 0.8

        return pd.concat(data)