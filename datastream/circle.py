
# Circle (Pesaranghader et al., 2016): This dataset contains two features x1, x2 drawn uniformly from the interval [0, 1].
# We use 100000 points (100 batches, 1000 points per batch) from this dataset. Each data point is labeled as per the
# condition (x1 − c1)2 + (x2 − c2)2 <= r where the center (c1, c2)) and radius r of the circular decision boundary
# changes gradually over a period of time introducing (gradual) concept drift. Specifically we change the center and
# radius at batch 25, 50, and 75. Each time the change happens gradually over the next 5 batches with the center and
# radius changing for 20% of the points each time

import pandas as pd
from osre.datastream.datastream import DataStream
import numpy as np

class CircleDataStream(DataStream):
    type = 'Circle'
    features = ['x1', 'x2']
    target_names = ['label']

    def generate_data(self):
        data = []
        c1, c2, r = 0.5, 0.5, 0.3
        c1_target, c2_target, r_target = c1, c2, r
        change_step = 0

        for batch in range(self.n_batches):
            x1 = np.random.uniform(0, 1, self.batch_size)
            x2 = np.random.uniform(0, 1, self.batch_size)
            labels = ((x1 - c1)**2 + (x2 - c2)**2 <= r**2).astype(int)
            batch_data = pd.DataFrame({'x1': x1, 'x2': x2, 'label': labels, 'window': batch})
            data.append(batch_data)

            # Gradual drift
            if batch in [25, 50, 75]:
                c1_target = np.random.uniform(0.2, 0.8)
                c2_target = np.random.uniform(0.2, 0.8)
                r_target = np.random.uniform(0.2, 0.3)
                change_step = 1

            if change_step > 0 and change_step <= 5:
                c1 += (c1_target - c1) / (6 - change_step)
                c2 += (c2_target - c2) / (6 - change_step)
                r += (r_target - r) / (6 - change_step)
                change_step += 1

        return pd.concat(data)