## This generator is an implementation of the dara stream with abrupt concept drift
## The classification function is y > sin(x), which abruptly changes upon specified indices

from osre.datastream.datastream import DataStream
import pandas as pd
import numpy as np

class SineDataStream(DataStream):
    type = 'Sine'
    features = ['x1', 'x2']
    target_names = ['label']

    def generate_data(self):
        data = []
        is_positive = True
        np.random.seed(self.seed)

        for batch in range(self.n_batches):
            x = np.random.uniform(0, 1, self.batch_size)
            y = np.random.uniform(0, 1, self.batch_size)

            if is_positive:
                labels = (y < np.sin(x * np.pi)).astype(int)
            else:
                labels = (y >= np.sin(x * np.pi)).astype(int)

            batch_data = pd.DataFrame({'x1': x, 'x2': y, 'label': labels, 'window': batch})
            data.append(batch_data)

            # Abrupt drift at batches 25, 50, and 75
            if batch in [25, 50, 75]:
                is_positive = not is_positive

        return pd.concat(data)