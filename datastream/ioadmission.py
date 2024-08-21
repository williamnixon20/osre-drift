from osre.datastream.datastream import DataStream
import pandas as pd

class IOAdmissionStream(DataStream):
    type = 'IOAdmission'
    features = ['prev_queue_len_1','prev_queue_len_2','prev_queue_len_3','prev_latency_1',
                'prev_latency_2','prev_latency_3','prev_throughput_1','prev_throughput_2','prev_throughput_3']
    target_names = ['reject']
    data_path = '6350_6650_readonly_10k_sampled.csv'

    def generate_data(self):
        return pd.read_csv(self.data_path)