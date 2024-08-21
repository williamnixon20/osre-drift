## This collects metrics from prediction and ground truth results
## This also contains plotting functions

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

class MetricEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, idx, dataset_name, algo_name, y_true, y_pred, retrained, inference_time, training_time, num_train_data):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5

        if dataset_name not in self.results:
            self.results[dataset_name] = pd.DataFrame(columns=['idx','algo_name', 'accuracy', 'auc', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives'])

        # Append the results to the DataFrame
        self.results[dataset_name] = self.results[dataset_name]._append({
            'idx': idx,
            'algo_name': algo_name,
            'accuracy': accuracy,
            'error-rate': 1 - accuracy,
            'auc': auc,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'retrained': retrained,
            'inference_time': inference_time,
            'training_time': training_time,
            'num_train_data': num_train_data
        }, ignore_index=True)

    def get_results(self):
        return self.results

    def plot_line(self, data, metric, dataset_name):
      # Plot: Classic Line Plot
      plt.figure(figsize=(10, 6))
      sns.lineplot(x='idx', y=metric, hue='algo_name', data=data, palette=self.algo_cmap)
      plt.title(f'{metric.capitalize()} over Chunks by Algorithm \n {dataset_name}')
      plt.xlabel('Chunk Index')
      plt.ylabel(metric)
      plt.legend(title='Algorithm')
      plt.show()


    def error_bar_plot(self, data, metric, time_type, dataset_name):
      plt.figure(figsize=(14, 8))


      # Define a list of markers for differentiation
      markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']
      marker_cycle = {algo: markers[i % len(markers)] for i, algo in enumerate(data['algo_name'].unique())}

      # Plot with error bars
      for algo_name, df in data.groupby('algo_name'):
          avg_time = df[df[time_type] > 1e-5][time_type].mean()
          avg_metric = df[metric].mean()
          std_deviation_metric = df[metric].std()

          plt.errorbar(
              x=avg_time, 
              y=avg_metric, 
              yerr=std_deviation_metric, 
              fmt=marker_cycle[algo_name],  # Use varying markers
              label=algo_name, 
              color=self.algo_cmap[algo_name], 
              markersize=8,  # Adjust marker size
              capsize=5,  # Add whiskers
              elinewidth=2,  # Thickness of error bars
              markeredgewidth=1.5,  # Thickness of marker edges
              markerfacecolor='white'  # Add a distinct border
          )

      # Adding a border to the plot
      plt.gca().spines['top'].set_linewidth(1.5)
      plt.gca().spines['right'].set_linewidth(1.5)
      plt.gca().spines['left'].set_linewidth(1.5)
      plt.gca().spines['bottom'].set_linewidth(1.5)

      # Customizing the title and labels
      plt.title(f'{metric.capitalize()} vs {time_type.capitalize()} by Algorithm \n {dataset_name}', fontsize=16)
      plt.xlabel(f'{time_type.capitalize()} (s)', fontsize=14)
      plt.ylabel(f'{metric.capitalize()}', fontsize=14)

      # Fine-tuning the legend
      plt.legend(title='Algorithm', title_fontsize='13', fontsize='11', loc='best', frameon=True, fancybox=True, borderpad=1)

      plt.show()

    def scatter_plot(self, data, metric, time_type, size, dataset_name):
      plt.figure(figsize=(12, 6))
      
      scatter_data = []

      # Gather data and sort by bubble size (largest first)
      for algo_name, df in data.groupby('algo_name'):
          avg_time = df[df[time_type] > 1e-5][time_type].mean()
          avg_metric = df[metric].mean()
          avg_size = df[df[size] > 0][size].mean() 
 
          scatter_data.append((avg_time, avg_metric, avg_size, algo_name))

      scatter_data.sort(key=lambda x: x[2], reverse=True)

      # Convert to ms
      if 'time' in size:
        scatter_data = [(x[0], x[1], x[2] * 1000, x[3]) for x in scatter_data]

      # Plot each bubble
      for i, (avg_time, avg_metric, _size, algo_name) in enumerate(scatter_data):
          plt.scatter(x=avg_time, y=avg_metric, s=_size, color=self.algo_cmap[algo_name], label=algo_name)
          plt.text(avg_time, avg_metric, str(int(_size)), fontsize=10, ha='center', va='center')

      # # Create a custom legend
      handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=algo_name)
                for i, (algo_name, color) in enumerate(self.algo_cmap.items())]
      plt.legend(handles=handles, title='Algorithm')

      added_str = '- ms' if 'time' in size else ''
      plt.title(f'{metric.capitalize()} vs {time_type.capitalize()} vs {size.capitalize()} (Size {added_str}) \n {dataset_name}')
      plt.xlabel(f'{time_type.capitalize()} (s)')
      plt.ylabel(f'{metric.capitalize()}')
      plt.show()


    def plot_results(self, metrics=['accuracy']):
        sns.set(style="whitegrid")
        for dataset_name, df in self.results.items():
            self.palette = sns.color_palette(n_colors = len(df['algo_name'].unique()))
            self.algo_cmap = {}
            for palette_val, algo_name in zip(self.palette, df['algo_name'].unique()):
                self.algo_cmap[algo_name] = palette_val

            for metric in metrics:
                self.plot_line(df, metric, dataset_name)
                self.error_bar_plot(df, metric, 'inference_time', dataset_name)
                self.error_bar_plot(df, metric, 'training_time', dataset_name)
                self.scatter_plot(df, metric, 'inference_time', 'training_time', dataset_name)
                self.scatter_plot(df, metric, 'training_time', 'num_train_data', dataset_name)

        for dataset_name, df in self.results.items():
            df.to_csv(f'{dataset_name}_results.csv', index=False)

    def load_results(self, dataset_name, path):
        self.results[dataset_name] = pd.read_csv(path)