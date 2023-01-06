import pandas as pd
from scipy.spatial.distance import jaccard
import numpy as np

class DataFilter:
    def __init__(self, io, second_io=' ', two_data_sets=False, file_type='xlsx'):
        self.io = io
        if file_type == 'csv':
            self.first_data = pd.read_csv(self.io, header=None)
        else:
            self.first_data = pd.read_excel(self.io)
        self.first_set, self.second_set = 0, 0
        if two_data_sets:
            self.second_io = second_io
            if file_type == 'csv':
                self.second_data = pd.read_csv(self.second_io, header=None)
            else:
                self.second_data = pd.read_excel(self.second_io)
            self.index = len(self.first_data.columns)
            self.data = pd.concat([self.first_data, self.second_data], axis=1)

    def remove_less_one_data(self):
        mean_rows = self.data.mean(axis=1)
        index = mean_rows.index[mean_rows < 1]
        self.data = self.data.drop(index)
        return self.data

    def remove_zeros(self):
        self.data = self.data.loc[(self.data != 0).any(axis=1)]
        return self.data

    def split_two_sets(self):
        self.first_set = self.data.iloc[:, :self.index]
        self.second_set = self.data.iloc[:, self.index:]

    #def remove_outliers(self):
    #    self.data.to_numpy()
    #    num_samples = np.size(self.data, 1)
    #    mean_dist_vector = np.zeros(num_samples)
    #    for i in range(0, num_samples):
    #        sample_dist = np.zeros(num_samples)
    #        for j in range(0, num_samples):
    #            dist = jaccard(self.data[:, j], self.data[:, i])
    #            sample_dist[j] = dist
    #        mean_dist_vector[i] = np.mean(sample_dist)
    #    minimal_mean_index = np.where(np.min(mean_dist_vector))
    #    index =