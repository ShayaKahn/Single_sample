import numpy as np

class Overlap:
    """
    This class calculates the overlap
    value between two given samples
    """
    def __init__(self, sample_1, sample_2):
        """
        :param sample_1: first sample.
        :param sample_2: second sample.
        """
        self.overlap = 0
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.normalized_sample_1 = self.sample_1 / np.sum(self.sample_1)  # Normalization of the first sample.
        self.normalized_sample_2 = self.sample_2 / np.sum(self.sample_2)  # Normalization of the second sample.
        self.nonzero_index_1 = np.nonzero(self.normalized_sample_1)  # Find the non-zero index of the first sample.
        self.nonzero_index_2 = np.nonzero(self.normalized_sample_2)  # Find the non-zero index of the second sample.
        self.S = np.intersect1d(self.nonzero_index_1, self.nonzero_index_2)  # Find the intersection of the non-zero
        # indexes.

    def calculate_overlap(self):
        # calculation of the overlap value between the two samples
        self.overlap = np.sum(self.normalized_sample_1[self.S] + self.normalized_sample_2[self.S])/2
        return self.overlap
