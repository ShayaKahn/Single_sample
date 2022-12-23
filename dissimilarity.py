import numpy as np

class Dissimilarity:
    """
    This class calculates the dissimilarity
    value between two given samples
    """
    def __init__(self, sample_1, sample_2):
        """
        :param sample_1: first sample.
        :param sample_2: second sample.
        """
        self.dissimilarity, self.normalized_sample_1, self.normalized_sample_2, self.\
            nonzero_index_1, self.nonzero_index_2, self.S, self.normalized_sample_1_hat, self.\
            normalized_sample_2_hat, self.z = 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.normalize()
        self.find_non_zero_index_and_intersection()
        self.calculate_normalised_in_s()

    def normalize(self):
        self.normalized_sample_1 = self.sample_1 / np.sum(self.sample_1)  # Normalization of the first sample.
        self.normalized_sample_2 = self.sample_2 / np.sum(self.sample_2)  # Normalization of the second sample.

    def find_non_zero_index_and_intersection(self):
        self.nonzero_index_1 = np.nonzero(self.normalized_sample_1)  # Find the non-zero index of the first sample.
        self.nonzero_index_2 = np.nonzero(self.normalized_sample_2)  # Find the non-zero index of the second sample.
        self.S = np.intersect1d(self.nonzero_index_1, self.nonzero_index_2)  # Find the intersection of the non-zero

    def calculate_normalised_in_s(self):
        self.normalized_sample_1_hat = self.normalized_sample_1[self.S] / np.sum(self.normalized_sample_1[self.S])
        self.normalized_sample_2_hat = self.normalized_sample_2[self.S] / np.sum(self.normalized_sample_2[self.S])
        self.z = (self.normalized_sample_1_hat + self.normalized_sample_2_hat) / 2  # define z variable

    def dkl(self, u_hat, z):
        # Calculate dkl
        return np.sum(u_hat*np.log(u_hat/z))

    def calculate_dissimilarity(self):
        # Calculate dissimilarity
        if ((self.dkl(self.normalized_sample_1_hat, self.z) + self.dkl(self.normalized_sample_2_hat, self.z)) / 2) > 0:
            self.dissimilarity = np.sqrt((self.dkl(self.normalized_sample_1_hat, self.z) +
                                          self.dkl(self.normalized_sample_2_hat, self.z)) / 2)
        else:
            self.dissimilarity = 0
        return self.dissimilarity



