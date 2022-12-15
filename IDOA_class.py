import numpy as np
from overlap import Overlap
from dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    def __init__(self, first_cohort, second_cohort):
        self.first_cohort = first_cohort
        self.second_cohort = second_cohort
        self.num_samples_first = np.size(first_cohort, 0)
        self.num_samples_second = np.size(second_cohort, 0)
        self.IDOA_vector = np.zeros(self.num_samples_second)

    def calc_idoa_vector(self):
        for i in range(0, self.num_samples_second):
            overlap_vector = np.zeros(self.num_samples_first)
            dissimilarity_vector = np.zeros(self.num_samples_first)
            for j in range(0, self.num_samples_first):
                o = Overlap(self.first_cohort[j, :], self.second_cohort[i, :])
                d = Dissimilarity(self.first_cohort[j, :], self.second_cohort[i, :])
                overlap_vector[j] = o.calculate_overlap()
                dissimilarity_vector[j] = d.calculate_dissimilarity()
            # Indexes of the overlap vector that are greater than 0.5.
            if np.max(overlap_vector) <= 0.6:
                self.IDOA_vector[i] = 0
            else:
                overlap_vector_index = np.where(np.logical_and(overlap_vector >= 0.5, overlap_vector <= 1))
                new_overlap_vector = overlap_vector[overlap_vector_index]
                new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
                slope = linregress(new_overlap_vector, new_dissimilarity_vector)[0]
                self.IDOA_vector[i] = slope
        return self.IDOA_vector
