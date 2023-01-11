import numpy as np
from overlap import Overlap
from dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    def __init__(self, first_cohort, second_cohort, self_cohort=False):
        self.first_cohort = first_cohort
        self.second_cohort = second_cohort
        self.num_samples_first = np.size(first_cohort, 0)
        self.num_samples_second = np.size(second_cohort, 0)
        self.IDOA_vector = np.zeros(self.num_samples_second)
        self.self_cohort = self_cohort
        if self_cohort:
            self.overlap_mat = np.zeros((self.num_samples_first, self.num_samples_first-1))
            self.dissimilarity_mat = np.zeros((self.num_samples_first, self.num_samples_first - 1))
        else:
            self.overlap_mat = np.zeros((self.num_samples_first, self.num_samples_first))
            self.dissimilarity_mat = np.zeros((self.num_samples_first, self.num_samples_first))
    def calc_idoa_vector(self):
        if self.self_cohort:
            for i in range(0, self.num_samples_second):
                o_vector = []
                d_vector = []
                for j in range(0, self.num_samples_first):
                    o = Overlap(self.first_cohort[j, :], self.second_cohort[i, :])
                    d = Dissimilarity(self.first_cohort[j, :], self.second_cohort[i, :])
                    o_vector.append(o)
                    d_vector.append(d)
                overlap_vector = np.array([o_vector[j].calculate_overlap()
                                          for j in range(0, self.num_samples_first) if j != i])
                dissimilarity_vector = np.array([d_vector[j].calculate_dissimilarity()
                                                for j in range(0, self.num_samples_first) if j != i])
                # Indexes of the overlap vector that are greater than 0.5.
                if np.max(overlap_vector) <= 0.6:
                    self.IDOA_vector[i] = 0
                else:
                    overlap_vector_index = np.where(np.logical_and(overlap_vector >= 0.5, overlap_vector <= 1))
                    new_overlap_vector = overlap_vector[overlap_vector_index]
                    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
                    slope = linregress(new_overlap_vector, new_dissimilarity_vector)[0]
                    self.IDOA_vector[i] = slope
                np.copyto(self.overlap_mat[i, :], overlap_vector)
                np.copyto(self.dissimilarity_mat[i, :], dissimilarity_vector)
            return self.IDOA_vector
        else:
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
                np.copyto(self.overlap_mat[i, :], overlap_vector)
                np.copyto(self.dissimilarity_mat[i, :], dissimilarity_vector)
            return self.IDOA_vector
