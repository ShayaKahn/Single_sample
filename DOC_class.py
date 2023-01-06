import numpy as np
from overlap import Overlap
from dissimilarity import Dissimilarity
from scipy.stats import linregress

class DOC:
    def __init__(self, cohort):
        self.cohort = cohort
        self.num_samples = np.size(cohort, 0)

    def calc_DOC(self):
        D = [[Dissimilarity(self.cohort[j, :], self.cohort[i, :]).calculate_dissimilarity() for i in range(
            j+1, self.num_samples)] for j in range(0, self.num_samples-1)]
        O = [[Overlap(self.cohort[j, :], self.cohort[i, :]).calculate_overlap() for i in range(
            j+1, self.num_samples)] for j in range(0, self.num_samples-1)]

        def flatten(lis):
            return [item for sublist in lis for item in sublist]

        D = np.array(flatten(D))
        O = np.array(flatten(O))
        DOC_mat = np.vstack((D, O))

        return DOC_mat

