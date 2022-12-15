import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from overlap import Overlap
from dissimilarity import Dissimilarity

def normalize_data(data):
    norm_factors = np.sum(data, axis=0)
    norm_data = np.array([data[:, i] / norm_factors[i] for i in range(0, np.size(norm_factors))])
    return norm_data.T

def find_most_abundant_rows(Data, num):
    mean_vector = np.mean(Data, axis=1)
    index_vector = np.argsort(mean_vector)
    most_abundant_index = index_vector[-num:]
    most_abundant_data = Data[most_abundant_index]
    return most_abundant_data

def create_shuffled(some_sample, data):
    indexes = np.nonzero(some_sample)
    for ind in indexes[0]:
        some_sample[ind] = np.random.choice(data[ind, :])
    return some_sample

def idoa(sample, cohort, overlap_vector, dissimilarity_vector):
    """
    :param sample: single sample
    :param cohort: cohort that consists of m samples
    :param overlap_vector: empty vector size m
    :param dissimilarity_vector: empty vector size m
    :return: overlap and dissimilarity vectors for larger than 0.5 overlap values
    """
    for i in range(0, np.size(cohort, axis=1)):
        o = Overlap(sample, cohort[:, i])
        d = Dissimilarity(sample, cohort[:, i])
        overlap_vector[i] = o.calculate_overlap()
        dissimilarity_vector[i] = d.calculate_dissimilarity()
    # Indexes of the overlap vector that are greater than 0.5.
    overlap_vector_index = np.where(np.logical_and(overlap_vector >= 0.5, overlap_vector <= 0.85))
    new_overlap_vector = overlap_vector[overlap_vector_index]
    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
    return new_overlap_vector, new_dissimilarity_vector

def create_randomized_cohort(data, samples):
    for j, row in enumerate(samples.T):
        row = create_shuffled(row, data)
        samples[:, j] = row
    return samples

def average_get_ed_vector(ref_samples, ref_cohort):
    euclidean_dist_mat = euclidean_distances(ref_samples, ref_cohort)
    average_euclidean_dist_vector = np.mean(euclidean_dist_mat, axis=0)
    return average_euclidean_dist_vector
