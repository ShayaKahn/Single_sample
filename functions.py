import numpy as np
from scipy.spatial.distance import braycurtis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from GLV_model_class import Glv
from sklearn.metrics.pairwise import euclidean_distances
from overlap import Overlap
from dissimilarity import Dissimilarity
from matplotlib.colors import ListedColormap

def calc_bray_curtis_dissimilarity(first_cohort, second_cohort, median=False, self_cohort=False):
    if self_cohort:
        num_samples = np.size(first_cohort, 0)
        mean_dist_vector = np.zeros(num_samples)
        for i in range(0, num_samples):
            sample_dist = np.array([braycurtis(first_cohort[j, :], second_cohort[i, :]
                                               ) for j in range(0, num_samples) if i != j])
            if median:
                mean_dist_vector[i] = np.median(sample_dist)
            else:
                mean_dist_vector[i] = np.mean(sample_dist)
    else:
        num_samples_first = np.size(first_cohort, 0)
        num_samples_second = np.size(second_cohort, 0)
        mean_dist_vector = np.zeros(num_samples_second)
        for i in range(0, num_samples_second):
            sample_dist = np.zeros(num_samples_first)
            for j in range(0, num_samples_first):
                dist = braycurtis(first_cohort[j, :], second_cohort[i, :])
                sample_dist[j] = dist
            if median:
                mean_dist_vector[i] = np.median(sample_dist)
            else:
                mean_dist_vector[i] = np.mean(sample_dist)
    return mean_dist_vector

def Confusion_matrix(v_neg_pos, v_pos_pos, v_pos_neg, v_neg_neg):
    """
    :param v_1: vector compares same type of subjects.
    :param v_2: vector compares different type of subjects.
    :return:
    """
    y_exp_pos = np.zeros(np.size(v_neg_pos))
    y_exp_neg = np.ones(np.size(v_pos_neg))
    y_exp_tot = np.concatenate((y_exp_pos, y_exp_neg))
    y_pred_pos = np.array([0 if measure1 < measure2 else 1 for measure1, measure2 in zip(v_pos_pos, v_neg_pos)])
    y_pred_neg = np.array([0 if measure1 < measure2 else 1 for measure1, measure2 in zip(v_pos_neg, v_neg_neg)])
    y_pred_tot = np.concatenate((y_pred_pos, y_pred_neg))
    conf_mat = confusion_matrix(y_exp_tot, y_pred_tot)
    return conf_mat, y_exp_tot, y_pred_tot

def Confusion_matrix_comb(v_neg_pos_method_1, v_pos_pos_method_1, v_pos_neg_method_1, v_neg_neg_method_1,
                          v_neg_pos_method_2, v_pos_pos_method_2, v_pos_neg_method_2, v_neg_neg_method_2, And=False):
    """
    :param v_1: vector compares same type of subjects.
    :param v_2: vector compares different type of subjects.
    :return:
    """
    y_exp_pos = np.zeros(np.size(v_neg_pos_method_1))
    y_exp_neg = np.ones(np.size(v_pos_neg_method_1))
    y_exp_tot = np.concatenate((y_exp_pos, y_exp_neg))
    if And:
        y_pred_pos = np.array([0 if measure1_method_1 < measure2_method_1 and measure1_method_2 < measure2_method_2 else
                               1 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
            v_pos_pos_method_1, v_neg_pos_method_1, v_pos_pos_method_2, v_neg_pos_method_2)])
        y_pred_neg = np.array([1 if measure1_method_1 > measure2_method_1 and measure1_method_2 > measure2_method_2 else
                               0 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
            v_pos_neg_method_1, v_neg_neg_method_1, v_pos_neg_method_2, v_neg_neg_method_2)])
    else:
        y_pred_pos = np.array([0 if measure1_method_1 < measure2_method_1 or measure1_method_2 < measure2_method_2 else
                               1 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
                                v_pos_pos_method_1, v_neg_pos_method_1, v_pos_pos_method_2, v_neg_pos_method_2)])
        y_pred_neg = np.array([0 if measure1_method_1 < measure2_method_1 and measure1_method_2 < measure2_method_2 else
                               1 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
                                v_pos_neg_method_1, v_neg_neg_method_1, v_pos_neg_method_2, v_neg_neg_method_2)])
    y_pred_tot = np.concatenate((y_pred_pos, y_pred_neg))
    conf_mat = confusion_matrix(y_exp_tot, y_pred_tot)
    return conf_mat, y_exp_tot, y_pred_tot

def plot_confusion_matrix(confusion_mat, Title, labels=('0', '1')):
    Labels = labels
    df_con_mat_distances_comb = pd.DataFrame(confusion_mat, index=Labels, columns=Labels)
    df_con_mat_distances_comb['TOTAL'] = df_con_mat_distances_comb.sum(axis=1)
    df_con_mat_distances_comb.loc['TOTAL'] = df_con_mat_distances_comb.sum()
    df_con_mat_distances_comb = df_con_mat_distances_comb.to_numpy()
    labels = (labels[0], labels[1], 'Total')
    fig, ax = plt.subplots()
    camp = ListedColormap(['#E0E0E0', '#BDBDBD', '#9E9E9E', '#757575'])
    im = ax.imshow(df_con_mat_distances_comb/df_con_mat_distances_comb[-1, -1], cmap=camp)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(len(labels)-1):
        for j in range(len(labels)-1):
            ax.text(j, i, df_con_mat_distances_comb[i, j],
                        ha="center", va="top", color="black", fontsize='large')
            ax.text(j, i, s=str(
                round(df_con_mat_distances_comb[i, j] / df_con_mat_distances_comb[i, -1] * 100, 2)) + ' \%',
                                 ha="center", va="bottom", color="black", fontsize='large')
            ax.text(j, len(labels) - 1, df_con_mat_distances_comb[len(labels) - 1, j],
                    ha="center", va="top", color="black", fontsize='large')
        ax.text(len(labels)-1, i, df_con_mat_distances_comb[i, len(labels)-1],
                ha="center", va="top", color="black", fontsize='large')
    ax.text(len(labels) - 1, len(labels) - 1, df_con_mat_distances_comb[len(labels) - 1, len(labels) - 1],
            ha="center", va="top", color="black", fontsize='large')
    ax.set_xlabel('Predicted condition', fontsize='x-large')
    ax.set_ylabel('Actual condition', fontsize='x-large')
    ax.set_title(Title, fontsize='x-large')

def create_PCA(combined_cohort, num_samples_first, title, first_cohort_label, second_cohort_label):
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    scaled = pca.fit(combined_cohort.T)
    ax.scatter(scaled.components_[0, 0:num_samples_first], scaled.components_[1, 0:num_samples_first], color='blue',
               label=first_cohort_label)
    ax.scatter(scaled.components_[0, num_samples_first:], scaled.components_[1, num_samples_first:], color='red',
               label=second_cohort_label)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend(loc='lower right')

def create_PCoA(dist_mat, num_samples_first, title, first_cohort_label, second_cohort_label):
    fig, ax = plt.subplots()
    mds = MDS(n_components=2, metric=True, max_iter=300, random_state=0, dissimilarity='precomputed')
    scaled = mds.fit_transform(dist_mat)
    ax.scatter(scaled[0:num_samples_first, 0], scaled[0:num_samples_first, 1], color='blue', label=first_cohort_label)
    ax.scatter(scaled[num_samples_first:, 0], scaled[num_samples_first:, 1], color='red', label=second_cohort_label)
    ax.set_xlabel('PCoA1')
    ax.set_ylabel('PCoA2')
    ax.set_title(title)
    ax.legend(loc='lower right')

def generate_cohort(n_samples, n_species, delta, r, s, A, Y, time_span, max_step, B=0,
                    noise=False, mixed_noise=False):
    """
    :param n_samples: The number of samples you are need to compute.
    :param n_species: The number of species at each sample.
    :param delta: This parameter is responsible for the stop condition at the steady state.
    :param r: growth rate vector.
    :param s: logistic growth term vector.
    :param A: interaction matrix.
    :param Y: set of initial conditions for each sample.
    :param time_span: the time interval.
    :param max_step: maximal allowed step size.
    :return: The final abundances' matrix.
    """
    glv = Glv(n_samples, n_species, delta, r, s, A, Y, time_span, max_step, B, mixed_noise)
    if noise:
        glv.solve_noisy()
        glv.normalize_results()
    else:
        glv.solve()
        glv.normalize_results()
    if n_samples > 1:
        return glv.norm_Final_abundances
    else:
        return glv.norm_Final_abundances_single_sample

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