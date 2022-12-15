import numpy as np
from scipy.spatial.distance import braycurtis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from GLV_model_class import Glv

def calc_bray_curtis_dissimilarity(first_cohort, second_cohort):
    num_samples_first = np.size(first_cohort, 0)
    num_samples_second = np.size(second_cohort, 0)
    mean_dist_vector = np.zeros(num_samples_second)
    dist_mat = np.zeros((num_samples_second, num_samples_first))
    for i in range(0, num_samples_second):
        sample_dist = np.zeros(num_samples_first)
        for j in range(0, num_samples_first):
            dist = braycurtis(first_cohort[j, :], second_cohort[i, :])
            sample_dist[j] = dist
            dist_mat[i, j] = dist
        mean_dist_vector[i] = np.mean(sample_dist)
    return mean_dist_vector, dist_mat

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
                          v_neg_pos_method_2, v_pos_pos_method_2, v_pos_neg_method_2, v_neg_neg_method_2):
    """
    :param v_1: vector compares same type of subjects.
    :param v_2: vector compares different type of subjects.
    :return:
    """
    y_exp_pos = np.zeros(np.size(v_neg_pos_method_1))
    y_exp_neg = np.ones(np.size(v_pos_neg_method_1))
    y_exp_tot = np.concatenate((y_exp_pos, y_exp_neg))
    y_pred_pos = np.array([0 if measure1_method_1 < measure2_method_1 or measure1_method_2 < measure2_method_2 else
                           1 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
                            v_pos_pos_method_1, v_neg_pos_method_1, v_pos_pos_method_2, v_neg_pos_method_2)])
    y_pred_neg = np.array([0 if measure1_method_1 < measure2_method_1 and measure1_method_2 < measure2_method_2 else
                           1 for measure1_method_1, measure2_method_1, measure1_method_2, measure2_method_2 in zip(
                            v_pos_neg_method_1, v_neg_neg_method_1, v_pos_neg_method_2, v_neg_neg_method_2)])
    y_pred_tot = np.concatenate((y_pred_pos, y_pred_neg))
    conf_mat = confusion_matrix(y_exp_tot, y_pred_tot)
    return conf_mat, y_exp_tot, y_pred_tot

def plot_confusion_matrix(confusion_mat, Title):
    Labels = ['0', '1']
    df_con_mat_distances_comb = pd.DataFrame(confusion_mat, index=Labels, columns=Labels)
    df_con_mat_distances_comb['TOTAL'] = df_con_mat_distances_comb.sum(axis=1)
    df_con_mat_distances_comb.loc['TOTAL'] = df_con_mat_distances_comb.sum()
    df_con_mat_distances_comb = df_con_mat_distances_comb.to_numpy()
    labels = ['0', '1', 'Total']
    fig, ax = plt.subplots()
    im = ax.imshow(df_con_mat_distances_comb)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, df_con_mat_distances_comb[i, j],
                           ha="center", va="top", color="black")
            ax.text(j, i, s=str(
                round(df_con_mat_distances_comb[i, j] / df_con_mat_distances_comb[-1, -1] * 100, 2)) + ' \%',
                                 ha="center", va="bottom", color="black")
    ax.set_xlabel('Predicted condition')
    ax.set_ylabel('Actual condition')
    ax.set_title(Title)
    plt.colorbar(im)

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
    ax.legend(loc='upper right')

def create_PCoA(dist_mat, num_samples_first, title, first_cohort_label, second_cohort_label):
    fig, ax = plt.subplots()
    mds = MDS(n_components=2, metric=True, max_iter=300, random_state=0, dissimilarity='precomputed')
    scaled = mds.fit_transform(dist_mat)
    ax.scatter(scaled[0:num_samples_first, 0], scaled[0:num_samples_first, 1], color='blue', label=first_cohort_label)
    ax.scatter(scaled[num_samples_first:, 0], scaled[num_samples_first:, 1], color='red', label=second_cohort_label)
    ax.set_xlabel('PCoA1')
    ax.set_ylabel('PCoA2')
    ax.set_title(title)
    ax.legend(loc='upper right')

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

