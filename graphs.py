import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from GLV_model_class import Glv
import defaults as de
import time
from overlap import Overlap
from dissimilarity import Dissimilarity
import seaborn as sns
import matplotlib
matplotlib.rcParams['text.usetex'] = True

start = time.time()

""" PCA rgraph """

def generate_cohort(n_samples, n_species, delta, r, s, A, Y, time_span, max_step):
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
    glv = Glv(n_samples, n_species, delta, r, s, A, Y, time_span, max_step)
    glv.solve()
    return glv.Final_abundances

# Random sample (generated from the first cohort parameters).
random_sample = Glv(1, de.n, de.delta, de.r, de.s, de.A1, de.Y0, de.time_span, de.max_step)
random_sample.solve_single_sample()
rand_sample = random_sample.Final_abundances_single
# Two cohorts
cohort1 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step)
cohort2 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step)


fig = plt.figure()
ax = fig.add_subplot(111)
combined_cohorts = np.concatenate((cohort1, cohort2), axis=0)  # Combine both cohorts to apply PCA.
u, s, v_T = np.linalg.svd(combined_cohorts, full_matrices=True)  # svd calculation.
# Plot PCA graph using SVD.
for j in range(combined_cohorts.shape[0]):
    pc1 = v_T[0, :] @ combined_cohorts[j, :].T
    pc2 = v_T[1, :] @ combined_cohorts[j, :].T
    if j < de.Y_row and j != 1:
        ax.scatter(pc1, pc2, marker='o', color='r')
    elif j == 1:  # Mark one arbitrary point.
        ax.scatter(pc1, pc2, marker='o', color='black', linewidths=6)
    else:
        ax.scatter(pc1, pc2, marker='o', color='b')
fig.suptitle('PCA graph', fontsize=16)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)

""" Graph of the 10 most abundant species in both samples """

def find_most_abundant(cohort, n_samples):
    """
    :param cohort: A given cohort.
    :param n_samples: The number of species at each sample.
    :return: Averaged abundances vector sorted and sorted standard deviation vector.
    """
    the_sum = np.sum(cohort, axis=0)
    std = np.std(cohort, axis=0)
    sorted_cohort_index = np.argsort(the_sum)
    average_sorted_cohort = the_sum[sorted_cohort_index] / n_samples
    std_sorted = std[sorted_cohort_index]
    return average_sorted_cohort, std_sorted

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

# Sort the random sample array.
sorted_rand_sample_index = np.argsort(rand_sample)
sorted_rand_sample = rand_sample[sorted_rand_sample_index]

# Set the averaged sorted cohorts and SD accordingly for both cohorts.
[average_sorted_cohort1, std_sorted1] = find_most_abundant(cohort1, de.Y_row)
[average_sorted_cohort2, std_sorted2] = find_most_abundant(cohort2, de.Y_row)

# Plots.
x_axis = np.linspace(1, 10, 10)
ax1.scatter(x_axis, np.flip(sorted_rand_sample[len(sorted_rand_sample)-10:]), marker='o', color='black')
ax1.errorbar(x_axis, np.flip(average_sorted_cohort1[len(average_sorted_cohort1)-10:]),
             yerr=np.flip(std_sorted1[len(std_sorted1)-10:]), marker='o', color='r', uplims=True, lolims=True)
ax1.errorbar(x_axis, np.flip(average_sorted_cohort2[len(average_sorted_cohort2)-10:]),
             yerr=np.flip(std_sorted2[len(std_sorted2)-10:]), marker='o', color='b', alpha=0.4,
             uplims=True, lolims=True)
ax1.set_xlabel('Most abundant species', fontsize=12)
ax1.set_ylabel('Averaged abundances', fontsize=12)
fig1.suptitle('Top 10 abundant species', fontsize=16)

""" Histograms of the Euclidean distances of the both cohorts graphs """

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

def get_ed_vector(cohort):
    euclidean_dist_matrix = euclidean_distances(cohort, cohort)
    euclidean_dist_vector = euclidean_dist_matrix.reshape(np.size(euclidean_dist_matrix), )
    return euclidean_dist_vector

euclidean_dist_vector_1 = get_ed_vector(cohort1)
euclidean_dist_vector_2 = get_ed_vector(cohort2)

sns.kdeplot(euclidean_dist_vector_1, ax=ax2, fill=True, alpha=0.5, common_norm=True)
sns.kdeplot(euclidean_dist_vector_2, ax=ax2, fill=True, alpha=0.5, common_norm=True)
plt.grid()

ax2.set_xlabel('sample-sample distance', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
fig2.suptitle('Euclidean distances distributions', fontsize=16)

# DOC.
# Initiation of the variables.
overlap_vector_1 = np.zeros(de.Y_row)
overlap_vector_2 = np.zeros(de.Y_row)
dissimilarity_vector_1 = np.zeros(de.Y_row)
dissimilarity_vector_2 = np.zeros(de.Y_row)

def IDOA(sample, cohort, overlap_vector, dissimilarity_vector):
    """
    :param sample: single sample
    :param cohort: cohort that consists of m samples
    :param overlap_vector: empty vector size m
    :param dissimilarity_vector: empty vector size m
    :return: overlap and dissimilarity vectors for larger than 0.5 overlap values
    """
    for i in range(0, de.Y_row):
        O = Overlap(sample, cohort[i, :])
        D = Dissimilarity(sample, cohort[i, :])
        overlap_vector[i] = O.calculate_overlap()
        dissimilarity_vector[i] = D.calculate_dissimilarity()
    # Indexes of the overlap vector that are greater than 0.5.
    overlap_vector_index = np.where(overlap_vector > 0.5)
    new_overlap_vector = overlap_vector[overlap_vector_index]
    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
    return new_overlap_vector, new_dissimilarity_vector

[new_overlap_vector_1, new_dissimilarity_vector_1] = IDOA(rand_sample, cohort1,
                                                          overlap_vector_1, dissimilarity_vector_1)
[new_overlap_vector_2, new_dissimilarity_vector_2] = IDOA(rand_sample, cohort2,
                                                          overlap_vector_2, dissimilarity_vector_2)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
# Plots.
ax3.scatter(new_overlap_vector_1, new_dissimilarity_vector_1, color='b')
ax3.scatter(new_overlap_vector_2, new_dissimilarity_vector_2, color='r')
ax3.set_xlabel('Overlap', fontsize=12)
ax3.set_ylabel('Dissimilarity', fontsize=12)
a1, b1 = np.polyfit(new_overlap_vector_1, new_dissimilarity_vector_1, 1)
a2, b2 = np.polyfit(new_overlap_vector_2, new_dissimilarity_vector_2, 1)
ax3.plot(new_overlap_vector_1, a1*new_overlap_vector_1+b1)
ax3.plot(new_overlap_vector_2, a2*new_overlap_vector_2+b2)
fig3.suptitle('IDOA graph', fontsize=16)

plt.show()

end = time.time()
print(end - start)