import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import kstest
import defaults as de
import time
import seaborn as sns
import matplotlib
from functions import generate_cohort, idoa, find_most_abundant

matplotlib.rcParams['text.usetex'] = True
start = time.time()

# Two cohorts
cohort1 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step)
cohort2 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step)

# Random sample (generated from the first cohorts GLV model).
rand_sample = generate_cohort(1, de.n, de.delta, de.r, de.s, de.A1, de.Y0, de.time_span, de.max_step)

""" PCA Graph """
fig = plt.figure()
ax = fig.add_subplot(111)
combined_cohorts = np.concatenate((cohort1, cohort2), axis=0)  # Combine both cohorts to apply PCA.
u, s, v_T = np.linalg.svd(combined_cohorts, full_matrices=True)  # svd calculation.
# Plot PCA graph using SVD.
for j in range(combined_cohorts.shape[0]):
    pc1 = v_T[0, :] @ combined_cohorts[j, :].T
    pc2 = v_T[1, :] @ combined_cohorts[j, :].T
    if j < de.Y_row:
        ax.scatter(pc1, pc2, marker='o', color='r')
    else:
        ax.scatter(pc1, pc2, marker='o', color='b')
pc1 = v_T[0, :] @ rand_sample.T
pc2 = v_T[1, :] @ rand_sample.T
ax.scatter(pc1, pc2, marker='o', color='k', linewidths=6)
fig.suptitle('PCA graph', fontsize=16)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.show()
""" Graph of the 10 most abundant species in both samples """

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
             yerr=np.flip(std_sorted1[len(std_sorted1)-10:]), marker='o', color='r', capsize=10)
ax1.errorbar(x_axis, np.flip(average_sorted_cohort2[len(average_sorted_cohort2)-10:]),
             yerr=np.flip(std_sorted2[len(std_sorted2)-10:]), marker='o', color='b', alpha=0.4, capsize=10)
ax1.set_xlabel('Most abundant species', fontsize=12)
ax1.set_ylabel('Averaged abundances', fontsize=12)
fig1.suptitle('Top 10 abundant species', fontsize=16)
plt.xticks(np.arange(0, len(x_axis)+1, 1))
plt.show()
""" Histograms of the Euclidean distances of the both cohorts graphs """

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

def get_ed_vector(random_sample, cohort):
    euclidean_dist_vector = euclidean_distances(random_sample.reshape(1, -1), cohort)
    return euclidean_dist_vector

euclidean_dist_vector_1 = get_ed_vector(rand_sample, cohort1)
euclidean_dist_vector_2 = get_ed_vector(rand_sample, cohort2)

# Calculate P value by Kolmogorov-Smirnov test
[statistic, pvalue] = kstest(euclidean_dist_vector_1[0], euclidean_dist_vector_2[0])
g = sns.kdeplot(euclidean_dist_vector_1[0], ax=ax2, fill=True, alpha=0.5, common_norm=True, label='same cohort')
sns.kdeplot(euclidean_dist_vector_2[0], ax=ax2, fill=True, alpha=0.5, common_norm=True, label='different cohorts')
plt.legend(loc='upper left')
g.text(x=0.07, y=30, s='P value = '+str(float("{:.3f}".format(pvalue))))
plt.grid()
ax2.set_xlabel('sample-sample distance', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
fig2.suptitle('Euclidean distances distributions', fontsize=16)
plt.show()

""" IDOA Graph """
# Initiation of the variables.
overlap_vector_1 = np.zeros(de.Y_row)
overlap_vector_2 = np.zeros(de.Y_row)
dissimilarity_vector_1 = np.zeros(de.Y_row)
dissimilarity_vector_2 = np.zeros(de.Y_row)

[new_overlap_vector_1, new_dissimilarity_vector_1] = idoa(rand_sample, cohort1.T,
                                                          overlap_vector_1, dissimilarity_vector_1)
[new_overlap_vector_2, new_dissimilarity_vector_2] = idoa(rand_sample, cohort2.T,
                                                          overlap_vector_2, dissimilarity_vector_2)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
# Plots.
ax3.scatter(new_overlap_vector_1, new_dissimilarity_vector_1, color='b')
ax3.scatter(new_overlap_vector_2, new_dissimilarity_vector_2, color='r')
ax3.set_xlabel('Overlap', fontsize=12)
ax3.set_ylabel('Dissimilarity', fontsize=12)
[a1, b1] = np.polyfit(new_overlap_vector_1, new_dissimilarity_vector_1, 1)
[a2, b2] = np.polyfit(new_overlap_vector_2, new_dissimilarity_vector_2, 1)
ax3.plot(new_overlap_vector_1, a1*new_overlap_vector_1+b1)
ax3.plot(new_overlap_vector_2, a2*new_overlap_vector_2+b2)
fig3.suptitle('IDOA graph', fontsize=16)

plt.show()
end = time.time()
print(end - start)
