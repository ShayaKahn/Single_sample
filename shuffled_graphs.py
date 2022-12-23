import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kstest, rv_histogram
from seaborn import kdeplot
from data_filter import DataFilter
import functions as fun
from IDOA_class import IDOA
from roc import Roc
import time

start = time.time()
matplotlib.rcParams['text.usetex'] = True

Tongue_data = DataFilter(r'C:\Users\shaya\OneDrive\Desktop\תואר ראשון\מעבדה ממוחשבת\Data.xlsx')
Tongue_data.remove_less_one_data()
Data = Tongue_data.data.to_numpy()

Data = fun.normalize_data(Data)

Data_temp = np.copy(Data)
num = 50

random_subjects = Data_temp[:, 60:90]

Tongue_cohort = fun.find_most_abundant_rows(random_subjects, num)
real_sample = Tongue_cohort[:, 25]
temp_sample = Tongue_cohort[:, 24]
shuffled_sample = fun.create_shuffled(temp_sample, Tongue_cohort)

fig, ax = plt.subplots(1, 3)
plt.subplots_adjust(left=0.1,
bottom=0.1,
right=0.9,
top=0.9,
wspace=0.4,
hspace=0.4)
shw = ax[0].imshow(Tongue_cohort)
ax[0].set_xlabel('Samples')
ax[0].set_ylabel('Species')
ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].yaxis.set_major_locator(plt.NullLocator())
shw1 = ax[1].imshow(np.expand_dims(real_sample, axis=1))
ax[1].set_ylabel('Random sample')
ax[1].xaxis.set_major_locator(plt.NullLocator())
ax[1].yaxis.set_major_locator(plt.NullLocator())
shw2 = ax[2].imshow(np.expand_dims(shuffled_sample, axis=1))
ax[2].set_ylabel('Shuffled sample')
ax[2].xaxis.set_major_locator(plt.NullLocator())
ax[2].yaxis.set_major_locator(plt.NullLocator())

Data_temp = np.copy(Data)
real_sample = Data_temp[:, 10]
random_sample = Data_temp[:, 0]
shuffled_sample = fun.create_shuffled(random_sample, Data_temp)

# Initiation of the variables.
overlap_vector_real = np.zeros(np.size(Data_temp, axis=1))
overlap_vector_shuffled = np.zeros(np.size(Data_temp, axis=1))
dissimilarity_vector_real = np.zeros(np.size(Data_temp, axis=1))
dissimilarity_vector_shuffled = np.zeros(np.size(Data_temp, axis=1))

[new_overlap_vector_real, new_dissimilarity_vector_real] = fun.idoa(real_sample, Data,
                                                                   overlap_vector_real, dissimilarity_vector_real)
[new_overlap_vector_shuffled, new_dissimilarity_vector_shuffled] = fun.idoa(shuffled_sample, Data,
                                                                           overlap_vector_shuffled,
                                                                           dissimilarity_vector_shuffled)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
# Plots.
ax3.scatter(new_overlap_vector_real, new_dissimilarity_vector_real, color='b', label='Real sample')
ax3.scatter(new_overlap_vector_shuffled, new_dissimilarity_vector_shuffled, color='r', label='Randomized sample')
ax3.set_xlabel('Overlap', fontsize=12)
ax3.set_ylabel('Dissimilarity', fontsize=12)
[a1, b1] = np.polyfit(new_overlap_vector_real, new_dissimilarity_vector_real, 1)
[a2, b2] = np.polyfit(new_overlap_vector_shuffled, new_dissimilarity_vector_shuffled, 1)
ax3.plot(new_overlap_vector_real, a1*new_overlap_vector_real+b1)
ax3.plot(new_overlap_vector_shuffled, a2*new_overlap_vector_shuffled+b2)
ax3.legend(loc='lower left')
fig3.suptitle('IDOA graph - Tongue', fontsize=16)
plt.ylim([0.3, 0.65])

########## Histograms of the average euclidean distances of shuffled and real samples graphs ##########

Data_temp = np.copy(Data)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

reference_cohort = Data_temp[:, 0:86]
samples = Data_temp[:, 86:]
temp_samples = np.copy(samples)

random_samples = fun.create_randomized_cohort(reference_cohort, temp_samples)

euclidean_dist_vector_real = fun.average_get_ed_vector(samples.T, reference_cohort.T)
euclidean_dist_vector_rand = fun.average_get_ed_vector(random_samples.T, reference_cohort.T)

# Calculate P value by Kolmogorov-Smirnov test
[statistic, pvalue] = kstest(euclidean_dist_vector_real, euclidean_dist_vector_rand)

g = kdeplot(euclidean_dist_vector_real, ax=ax2, fill=True, alpha=0.5, common_norm=True, label='Real samples')
kdeplot(euclidean_dist_vector_rand, ax=ax2, fill=True, alpha=0.5, common_norm=True, label='Random samples')
plt.legend(loc='upper right')
g.text(x=0.35, y=11, s='P value = '+str(float("{:.3f}".format(pvalue))))

ax2.set_xlabel('Average sample-sample distance', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
fig2.suptitle('Average euclidean distances distributions', fontsize=16)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

idoa_real = IDOA(samples.T, reference_cohort.T)
idoa_vector_real = idoa_real.calc_idoa_vector()

idoa_rand = IDOA(random_samples.T, reference_cohort.T)
idoa_vector_rand = idoa_rand.calc_idoa_vector()

g0 = kdeplot(idoa_vector_real, ax=ax3, fill=True, alpha=0.5, common_norm=True, label='Real samples')
kdeplot(idoa_vector_rand, ax=ax3, fill=True, alpha=0.5, common_norm=True, label='Random samples')
plt.legend(loc='upper left')

ax3.set_xlabel('IDOA', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
fig3.suptitle('IDOA distributions for shuffled and real cohorts', fontsize=16)

########## ROS graph ###########

# Calculate IDOA values vectors for real and shuffled samples.
data = np.copy(Data).T
ref_cohort = data[0:86, :]
real_data = data[86:, :]
idoa_data_ref = IDOA(real_data, ref_cohort)#, data)
idoa_vector_data_ref = idoa_data_ref.calc_idoa_vector()
temp_shuffled_data = np.copy(real_data)
shuffled_data = fun.create_randomized_cohort(temp_shuffled_data.T, ref_cohort.T)#, temp_shuffled_data.T)
shuffled_data = shuffled_data.T
idoa_shuffled_data = IDOA(shuffled_data, data)
idoa_vector_shuffled_data = idoa_shuffled_data.calc_idoa_vector()
idoa_real_data = IDOA(real_data, data)
idoa_vector_real_data = idoa_real_data.calc_idoa_vector()

# Calculate Euclidian distances vectors for real and shuffled samples
ed_vector_shuffled_data = fun.average_get_ed_vector(shuffled_data, data)
ed_vector_real_data = fun.average_get_ed_vector(real_data, data)

# Create Roc curve for ED.
roc_curve_ed = Roc(ed_vector_real_data, ed_vector_shuffled_data, 10000)
tpr_ed = roc_curve_ed.tpr_vector
fpr_ed = roc_curve_ed.fpr_vector
Auc_ed = roc_curve_ed.Auc

# Create Roc curve for IDOA.
roc_curve_idoa = Roc(idoa_vector_shuffled_data, idoa_vector_real_data, 10000)
tpr_idoa = roc_curve_idoa.tpr_vector
fpr_idoa = roc_curve_idoa.fpr_vector
Auc_idoa = roc_curve_idoa.Auc

fig, ax = plt.subplots()
ax.plot(fpr_idoa, tpr_idoa, 'ro', label='IDOA')
ax.plot(fpr_ed, tpr_ed, 'bo', label='Distances')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC curve')
ax.legend(loc='lower right')
ax.plot([0, 1], [0, 1], ls="--", c=".3")
ax.text(x=0.8, y=0.15, s='AUC = ' + str(float("{:.3f}".format(Auc_ed))), color='blue')
ax.text(x=0.8, y=0.2, s='AUC = ' + str(float("{:.3f}".format(Auc_idoa))), color='red')

plt.show()
end = time.time()
print(end - start)

