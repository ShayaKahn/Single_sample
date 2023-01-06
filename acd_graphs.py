import numpy as np
import matplotlib.pyplot as plt
from functions import normalize_data
from data_filter import DataFilter
from functions import calc_bray_curtis_dissimilarity, create_PCoA, Confusion_matrix, Confusion_matrix_comb,\
    plot_confusion_matrix
from IDOA_class import IDOA
from scipy.spatial.distance import cdist
import matplotlib
from DOC_class import DOC
matplotlib.rcParams['text.usetex'] = True
import os

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\ASD data')
# The data without filtering.
#ACD_data = DataFilter('ACD.xlsx')
#control_data = DataFilter('Control_(ACD).xlsx')

# Convert the data to numpy array.
#ACD_data = ACD_data.first_data.to_numpy()
#control_data = control_data.first_data.to_numpy()

# Filter the data.
total_data = DataFilter('ACD.xlsx', 'Control_(ACD).xlsx', two_data_sets=True)
total_data.remove_less_one_data()
total_data.split_two_sets()
ACD_data = total_data.first_set
control_data = total_data.second_set

ACD_data = ACD_data.to_numpy()
control_data = control_data.to_numpy()

# Normalization of the data.
ACD_data = normalize_data(ACD_data)
control_data = normalize_data(control_data)

########## Calculate DOC ##########
Doc = DOC(control_data.T)
Doc_mat = Doc.calc_DOC()
Doc_ACD = DOC(ACD_data.T)
Doc_mat_ACD = Doc_ACD.calc_DOC()

########## IDOA ##########
idoa_ACD_ACD = IDOA(ACD_data.T, ACD_data.T, self_cohort=True)
idoa_ACD_ACD_vector = idoa_ACD_ACD.calc_idoa_vector()
idoa_control_control = IDOA(control_data.T, control_data.T, self_cohort=True)
idoa_control_control_vector = idoa_control_control.calc_idoa_vector()
idoa_ACD_control = IDOA(ACD_data.T, control_data.T)
idoa_ACD_control_vector = idoa_ACD_control.calc_idoa_vector()
idoa_control_ACD = IDOA(control_data.T, ACD_data.T)
idoa_control_ACD_vector = idoa_control_ACD.calc_idoa_vector()

fig, ax = plt.subplots()
ax.scatter(idoa_control_ACD_vector, idoa_ACD_ACD_vector, color='blue', label='ACD')
ax.scatter(idoa_control_control_vector, idoa_ACD_control_vector, color='red', label='Control')
ax.legend(loc='lower right')
ax.set_xlabel('IDOA w.r.t control')
ax.set_ylabel('IDOA w.r.t ACD')
ax.set_aspect('equal', adjustable='box')
ax.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax.set_xlim([-4, 1])
ax.set_ylim([-4, 1])
plt.show()

########## Bray Curtis ##########
dist_ACD_control_vector = calc_bray_curtis_dissimilarity(ACD_data.T, control_data.T)
dist_control_ACD_vector = calc_bray_curtis_dissimilarity(control_data.T, ACD_data.T)
dist_control_control_vector = calc_bray_curtis_dissimilarity(control_data.T, control_data.T, self_cohort=True)
dist_ACD_ACD_vector = calc_bray_curtis_dissimilarity(ACD_data.T, ACD_data.T, self_cohort=True)

fig1, ax1 = plt.subplots()
ax1.scatter(dist_control_ACD_vector, dist_ACD_ACD_vector, color='blue', label='ACD')
ax1.scatter(dist_control_control_vector, dist_ACD_control_vector, color='red', label='Control')
ax1.legend(loc='lower right')
ax1.set_xlabel('Mean distance to control cohort')
ax1.set_ylabel('Mean distance to ACD cohort')
ax1.set_aspect('equal', adjustable='box')
ax1.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax1.set_xlim([0.35, 0.7])
ax1.set_ylim([0.35, 0.7])
plt.show()

########## PCoA graph ##########
combined_data = np.concatenate((ACD_data.T, control_data.T), axis=0)
dist_mat = cdist(combined_data, combined_data, 'braycurtis')
create_PCoA(dist_mat, np.size(ACD_data, axis=1), 'PCoA graph', 'ACD', 'Control')

########## Combination of the methods ##########
fig3, ax3 = plt.subplots()
ax3.scatter(idoa_ACD_ACD_vector-idoa_control_ACD_vector,
            dist_ACD_ACD_vector-dist_control_ACD_vector, color='blue', label='ACD')
ax3.scatter(idoa_ACD_control_vector-idoa_control_control_vector,
            dist_ACD_control_vector-dist_control_control_vector, color='red', label='Control')
ax3.legend(loc='lower right')
ax3.set_xlabel('IDOA differance')
ax3.set_ylabel('Distance differance')
ax3.axhline(y=0, color='black', linestyle='--')
ax3.axvline(x=0, color='black', linestyle='--')
ax3.set_xlim([-3, 3])
ax3.set_ylim([-0.065, 0.05])
plt.show()



########## Confusion matrices ##########
con_mat_distances, y_exp_dist, y_pred_dist = Confusion_matrix(
    dist_ACD_control_vector, dist_control_control_vector, dist_control_ACD_vector, dist_ACD_ACD_vector)
con_mat_IDOA, y_exp_IDOA, y_pred_IDOA = Confusion_matrix(
    idoa_ACD_control_vector, idoa_control_control_vector, idoa_control_ACD_vector, idoa_ACD_ACD_vector)
plot_confusion_matrix(con_mat_distances, 'Confusion matrix - distances', labels=('NACD', 'ACD'))
plot_confusion_matrix(con_mat_IDOA, 'Confusion matrix - IDOA', labels=('NACD', 'ACD'))

########## Confusion matrix - combination of the methods ##########
con_mat_distances_comb_or, y_exp_dist_comb_or, y_pred_dist_comb_or = Confusion_matrix_comb(
    dist_ACD_control_vector, dist_control_control_vector, dist_control_ACD_vector, dist_ACD_ACD_vector,
    idoa_ACD_control_vector, idoa_control_control_vector, idoa_control_ACD_vector, idoa_ACD_ACD_vector)

plot_confusion_matrix(con_mat_distances_comb_or, r'Confusion matrix - IDOA or distances', labels=('NACD', 'ACD'))

con_mat_distances_comb_and, y_exp_dist_comb_and, y_pred_dist_comb_and = Confusion_matrix_comb(
    dist_ACD_control_vector, dist_control_control_vector, dist_control_ACD_vector, dist_ACD_ACD_vector,
    idoa_ACD_control_vector, idoa_control_control_vector, idoa_control_ACD_vector, idoa_ACD_ACD_vector, And=True)

plot_confusion_matrix(con_mat_distances_comb_and, r'Confusion matrix - IDOA and distances', labels=('NACD', 'ACD'))

########## Find the samples that distance method failed ##########
ind_control = np.where(dist_ACD_ACD_vector > dist_control_ACD_vector)
failed_ACD = ACD_data[:, ind_control[0]]

idoa_ACD_ACD_failed = IDOA(failed_ACD.T, failed_ACD.T)
idoa_ACD_ACD_vector_failed = idoa_ACD_ACD_failed.calc_idoa_vector()
idoa_control_control_failed = IDOA(control_data.T, control_data.T)
idoa_control_control_vector_failed = idoa_control_control_failed.calc_idoa_vector()
idoa_ACD_control_failed = IDOA(failed_ACD.T, control_data.T)
idoa_ACD_control_vector_failed = idoa_ACD_control_failed.calc_idoa_vector()
idoa_control_ACD_failed = IDOA(control_data.T, failed_ACD.T)
idoa_control_ACD_vector_failed = idoa_control_ACD_failed.calc_idoa_vector()

########## Confusion matrices for failed distances samples ##########
con_mat_IDOA_failed, y_exp_IDOA_failed, y_pred_IDOA_failed = Confusion_matrix(
    idoa_ACD_control_vector_failed, idoa_control_control_vector_failed, idoa_control_ACD_vector_failed,
    idoa_ACD_ACD_vector_failed)
plot_confusion_matrix(con_mat_IDOA_failed, 'Confusion matrix - IDOA for failed samples by distances',
                      labels=('NACD', 'ACD'))
plt.show()
