import numpy as np
import matplotlib.pyplot as plt
from functions import normalize_data
from data_filter import DataFilter
from functions import calc_bray_curtis_dissimilarity, create_PCoA, plot_confusion_matrix
from IDOA_class import IDOA
import os
from functions import Confusion_matrix, Confusion_matrix_comb
from scipy.spatial.distance import cdist
import matplotlib
from DOC_class import DOC
matplotlib.rcParams['text.usetex'] = True

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\CD data')

# The data without filtering.
#cd_data = DataFilter('CD_data.xlsx')
#control_data = DataFilter('Control.xlsx')

# Convert the data to numpy array.
#cd_data = cd_data.data.to_numpy()
#control_data = control_data.data.to_numpy()

# Filter the data.
total_data = DataFilter('CD_data.xlsx', 'Control_(CD).xlsx', two_data_sets=True)
total_data.remove_less_one_data()
total_data.split_two_sets()
cd_data = total_data.first_set
control_data = total_data.second_set

cd_data = cd_data.to_numpy()
control_data = control_data.to_numpy()

# Normalization of the data.
cd_data = normalize_data(cd_data)
control_data = normalize_data(control_data)

########## Calculate DOC ##########
Doc = DOC(control_data.T)
Doc_mat = Doc.calc_DOC()
Doc_CD = DOC(cd_data.T)
Doc_mat_CD = Doc_CD.calc_DOC()

########## IDOA ##########
idoa_cd_cd = IDOA(cd_data.T, cd_data.T, self_cohort=True)
idoa_cd_cd_vector = idoa_cd_cd.calc_idoa_vector()
idoa_control_control = IDOA(control_data.T, control_data.T, self_cohort=True)
idoa_control_control_vector = idoa_control_control.calc_idoa_vector()
idoa_cd_control = IDOA(cd_data.T, control_data.T)
idoa_cd_control_vector = idoa_cd_control.calc_idoa_vector()
idoa_control_cd = IDOA(control_data.T, cd_data.T)
idoa_control_cd_vector = idoa_control_cd.calc_idoa_vector()

fig, ax = plt.subplots()
ax.scatter(idoa_control_cd_vector, idoa_cd_cd_vector, color='blue', label='CD')
ax.scatter(idoa_control_control_vector, idoa_cd_control_vector, color='red', label='Control')
ax.legend(loc='lower right')
ax.set_xlabel('IDOA w.r.t control')
ax.set_ylabel('IDOA w.r.t CD')
ax.set_aspect('equal', adjustable='box')
ax.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax.set_xlim([-1.2, 1])
ax.set_ylim([-1.2, 1])
plt.show()

########## Bray Curtis ##########
dist_cd_control_vector = calc_bray_curtis_dissimilarity(cd_data.T, control_data.T, median=False)
dist_control_cd_vector = calc_bray_curtis_dissimilarity(control_data.T, cd_data.T, median=False)
dist_control_control_vector = calc_bray_curtis_dissimilarity(control_data.T, control_data.T, self_cohort=True,
                                                             median=False)
dist_cd_cd_vector = calc_bray_curtis_dissimilarity(cd_data.T, cd_data.T, self_cohort=True, median=False)

fig1, ax1 = plt.subplots()
ax1.scatter(dist_control_cd_vector, dist_cd_cd_vector, color='blue', label='CD')
ax1.scatter(dist_control_control_vector, dist_cd_control_vector, color='red', label='Control')
ax1.legend(loc='lower right')
ax1.set_xlabel('Mean distance to control cohort')
ax1.set_ylabel('Mean distance to CD cohort')
ax1.set_aspect('equal', adjustable='box')
ax1.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax1.set_xlim([0.6, 1])
ax1.set_ylim([0.6, 1])
ax1.set_ylim([-2, 2])
plt.show()

########## PCoA graph ##########
combined_data = np.concatenate((cd_data.T, control_data.T), axis=0)
dist_mat = cdist(combined_data, combined_data, 'braycurtis')
create_PCoA(dist_mat, np.size(cd_data, axis=1), 'PCoA graph', 'CD', 'Control')

########## Combination of the methods ##########
fig3, ax3 = plt.subplots()
ax3.scatter(idoa_cd_cd_vector-idoa_control_cd_vector,
            dist_cd_cd_vector-dist_control_cd_vector, color='blue', label='CD')
ax3.scatter(idoa_cd_control_vector-idoa_control_control_vector,
            dist_cd_control_vector-dist_control_control_vector, color='red', label='Control')
ax3.legend(loc='lower right')
ax3.set_xlabel('IDOA differance')
ax3.set_ylabel('Distance differance')
ax3.axhline(y=0, color='black', linestyle='--')
ax3.axvline(x=0, color='black', linestyle='--')
ax3.set_xlim([-2, 2])
ax3.set_ylim([-0.075, 0.1])
plt.show()

########## Confusion matrices ##########
con_mat_distances, y_exp_dist, y_pred_dist = Confusion_matrix(
    dist_cd_control_vector, dist_control_control_vector, dist_control_cd_vector, dist_cd_cd_vector)
con_mat_IDOA, y_exp_IDOA, y_pred_IDOA = Confusion_matrix(
    idoa_cd_control_vector, idoa_control_control_vector, idoa_control_cd_vector, idoa_cd_cd_vector)
plot_confusion_matrix(con_mat_distances, 'Confusion matrix - distances', labels=('NCD', 'CD'))
plot_confusion_matrix(con_mat_IDOA, 'Confusion matrix - IDOA', labels=('NCD', 'CD'))

########## Confusion matrix - combination of the methods ##########
con_mat_distances_comb_or, y_exp_dist_comb_or, y_pred_dist_comb_or = Confusion_matrix_comb(
    dist_cd_control_vector, dist_control_control_vector, dist_control_cd_vector, dist_cd_cd_vector,
    idoa_cd_control_vector, idoa_control_control_vector, idoa_control_cd_vector, idoa_cd_cd_vector)

plot_confusion_matrix(con_mat_distances_comb_or, r'Confusion matrix - IDOA or distances', labels=('NCD', 'CD'))

con_mat_distances_comb_and, y_exp_dist_comb_and, y_pred_dist_comb_and = Confusion_matrix_comb(
    dist_cd_control_vector, dist_control_control_vector, dist_control_cd_vector, dist_cd_cd_vector,
    idoa_cd_control_vector, idoa_control_control_vector, idoa_control_cd_vector, idoa_cd_cd_vector, And=True)

plot_confusion_matrix(con_mat_distances_comb_and, r'Confusion matrix - IDOA and distances', labels=('NCD', 'CD'))

########## Find the samples that distance method failed ##########
ind_control = np.where(dist_cd_cd_vector > dist_control_cd_vector)
failed_cd = cd_data[:, ind_control[0]]

idoa_cd_cd_failed = IDOA(failed_cd.T, failed_cd.T)
idoa_cd_cd_vector_failed = idoa_cd_cd_failed.calc_idoa_vector()
idoa_control_control_failed = IDOA(control_data.T, control_data.T)
idoa_control_control_vector_failed = idoa_control_control_failed.calc_idoa_vector()
idoa_cd_control_failed = IDOA(failed_cd.T, control_data.T)
idoa_cd_control_vector_failed = idoa_cd_control_failed.calc_idoa_vector()
idoa_control_cd_failed = IDOA(control_data.T, failed_cd.T)
idoa_control_cd_vector_failed = idoa_control_cd_failed.calc_idoa_vector()

########## Confusion matrices for failed distances samples ##########
con_mat_IDOA_failed, y_exp_IDOA_failed, y_pred_IDOA_failed = Confusion_matrix(
    idoa_cd_control_vector_failed, idoa_control_control_vector_failed, idoa_control_cd_vector_failed,
    idoa_cd_cd_vector_failed)
plot_confusion_matrix(con_mat_IDOA_failed, 'Confusion matrix - IDOA for failed samples by distances',
                      labels=('NCD', 'CD'))
plt.show()
