import numpy as np
import matplotlib.pyplot as plt
from functions import normalize_data
from data_filter import DataFilter
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from functions import calc_bray_curtis_dissimilarity
from IDOA_class import IDOA
from scipy.spatial.distance import cdist
import os

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\Research\ASD data')
# The data without filtering.
#ACD_data = DataFilter('ACD.xlsx')
#control_data = DataFilter('Control_(ACD).xlsx')

# Convert the data to numpy array.
#ACD_data = ACD_data.data.to_numpy()
#control_data = control_data.data.to_numpy()

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

########## IDOA ##########
idoa_ACD_ACD = IDOA(ACD_data.T, ACD_data.T)
idoa_ACD_ACD_vector = idoa_ACD_ACD.calc_idoa_vector()
idoa_control_control = IDOA(control_data.T, control_data.T)
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

########## Bray Curtis ##########
dist_ACD_control_vector = calc_bray_curtis_dissimilarity(ACD_data.T, control_data.T)
dist_control_ACD_vector = calc_bray_curtis_dissimilarity(control_data.T, ACD_data.T)
dist_control_control_vector = calc_bray_curtis_dissimilarity(control_data.T, control_data.T)
dist_ACD_ACD_vector = calc_bray_curtis_dissimilarity(ACD_data.T, ACD_data.T)

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

########## PCoA graph ##########
combined_data = np.concatenate((ACD_data.T, control_data.T), axis=0)
dist_mat = cdist(combined_data, combined_data, 'braycurtis')

fig0, ax0 = plt.subplots()
mds = MDS(n_components=2, metric=True, max_iter=300, random_state=0, dissimilarity='precomputed')
scaled = mds.fit_transform(dist_mat)
ax0.scatter(scaled[0:np.size(ACD_data, axis=1), 0], scaled[0:np.size(ACD_data, axis=1), 1], color='blue', label='ACD')
ax0.scatter(scaled[np.size(ACD_data, axis=1):, 0], scaled[np.size(ACD_data, axis=1):, 1], color='red', label='Control')
ax0.set_xlabel('PCoA1')
ax0.set_ylabel('PCoA2')
ax0.set_title('PCoA graph')
ax0.legend(loc='upper right')

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