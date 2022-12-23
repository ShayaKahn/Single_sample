from functions import calc_bray_curtis_dissimilarity
import defaults as de
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist
from IDOA_class import IDOA
from functions import Confusion_matrix, Confusion_matrix_comb, plot_confusion_matrix,\
    create_PCA, create_PCoA, generate_cohort

matplotlib.rcParams['text.usetex'] = True

# Two cohorts --> case: r1=r2 sigma=0.0005.
cohort1 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step)
cohort2 = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step)

combined_cohorts_1_2 = np.concatenate((cohort1, cohort2), axis=0)
mean_dist_1_1 = calc_bray_curtis_dissimilarity(cohort1, cohort1, self_cohort=True)
mean_dist_2_2 = calc_bray_curtis_dissimilarity(cohort2, cohort2, self_cohort=True)
dist_mat_1 = cdist(combined_cohorts_1_2, combined_cohorts_1_2, 'braycurtis')

########## PCoA ##########
create_PCoA(dist_mat_1, de.Y_row, f'PCoA graph $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}', '$cohort_1$', '$cohort_2$')

########## Bray Curtis ##########
mean_dist_1_2 = calc_bray_curtis_dissimilarity(cohort1, cohort2)
mean_dist_2_1 = calc_bray_curtis_dissimilarity(cohort2, cohort1)
fig1, ax1 = plt.subplots()
ax1.scatter(mean_dist_1_1, mean_dist_2_1, color='blue', label='$cohort_1$')
ax1.scatter(mean_dist_1_2, mean_dist_2_2, color='red', label='$cohort_2$')
ax1.legend(loc='lower right')
ax1.set_xlabel('Mean distance to cohort 1')
ax1.set_ylabel('Mean distance to cohort 2')
ax1.set_title(f'Bray Curtis distances $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')
ax1.set_aspect('equal', adjustable='box')
ax1.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax1.set_xlim([0.22, 0.46])
ax1.set_ylim([0.22, 0.46])

########## PCA graph ##########
create_PCA(combined_cohorts_1_2, de.Y_row, f'PCA graph $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}',
           '$cohort_1$', '$cohort_2$')

########## IDOA ##########
idoa_1_1 = IDOA(cohort1, cohort1)
idoa_1_1_vector = idoa_1_1.calc_idoa_vector()
idoa_2_2 = IDOA(cohort2, cohort2)
idoa_2_2_vector = idoa_2_2.calc_idoa_vector()
idoa_1_2 = IDOA(cohort1, cohort2)
idoa_1_2_vector = idoa_1_2.calc_idoa_vector()
idoa_2_1 = IDOA(cohort2, cohort1)
idoa_2_1_vector = idoa_2_1.calc_idoa_vector()

fig3, ax3 = plt.subplots()
ax3.scatter(idoa_1_1_vector, idoa_2_1_vector, color='blue', label='$cohort_1$')
ax3.scatter(idoa_1_2_vector, idoa_2_2_vector, color='red', label='$cohort_2$')
ax3.legend(loc='lower right')
ax3.set_xlabel('IDOA w.r.t $cohort_1$')
ax3.set_ylabel('IDOA w.r.t $cohort_2$')
ax3.set_aspect('equal', adjustable='box')
ax3.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax3.set_title(f'IDOA graph $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')
ax3.set_xlim([-0.03, 0.03])
ax3.set_ylim([-0.03, 0.03])

########## Confusion matrices ##########
con_mat_distances_1, y_exp_dist_1, y_pred_dist_1 = Confusion_matrix(mean_dist_1_2, mean_dist_2_2, mean_dist_2_1,
                                                                    mean_dist_1_1)
con_mat_IDOA_1, y_exp_IDOA_1, y_pred_IDOA_1 = Confusion_matrix(idoa_1_2_vector, idoa_2_2_vector, idoa_2_1_vector,
                                                               idoa_1_1_vector)

plot_confusion_matrix(con_mat_distances_1,
                      fr'Confusion matrix - distances for $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')
plot_confusion_matrix(con_mat_IDOA_1, f'Confusion matrix - IDOA $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')

########## Confusion matrix - combination of the methods ##########
con_mat_distances_comb_or, y_exp_dist_comb_or, y_pred_dist_comb_or = Confusion_matrix_comb(mean_dist_1_2, mean_dist_2_2,
                                                                                  mean_dist_2_1, mean_dist_1_1,
                                                                                  idoa_1_2_vector, idoa_2_2_vector,
                                                                                  idoa_2_1_vector, idoa_1_1_vector)
plot_confusion_matrix(con_mat_distances_comb_or,
                      f'Confusion matrix - IDOA or distances for $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')

con_mat_distances_comb_and, y_exp_dist_comb_and, y_pred_dist_comb_and = Confusion_matrix_comb(mean_dist_1_2,
                                                                                              mean_dist_2_2,
                                                                                mean_dist_2_1, mean_dist_1_1,
                                                                                idoa_1_2_vector, idoa_2_2_vector,
                                                                                idoa_2_1_vector, idoa_1_1_vector,
                                                                                And=True)
plot_confusion_matrix(con_mat_distances_comb_and,
                      f'Confusion matrix - IDOA and distances for $r_1$ = $r_2$ and $sigma$ = {de.sigma_value}')

# Two cohorts --> case: r1!=r2 sigma=0.0005.
cohort4 = generate_cohort(de.Y_row, de.n, de.delta, de.r_new, de.s, de.A2, de.Y2, de.time_span, de.max_step)

combined_cohorts_3_4 = np.concatenate((cohort1, cohort4), axis=0)

mean_dist_3_3 = calc_bray_curtis_dissimilarity(cohort1, cohort1, self_cohort=True)
mean_dist_4_4 = calc_bray_curtis_dissimilarity(cohort4, cohort4, self_cohort=True)
dist_mat_2 = cdist(combined_cohorts_3_4, combined_cohorts_3_4, 'braycurtis')

########## PCoA ##########
create_PCoA(dist_mat_2, de.Y_row,
            fr'PCoA graph $r_3$ $\neq$ $r_4$ and $sigma$ = {de.sigma_value}', '$cohort_3$', '$cohort_4$')

########## PCA graph ##########
create_PCA(combined_cohorts_3_4, de.Y_row,
           fr'PCA graph $r_3$ $\neq$ $r_4$ and $sigma$ = {de.sigma_value}', '$cohort_3$', '$cohort_4$')

########## Bray Curtis ##########
mean_dist_3_4 = calc_bray_curtis_dissimilarity(cohort1, cohort4)
mean_dist_4_3 = calc_bray_curtis_dissimilarity(cohort4, cohort1)
fig5, ax5 = plt.subplots()
ax5.scatter(mean_dist_3_3, mean_dist_4_3, color='blue', label='$cohort_3$')
ax5.scatter(mean_dist_3_4, mean_dist_4_4, color='red', label='$cohort_4$')
ax5.legend(loc='lower right')
ax5.set_xlabel('Mean distance to $cohort_3$')
ax5.set_ylabel('Mean distance to $cohort_4$')
ax5.set_title(fr'Bray Curtis distances $r_1$ $\neq$ $r_2$ and $sigma$ = {de.sigma_value}')
ax5.set_aspect('equal', adjustable='box')
ax5.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax5.set_xlim([0.2, 0.7])
ax5.set_ylim([0.2, 0.7])

########## PCA graph ##########
create_PCA(combined_cohorts_3_4, de.Y_row, fr'PCA graph $r_3$ $\neq$ $r_4$ and $sigma$ = {de.sigma_value}', '$cohort_3$',
           '$cohort_4$')

plt.show()