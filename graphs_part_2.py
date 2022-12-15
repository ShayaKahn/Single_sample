import matplotlib.pyplot as plt
import matplotlib
import defaults as de
from GLV_model_class import Glv
from IDOA_class import IDOA

matplotlib.rcParams['text.usetex'] = True

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

""" IDOA graph for two Homogeneous cohorts """

cohort_A = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step, noise=False)
cohort_B = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step, noise=False)

idoa_A_B = IDOA(cohort_A, cohort_B)
idoa_vector_A_B = idoa_A_B.calc_idoa_vector()
slope_vector = idoa_A_B.IDOA_vector
idoa_A_A = IDOA(cohort_A, cohort_A)
idoa_vector_A_A = idoa_A_A.calc_idoa_vector()
idoa_B_B = IDOA(cohort_B, cohort_B)
idoa_vector_B_B = idoa_B_B.calc_idoa_vector()

fig, ax = plt.subplots()
ax.scatter(idoa_vector_A_A, idoa_vector_A_B, color='red', label='Cohort A')
ax.scatter(idoa_vector_A_B, idoa_vector_B_B, color='blue', label='Cohort B')
ax.legend(loc='upper right')
ax.set_xlabel('IDOA w.r.t A')
ax.set_ylabel('IDOA w.r.t B')
ax.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])

""" IDOA graph for two Homogeneous + noise cohorts """

cohort_A = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step, noise=True)
cohort_B = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step, noise=True)

idoa_A_B = IDOA(cohort_A, cohort_B)
idoa_vector_A_B = idoa_A_B.calc_idoa_vector()
idoa_A_A = IDOA(cohort_A, cohort_A)
idoa_vector_A_A = idoa_A_A.calc_idoa_vector()
idoa_B_B = IDOA(cohort_B, cohort_B)
idoa_vector_B_B = idoa_B_B.calc_idoa_vector()

fig1, ax1 = plt.subplots()
ax1.scatter(idoa_vector_A_A, idoa_vector_A_B, color='red', label='Cohort A')
ax1.scatter(idoa_vector_A_B, idoa_vector_B_B, color='blue', label='Cohort B')
ax1.legend(loc='upper right')
ax1.set_xlabel('IDOA w.r.t A')
ax1.set_ylabel('IDOA w.r.t B')
ax1.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax1.set_xlim([-0.25, 0.25])
ax1.set_ylim([-0.25, 0.25])

""" IDOA graph for one Homogeneous + noise and the other mixed + noise cohorts """

cohort_A = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A1, de.Y1, de.time_span, de.max_step, B=de.A2,
                           noise=True, mixed_noise=True)
cohort_B = generate_cohort(de.Y_row, de.n, de.delta, de.r, de.s, de.A2, de.Y2, de.time_span, de.max_step, noise=True)

idoa_A_B = IDOA(cohort_A, cohort_B)
idoa_vector_A_B = idoa_A_B.calc_idoa_vector()
idoa_A_A = IDOA(cohort_A, cohort_A)
idoa_vector_A_A = idoa_A_A.calc_idoa_vector()
idoa_B_B = IDOA(cohort_B, cohort_B)
idoa_vector_B_B = idoa_B_B.calc_idoa_vector()

fig2, ax2 = plt.subplots()
ax2.scatter(idoa_vector_A_A, idoa_vector_A_B, color='red', label='Cohort A')
ax2.scatter(idoa_vector_A_B, idoa_vector_B_B, color='blue', label='Cohort B')
ax2.legend(loc='upper right')
ax2.set_xlabel('IDOA w.r.t A')
ax2.set_ylabel('IDOA w.r.t B')
ax2.plot([-5, 5], [-5, 5], ls="--", c=".3")
ax2.set_xlim([-0.25, 0.25])
ax2.set_ylim([-0.25, 0.25])

plt.show()
