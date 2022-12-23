import numpy as np
import math

# Parameters.
n = 10
s = np.ones(n)
s_new = np.ones(n)
r = np.random.uniform(0, 1, n)
r_new = np.random.uniform(0, 1, n)
delta = 10**-4
Y_row = 10
time_span = 50
max_step = 3
sigma_value = 0.003

#%% Calculation of the matrix A for the simulation.
def calc_matrix(num_of_species, sigma=sigma_value):
    interaction_matrix = np.zeros([num_of_species, num_of_species])
    p = 0.1
    for row, col in np.ndindex(interaction_matrix.shape):
        if np.random.uniform(0, 1) < p:
            interaction_matrix[row, col] = np.random.uniform(-sigma, sigma)
        else:
            interaction_matrix[row, col] = 0
    return interaction_matrix

# Two different A matrices for different  cohorts.
A1 = calc_matrix(n)
A2 = calc_matrix(n)
A3 = calc_matrix(n, sigma=0.01)
A4 = calc_matrix(n, sigma=0.01)

def calc_matrix_vector_with_noise(number_of_samples, interaction_matrix, noise_off=False, maximum_noise=False):
    if noise_off:
        delta_vector = np.zeros(number_of_samples)
    elif maximum_noise:
        delta_vector = np.ones(number_of_samples)
    else:
        delta_vector = np.random.uniform(0, 0.7, number_of_samples)
    eye = np.ones(number_of_samples)
    eta = [calc_matrix(np.size(interaction_matrix, 0)) for i in range(0, number_of_samples)]
    first_term = [value * interaction_matrix for value in eye-delta_vector]
    second_term = [delta_value * mat for (delta_value, mat) in zip(delta_vector, eta)]
    tot = [first_term, second_term]
    array_of_matrices = [sum(x) for x in zip(*tot)]
    return array_of_matrices

def calc_matrix_vector_with_noise_non_homogeneous(number_of_samples, first_interaction_mat,
                                                  second_interaction_mat, noise_off=False, maximum_noise=False):
    first_interaction_mat_list = calc_matrix_vector_with_noise(int(math.ceil(number_of_samples/2)),
                                                               first_interaction_mat, noise_off, maximum_noise)
    second_interaction_mat_list = calc_matrix_vector_with_noise(int(math.floor(number_of_samples/2)),
                                                                second_interaction_mat, noise_off, maximum_noise)
    combined_vector = first_interaction_mat_list + second_interaction_mat_list
    return combined_vector

# Calculation of initial condition vector.
def calc_initial_condition(number_of_species):
    prob_vector = np.random.uniform(0.6, 0.9, number_of_species)
    y0 = np.zeros(number_of_species)
    for i in range(0, number_of_species):
        if np.random.uniform(0, 1) < prob_vector[i]:
            y0[i] = np.random.uniform(0, 1)
        else:
            y0[i] = 0
    return y0

Y0 = calc_initial_condition(n)

def clac_set_of_initial_conditions(num_species, num_samples):
    init_cond_set = np.zeros([num_samples, num_species])
    for i in range(0, num_samples):
        init_cond_set[:][i] = calc_initial_condition(num_species)
    return init_cond_set

Y1 = clac_set_of_initial_conditions(n, Y_row)
Y2 = clac_set_of_initial_conditions(n, Y_row)
