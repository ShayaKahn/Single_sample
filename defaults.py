import numpy as np

# Parameters.
n = 100
s = np.ones(n)
r = np.random.uniform(0, 1, (n,))
delta = 10**-5
epsilon = 10**-5
Y_row = 100
time_span = 100
max_step = 0.5

# Calculation of the matrix A for the simulation.
def calc_A(n):
    A = np.zeros([n, n])
    for row, col in np.ndindex(A.shape):
        p = 0.25
        if np.random.uniform(0, 1) < p:
            A[row, col] = np.random.uniform(-0.05, 0.05)
        else:
            A[row, col] = 0
    return A

# Two different A matrices for different  cohorts.
A1 = calc_A(n)
A2 = calc_A(n)

# Calculation of initial condition vector.
def calc_y0(n):
    prob_vector = np.random.uniform(0.6, 0.9, n) # Option to set all the initial abundances non-zero.
    #prob_vector = np.ones(n, dtype=(int))
    y0 = np.zeros(n)
    for i in range(0, n):
        if np.random.uniform(0, 1) < prob_vector[i]:
            y0[i] = np.random.uniform(0, 1)
        else:
            y0[i] = 0
    return y0

Y0 = calc_y0(n)
Y1 = np.zeros([Y_row, n])
Y2 = np.zeros([Y_row, n])

for i in range(0, Y_row):
    Y1[:][i] = calc_y0(n)
    Y2[:][i] = calc_y0(n)





