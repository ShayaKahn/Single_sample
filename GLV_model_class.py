import numpy as np
from scipy.integrate import solve_ivp

class Glv:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, r, s, A, Y, time_span, max_step):
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
        """
        self.smp = n_samples
        self.n = n_species
        self.delta = delta
        self.r = r
        self.s = s
        self.A = A
        self.Y = Y
        self.time_span = time_span
        self.max_step = max_step

        # Initiation.
        self.Final_abundances = np.zeros((self.smp, self.n))
        self.Final_abundances_single = np.zeros(self.n)

    def solve(self):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """
        # GLV formula.
        def f(t, x):
            return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 +
                             sum([self.A[i][p] * x[i] * x[p] for p in
                                  range(0, self.n) if p != i]) for i in range(0, self.n)])

        for m in range(0, self.smp):
            print(m)
            time_control = 0
            solutions = solve_ivp(f, (time_control, time_control + self.time_span),
                                  self.Y[m][:], max_step=self.max_step)
            abundances = solutions.y.T

            while max(abs(abundances[-1][:] - abundances[-2][:])) > self.delta:
                time_control += self.time_span
                new_solutions = solve_ivp(f, (time_control, time_control + self.time_span),
                                          abundances[-1][:], max_step=self.max_step)
                abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)

            self.Final_abundances[m][:] = abundances[-1][:]

    def solve_single_sample(self):
        """
        Optional solution for single sample.
        """
        # GLV formula.
        def f(t, x):
            return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 +
                             sum([self.A[i][p] * x[i] * x[p] for p in
                                  range(0, self.n) if p != i]) for i in range(0, self.n)])

        time_control = 0
        solutions = solve_ivp(f, (time_control, time_control + self.time_span),
                              self.Y, max_step=self.max_step)
        abundances = solutions.y.T

        while max(abs(abundances[-1][:] - abundances[-2][:])) > self.delta:
            time_control += self.time_span
            new_solutions = solve_ivp(f, (time_control, time_control + self.time_span),
                                      abundances[-1][:], max_step=self.max_step)
            abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)

        self.Final_abundances_single[:] = abundances[-1][:]
