import numpy as np
from scipy.integrate import solve_ivp
import defaults as de
import cython.parallel as par
import concurrent.futures
class Glv:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, r, s,
                 interaction_matrix, initial_cond, time_span, max_step, second_interaction_matrix=0, mixed_noise=False):
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
        :param B: Optional parameter, second interaction matrix for mixed cohort with noise.
        :param mixed_noise: If true, it solves with noise and mixing samples from A and B.
        """
        self.smp = n_samples
        self.n = n_species
        self.delta = delta
        self.r = r
        self.s = s
        self.A = interaction_matrix
        self.Y = initial_cond
        self.time_span = time_span
        self.max_step = max_step
        self.mixed_noise = mixed_noise
        self.B = second_interaction_matrix
        # Initiation.
        self.Final_abundances = np.zeros((self.smp, self.n))
        self.Final_abundances_single_sample = np.zeros(self.n)
        self.norm_Final_abundances = 0
        self.norm_Final_abundances_single_sample = 0

    def solve(self):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """

        def f(t, x):
            """
            GLV formula.
            """
            return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 + sum([self.A[i, p] * x[
                i] * x[p] for p in par.prange(self.n, nogil=True) if p != i]) for i in par.prange(self.n, nogil=True)])

        if self.smp > 1:  # Solution for cohort
            def execute_process(num_samples):
                """
                This function represents single process of GLV model solution.
                :param num_samples: The number of samples
                :return:
                """
                time_control = 0
                # solve GLV up to time span.
                solutions = solve_ivp(f, (time_control, time_control + self.time_span), self.Y[num_samples][:],
                                      max_step=self.max_step)
                abundances = solutions.y.T
                # Keep the integration until the stop condition.
                while max(abs(abundances[-1][:] - abundances[-2][:])) > self.delta:
                    time_control += self.time_span
                    new_solutions = solve_ivp(f, (time_control, time_control + self.time_span
                                                  ), abundances[-1][:], max_step=self.max_step)
                    abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)
                self.Final_abundances[num_samples][:] = abundances[-1][:]
            #  Create a ThreadPoolExecutor with 1000 worker threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
                # Process the elements of the list in parallel
                results = [executor.submit(execute_process, num_samples) for num_samples in range(self.smp)]
                for future in concurrent.futures.as_completed(results):
                    result = future.result()

        else:  # Solution for single sample
            time_control = 0
            solutions_single = solve_ivp(f, (time_control, time_control + self.time_span),
                                  self.Y[:], max_step=self.max_step)
            abundances_single = solutions_single.y.T
            # Keep the integration until the stop condition.
            while max(abs(abundances_single[-1][:] - abundances_single[-2][:])) > self.delta:
                time_control += self.time_span
                new_solutions_single = solve_ivp(f, (time_control, time_control + self.time_span
                                                     ), abundances_single[-1][:], max_step=self.max_step)
                abundances_single = np.concatenate((abundances_single, new_solutions_single.y.T), axis=0)
            self.Final_abundances_single_sample[:] = abundances_single[-1][:]

    def solve_noisy(self):  # Solution with noise
        """
        This function represents single process of GLV model solution for noisy matrix.
        :param num_samples: The number of samples
        :return:
        """
        if self.mixed_noise:  # Creates noisy matrix from mixed samples of the original matrices A and B.
            noisy_matrix_list = de.calc_matrix_vector_with_noise_non_homogeneous(self.smp, self.A, self.B)
        else:  # Creates noisy matrix from a single matrix A
            noisy_matrix_list = de.calc_matrix_vector_with_noise(self.smp, self.A)

        def execute_process_noisy(num_samples):
            """
            :param num_samples:
            :return:
            """
            noisy_matrix = noisy_matrix_list[num_samples]
            # GLV formula.

            def f_noisy(t, x):
                """
                GLV formula.
                """
                return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 +
                                 sum([noisy_matrix[i][p] * x[i] * x[p] for p in
                                      range(0, self.n) if p != i]) for i in range(0, self.n)])
            time_control_noisy = 0
            solutions = solve_ivp(f_noisy, (time_control_noisy, time_control_noisy + self.time_span),
                                  self.Y[num_samples][:], max_step=self.max_step)
            abundances = solutions.y.T
            # Keep the integration until the stop condition.
            while max(abs(abundances[-1][:] - abundances[-2][:])) > self.delta:
                time_control_noisy += self.time_span
                new_solutions = solve_ivp(f_noisy, (time_control_noisy, time_control_noisy + self.time_span),
                                          abundances[-1][:], max_step=self.max_step)
                abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)
            self.Final_abundances[num_samples][:] = abundances[-1][:]
            #  Create a ThreadPoolExecutor with 1000 worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
            # Process the elements of the list in parallel
            results = [executor.submit(execute_process_noisy, num_samples) for num_samples in range(self.smp)]
            for future in concurrent.futures.as_completed(results):
                result = future.result()

    def normalize_results(self):
        """
        Normalization of the final abundances.
        """
        if self.smp > 1:  # Normalization for cohort
            norm_factors = np.sum(self.Final_abundances, axis=1)
            self.norm_Final_abundances = np.array([self.Final_abundances[:][i] /
                                                   norm_factors[i] for i in range(0, np.size(norm_factors))])
        else:  # Normalization for single sample
            norm_factor = np.sum(self.Final_abundances_single_sample)
            self.norm_Final_abundances_single_sample = self.Final_abundances_single_sample/norm_factor
