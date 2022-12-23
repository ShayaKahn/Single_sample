import defaults as de
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import cython.parallel as p
import concurrent.futures

# Count time.
start = time.time()

# GLV formula.
def f(t, x): return np.array([de.r[i] * x[i] - de.s[i] * x[i] ** 2 + sum(
    [de.A1[i][p] * x[i] * x[p] for p in p.prange(
        de.n, nogil=True) if p != i]) for i in p.prange(de.n, nogil=True)])

# Initiation.
Final_abundances = np.zeros((de.Y_row, de.n))
Initial_abundances = np.zeros((de.Y_row, de.n))
middle_abundances = np.zeros((de.Y_row, de.n))

fig1, ax = plt.subplots(1, de.Y_row)
def solve_glv_model(num_samples):
    time_control = 0
    solutions = solve_ivp(f, (time_control, time_control + de.time_span), de.Y1[num_samples][:], max_step=de.max_step)
    abundances = solutions.y.T
    t = solutions.t
    while max(abs(abundances[-1][:] - abundances[-2][:])) > de.delta:
        time_control += de.time_span
        new_solutions = solve_ivp(f, (time_control, time_control + de.time_span),
                                  abundances[-1][:], max_step=de.max_step)
        abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)
        t = np.concatenate((t, new_solutions.t), axis=0)
    for i in range(0, de.n):
        # Plots
        ax[num_samples].plot(t, abundances[:, i])
        ax[num_samples].set_xlabel("time", fontsize=15)
        ax[num_samples].set_ylabel("population", fontsize=15)
        fig1.suptitle(f'Solution for: Number of species = {de.n}', fontsize=16)

        Final_abundances[num_samples][:] = abundances[-1][:]
        Initial_abundances[num_samples][:] = abundances[0][:]
        middle_abundances[num_samples][:] = abundances[int(len(abundances) / 2)][:]
    return Final_abundances, Initial_abundances, middle_abundances

# Create a ThreadPoolExecutor with 5 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Process the elements of the list in parallel
    results = [executor.submit(solve_glv_model, num_samples) for num_samples in range(de.Y_row)]
    for future in concurrent.futures.as_completed(results):
        result = future.result()

""" Plots """
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=2)

fig2, axs = plt.subplots(3, 1)
shw1 = axs[0].imshow(Initial_abundances, aspect='auto')
cax = fig2.add_axes([axs[0].get_position().x1 + 0.01, axs[0].get_position().y0,
                     0.01, axs[0].get_position().height])
axs[0].set_title('Initial configuration')
axs[0].set_xlabel('species')
axs[0].set_ylabel('initial conditions')
bar = plt.colorbar(shw1, cax=cax)
bar.set_label('ColorBar')

shw2 = axs[1].imshow(middle_abundances, aspect='auto')
cax = fig2.add_axes([axs[1].get_position().x1 + 0.01, axs[1].get_position().y0,
                     0.01, axs[1].get_position().height])
axs[1].set_title('Middle of the process configuration')
axs[1].set_xlabel('species')
axs[1].set_ylabel('initial conditions')
bar = plt.colorbar(shw2, cax=cax)
bar.set_label('ColorBar')

shw3 = axs[2].imshow(Final_abundances, aspect='auto')
cax = fig2.add_axes([axs[2].get_position().x1 + 0.01, axs[2].get_position().y0,
                     0.01, axs[2].get_position().height])
axs[2].set_title('Steady state configuration')
axs[2].set_xlabel('species')
axs[2].set_ylabel('initial conditions')
bar = plt.colorbar(shw3, cax=cax)
bar.set_label('ColorBar')

fig2.suptitle('Species abundances for different initial conditions', fontsize=16)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)

plt.show()

# Count time.
end = time.time()
print(end - start)
