import defaults as de
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

# Count time.
start = time.time()

# GLV formula.
def f(t, x): return np.array([de.r[i] * x[i] - de.s[i] * x[i] ** 2 +
                           sum([de.A1[i][p] * x[i] * x[p] for p in
                                range(0, de.n) if p != i]) for i in range(0, de.n)])

# Stop condition if x[i] != 0, not working at the moment. but I prefer the other idea of stop condition.
def g(x): return np.array([x[i] - 1/de.s[i]*(de.r[i] + sum([de.A1[i][p] * x[p]
                                for p in range(0, de.n) if p != i])) for i in range(0, de.n)])

# Initiation.
Final_abundances = np.zeros((de.Y_row, de.n))
Initial_abundances = np.zeros((de.Y_row, de.n))
middle_abundances = np.zeros((de.Y_row, de.n))

fig1, ax = plt.subplots(1, de.Y_row)

for m in range(0, de.Y_row):
    time_control = 0
    solutions = solve_ivp(f, (time_control, time_control + de.time_span), de.Y1[m][:], max_step=de.max_step)
    abundances = solutions.y.T
    t = solutions.t
    non_zero_abundances_index = np.nonzero(abundances[-1][:])
    #non_zero_abundances_index = np.where(abundances[-1][:] > de.epsilon)
    print(abundances[-1][:])
    while max(abs(abundances[-1][:] - abundances[-2][:])) > de.delta:
    #while max(abs(g(abundances[-1][:])[non_zero_abundances_index])) > de.delta:
        print(abundances[-1][:])
        print(g(abundances[-1][:])[non_zero_abundances_index])
        print(max(abs(g(abundances[-1][:])[non_zero_abundances_index])))
        print(min(abundances[-1][:][non_zero_abundances_index]))
        if np.where(abundances[-1][:] < de.epsilon):
            abundances[-1][:][np.where(abundances[-1][:] < de.epsilon)] = 0
        time_control += de.time_span
        new_solutions = solve_ivp(f, (time_control, time_control + de.time_span),
                                  abundances[-1][:], max_step=de.max_step)
        abundances = np.concatenate((abundances, new_solutions.y.T), axis=0)
        t = np.concatenate((t, new_solutions.t), axis=0)
        #non_zero_abundances_index = np.nonzero(abundances[-1][:])
        #non_zero_abundances_index = np.where(abundances[-1][:] > de.epsilon)

    for i in range(0, de.n):
        # Plots
        ax[m].plot(t, abundances[:, i])
        ax[m].set_xlabel("time", fontsize=15)
        ax[m].set_ylabel("population", fontsize=15)
        fig1.suptitle(f'Solution for: Number of species = {de.n}', fontsize=16)

    Final_abundances[m][:] = abundances[-1][:]
    Initial_abundances[m][:] = abundances[0][:]
    middle_abundances[m][:] = abundances[int(len(abundances)/2)][:]

# plots.
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




