import defaults as de
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
solution to the Lotka Volterra equation for n species.
"""
""""
dydt = lambda y,t: np.array([de.r[i] * y[i] -
                             de.s[i] * y[i] ** 2 + sum([de.A[i][j] * y[i] * y[j] for j in range(0,
                                                                                                de.n) if j != i])
                             for i in range(0, de.n)])
"""
def f(x, T): return np.array([de.r[i] * x[i] - de.s[i] * x[i] ** 2 +
                           sum([de.A[i][p] * x[i] * x[p]
                                for p in range(0, de.n) if p != i])
                           for i in range(0, de.n)])
###
Final_abundances = np.zeros((de.Y_row, de.n))
###
abundances = [0] * de.Y_row
fig, ax = plt.subplots()
plt.ion()
###
for m in range(0, de.Y_row):
    # Variables:
    # initialization of arrays.
    t = np.array([0, de.h])
    y = np.zeros([2, de.n])
    y[0][:] = de.Y[m][:]
    k = 1
    y[:][k] = y[:][k - 1] + f(y[:][k - 1], de.t[:]) * de.h

    # solution by Euler method:
    while max(abs(y[k][:] - y[k - 1][:])) > de.delta:
        y = np.concatenate((y, np.zeros((1, de.n))), axis=0)
        t = np.append(t, de.h * (k + 2))
        y[:][k + 1] = y[:][k] + f(y[:][k], de.t[:]) * de.h
        k = k + 1
        # abundances[m][:] = y[-1][:]

    abundances[m] = y

for i in range(0, de.Y_row):
    print(np.shape(abundances[i]))
    Final_abundances[m][:] = y[-1][:]
"""
for j in range(0, len(t)):
    plt.clf()
    shw = ax.imshow(abundances[:][:][j])
    plt.xlabel('species', fontsize=20)
    plt.ylabel('initial conditions', fontsize=20)
    plt.title('Species abundances at the steady state for diffrerent initial conditions',
              fontsize=20)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0,
                        0.01, ax.get_position().height])
    plt.show()
    plt.pause(0.01)
    bar = plt.colorbar(shw, cax=cax)
    bar.set_label('ColorBar')
###

"""
#fig, ax = plt.subplots()
shw = ax.imshow(Final_abundances) 
plt.xlabel('species', fontsize=20)
plt.ylabel('initial conditions', fontsize=20)
plt.title('Species abundances at the steady state for different initial conditions',
          fontsize=20)
cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0,
                    0.01, ax.get_position().height])
bar = plt.colorbar(shw, cax=cax)
bar.set_label('ColorBar')
plt.show()
"""
fig, ax = plt.subplots(111)

def plot_animation(fig, xlabel='species', ylabel='initial conditions',
                   title='Species abundances at the steady state for diffrerent initial conditions'):
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.title(title, fontsize = 20)
    line, = ax.imshow(Final_abundances)

    def animate(i):
        if i >= len(t):
            return line,
        line.set_ydata(Final_abundances[i, :])
        return line,

        ani = animation.FuncAnimation(fig, animate, interval=25, blit=False, frames=200, save_count=50)
        return ani


plot_animation(fig)
"""
"""
fig, ax = plt.subplots()

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

shw = ax.imshow(Final_abundances, animated = True)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
"""