r1 = 1
r2 = 1
r = np.array([r1, r2])
s1 = 0.05
s2 = 0.05
s = np.array([s1, s2])
A12 = 0.05
A21 = -0.05
A = np.array([A12, A21])
y10 = 10
y20 = 10
y0 = np.array([y10, y20])
t0 = 0
tf = 15
nt = 1000
t = np.linspace(t0, tf, nt)

dy1dt = lambda y,t:np.array([r[0]*y[0]-s[0]*y[0]**2+A[0]*y[0]*y[1],
                    r[1]*y[1]-s[1]*y[1]**2+A[1]*y[1]*y[0]])

y = np.zeros([len(t), 2])
y[0][:] = y0
for n in range(0, len(t)-1):
    y[:][n+1] = y[:][n] + dy1dt(y[:][n], t[n])*(t[n+1] - t[n])

plt.plot(t, y[:, 0], color = "blue", label = "species 1")
plt.plot(t, y[:, 1], color = "red", label = "species 2")
plt.legend(loc = "right")
plt.xlabel("time")
plt.ylabel("Population")
plt.title('Results for: A12=0.05, A21=-0.05, r1=r2=1, s1=s2=0.05')
plt.show() 