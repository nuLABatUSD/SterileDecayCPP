import numpy as np
import matplotlib.pyplot as plt
import ODESolve_slow as ODE
#import ODESolve as ODE


y0 = np.ones(3)
p = np.zeros(4)

p[0] = 0
p[1:] = np.random.rand(3)

x0 = 0
dx0 = 0.01


results = ODE.ODEOneRun(x0, y0, dx0, p, 100, 2, 20)


plt.figure()
for i in range(3):
    plt.plot(results[0], results[1][:,i], 'o-',markersize=3,label=r"$\lambda = -{}$".format(np.round(p[i+1], 3)))
plt.legend(loc='upper right')
plt.title(r"Solving $y' = - \lambda y^2$ with $y(0) = 1$")
plt.show()


y1 = np.array([1, 0, -np.pi**2, 0])
p1 = np.array([1, np.pi**4])

x0 = 0
dx0 = 0.01


results1 = ODE.ODEOneRun(x0, y1, dx0, p1, 100, 3, 10)




plt.figure()
plt.plot(results1[0], results1[1][:,0], 'o-')
plt.show()




f, axs = plt.subplots(nrows=2, sharex='col', height_ratios=[3,1])
plt.subplots_adjust(hspace=0)

axs[0].plot(results1[0], results1[1][:,0], 'o-')
axs[1].plot(results1[0][1:], results1[2][1:], 'o')
axs[1].set_ylim(0.02, 0.045)






