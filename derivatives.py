import numpy as np
import numba as nb

######################
# This wrapper, @nb.jit(nopython=True), is necessary to use ODESolve (uncomment to use). ODESolve_slow will work with or without the wrapper.
#
#@nb.jit(nopython=True)
def f(x, y, p):
    if p[0] == 0:
        der = np.zeros(3)

        for i in range(3):
            der[i] = - p[i+1] * y[i]**2

        return der
    elif p[0] == 1:
        der = np.zeros(4)

        for i in range(3):
            der[i] = y[i+1]
        der[3] = p[1] * y[0]

        return der
    else:
        print("Only works for p[0] = 0 or 1")