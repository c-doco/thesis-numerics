import numpy as np
import scipy
import matplotlib.pyplot as plt

N = 50
k = 3**2

def V(x,k):
    return x**4/4 - k*x**2/2

def EquilibriumFunction(x, k):
    ret = np.zeros(N, dtype = float)
    for i in range(N):
        sum = 0
        for j in range(N):
            if i != j:
                sum += 1/np.abs(x[i] - x[j])
        ret[i] = -N * (x[i]**3 - k * x[i]) + sum
    return ret

def EquilibriumFunctionJacobian(x, k):
    ret = np.zeros((N,N), dtype = float)
    for i in range(N):
        sum = 0
        for j in range(N):
            if i > j:
                ret[i][j] = -1/(x[i] - x[j])**2
                sum += 1/(x[i] - x[j])**2
            if i < j:
                ret[i][j] = 1/(x[i] - x[j])**2
                sum += -1/(x[i] - x[j])**2
        ret[i][i] = -N * (3 * x[i]**2 - k) + sum
    return ret

initialGuess = np.zeros(N, dtype = float)
for i in range(N):
    if i < N/2:
        initialGuess[i] = np.random.normal(-np.sqrt(k), 1)
    else:
        initialGuess[i] = np.random.normal(np.sqrt(k), 1)

sol  = scipy.optimize.root(EquilibriumFunction, initialGuess, args=(k), jac=EquilibriumFunctionJacobian)

maxx = max(abs(min(sol.x)), abs(max(sol.x)))
xlinsp = np.linspace(-1.5 * maxx, 1.5 * maxx, 10000)

plt.plot(xlinsp, [V(t, k) for t in xlinsp])
plt.scatter(sol.x, [V(t,k) for t in sol.x], color='red')
plt.show()