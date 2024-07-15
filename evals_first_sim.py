import numpy as np
import scipy
import matplotlib.pyplot as plt

#==================================PARAMETERS==========================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 100
k = 4**2
t = 1000
dt = 1/10000
noiseVariance = 0.1

#==================================EIGENVALUES_&_EIGENVECTORS==================================

def EquilibriumFunction(x):
    ret = np.zeros(N, dtype = float)
    for i in range(N):
        sum = 0
        for j in range(N):
            if i != j:
                sum += 1/np.abs(x[i] - x[j])
        ret[i] = -N * (x[i]**3 - k * x[i]) + sum
    return ret

def EquilibriumFunctionJacobian(x):
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

def Eigens():
    initialGuess = np.zeros(N, dtype = float)
    for i in range(N):
        if i < N/2:
            initialGuess[i] = np.random.normal(-np.sqrt(k), 1)
        else:
            initialGuess[i] = np.random.normal(np.sqrt(k), 1)

    sol  = scipy.optimize.root(EquilibriumFunction, initialGuess, jac=EquilibriumFunctionJacobian)     
    ret = sol.x
    ret.sort()
    return ret

#==================================EVOLUTION_FUNCTION_&_AUXILIARY_FUNCTIONS====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def potentialDerivative(x):
    return x**3 - k*x

def EigensEvolution(evals, evecs):
    dB = np.random.normal(0, noiseVariance, N)
    dW = np.random.normal(0, noiseVariance/2, (N,N)) + 1j * np.random.normal(0, noiseVariance/2, (N,N))
    dW = ( dW + dW.conj().T ) / 2

    dEvecs = np.zeros((N, N), dtype = complex)
    dEvals = np.zeros(N, dtype=float)

    for i in range(N):
        suma1 = 0
        suma2 = 0
        suma3 = np.zeros(N, dtype = complex)
        for j in range(N):
            if i != j:
                delta = 1/(evals[i] - evals[j])
                delta_stoch = dW[i,j] * evecs[j] * delta

                suma1 += delta
                suma2 += delta**2
                suma3 += delta_stoch

        dEvals[i] = - potentialDerivative(evals[i]) * dt + (1/N) * suma1 * dt + (1/np.sqrt(N)) * dB[i]
        dEvecs[i] = - (1/(2*N)) * suma2 * evecs[i] * dt + (1/np.sqrt(N)) * suma3

    return (dEvals, dEvecs)

#==================================PLOT_FUNCTIONS======================================

def EigensPlot(evals, evecs):
    rez = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            rez[i,j] = abs(evecs[i,j])

    fig,ax = plt.subplots(1,2)
    cax = ax[0].matshow(rez)

    x = np.linspace(evals[0] * 1.2, evals[N-1] * 1.2,  1001)
    ax[1].plot(x, potentialV(x), color='red')
    ax[1].scatter(evals, [potentialV(x) for x in evals])

    fig.colorbar(cax)

#==================================MAIN============================================

evals = Eigens()
evecs = np.identity(N, dtype=complex)

for i in range(t):
    print(i)
    dEvals, dEvecs = EigensEvolution(evals, evecs)
    evals += dEvals
    evecs += dEvecs

#sorts the eigenvectors by ascending eigenvalues
    evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
    evals.sort()
    evecs, _ = np.linalg.qr(np.transpose(evecs))
    evecs = np.transpose(evecs)

EigensPlot(evals, evecs)
plt.show()