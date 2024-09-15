import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
import scipy.stats as st
from scipy.integrate import quad

#==================================PARAMETERS==========================================
large_width = 400
np.set_printoptions(linewidth=large_width)

N = 500
k = 16
evalsSteps = 20000
dt = 1/5000
noiseVarianceBase = 0
iterations = 1

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#================================AUXILIARY_FUNCTIONS====================================

def potentialV(x):
    return x**2/2

def potentialDerivative(x):
    return x

def ChristoffelDarboux(x):
    return 1

norm = quad(ChristoffelDarboux, -np.inf, np.inf)[0]

def NormedChristoffelDarboux(x):
    return ChristoffelDarboux(x) / norm

#================================EVOLUTION_FUNCTIONS=====================================

def EvalsEvolution(evals, evalsSteps):
    for t in range(evalsSteps):
        noiseVariance = noiseVarianceBase
        if t % 100 == 0:
            print(t)
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - potentialDerivative(evals) * dt + dt * np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
    return evals

#==================================PLOT_FUNCTIONS======================================

def GaussProfile(x):
    if abs(x) < np.sqrt(2) - 1e-5:
        ret = np.sqrt(2 - x**2) / np.pi
    else:
        ret = 0
    # ret = 0
    # for t in evals:
    #     ret += np.exp(-(x - t)**2 * N**2/2) / np.sqrt(2 * np.pi)
    return ret

def EigensPlot(evals, title):

    fig,ax = plt.subplots(1,1)
    gaussNorm = quad(GaussProfile, -np.inf, np.inf)[0]
    x = np.linspace(evals[0] * 1.2, evals[N-1] * 1.2,  1001)
    ax.hist(evals, bins = 100, density=True)    
    ax.plot(x, [GaussProfile(u) for u in x], color='red')
    ax.scatter(evals, [potentialV(x) for x in evals])
    ax.set_title(title)
    

#==================================MAIN============================================

evals, _ = Eigens()
evals = EvalsEvolution(evals, evalsSteps)
evals.sort()

EigensPlot(evals, 'N={}, k={}'.format(N,k))
plt.show()
