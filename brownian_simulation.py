import numpy as np
import matplotlib.pyplot as plt

#===============================PARAMETERS===================================

large_width = 800
np.set_printoptions(linewidth=large_width)

N = 50
k = 3
t = 1000
dt = 1/1000
noiseVariance = 1

#=======================GAUSSIAN_RANDOM_MATRIX=========================

def Eigens():

    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          #   EVAL0   EVAL1   EVAL2
    M = ( M + M.conj().T ) / 2                                                                          #   EVEC0_0 EVEC1_0 EVEC2_0
    (evals, evecs) = np.linalg.eigh(M)                                                                  #   EVEC0_1 EVEC1_1 EVEC2_1
    evals = evals/(np.sqrt(N))                                                                          #   EVEC0_2 EVEC1_2 EVEC2_2

    formatEvecs = np.zeros((N,N), dtype=complex)                                                        #arranges eigenvectors so that eigenvector of evals[i] is evecs[i] 
    for i in range(N):                                                                                  #instead of evecs[:,i] which is default in numpy
        formatEvecs[i] = evecs[:,i].copy()

    return (evals, formatEvecs)

#===========================EVOLUTION_FUNCTIONS===============================
def potentialV(x):
    return x**4/4 - k*x**2/2

def potentialDerivative(x):
    return x**3 - k*x

def EvalEvolution(evals, evecs):
    dB = np.random.normal(0, noiseVariance, N)
    dW = np.random.normal(0, noiseVariance/2, (N,N)) + 1j * np.random.normal(0, noiseVariance/2, (N,N))
    dW = ( dW + dW.conj().T ) / 2

    dEvecs = np.zeros((N, N), dtype = complex)
    dEvals = np.zeros(N, dtype = float)

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

#========================PLOT_FUNCTIONS=================================

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
    #ax[1].scatter(evals, [0 for x in range(N)])
    #ax[1].scatter([np.sqrt(k), -np.sqrt(k)], [0,0])

    fig.colorbar(cax)

    fig.show()                                                

#======================EVALS_&_EVECS_EVOLUTION=============================

evals, evecs = Eigens()

for i in range(t):
    (dEvals, dEvecs) = EvalEvolution(evals, evecs)
    evals += dEvals
    evecs += dEvecs

    evecs = evecs[np.argsort(evals)]                                                                        #sorts the eigenvectors by ascending eigenvalues
    evals = evals[np.argsort(evals)]                                                                        #sorts eigenvalues in case some jumps occured
    evecs, _ = np.linalg.qr(evecs)

EigensPlot(evals, evecs)
plt.show()
