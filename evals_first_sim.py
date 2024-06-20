import numpy as np
import matplotlib.pyplot as plt

#==================================PARAMETERS==========================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 50
k = 3
t = 100
dt = 1/1000

#==================================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():

    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          #   EVAL0   EVAL1   EVAL2
    M = ( M + M.conj().T ) / 2                                                                          #   EVEC0_0 EVEC1_0 EVEC2_0
    (evals, evecs) = np.linalg.eigh(M)                                                                  #   EVEC0_1 EVEC1_1 EVEC2_1
    evals = evals/(np.sqrt(N))                                                                          #   EVEC0_2 EVEC1_2 EVEC2_2
        
    return np.concatenate(([evals], evecs), axis = 0)

#==================================EVOLUTION_FUNCTION_&_AUXILIARY_FUNCTIONS====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def potentialDerivative(x):
    return x**3 - k*x

def EigensEvolution(eigens):
    dB = np.random.normal(0, 1, N)
    dW = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))
    dW = ( dW + dW.conj().T ) / 2

    dE = np.zeros((N+1, N), dtype = complex)

    for i in range(N):
        suma1 = 0
        suma2 = 0
        suma3 = np.zeros(N, dtype = complex)
        for j in range(N):
            if i != j:
                delta = 1/(eigens[0,i] - eigens[0,j])
                delta_stoch = dW[i,j] * eigens[1:N+1, j] * delta

                suma1 += delta
                suma2 += delta**2
                suma3 += delta_stoch

        dE[0,i] = - potentialDerivative(eigens[0,i]) * dt + (1/N) * suma1 * dt + (1/np.sqrt(N)) * dB[i]
        dE[1:N+1, i] = - (1/(2*N)) * suma2 * eigens[1:N+1, i] * dt + (1/np.sqrt(N)) * suma3

    return dE

def EvalsEvolution(eigens):
    dB = np.random.normal(0, 1, N)
    dE = np.zeros(N, dtype = complex)

    for i in range(N):
        suma1 = 0
        for j in range(N):
            if i != j:
                delta = 1/(eigens[i] - eigens[j])
                suma1 += delta
        dE[i] = - potentialDerivative(eigens[i]) * dt + (1/N) * suma1 * dt + (1/np.sqrt(N)) * dB[i]

    return dE

def EigensNorm(eigens):
    for i in range(N):
        eigens[1:N+1, i] = eigens[1:N+1, i] / np.linalg.norm(eigens[1:N+1, i])
    return eigens    

#==================================PLOT_FUNCTIONS=================================

def EigensPlot(eigens):
    rez = np.zeros((N,N))
    for i in range(1,N+1):
        for j in range(N):
            rez[i-1,j] = abs(eigens[i,j])

    fig,ax = plt.subplots(1,2)
    cax = ax[0].matshow(rez)

    x = np.linspace(eigens[0,0] * 1.2, eigens[0,N-1] * 1.2,  1001)
    ax[1].plot(x, potentialV(x), color='red')
    #ax[1].scatter(eigens[0,:], [potentialV(x) for x in eigens[0,:]])
    ax[1].scatter(eigens[0,:], [0 for x in range(N)])
    ax[1].scatter([np.sqrt(k), -np.sqrt(k)], [0,0])

    fig.colorbar(cax)

    fig.show()

#==================================MAIN============================================

eig = Eigens()
evals = eig[0,:]

for i in range(t):
    evals += EvalsEvolution(evals)

eig = np.concatenate(([evals], np.identity(N)), axis = 0)

for i in range(t):
    eig += EigensEvolution(eig)
    eig = EigensNorm(eig)

eig = eig[:, np.argsort(eig[0,:].real)]

EigensPlot(eig)
plt.show()