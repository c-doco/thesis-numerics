import numpy as np
import scipy
import matplotlib.pyplot as plt

#==================================PARAMETERS==========================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 50
k = 8
steps = 20000
evalsSteps = 2000
dt = 1/2000
noiseVariance = 10

#==================================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = evals/(np.sqrt(N))                                                                          
    
#arranges eigenvectors so that eigenvector of evals[i] is evecs[i] 
#instead of evecs[:,i] which is default in numpy
    formatEvecs = np.zeros((N,N), dtype=complex)                                                        
    for i in range(N):                                                                                  
        formatEvecs[i] = evecs[:,i].copy()

    return (evals, formatEvecs)

#==================================EVOLUTION_FUNCTION_&_AUXILIARY_FUNCTIONS====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def potentialDerivative(x):
    return x**3 - k*x

def EigensEvolution(evals, evecs, steps):
    for t in range(steps):

        if t % 100 == 0:
            print(t)

        dB = np.random.normal(0, noiseVariance, N)
        dW = np.random.normal(0, noiseVariance/2, (N,N)) + 1j * np.random.normal(0, noiseVariance/2, (N,N))
        dW = ( dW + dW.conj().T ) / 2

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

            evals[i] += - potentialDerivative(evals[i]) * dt + (1/N) * suma1 * dt + (1/np.sqrt(N)) * dB[i] * dt
            evecs[i] += - (1/(2*N)) * suma2 * evecs[i] * dt + (1/np.sqrt(N)) * suma3 * dt
        
    #sorts the eigenvectors by ascending eigenvalues
        evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
        evals.sort()
        evecs, _ = np.linalg.qr(np.transpose(evecs))
        evecs = np.transpose(evecs)

    return (evals, evecs)

def EvalsEvolution(evals, evalsSteps):
    for t in range(evalsSteps):

        if t % 100 == 0:
            print("evals", t)

        dB = np.random.normal(0, noiseVariance, N)

        for i in range(N):
            suma1 = 0
            for j in range(N):
                if i != j:
                    suma1 += 1/(evals[i] - evals[j])

            evals[i] += - potentialDerivative(evals[i]) * dt + (1/N) * suma1 * dt + (1/np.sqrt(N)) * dB[i] * dt
            
        evals.sort()
    return evals

#==================================PLOT_FUNCTIONS======================================

def EigensPlot(evals, evecs, title):
    rez = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            rez[i,j] = abs(evecs[i,j])

    fig,ax = plt.subplots(1,2)
    cax = ax[0].matshow(rez)

    x = np.linspace(evals[0] * 1.2, evals[N-1] * 1.2,  1001)
    ax[1].plot(x, potentialV(x), color='red')
    ax[1].scatter(evals, [potentialV(x) for x in evals])
    ax[1].set_title(title)

    fig.colorbar(cax)

#==================================MAIN============================================

evals, _ = Eigens()
evecs = np.identity(N, dtype=complex)

evals = EvalsEvolution(evals, evalsSteps)
evals, evecs = EigensEvolution(evals, evecs, steps)

EigensPlot(evals, evecs, 'N={}, k={}, noiseVariance={}'.format(N,k,noiseVariance))
plt.show()