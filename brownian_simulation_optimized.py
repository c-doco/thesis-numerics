import numpy as np
import time
import matplotlib.pyplot as plt

#==================================PARAMETERS==========================================
start_time = time.time()
large_width = 400
np.set_printoptions(linewidth=large_width)

N = 1000
k = 5
steps = 100000
evalsSteps = 2000
dt = 1/2000
noiseVariance = 1

#==================================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#==================================EVOLUTION_FUNCTION_&_AUXILIARY_FUNCTIONS====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def potentialDerivative(x):
    return x*x*x - k*x

def EigensEvolution(evals, evecs, steps):
    for t in range(steps):

        noiseVariance = 1
        if t < steps/10:
            noiseVariance = 10
        elif t < steps/4:
            noiseVariance = 5

        if t % 100 == 0:
            print(t)

        dB = np.random.normal(0, noiseVariance, N)
        dW = np.random.normal(0, noiseVariance/2, (N,N)) + 1j * np.random.normal(0, noiseVariance/2, (N,N))
        dW = ( dW + dW.conj().T ) / 2

        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)

        evals += - potentialDerivative(evals) * dt + dt * np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
        evecs += - np.matmul( evecs, np.multiply(D, D)/(2*N) + np.multiply(dW, D)/np.sqrt(N)  ) * dt

#check if evals is sorted, if not sorts evals and evecs by evals
        if np.all(evals[:-1] <= evals[1:]) == False:
            evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
            evals = np.sort(evals)
        evecs, _ = np.linalg.qr(evecs)

    return (evals, evecs)

def EvalsEvolution(evals, evalsSteps):
    for t in range(evalsSteps):
        if t % 100 == 0:
            print(t)
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - potentialDerivative(evals) * dt + dt * np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
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
evals.sort()
evecs = np.identity(N, dtype=complex)

evals = EvalsEvolution(evals, evalsSteps)
evals.sort()
evals, evecs = EigensEvolution(evals, evecs, steps)

end_time = time.time()
print(end_time - start_time)

EigensPlot(evals, evecs, 'N={}, k={}, noiseVariance={}'.format(N,k,noiseVariance))
plt.show()