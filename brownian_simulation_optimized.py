import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
import scipy.stats as st
from scipy.integrate import quad
from scipy.optimize import minimize

#==================================PARAMETERS==========================================
large_width = 400
np.set_printoptions(linewidth=large_width, precision=2)

N = 100

#PUNKKLA
kmin =10
kmax =10
#PUNKLA
ksteps = int((kmax - kmin) * 4 + 1)

steps = 1
evalsSteps = 0
dt = 1/5000
noiseVarianceBase = 100

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#================================AUXILIARY_FUNCTIONS====================================

def V(x, k):
    return x**4/4 - k*x**2/2

def Vd(x, k):
    return x*x*x - k*x

def Vdd(x, k):
    return 0*x

def negLogProb(N, data):
    n = len(data)
    if N <= 0:  # To ensure N remains positive during optimization
        return np.inf
    return -n * np.log(N) + N * np.sum(data)

class PorterThomas(st.rv_continuous):
    def _pdf(self, x, M):
        return M * np.exp(-M * x)
    
    def _argcheck(self, M):
        check = M >= 0
        return check

#================================EVOLUTION_FUNCTIONS=====================================

def EigensEvolution(evals, evecs, steps, k):
    W = np.zeros((N,N), dtype=complex)
    D = np.zeros((N,N), dtype=float)

    for t in range(steps):
        
        #PUNKLA
        noiseVariance = noiseVarianceBase * 0.35
        #PUNKLA
        
        if t % 10000 == 0:
            print(t)
           
        try:
            W = (np.random.normal(0, noiseVariance, (N,N)) + 1j * np.random.normal(0, noiseVariance, (N,N)))/np.sqrt(2)
            W = ( W + W.conj().T ) / 2
            
            D = evals[:, np.newaxis] - evals
            D += np.identity(N)
            D = np.reciprocal(D)
            np.fill_diagonal(D, 0)

            tmp = - Vd(evals, k) * dt + dt * 2 * np.sum(D, axis=1)/N + dt * W.diagonal().real
            if(np.any(np.isnan(tmp)) or np.any(np.isinf(tmp))):
                raise ValueError('Nan or Inf')
            evals += tmp
            evecs += - ( np.multiply( evecs, Vdd(evals,k) + np.sum( np.multiply(D,D), axis=1 ) / (2*N) ) + np.matmul(evecs, np.multiply(W,D) ) ) * dt
        except ValueError:
           print('Nan or Inf')
           t = t-1
           continue

#check if evals is sorted, if not sorts evals and evecs by evals
        if np.all(evals[:-1] <= evals[1:]) == False:
            evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
            evals = np.sort(evals)
        # evecs, _ = np.linalg.qr(evecs)
        print('\nevecs:\n',evecs)
        print('\nOne:\n',np.matmul(evecs, np.conj(np.transpose(evecs))).diagonal().real)

    return (evals, evecs)

def EvalsEvolution(evals, evalsSteps, k):
    noiseVariance = 0
    for t in range(evalsSteps):
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - Vd(evals, k) * dt + dt * 2 * np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
    return evals
#============================STATISTICAL_ANALYSIS_FUNCTIONS============================

def PorterThomasFit(evals, evecs):
    n = bisect(evals, 0)
    diag1 = np.square( np.abs(evecs[0:n, 0:n]) ).ravel()
    diag2 = np.square( np.abs(evecs[n:N, n:N]) ).ravel()
    offur = np.square( np.abs(evecs[0:n, n:N]) ).ravel()
    offdl = np.square( np.abs(evecs[n:N, 0:n]) ).ravel()

    diag = np.concatenate((diag1, diag2))
    offd = np.concatenate((offur, offdl))

    porterThomas = PorterThomas(name='Porter-Thomas', a=1e-14, b=1e14)
    N1 = porterThomas.fit(diag, N, floc=0, fscale=1)
    N2 = porterThomas.fit(offd, N, floc=0, fscale=1)
    print('PT_diag={}, PD_offd={}'.format(N1[0], N2[0]))

    initialGuess = N
    diagRes = minimize(negLogProb, initialGuess, args=(diag,), bounds=[(1e-14, None)])
    diagHessianInv = diagRes.hess_inv.todense()
    N1n = diagRes.x[0]
    N1_error = np.sqrt(np.diag(diagHessianInv))[0]
    
    offdRes = minimize(negLogProb, initialGuess, args=(offd,), bounds=[(1e-14, None)])
    offdHessianInv = offdRes.hess_inv.todense()
    N2n = offdRes.x[0]
    N2_error = np.sqrt(np.diag(offdHessianInv))[0]

    print('PT_diag={} pm {}, PD_offd={} pm {}'.format(N1n, N1_error, N2n, N2_error))
    
    return 0   
    

#==================================PLOT_FUNCTIONS======================================

def GaussProfile(x):
    ret = 0
    for t in evals:
        ret += np.exp(-(x - t)**2 * N**2/2) / np.sqrt(2 * np.pi)
    return ret

def EigensPlot(evals, evecs, title):
    rez = np.abs(evecs)

    fig,ax = plt.subplots(1,2)
    cax = ax[0].matshow(rez)


    # gaussNorm = quad(GaussProfile, -np.inf, np.inf)[0]
    x = np.linspace(evals[0] * 1.2, evals[N-1] * 1.2,  1001)
    ax[1].plot(x, V(x, k), color='red')    
    ax[1].scatter(evals, [V(x, k) for x in evals])
    ax[1].set_title(title)

    fig.colorbar(cax)

def EvecsPlot(evecs):
    print(np.shape(evecs))
    rez = np.abs(evecs)

    fig,ax = plt.subplots(1,1)
    cax = ax.matshow(rez)
    fig.colorbar(cax)
    

#==================================MAIN============================================

krange = np.linspace(kmin, kmax, ksteps)
print(krange)

for i in krange:
    k = i
    print(k,':', end=' ')
    evals, _ = Eigens()
    evals.sort()
    evecs = np.identity(N, dtype=complex)

    evals = EvalsEvolution(evals, evalsSteps, k)
    evals.sort()
    evals, evecs = EigensEvolution(evals, evecs, steps, k)

    PorterThomasFit(evals, evecs)
EigensPlot(evals, evecs, 'N={}, k={}'.format(N,k))
plt.show()
