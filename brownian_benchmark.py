import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
import scipy.stats as st
from scipy.integrate import quad
from scipy.optimize import minimize
# import scipy as sc
from mpmath import mp
import flint as fl
import math

np.set_printoptions(linewidth = 500)
mp.dps = 300
fl.ctx.dps = 300
mp.pretty = True
fl.ctx.pretty = True

#====================================PARAMETERS=======================================

N = 100
k = 1e-50
mpN = mp.mpf(N)
mpk = mp.mpf(k)

steps = 10000
evalsSteps = 10000
dt = 1/2000
noiseVarianceBase = 4

a = -100
b = 100

#=========================================POLYNOMIAL_GENERATION=======================================

def wxp(p):
    dp = mp.mpf(p)
    if dp != 0:
        t = mp.sqrt( dp / 2 )
    else:
        t = 1
    s = lambda x : mp.exp( - x**2/2 + (t**2/2)) * (x/t)**dp
    result = mp.quad(s, mp.linspace(a, b, 500))
    return result

def ppw(p1, p2):
    p = p1 * p2
    coef = [mp.mpf(i) for i in p.coeffs()]
    sum = mp.mpf(0)
    for i in range(0,p.degree() + 1, 2):
        if i != 0:
            t = mp.sqrt( i / 2 )
        else:
            t = 0
        sum += coef[i] * xpwint[i] * mp.power(t,i) * mp.exp(- t**2/2)
    return fl.arb(sum)

xpwint = []
for i in range(2*N+1):
    if i % 2 == 0:
        xpwint.append(wxp(i))
    else:
        xpwint.append(0)
            
def DensityProfile():
            
    polys = []
    one = fl.arb_poly([1]); x = fl.arb_poly([0,1])

    polys.append(one); polys.append(x)

    sqnorms = []
    sqnorm1 = ppw(x, x); sqnorm2 = ppw(one,one)
    sqnorms.append(sqnorm2); sqnorms.append(sqnorm1)

    for i in range(2, N+1):
        p1 = polys[i-1]; p2 = polys[i-2]
        c = sqnorm1 / sqnorm2
        p = x * p1 - c * p2
    
        sqnormp = ppw(p,p)
        sqnorms.append(sqnormp)
        sqnorm2 = sqnorm1
        sqnorm1 = sqnormp
        polys.append(p)

    p1 = polys[N]; p2 = polys[N-1]

    p = p1.derivative() * p2 - p1 * p2.derivative()
    pcoef = [z/sqnorms[N-1] for z in p.coeffs()]
    p = fl.arb_poly(pcoef)

    return p

#=============================PLOT_FUNCTIONS===============================

def GaussProfile(x):
    ret = 0
    for z in evals:
        ret += N * mp.exp(-(x - z)**2 * N**2/2) / mp.sqrt(2 * mp.pi)
    return ret

def EigensPlot(evals, density, title):
    rez = np.abs(evecs)

    fig,ax = plt.subplots(1,2)
    cax = ax[0].matshow(rez)

    norm = mp.quad(GaussProfile, mp.linspace(a,b, 200))
    x = mp.linspace(-10, 10, 1001)
    xx = np.linspace(-10,10, 1001)
    # ax.hist(evals, bins = 100, density = True)
    ax[1].plot(x, [GaussProfile(z)/norm for z in x], linestyle='dotted', color='red', label='Numerical Results')
    ax[1].plot(x, [density(z) for z in x], linestyle='dotted', color='purple', label='Kernel Density')
    ax[1].plot(xx, [Semicircle(z) for z in xx], linestyle='dotted', color='steelblue', label='Semicircle Law')
    ax[1].legend()
    # ax.scatter(evals, [potentialV(x) for x in evals])
    # ax.set_title(title)

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#================================AUXILIARY_FUNCTIONS====================================

def V(x, k):
    return x**2/2

def Vd(x, k):
    return x

def Vdd(x, k):
    return 1

def ScaledWeight(x):
    return mp.exp(-mpN*(x**4/4 - mpk * x**2/2 + mpk**2/4))

def Semicircle(x):
    if np.abs(x) < np.sqrt(2):
        return np.sqrt(2-x*x)/np.pi
    else:
        return 0

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
        # if t < steps/5:
        #     noiseVariance = noiseVarianceBase * 0.75
        # elif t < steps/2:
        #     noiseVariance = noiseVarianceBase * 0.5
        # else:
        #     noiseVariance = noiseVarianceBase * 0.15
        #PUNKLA
        noiseVariance = noiseVarianceBase * 2
           
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
            # evecs += - np.matmul( evecs, - np.diag(Vdd(evals,k)) - 2*np.diag( np.sum( np.multiply(D,D), axis=1 ) )/N + np.multiply(W, D)  ) * dt
            evecs += - ( np.multiply( evecs, Vdd(evals,k) + 2*np.sum( np.multiply(D,D), axis=1 ) / N ) + np.matmul(evecs, np.multiply(W,D) ) ) * dt
        except ValueError:
           print('Nan or Inf')
           t = t-1
           continue

#check if evals is sorted, if not sorts evals and evecs by evals
        if np.all(evals[:-1] <= evals[1:]) == False:
            evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
            evals = np.sort(evals)
        evecs, _ = np.linalg.qr(evecs)

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
    

#==================================MAIN============================================

evals, _ = Eigens()
evals.sort()
evecs = np.identity(N, dtype=complex)

evals = EvalsEvolution(evals, evalsSteps, k)
evals.sort()
evals, evecs = EigensEvolution(evals, evecs, steps, k)

p = DensityProfile()
Density = lambda x : mp.mpf(p(fl.arb(x))) * ScaledWeight(x)
norm = mp.quad(Density, mp.linspace(a, b, 100))
ScaledDensity = lambda x : Density(x) / norm

EigensPlot(evals, ScaledDensity, 'N={}, k={}'.format(N, k))
plt.show()

