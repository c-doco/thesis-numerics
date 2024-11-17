import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
import scipy.stats as st
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
from mpmath import mp
import flint as fl

np.set_printoptions(linewidth = 500)
mp.dps = 100
fl.ctx.dps = 100
mp.pretty = True
fl.ctx.pretty = True

#====================================PARAMETERS=======================================

N = 100
mpN = mp.mpf(N)

steps = 1000001
# steps = 1
evalsSteps = 0
dt = 1/2000
noiseVarianceBase = 20

a = -20
b = 20

#=========================================POLYNOMIAL_GENERATION=======================================

def wxp(p):
    dp = mp.mpf(p)
    if dp != 0:
        t = mp.sqrt( dp / 2 )
    else:
        t = 1
    s = lambda x : mp.exp( - x**2/2 + (t**2/2)) * (x/t)**dp
    result = mp.quad(s, mp.linspace(a, b, 10))
    return result

def ppw(p1, p2):
    p = p1 * p2
    coef = [mp.mpf(i) for i in p.coeffs()]
    sum = mp.mpf(0)
    for i in range(0,p.degree() + 1, 2):
        if i != 0:
            t = mp.sqrt( i / 2 )
        else:
            t = 1
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
    sigma = 2/N #2/mp.sqrt(N)
    t = mp.sqrt(4*N)
    ret = 0
    for z in evals / t:
        ret += mp.exp(-(x - z)**2 / (2*sigma**2)) / mp.sqrt(2 * sigma**2 * mp.pi)
    return float(ret / N)

def Semicircle(x):
    t = 1/np.sqrt(2)
    if np.abs(x**2 / t**2) < 2:
        return np.sqrt(2-x**2/t**2)/(np.pi * t)
    else:
        return 0

def EigensPlot(evecs, evals, density, title):
    rez = np.abs(evecs)
    t = mp.sqrt(4*N)

    plotMax = 1.3
    
    x = np.linspace(-plotMax, plotMax, 1001)

    fig,ax = plt.subplots(2,2)
    ax[0][1].plot(x*t, [V(q) for q in x*t], label='Gaussian Potential', color='blue', zorder=1, linewidth=1)
    ax[0][1].scatter(evals, [V(q) for q in evals], label='Eigenvalues', color='orange', zorder=2, s=14)
    ax[0][1].legend(loc=1)
    ax[0][1].set_xlabel('x')
    ax[0][1].set_ylabel('V(x)')
    ax[0][1].title.set_text('b')
    
    ax[1][0].plot(x, [density(z) for z in x], color='red', label='Marginal Density')
    ax[1][0].plot(x, [GaussProfile(z) for z in x], color='orange', label='Numerical Density', linestyle='--')
    KSFit(evals, GaussProfile, t)
    ax[1][0].set_ylim([0,1])
    ax[1][0].set_xlim([-plotMax,plotMax])
    ax[1][0].legend(loc=1)
    ax[1][0].set_xlabel('x')
    ax[1][0].set_ylabel('Probability Density')
    ax[1][0].title.set_text('c)')

    cax = ax[0][0].matshow(rez,cmap='magma')
    fig.colorbar(cax, ax=ax[0][0])
    ax[0][0].title.set_text('a)')
    ax[0][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top=False, labeltop=False)


    ax[1][1].plot(x, [Semicircle(z) for z in x], color='blue', label='Semicircle Law')
    ax[1][1].plot(x, [GaussProfile(z) for z in x], color='orange', label='Numerical Density', linestyle='--')
    KSFit(evals, Semicircle, t)
    ax[1][1].set_ylim([0,1])
    ax[1][1].set_xlim([-plotMax,plotMax])
    ax[1][1].legend(loc=1)
    ax[1][1].set_xlabel('x')
    ax[1][1].set_ylabel('Probability Density')
    ax[1][1].title.set_text('d)')

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)
    return (evals, evecs)

#================================AUXILIARY_FUNCTIONS====================================

def V(x):
    return x**2/2

def Vd(x):
    return x

def ScaledWeight(x):
    return mp.exp(-x**2/2)


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

def EigensEvolution(evals, evecs, steps):
    W = np.zeros((N,N), dtype=complex)
    D = np.zeros((N,N), dtype=float)

    for t in range(steps):

        if t < steps/1.5:
            noiseVariance = noiseVarianceBase *5
        else:
            noiseVariance = 0

        if t% 10000 == 0:
            print(t/steps*100, '%')
                       
        W = (np.random.normal(0, noiseVariance, (N,N)) + 1j * np.random.normal(0, noiseVariance, (N,N)))/np.sqrt(2)
        W = ( W + W.conj().T ) / 2
        
        D = evals[:, np.newaxis] - evals
        D += np.identity(N)
        D = np.reciprocal(D)
        np.fill_diagonal(D, 0)

        evals += ( - Vd(evals)/N +  2 * np.sum(D, axis=1)/N +  W.diagonal().real/np.sqrt(N) ) * dt
        evecs += - ( np.multiply( evecs, np.sum( np.multiply(D,D), axis=1 )/N ) + np.matmul(evecs, np.multiply(W,D) )/np.sqrt(N) ) * dt

        #check if evals is sorted, if not sorts evals and evecs by evals
        if np.all(evals[:-1] <= evals[1:]) == False:
            evecs = np.array([x for _, x in sorted(zip(evals, evecs))])             
            evals = np.sort(evals)
        evecs, _ = np.linalg.qr(evecs)

    return (evals, evecs)

def EvalsEvolution(evals, evalsSteps):
    noiseVariance = 0
    for t in range(evalsSteps):
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - Vd(evals) * dt + dt * 2 * np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
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

def PDF_to_CDF(pdf, x_range=(-1,1), num_points=1001):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    cdf_values = np.zeros(num_points)
    for i in range(num_points):
        cdf_values[i] = cdf_values[i-1] + pdf(x_values[i])
    norm = cdf_values[num_points-1]
    cdf_values = cdf_values/norm
    
    cdf_function = interp1d(x_values, cdf_values, bounds_error=False, fill_value=(0, 1))
    return cdf_function    

def KSFit(evals, density, t):
    cdf = PDF_to_CDF(density)
    evals = evals / float(t)
    ret = st.kstest(evals, cdf)
    print(ret)
    return 0


#==================================MAIN============================================

evals, _ = Eigens()
evals.sort()
evecs = np.identity(N, dtype=complex)

evals = EvalsEvolution(evals, evalsSteps)
evals.sort()
evals, evecs = EigensEvolution(evals, evecs, steps)

t = mp.sqrt(4*N)
p = DensityProfile()
Density = lambda x : mp.mpf(p(fl.arb(x))) * ScaledWeight(x)
norm = mp.quad(Density, mp.linspace(a, b, 100))
ScaledDensity = lambda x : float(t * Density(t * x) / norm)

KSFit(evals, ScaledDensity, t)
PorterThomasFit(evals, evecs)
EigensPlot(evecs, evals, ScaledDensity, 'N={}'.format(N))
plt.show()

