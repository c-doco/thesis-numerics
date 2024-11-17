import numpy as np
import scipy as sc
from mpmath import mp
import flint as fl
import math
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interp1d

np.set_printoptions(linewidth = 500)
mp.dps = 300
fl.ctx.dps = 300
mp.pretty = True
fl.ctx.pretty = True

#====================================PARAMETERS=======================================

N = 2
mpN = mp.mpf(N)
beta = 2
sqrtN = np.sqrt(N)

evalsSteps = 20000
steps = 600001
dt = 1/5000
noiseVarianceBase = 20

a = -20
b = 20

#=================================AUXILIARY_FUNCTIONS=================================

def potentialV(x, k):
    return x**4/4 - k * x**2 / 2

def potentialDerivative(x, k):
    return x*x*x - k * x

def ScaledWeight(x, mpk):
    return mp.exp(-mpN*(x**4/4 - mpk * x**2/2 + mpk**2/4))

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#================================EVOLUTION_FUNCTIONS=====================================

def EigensEvolution(evals, evecs, steps, k):
    W = np.zeros((N,N), dtype=complex)
    D = np.zeros((N,N), dtype=float)

    noiseVariance = noiseVarianceBase
    
    for t in range(steps):
        if t % 10000 == 0:
          print(t/steps * 100, '%')
        
        W = (np.random.normal(0, noiseVariance, (N,N)) + 1j * np.random.normal(0, noiseVariance, (N,N))) / np.sqrt(2)
        W = ( W + W.conj().T ) / 2

        D = np.reciprocal((evals[:, np.newaxis] - evals) + np.identity(N))
        np.fill_diagonal(D, 0)

        first_order1  = - np.matmul(evecs, np.multiply(W,D)) / sqrtN
        second_order1 = - beta * np.multiply(evecs, np.sum(np.multiply(D,D), axis=1)) / (2*N)
        
        evals += (- potentialDerivative(evals, k) + beta * np.sum(D, axis=1)/N + W.diagonal().real/sqrtN ) * dt
        evecs += (first_order1 + second_order1) * dt 
        
        if np.all(evals[:-1] <= evals[1:]) == False:
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
        if t % 5 == 0:
            evecs /= np.linalg.norm(evecs, axis=0)
            if t % 25 == 0:
                evecs, _ = np.linalg.qr(evecs)
            
    return (evals, evecs)



def EvalsEvolution(evals, evalsSteps, k):
    for t in range(evalsSteps):
        if t < 0.8 * evalsSteps:
            noiseVariance = noiseVarianceBase
        else:
            noiseVariance = 0
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - potentialDerivative(evals, k) * dt + dt * 2*np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
    return evals

#==============================ORTHOGONAL_POLYNOMIALS_GENERATION===============================

def wxp(p, mpk):
    dp = mp.mpf(p)
    if mpk != 0:
        t = mp.sqrt( (mpk + mp.sqrt(mpk**2 + 4*dp/mpN)) / 2 )
    else:
        t = 1
    s = lambda x : mp.exp( -mpN * ( (x**4/4 - mpk * x**2/2) - (t**4/4 - mpk * t**2/2))) * (x/t)**dp
    result = mp.quad(s, mp.linspace(a, b, 150))
    return result

def ppw(p1, p2, mpk, xpwint):
    p = p1 * p2
    coef = [mp.mpf(i) for i in p.coeffs()]
    sum = mp.mpf(0)
    for i in range(0,p.degree() + 1, 2):
        if mpk != 0:
            t = mp.sqrt( (mpk + mp.sqrt(mpk**2 + 4*i/mpN)) / 2 )
        else:
            t = 1
        sum += coef[i] * xpwint[i] * mp.power(t,i) * mp.exp(-mpN*(t**4/4 - mpk * t**2 / 2 + mpk**2/4))
    return fl.arb(sum)

def DensityProfile(mpk):
    
    xpwint = []
    for i in range(2*N+1):
        if i % 2 == 0:
            xpwint.append(wxp(i, mpk))
        else:
            xpwint.append(0)
            
    polys = []
    one = fl.arb_poly([1]); x = fl.arb_poly([0,1])

    polys.append(one); polys.append(x)

    sqnorms = []
    sqnorm1 = ppw(x, x, mpk, xpwint); sqnorm2 = ppw(one,one, mpk, xpwint)
    sqnorms.append(sqnorm2); sqnorms.append(sqnorm1)

    for i in range(2, N+1):
        p1 = polys[i-1]; p2 = polys[i-2]
        c = sqnorm1 / sqnorm2
        p = x * p1 - c * p2
    
        sqnormp = ppw(p,p, mpk, xpwint)
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

def GaussProfile(x, evals):
    sigma = 0.3/N
    t = mp.sqrt(2*N)
    ret = 0
    for z in evals / mp.sqrt(4*N):
        ret += mp.exp(-(x - z)**2 / (2*sigma**2)) / mp.sqrt(2 * sigma**2 * mp.pi)
    return ret / N

def EigensPlot(evalsList, evecsList,  densityList, krange):
    t = mp.sqrt(4*N)
    x = mp.linspace(-1.25, 1.25, 1001)
    plotMax = 1.25

    fig,ax = plt.subplots(len(krange),2)
    for i in range(len(krange)):
        cax = ax[i][0].matshow(np.abs(evecsList[i]), label='Absolute Value Eigenvector Matrix', cmap='magma')
        fig.colorbar(cax, ax=ax[i][0])
        if i == 0:
            ax[i][0].title.set_text('Absolute value of Eigenvector matrix \n\n k={}'.format(krange[i]))
        else:
            ax[i][0].title.set_text('k={}'.format(krange[i]))
        
        ax[i][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top=False, labeltop=False)

    for j in range(len(krange)):
        KSFit(evalsList[j], densityList[j], t)
        ax[j][1].plot(x, [GaussProfile(z, evalsList[j]) for z in x], color='red', label='Numerical Density')
        ax[j][1].plot(x, [densityList[j](q) for q in x], color='blue', label='Marginal Density', linestyle='-')
        ax[j][1].set_ylim([0,15])
        ax[j][1].set_xlim([-0.3,0.3])
        ax[j][1].set_ylabel('Probability Density')
        ax[j][1].set_xlabel('x')
        ax[j][1].legend(loc=1, prop={'size': 10})
        if j == 0:
            ax[j][1].title.set_text('Eigenvalue distribution \n\n k={}'.format(krange[j]))
        else:
            ax[j][1].title.set_text('k={}'.format(krange[j]))

def PDF_to_CDF(pdf, x_range=(-1,1), num_points=1001):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    cdf_values = np.zeros(num_points)
    for i in range(num_points):
        cdf_values[i] = cdf_values[i-1] + float(pdf(mp.mpf(x_values[i])))
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

    
#====================================MAIN======================================

krange= [0, 2, 4, 8]
evalsList = []
evecsList = []
densityList = []
x = mp.linspace(-1.25, 1.25, 10001)

for k in krange:
    mpk = mp.mpf(k)
    p = DensityProfile(mpk)
    t = mp.sqrt(4*N)
    Density = lambda x : mp.mpf(p(fl.arb(x))) * ScaledWeight(x, mpk)
    norm = mp.quad(Density, mp.linspace(a, b, 40))
    ScaledDensity = lambda x : t * Density(t * x) / norm
    densityList.append(ScaledDensity)

    evals, _ = Eigens()
    evals.sort()
    evecs = np.identity(N, dtype=complex)

    evals = EvalsEvolution(evals, evalsSteps, k)
    evals.sort()
    evals, evecs = EigensEvolution(evals, evecs,  steps, k)
    evalsList.append(evals)
    evecsList.append(evecs)

EigensPlot(evalsList, evecsList, densityList, krange)
plt.show()
        

