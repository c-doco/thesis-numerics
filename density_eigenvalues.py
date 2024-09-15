import numpy as np
import scipy as sc
from mpmath import mp
import flint as fl
import math
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 500)
mp.dps = 300
fl.ctx.dps = 300
mp.pretty = True
fl.ctx.pretty = True

#====================================PARAMETERS=======================================

N = 30
k = 0.00001
mpN = mp.mpf(N)
mpk = mp.mpf(k)
evalsSteps = 20000
dt = 1/5000
noiseVarianceBase = 10

a = -100
b = 100

#=================================AUXILIARY_FUNCTIONS=================================

def potentialV(x):
    return x**4/4 - k * x**2 / 2

def potentialDerivative(x):
    return x*x*x - k * x

def ScaledWeight(x):
    return mp.exp(-mpN*(x**4/4 - mpk * x**2/2 + mpk**2/4))

#===========================EIGENVALUES_&_EIGENVECTORS==================================

def Eigens():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2                                                                          
    (evals, evecs) = np.linalg.eigh(M)                                                                  
    evals = (evals/np.sqrt(N)).copy()                                                                 
    return (evals, evecs)

#================================EVOLUTION_FUNCTIONS=====================================

def EvalsEvolution(evals, evalsSteps):
    for t in range(evalsSteps):
        if t < 0.8 * evalsSteps:
            noiseVariance = noiseVarianceBase
        else:
            noiseVariance = 0
        dB = np.random.normal(0, noiseVariance, N)
        D = evals[:, np.newaxis] - evals
        D = np.reciprocal(D, where= np.isclose(D,0) == False)
        evals += - potentialDerivative(evals) * dt + dt * 2*np.sum(D, axis=1)/N + dt * dB/np.sqrt(N)
    return evals

#==============================ORTHOGONAL_POLYNOMIALS_GENERATION===============================

def wxp(p):
    dp = mp.mpf(p)
    t = mp.sqrt( (mpk + mp.sqrt(mpk**2 + 4*dp/mpN)) / 2 )
    s = lambda x : mp.exp( -mpN * ( (x**4/4 - mpk * x**2/2) - (t**4/4 - mpk * t**2/2))) * (x/t)**dp
    result = mp.quad(s, mp.linspace(a, b, 500))
    return result

def ppw(p1, p2):
    p = p1 * p2
    coef = [mp.mpf(i) for i in p.coeffs()]
    sum = mp.mpf(0)
    for i in range(0,p.degree() + 1, 2):
        t = mp.sqrt( (mpk + mp.sqrt(mpk**2 + 4*i/mpN)) / 2 )
        sum += coef[i] * xpwint[i] * mp.power(t,i) * mp.exp(-mpN*(t**4/4 - mpk * t**2 / 2 + mpk**2/4))
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

    # for i in range(N+1):
    #     for j in range(N+1):
    #         integrand = lambda x : mp.mpf(polys[i](fl.arb(x)) * polys[j](fl.arb(x)) ) * ScaledWeight(x)
    #         res = mp.quad(integrand, np.linspace(a, b, 100))
    #         print('{:+.14f}'.format(float(res)), end = ' ')
    #     print('', end = '\n')

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

    norm = mp.quad(GaussProfile, mp.linspace(a,b, 200))
    fig,ax = plt.subplots(1,1)
    x = mp.linspace(-10, 10,  1001)
    # ax.hist(evals, bins = 100, density = True)
    ax.plot(x, [GaussProfile(z)/norm for z in x], color='red')
    ax.plot(x, [density(z) for z in x], linestyle='dashed', color='green')
    # ax.scatter(evals, [potentialV(x) for x in evals])
    ax.set_title(title)

#====================================MAIN======================================

p = DensityProfile()
Density = lambda x : mp.mpf(p(fl.arb(x))) * ScaledWeight(x)
norm = mp.quad(Density, mp.linspace(a, b, 100))
ScaledDensity = lambda x : Density(x) / norm

evals, _ = Eigens()
evals = EvalsEvolution(evals, evalsSteps)
evals.sort()

EigensPlot(evals, ScaledDensity, 'N={}, k={}'.format(N, k))
plt.show()
        

