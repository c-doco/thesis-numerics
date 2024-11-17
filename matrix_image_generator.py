import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.stats as st
from math import floor
from bisect import bisect
from scipy.optimize import minimize
from mpmath import mp
import flint as fl
from scipy.interpolate import interp1d
from functools import partial

np.set_printoptions(linewidth = 500)
mp.dps = 300
fl.ctx.dps = 300
mp.pretty = True
fl.ctx.pretty = True


#=================================PARAMETERS&GLOBAL_VARIABLES=================================

N = 100
mpN = mp.mpf(N)
p = 3

pertBase = 1#int(np.sqrt(N))
refSteps = int(600000 * (N/100)**3)
steps = int(1000000 * (N/100)**3)
# refSteps = 1
# steps = 1

a = -20
b = 20

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

#===================================AUXILIARY_FUNCTIONS=====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def ScaledWeight(x, mpk):
    return mp.exp(-mpN*(x**4/4 - mpk * x**2/2 + mpk**2/4))

def PertBase(t, p, steps):
    ret = 1 + (pertBase - 1) * (1 - t/steps)**p
    return floor(int(ret))

def timeTaper(t, steps):
    return np.sqrt(t / steps)

def GenerateMatrix():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2
    return M                                                            

def Potential(M, k):
    return np.linalg.matrix_power(M, 4)/4 - k * np.linalg.matrix_power(M, 2) / 2

def LnAcceptanceProbabilityFunction(A, k):                                                           
    tmp = np.real( np.trace(Potential(A, k)) )
    return - N * tmp

def GaussProfile(x, evals, k):
    sigma = 0.3/N
    t = mp.sqrt(4*N)
    ret = 0
    for z in evals / t:
        ret += mp.exp(-(x - z)**2 / (2*sigma**2)) / mp.sqrt(2 * sigma**2 * mp.pi)
    return ret / N

def EigensPlot(evalsList, evecsList,  densityList, krange):
    t = mp.sqrt(4*N)
    x = mp.linspace(-1.25, 1.25, 10001)
    plotMax = 1.25

    fig,ax = plt.subplots(len(krange),2)
    if len(krange) == 1:
        mpk = mp.mpf(krange[0])
        cax = ax[0].matshow(np.abs(evecsList[0]), label='Absolute Value Eigenvector Matrix', cmap='magma')
        fig.colorbar(cax, ax=ax[0])
        ax[0].title.set_text('Absolute value of Eigenvector matrix \n\n k={}'.format(krange[0]))
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top=False, labeltop=False)
        KSFit(evalsList[0], densityList[0], t, mpk)
        ax[1].plot(x, [GaussProfile(z, evalsList[0], krange[0]) for z in x], color='red', label='Numerical Density')
        ax[1].plot(x, [densityList[0](q, mpk) for q in x], color='blue', label='Marginal Density', linestyle='-')
        ax[1].set_ylim([0,15])
        ax[1].set_xlim([-0.3,0.3])
        ax[1].set_ylabel('Probability Density')
        ax[1].set_xlabel('x')
        ax[1].legend(loc=1, prop={'size': 10})
        ax[1].title.set_text('Eigenvalue distribution \n\n k={}'.format(krange[0]))

    else:
        for i in range(len(krange)):
            cax = ax[i][0].matshow(np.abs(evecsList[i]), label='Absolute Value Eigenvector Matrix', cmap='magma', vmin=0, vmax=0.5)
            fig.colorbar(cax, ax=ax[i][0])
            if i == 0:
                ax[i][0].title.set_text('Absolute value of Eigenvector matrix \n\n k={}'.format(krange[i]))
            else:
                ax[i][0].title.set_text('k={}'.format(krange[i]))
    
            ax[i][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top=False, labeltop=False)

        for j in range(len(krange)):
            mpk = mp.mpf(krange[j])
            KSFit(evalsList[j],densityList[j], t, mpk)
            ax[j][1].plot(x, [GaussProfile(z, evalsList[j], krange[j]) for z in x], color='red', label='Numerical Density')
            ax[j][1].plot(x, [densityList[j](q, mpk) for q in x], color='blue', label='Marginal Density', linestyle='-')
            ax[j][1].set_ylim([0,15])
            ax[j][1].set_xlim([-0.3,0.3])
            ax[j][1].set_ylabel('Probability Density')
            ax[j][1].set_xlabel('x')
            ax[j][1].legend(loc=1, prop={'size': 10})
            if j == 0:
                ax[j][1].title.set_text('Eigenvalue distribution \n\n k={}'.format(krange[j]))
            else:
                ax[j][1].title.set_text('k={}'.format(krange[j]))


def Transformation(A,B):
    _, U = np.linalg.eigh(A)

    Udag = np.transpose( U.conj() )                                                       
    M = np.matmul( np.matmul(Udag, B), U)

    _, V = np.linalg.eigh(M)

    return V

def negLogProb(N, data):
    n = len(data)
    if N <= 0:  # To ensure N remains positive during optimization
        return np.inf
    return -n * np.log(N) + N * np.sum(data)


def PorterThomasFit(evals, evecs):
    n = bisect(evals, 0)
    diag1 = np.square( np.abs(evecs[0:n, 0:n]) ).ravel()
    diag2 = np.square( np.abs(evecs[n:N, n:N]) ).ravel()
    offur = np.square( np.abs(evecs[0:n, n:N]) ).ravel()
    offdl = np.square( np.abs(evecs[n:N, 0:n]) ).ravel()

    diag = np.concatenate((diag1, diag2))
    offd = np.concatenate((offur, offdl))

    initialGuess = N/2
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

def PDF_to_CDF(pdf, mpk, x_range=(-1,1), num_points=1001):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    cdf_values = np.zeros(num_points)
    for i in range(num_points):
        cdf_values[i] = cdf_values[i-1] + float(pdf(mp.mpf(x_values[i]), mpk))
    norm = cdf_values[num_points-1]
    cdf_values = cdf_values/norm
    
    cdf_function = interp1d(x_values, cdf_values, bounds_error=False, fill_value=(0, 1))
    return cdf_function    

def KSFit(evals, density, t, mpk):
    cdf = PDF_to_CDF(density, mpk)
    evals = evals / float(t)
    ret = st.kstest(evals, cdf)
    print(ret)
    return 0

def DensityCreator(p, mpk):
    t = mp.sqrt(4*N)
    Density = lambda x, mpk : mp.mpf(p(fl.arb(x))) * ScaledWeight(x, mpk)
    mpk = mpk
    norm = mp.quad(lambda x: Density(x, mpk), mp.linspace(a, b, 40))
    ScaledDensity = lambda x, mpk : t * Density(t * x, mpk) / norm
    return ScaledDensity
#===================================METROPOLIS_EVOLUTION===============================================

def Metropolis(S, step, k):
    acceptanceProbability = 0
    count = 0
    acceptCount = 0
    
    origVec = []
    for i in range(pertBase):
        origVec.append((0,0,0))
        
    while(count < step):
    # TODO: find better way to determine when minimum is found
        pert = pertBase #PertBase(count, p, step)

        # if count % 10000 == 0:
        #     print(count)

        lnAccProb = -LnAcceptanceProbabilityFunction(S, k)

        for i in range(pert):
            index1, index2 = np.random.randint(0, N, size = 2)
            noise = np.random.normal(0, 1/2) + 1j * np.random.normal(0, 1/2)
            origVec[i] = (index1, index2, S[index1, index2])
            S[index1, index2] += noise
            S[index2, index1] += np.conjugate(noise)

        lnAccProb += LnAcceptanceProbabilityFunction(S, k)

        # u = min(np.random.random() +  timeTaper(count, step), 1)
        u = np.random.random()
        if lnAccProb > 700:
            acceptCount += pert
        elif lnAccProb < -700:
            for i in range(pert):
                ind1, ind2, val = origVec[i]
                S[ind1, ind2] = val
                S[ind2, ind1] = np.conjugate(val)
        elif u < np.exp(lnAccProb) :
            acceptCount += pert
        else:
            for i in range(pert):
                ind1, ind2, val = origVec[i]
                S[ind1, ind2] = val
                S[ind2, ind1] = np.conjugate(val)
                
        count += 1
    print('Accepted Steps:',acceptCount)

    return S

#============================================MAIN======================================================
krange= [0,2,4,8]
evalsList = []
evecsList = []
densityList = []
x = mp.linspace(-1.25, 1.25, 10001)
t = mp.sqrt(4*N)

for i in krange:
    k = i
    mpk = mp.mpf(i)
    p = DensityProfile(mpk)
    ScaledDensity = DensityCreator(p,mpk)
    densityList.append(ScaledDensity)
    
    M = GenerateMatrix()
    Metropolis(M, refSteps, k)
    F = M.copy()
    Metropolis(F, steps, k)
    Q = Transformation(M, F)
    evals, _ = np.linalg.eigh(F)
    evalsList.append(evals)
    evecsList.append(Q)

EigensPlot(evalsList, evecsList, densityList, krange)
plt.show()
