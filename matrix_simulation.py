import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import floor
from bisect import bisect
from scipy.optimize import minimize


#=================================PARAMETERS&GLOBAL_VARIABLES=================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 75
kmin = 0
kmax = 6
p = 3
kStep = ((kmax - kmin) * 4 + 1)
pertBase = int(np.sqrt(N))
refSteps = 600000 * (N / 100)**3
steps = 1000000 * (N/100)**3

# stavi PERTBASE = 1

#===================================AUXILIARY_FUNCTIONS=====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def PertBase(t, p, steps):
    ret = 1 + (pertBase - 1) * (1 - t/steps)**p
    return floor(int(ret))

def GenerateMatrix():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2
    return M                                                            

def Potential(M, k):
    return np.linalg.matrix_power(M, 4)/4 - k * np.linalg.matrix_power(M, 2) / 2

def LnAcceptanceProbabilityFunction(A, k):                                                           
    tmp = np.real( np.trace(Potential(A, k)) )
    return - N * tmp

def PlotMatrix(matrix, title, cond):
    ret = np.absolute(matrix)
    if cond == 1:
        fig,ax = plt.subplots(1,2)
        cax = ax[0].matshow(ret)
        ax[0].set_title(title)
        fig.colorbar(cax)

        evals, _ = np.linalg.eigh(matrix)
        bound = max(abs(evals))
        xlinsp = np.linspace(-1.2 * bound, 1.2 * bound, 10001)
        ax[1].scatter(evals, [potentialV(t) for t in evals], color='red')
        ax[1].plot(xlinsp, potentialV(xlinsp))
    else:
        fig,ax = plt.subplots()
        cax = ax.matshow(ret)
        ax.set_title(title)
        fig.colorbar(cax)

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
        pert = 1#PertBase(count, p, step)

        # if count % 10000 == 0:
        #     print(round(count/step,2)*100, '%')

        lnAccProb = -LnAcceptanceProbabilityFunction(S, k)

        for i in range(pert):
            index1, index2 = np.random.randint(0, N, size = 2)
            noise = np.random.normal(0, 1/2) + 1j * np.random.normal(0, 1/2)
            origVec[i] = (index1, index2, S[index1, index2])
            S[index1, index2] += noise
            S[index2, index1] += np.conjugate(noise)

        lnAccProb += LnAcceptanceProbabilityFunction(S, k)

        u = np.random.random()
        if lnAccProb > 700:
            acceptCount += pert
        elif lnAccProb < -700:
            for i in range(pert):
                ind1, ind2, val = origVec[i]
                S[ind1, ind2] = val
                S[ind2, ind1] = np.conjugate(val)
        elif u < min(1, np.exp(lnAccProb)) :
            acceptCount += pert
        else:
            for i in range(pert):
                ind1, ind2, val = origVec[i]
                S[ind1, ind2] = val
                S[ind2, ind1] = np.conjugate(val)
                
        count += 1
    # print('Accepted Steps:',acceptCount)

    return S

#============================================MAIN======================================================

krange = np.linspace(kmin, kmax, kStep)
print(krange)

for t in krange:
    k = t
    
    M = GenerateMatrix()
    Metropolis(M, refSteps, k)
    F = M.copy()
    Metropolis(F, steps, k)

    PlotMatrix(Transformation(M, F), 'Q F, N={}, k={}'.format(N,k), 0)
    Q = Transformation(M, F)
    evals, evecs = np.linalg.eigh(Q)

    print(k,end=':')
    PorterThomasFit(evals, Q)
plt.show()
