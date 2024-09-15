#source on Metropolis-Hastings algorithm: https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import quad

#=================================PARAMETERS&GLOBAL_VARIABLES=================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 100
k = 16
pertBase = 1
refSteps = 500000
steps = 1000000

#===================================AUXILIARY_FUNCTIONS=====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def GenerateMatrix():
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                          
    M = ( M + M.conj().T ) / 2
    return M                                                            

def Potential(M):
    return np.linalg.matrix_power(M, 4)/4 - k * np.linalg.matrix_power(M, 2) / 2

def LnAcceptanceProbabilityFunction(A):                                                           
    tmp = np.real( np.trace(Potential(A)) )
    return - N * tmp

def PlotMatrix(matrix, title, cond):
    ret = np.absolute(matrix)
    if cond == 1:
        fig,ax = plt.subplots(1,2)
        cax = ax[0].matshow(ret)
        ax[0].set_title(title)
        fig.colorbar(cax)
        
        evals, _ = np.linalg.eigh(matrix)
        xlinsp = np.linspace(-1.3*min(evals), 1.3*max(evals))
        ax[1].scatter(evals, [potentialV(t) for t in evals], color='red')
    else:
        fig,ax = plt.subplots()
        cax = ax.matshow(ret)
        ax.set_title(title)
        fig.colorbar(cax)

def Transformation(A,B):
    (evals, U) = np.linalg.eigh(A)

    Udag = np.transpose( np.matrix.conjugate(U) )                                                       
    M = np.matmul( np.matmul(Udag, B), U)

    (evals, V) = np.linalg.eigh(M)

    return V
        

#===================================METROPOLIS_EVOLUTION===============================================

def Metropolis(S, step):
    acceptanceProbability = 0
    count = 0
    acceptCount = 0
    
    origVec = []
    for i in range(pertBase):
        origVec.append((0,0,0))
        
    while(count < step):
    # TODO: find better way to determine when minimum is found
        pert = pertBase

        if count % 1000 == 0:
            print(count)

        lnAcceptanceProbability = -LnAcceptanceProbabilityFunction(S)

        for i in range(pert):
            index1, index2 = np.random.randint(0, N, size = 2)
            noise = np.random.normal(0, 1/2) + 1j * np.random.normal(0, 1/2)
            origVec[i] = (index1, index2, S[index1, index2])
            S[index1, index2] += noise
            S[index2, index1] += np.conjugate(noise)

        lnAcceptanceProbability += LnAcceptanceProbabilityFunction(S)

        u = np.random.random()
        if u < min(1, np.exp(lnAcceptanceProbability)) :
            acceptCount += 1
        else:
            for i in range(pert):
                ind1, ind2, val = origVec[i]
                S[ind1, ind2] = val
                S[ind2, ind1] = np.conjugate(val)
                
        count += 1
    print(acceptCount)

    return S

#============================================MAIN======================================================

M = GenerateMatrix().copy()
PlotMatrix(M, 'M', 1)

Q = Metropolis(M, refSteps).copy()
PlotMatrix(Q, 'Q', 1)

# L, _ = np.linalg.eigh(Q)
# K = np.diag(L)
# F = Metropolis(K, steps).copy()

F = Metropolis(Q, steps).copy()
PlotMatrix(F, 'F', 1)

PlotMatrix(Transformation(K, F), 'Q F, N={}, k={}'.format(N,k), 0)
plt.show()
