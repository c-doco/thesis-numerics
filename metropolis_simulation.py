#source on Metropolis-Hastings algorithm: https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/

import numpy as np
import scipy
import matplotlib.pyplot as plt

#=================================PARAMETERS&GLOBAL_VARIABLES=================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 100
k = 2.2
order = 1

#===================================AUXILIARY_FUNCTIONS=====================================

def potentialV(x):
    return x**4/4 - k*x**2/2

def EquilibriumFunction(x):
    ret = np.zeros(N, dtype = float)
    for i in range(N):
        sum = 0
        for j in range(N):
            if i != j:
                sum += 1/np.abs(x[i] - x[j])
        ret[i] = -N * (x[i]**3 - k * x[i]) + sum
    return ret

def EquilibriumFunctionJacobian(x):
    ret = np.zeros((N,N), dtype = float)
    for i in range(N):
        sum = 0
        for j in range(N):
            if i > j:
                ret[i][j] = -1/(x[i] - x[j])**2
                sum += 1/(x[i] - x[j])**2
            if i < j:
                ret[i][j] = 1/(x[i] - x[j])**2
                sum += -1/(x[i] - x[j])**2
        ret[i][i] = -N * (3 * x[i]**2 - k) + sum
    return ret

def GenerateMatrix():
    initialGuess = np.zeros(N, dtype = float)
    for i in range(N):
        if i < N/2:
            initialGuess[i] = np.random.normal(-np.sqrt(k), 1)
        else:
            initialGuess[i] = np.random.normal(np.sqrt(k), 1)

    sol  = scipy.optimize.root(EquilibriumFunction, initialGuess, jac=EquilibriumFunctionJacobian)     
    diag = sol.x
    diag.sort()
    M = np.zeros((N,N), dtype=complex)
    for i in range(N):
        M[i][i] = diag[i]
    return M                                                            

def Potential(M):
    return np.linalg.matrix_power(M, 4)/4 - k * np.linalg.matrix_power(M, 2) / 2

# acceptance probability is given as p(x_{n+1})/p(x_n)
# where x_n is matrix M at n-th step
def LnAcceptanceProbabilityFunction(A,B):                                                           
    tmp = np.real( np.trace( Potential(A) - Potential(B) ) )                                        
    return - N * tmp

def PlotMatrix(matrix, title):
    ret = np.absolute(matrix)
    fig1,ax1 = plt.subplots()
    cax = ax1.matshow(ret)
    ax1.set_title(title)
    fig1.colorbar(cax)

#===================================METROPOLIS_STEP_FUNCTION=====================================

M = GenerateMatrix()
originalM = M.copy()
nextM = np.zeros((N,N), dtype = complex)

acceptanceProbability = 0
terminationCounter = 0
counter = 0

while(1):
# TODO: find better way to determine when minimum is found
    if terminationCounter == 1000000:                                                                            
        break
    nextM = M.copy()
    print(terminationCounter)

    for i in range(order):
        index1, index2 = np.random.randint(0, N, size = 2)
        noise = np.random.normal(0, 1/2) + 1j * np.random.normal(0, 1/2)
        nextM[index1, index2] += noise
        nextM[index2, index1] += np.conjugate(noise)

    lnAcceptanceProbability = LnAcceptanceProbabilityFunction(nextM, M)

    u = np.random.random()
    if u < min(1, np.exp(lnAcceptanceProbability)) :
        M = nextM.copy()
        counter += 1
        #print(terminationCounter, counter)
    terminationCounter += 1

# V diagonalizes nextM, U diagonalizes M
(evals, U) = np.linalg.eigh(originalM)
(evals, V) = np.linalg.eigh(M)                                                                      

# V in the basis where M is diagonal is of the form U.conj().T * V * U
Udag = np.transpose( np.matrix.conjugate(U) )                                                       
rez = np.matmul( np.matmul(Udag, V), U)


#######################################PLOTS#########################################

#PlotMatrix(originalM, 'Original matrix')
PlotMatrix(rez, 'Transformation matrix')
PlotMatrix(M, 'Final matrix')
evals, _ = np.linalg.eigh(M)
xlinsp = np.linspace(-1.3*min(evals), 1.3*max(evals))
figh, axh = plt.subplots()
axh.plot(xlinsp, [potentialV(t) for t in xlinsp])
axh.scatter(evals, [potentialV(t) for t in evals], color='red')
plt.show()