#source on Metropolis-Hastings algorithm: https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/

import numpy as np
import matplotlib.pyplot as plt

#===================================PARAMETERS=================================================

large_width = 400
np.set_printoptions(linewidth=large_width)

N = 100
k = 3
order = int(np.sqrt(N) / 2)

#===================================AUXILIARY_FUNCTIONS=====================================

def GenerateMatrix():
    M = np.zeros((N,N), dtype = complex)
    M = np.random.normal(0, 1/2, (N,N)) + 1j * np.random.normal(0, 1/2, (N,N))                     
    M = ( M + M.conj().T ) / 2      
    return M                                                                   

def Potential(M):
    return np.linalg.matrix_power(M, 4)/4 - k * np.linalg.matrix_power(M, 2) / 2


def LnAcceptanceProbabilityFunction(A,B):                                       #acceptance probability is given as p(x_{n+1})/p(x_n)
    tmp = np.real( np.trace( Potential(A) - Potential(B) ) )                    #where x_n is matrix M at n-th step
    return - N * tmp

def PlotMatrix(matrix):
    ret = np.absolute(matrix)
    fig1,ax1 = plt.subplots()
    cax = ax1.matshow(ret)
    fig1.colorbar(cax)

#===================================METROPOLIS_STEP_FUNCTION=====================================

M = GenerateMatrix()
originalM = M.copy()
nextM = np.zeros((N,N), dtype = complex)

acceptanceProbability = 0
terminationCounter = 0
counter = 0

while(1):
    counter += 1
    #print(terminationCounter)
    #print(counter)
    if counter == 20000:
        break

    nextM = M.copy()

    for i in range(order):
        index1, index2 = np.random.randint(0, N, size = 2)
        noise = np.random.normal(0, 1/2) + 1j * np.random.normal(0, 1/2)
        nextM[index1, index2] += noise
        nextM[index2, index1] += np.conjugate(noise)

    lnAcceptanceProbability = LnAcceptanceProbabilityFunction(nextM, M)

    if lnAcceptanceProbability > 0 :        #defined as it is now, lnAcceptanceProbability > 0 means nextM is nearer to the local minimum of potential V than M
        M = nextM
        terminationCounter = 0
    else:
        u = np.random.random()
        if u > 0.5 :
            M = nextM
    terminationCounter += 1


#U diagonalizes M, V diagonalizes nextM
#V in the basis where M is diagonal is of the form U.conj().T * V * U

(evals, U) = np.linalg.eigh(originalM)
(evals, V) = np.linalg.eigh(M)

Udag = np.transpose( np.matrix.conjugate(U) )
rez = np.matmul( np.matmul(Udag, V), U)
rez = np.matmul( np.transpose( np.matrix.conjugate(rez) ), rez )


#######################################PLOTS#########################################

PlotMatrix(rez)
plt.show()
input()