import numpy as np
from math import sqrt



def Cholesky(A):
    n=len(A)
    L=np.zeros((n,n))
    for j in range(len(A)):

        for i in range(len(A[j])):
            if i==0:
                if j==0:
                    L[j][i]=sqrt(A[j][i])
                else:
                    L[j][i]=A[0][j]/L[0][0]
            elif j==i:
                sum=0
                for k in range(0,i):
                    sum+=L[i][k]**2
                L[i][i]=sqrt(A[i][i]-sum)
            elif i+1<=j<=n:
                sum=0
                for k in range(0,i):
                    sum+=L[i][k]*L[j][k]
                L[j][i]=(A[i][j]-sum)/L[i][i]
    return L


def ResolCholesky(A,B):
    n=len(A)
    L=Cholesky(A)
    LT=L.T
    Y=np.zeros((n,1))
    X=np.zeros((n,1))
    for k in range(len(Y)):
        if k==0:
            Y[k]=B[k]/L[k][k]
        else:
            sum=0
            for i in range(k):
                sum+=L[k][i]*Y[i]
            Y[k]=(B[k]-sum)/L[k][k]

    for k in range(-1,-len(X)-1,-1):

        if k==-1:
            X[k]=Y[k]/LT[k][k]

        else:
            sum=0
            for i in range(-1,k-1,-1):
                sum+=LT[k][i]*X[i]
            X[k]=(Y[k]-sum)/LT[k][k]

    return X




def CholeskyAlternative(A):
    n = len(A)
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            sum_2 = 0
            for j in range(k):
                sum_2 += D[j][j] * L[k][j] ** 2
            D[k][k] = A[k][k] - sum_2
            if i>k:
                sum=0

                for j in range(k):
                    sum+=L[i][j]*L[k][j]*D[j][j]
                L[i][k]=(1/D[k][k])*(A[i][k]-sum)
            elif i==k:
                L[k][k]=1
    return L,D

def ResolCholeskyAlternative(A,B):
    n = len(A)
    L,D = CholeskyAlternative(A)
    LT = L.T
    Y = np.zeros((n, 1))
    X = np.zeros((n, 1))
    DLT = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            DLT[i][j]=D[i][i]*LT[i][j]

    for k in range(len(Y)):
        if k == 0:
            Y[k] = B[k] / L[k][k]
        else:
            sum = 0
            for i in range(k):
                sum += L[k][i] * Y[i]
            Y[k] = (B[k] - sum) / L[k][k]

    for k in range(-1, -len(X) - 1, -1):

        if k == -1:
            X[k] = Y[k] / DLT[k][k]

        else:
            sum = 0
            for i in range(-1, k - 1, -1):
                sum += DLT[k][i] * X[i]
            X[k] = (Y[k] - sum) / DLT[k][k]

    return X


import time

A=np.random.randint(5,size=(50,50))
B=np.random.randint(5,size=(50,1))

# the following two functions are from the internet to create a positive definite matrix A
from numpy import linalg as la
import numpy as np


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


A=nearestPD(A)



#this is mine again:
A = (A + A.T)/2  #make it symetric


start=[]
end=[]

start.append(time.time())
ResolCholesky(A,B)
end.append(time.time())


start.append(time.time())
ResolCholeskyAlternative(A,B)
end.append(time.time())


print("ResolCholesky took: ", end[0] - start[0],"seconds")

print("ResolCholeskyAlternative took: ", end[1] - start[1],"seconds")



"""
A=[[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]]
B=[[1],[2],[3],[4]]


print(ResolCholeskyAlternative(A,B))
print(ResolCholesky(A,B))



exo 2 question 1:
A=LDLT
A(LDLT)^-1=LDLT(LDLT)^-1
A(LDLT)^-1=LDLT(LT^-1)(D^-1)(L^-1)
A(LDLT)^-1=LDI(D^-1)(L^-1)
A(LDLT)^-1=LI(L^-1)
A(LDLT)^-1=I
so (LDLT)^-1 is A^-1, LT being the transpose of L and thus A is inversible and symetric


exo 2 question 3:
As we can see in the code, L[0][0] is the square root of A[0][0] so if A[0][0] is negative the cholesky method doesn't work, also a matrix that is not non negative will not work for this reason

exo 3
as we can see both are very fast, but the normal cholesky is a bit faster, but the cholesky alternative takes a bit more time to create the LDLT matrices, and they are simpler to look at
so (the time we calculated also takes into account the formation of the LDLT and LLT matrices)

"""
