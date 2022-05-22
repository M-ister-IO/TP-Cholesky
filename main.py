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


A=[[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]]
B=[[1],[2],[3],[4]]


print(ResolCholeskyAlternative(A,B))
print(ResolCholesky(A,B))


"""
exo 2 question 1:
A=LDLT
A(LDLT)^-1=LDLT(LDLT)^-1
A(LDLT)^-1=LDLT(LT^-1)(D^-1)(L^-1)
A(LDLT)^-1=LDI(D^-1)(L^-1)
A(LDLT)^-1=LI(L^-1)
A(LDLT)^-1=I
so (LDLT)^-1 is A^-1, LT being the transpose of L and thus A is inversible and symetric


exo 2 question 3:
As we can see in the code, L[0][0] is the square root of A[0][0] so if A[0][0] is negative the cholesky method doesn't work
so A=[-1 2 3]
     [ 2 1 2]
     [ 3 2 1] doesn't work
"""
