import numpy as np
import copy
import timeit
import matplotlib.pyplot as plt

##########################################  Question 3  ##########################################
###################################### Fait par Tudal CREQUY #####################################
def MatriceA(n,tau):
    A=np.eye(n)
    for i in range(1,n):
        A+=np.diag((n-i)*[1/(1+(i*tau)**2)],i)+np.diag((n-i)*[1/(1+(i*tau)**2)],-i)
    return A
  
def remontee(T0,b0):
    """
    donne la solution du système triangulaire avec T0 matrice triangulaire supérieure
    et b0 vecteur colonne, fonctionne par remontée
    """
    T=copy.deepcopy(T0) #crée un deepcopy de la matrice argument
    T=T.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(T)[0] #enregistre la taille du système
    b=copy.deepcopy(b0) #crée un deepcopy du vecteur argument
    b=b.astype(float) #convertit les éléments en float pour éviter les arrondis
    for i in range(n-1,-1,-1):
        if i<n-1: #quand on est sur la derniere ligne, on ne fait pas de soustraction
            b[i]-=float(sum(T[i,(i+1):n]*b[(i+1):n]))
        b[i]=b[i]/T[i,i]
    return b
  
def descente(T0,b0):
    """
    donne la solution du système triangulaire avec T0 matrice triangulaire inférieure
    et b0 vecteur colonne, fonctionne par descente
    """
    T=copy.deepcopy(T0) #crée un deepcopy de la matrice argument
    T=T.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(T)[0] #enregistre la taille du système
    b=copy.deepcopy(b0) #crée un deepcopy du vecteur argument
    b=b.astype(float) #convertit les éléments en float pour éviter les arrondis
    for i in range(0,n,1):
        if i>0: #quand on est sur la premiere ligne, on ne fait pas de soustraction
            b[i]-=float(sum(T[i,0:i]*b[0:i]))
        b[i]=b[i]/T[i,i]
    return b
  
def LU(A0):
    A=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    A=A.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(A)[0] #enregistre la taille de la matrice
    L=np.eye(n)
    U=np.eye(n)
    for j in range(n):
        for k in range(j,n):
            U[j,k]=A[j,k]-sum(L[j,:j]*U[:j,k].transpose())
        for k in range(j+1,n):
            L[k,j]=(1/U[j,j])*(A[k,j]-sum(L[k,:j]*U[:j,j].transpose()))
    return [L,U]
##########################################  Question 5  ##########################################

panne=[7,8,15,16,17,18]
n=20
P=np.eye(n)
for i in panne:
  P[i-1][i-1]=0

A=MatriceA(n,1)
pi=np.array(n*[1.]).transpose()

A_new=np.dot(P,A)

for i in panne:
  A_new[i-1]=[0.]*n
  A_new[i-1][i-1]=1

pi_new=np.dot(P,pi)

LU=LU2(A_new)
L=LU[0]
U=LU[1]
y=descente(L,pi_new)
x=remontee(U,y)
print(x)
print(np.dot(A_new,x))
plt.bar(range(1,n+1),x,width = 0.6,label="x : la puissance émise par l'antenne")
yverif=np.dot(A,x)
plt.bar(range(1,n+1),yverif,width = 0.1,color='red',label=f"$\pi = Ax :$ la puissance reçue au pied de l'antenne")
plt.scatter(range(1,n+1),pi_new,color='green',label=f"$P\pi = PAPx :$ la puissance qu'on veut recevoir au pied de l'antenne")
plt.xlabel('Les antennes')
plt.xticks(range(21))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='lower left', mode="expand", borderaxespad=0.)

##########################################  Question 6  ##########################################

panne=[7,8,15,16,17,18]
n=20
P=np.eye(n)
for i in panne:
  P[i-1][i-1]=0

A=MatriceA(n,1)
pi=np.array(n*[1.]).transpose()

A_new=np.dot(A,P)
G=np.dot(A_new.T,A_new)

for i in panne:
  G[i-1]=[0.]*n
  G[i-1][i-1]=1

pi_new=np.dot(P,np.dot(A_new.T,pi))

LU=LU2(G)
L=LU[0]
U=LU[1]
y=descente(L,pi_new)
x=remontee(U,y)
print(x)
plt.bar(range(1,n+1),x,width = 0.6,label="x : la puissance émise par l'antenne")
yverif=np.dot(A,x)
plt.bar(range(1,n+1),yverif,width = 0.1,color='red',label=f"$\pi = Ax :$ la puissance reçue au pied de l'antenne")
plt.xlabel('Les antennes')
plt.xticks(range(1,n+1))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='lower left', mode="expand", borderaxespad=0.)
print(np.mean(yverif))

##########################################  Question 7_1  ##########################################

import scipy.linalg as lg
from numpy.linalg import norm

def Grad(M,P,b):
  x0=np.array([0.]*len(P)).T
  ro_0=np.dot(P,b)-np.dot(np.dot(P,M),np.dot(P,x0))
  g0=np.copy(ro_0)

  PMP=np.dot(np.dot(P,M),P)

  ro,g,x,k=ro_0,g0,x0,0
  
  while norm(ro,2)!=0:
    alpha=np.dot(ro,ro)/np.dot(np.dot(PMP,g),g)
    x+=alpha*g
    ro_old=np.copy(ro)
    
    ro-=alpha*np.dot(PMP,g)
    beta=np.dot(ro,ro)/np.dot(ro_old,ro_old)

    g=ro+beta*g
    k+=1
  return x

A=MatriceA(n,1)
M=A

panne=[7,8,15,16,17,18]
n=20
P=np.eye(n)
for i in panne:
  P[i-1][i-1]=0

pi=np.array(n*[1.]).transpose()
pi_new=np.dot(P,pi)

x=Grad(M,P,pi_new)
plt.bar(range(1,n+1),x,width = 0.6,label="x : la puissance émise par l'antenne")
yverif=np.dot(A,x)
plt.bar(range(1,n+1),yverif,width = 0.1,color='red',label=f"$\pi = Ax :$ la puissance reçue au pied de l'antenne")
plt.scatter(range(1,n+1),pi_new,color='green',label=f"$P\pi = PAPx :$ la puissance qu'on veut recevoir au pied de l'antenne")
plt.xlabel('Les antennes')
plt.xticks(range(21))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='lower left', mode="expand", borderaxespad=0.)

##########################################  Question 7_2  ##########################################

panne=[7,8,15,16,17,18]
n=20
P=np.eye(n)
for i in panne:
  P[i-1][i-1]=0

M=np.dot(A.T,A)

pi=np.array(n*[1.]).transpose()
pi_new=np.dot(P,np.dot(A.T,pi))

x=Grad(M,P,pi_new)
plt.bar(range(1,n+1),x,width = 0.6,label="x : la puissance émise par l'antenne")
yverif=np.dot(A,x)
plt.bar(range(1,n+1),yverif,width = 0.1,color='red',label=f"$\pi = Ax :$ la puissance reçue au pied de l'antenne")
plt.xlabel('Les antennes')
plt.xticks(range(1,n+1))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='lower left', mode="expand", borderaxespad=0.)
