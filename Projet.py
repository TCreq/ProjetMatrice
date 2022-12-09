# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:34:27 2022

@author: HP
"""


import numpy as np
import copy
import timeit
import matplotlib.pyplot as plt

### --------------------------------------------

def remontee(T0,b0): #Fonction du TP
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

def descente(T0,b0): #Fonction du TP
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

def LU2(A0): #Fonction du TP
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

### Fonction MatriceA  ------------------------

def MatriceA(n,tau=1):
    A=np.eye(n)
    for i in range(1,n):
        A+=np.diag((n-i)*[1/(1+(i*tau)**2)],i)+np.diag((n-i)*[1/(1+(i*tau)**2)],-i)
    return A

#print(MatriceA(4,1))

### Resolution probleme par LU ----------------

n=20
A=MatriceA(n,1)
pi=np.array(n*[1.]).transpose()
LU=LU2(A)
L=LU[0]
U=LU[1]
y=descente(L,pi)
x=remontee(U,y)
#print(x)
print(np.dot(A,x))
plt.scatter(list(range(1,n+1)),x)
plt.plot(list(range(1,n+1)),np.dot(A,x),color="red")
plt.legend(["Puissance émise","Puissance reçue"],loc="right")

### Vecteur t ----------------------------------

def t(n,tau=1):
    return np.array([1/(1+(i*tau)**2) for i in range(n)])
t1=t(20,1)
#print(t1)

### Etape 1 ------------------------------------

def Etape1(t):
    """
    Version améliorée à la fin, avec explications plus détaillées
    Etape 1 de l'algorithme défini dans le sujet
    """
    listf=len(t)*[None] #définition de la liste contenant les f (allocation espace stockage)
    fk=np.array([1/t[0]]) # f0
    listf[0]=fk # changement des fk
    for k in range(2,len(t)+1):
        delta=sum(t[1:k]*fk)
        beta=1/(1-delta**2)
        alpha=-delta*beta
        fk1=alpha*np.hstack((np.flip(fk,0),np.array([0])))+beta*np.hstack((np.array([0]),fk))
        fk=fk1
        listf[k-1]=fk
    return listf

### Bloc de test de l'étape 1
#l=Etape1(t1)
#print(l)
#print([len(u) for u in l])
#print([MatriceA(k+1).dot(l[k]).astype(int) for k in range(n)])

### Etape 2 -----------------------------------

def Etape2(t,b):
    """
    Etape 2 de l'algorithme défini dans le sujet
    """
    listf=Etape1(t) # récupération de la liste des f de l'étape 1
    x=listf[0]*b[0] # definition du premier x
    for k in range(1,n):
      theta=b[k]-sum(np.flip(t[1:k+1],0)*x)
      x=np.hstack((x,np.array([0])))+theta*listf[k]
    return x

n=20
b=np.array(n*[1.])
#print(b)
#print(b[0:1])
x=Etape2(t1,b)
#print(x)
print(np.dot(A,x))
plt.scatter(list(range(1,n+1)),x)
plt.plot(list(range(1,n+1)),np.dot(A,x),color="red")
plt.legend(["Puissance émise","Puissance reçue"],loc="right")

### Etape 1 et Etape 2 vesion améliorée qui minimise le stockage

def Etape12(t,b):
    """
    les vecteurs utilisés (t,fk,xk) sont en ligne dans le programme,
    mais c'est tout à fait identique à la méthode utilisant des vecteurs colonnes
    """
    fk=np.array([1/t[0]]) #premier fk
    x=fk*b[0] #premier xk
    for k in range(1,len(t)): #on varie entre k=2 et n (sur Python, on utilise les indices de 1 à n-1)
        delta=sum(t[1:k+1]*fk) #definition du delta k
        beta=1/(1-delta**2) #definition du beta k
        alpha=-delta*beta #definition du alpha k
        fk=alpha*np.hstack((np.flip(fk,0),np.array([0])))+beta*np.hstack((np.array([0]),fk)) # calcul du fk
        theta=b[k]-sum(np.flip(t[1:k+1],0)*x) # calcul du theta k
        x=np.hstack((x,np.array([0])))+theta*fk # calcul du xk
    return x
        
x=Etape12(t1,b)
#print(x)
print(np.dot(A,x))
plt.scatter(list(range(1,n+1)),x)
plt.plot(list(range(1,n+1)),np.dot(A,x),color="red")
plt.legend(["Puissance émise","Puissance reçue"],loc="best")

# on obtient les memes résultats que la méthode LU mais Nop inférieur

