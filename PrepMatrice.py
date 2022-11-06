# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""




import numpy as np
import copy
import timeit


### Fonctions Utiles support pour les generations de matrices ou les trigonalisations ou les vérifications

def matAleaN(N,e=7):
    """
    génère une matrice carrée aléatoire NxN avec des entiers entre 1 et e
    """
    l=(N**2)*[1]
    for i in range(N**2):
        l[i]=float(int(np.random.rand()*e))    
    a=np.array(l).reshape(N,N)
    return a



def Trig(A0):
    """
    methode 1 renvoie une matrice triangle avec les 0 sur la partie inf (utile pour verifications avec produit matriciel)
    """
    A=copy.deepcopy(A0)
    A=A.astype(float)
    #B=copy.deepcopy(A0)
    #B=B.astype(float)
    n=np.shape(A)[0]
    for k in range(n):
        #print(f"k={k}")
        ip=0
        while A[k,k]==0:
            ip+=1
            if k+ip<n:
                #print(f"A=\n{A}")
                #print(f"permutation entre ligne {k} et ligne {k+ip} à la {k+1}-ieme etape")
                A[k,:],A[k+ip,:]=copy.deepcopy(A[k+ip,:]),copy.deepcopy(A[k,:])
                #print(f"A=\n{A}")
            else:
                raise Exception("matrice non trigonalisable")
                break
        for i in range(k+1,n):
            mik=-(A[i,k]/A[k,k])
            #print(f"Coefficient {mik}")
            #B[i,k]=mik
            #print(f"Operation {A[i,:]}+{mik}*{A[k,:]}")
            A[i,:]=A[i,:]+mik*A[k,:]
            #print(A[i,:])
            #B[i,(k+1):n]=B[i,(k+1):n]+B[i,k]*B[k,(k+1):n]
    return A

def Trig2(A0):
    """
    methode 2 renvoie une matrice triangle avec les coefs à la place des 0
    """
    #A=copy.deepcopy(A0)
    #A=A.astype(float)
    B=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    B=B.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(B)[0] #enregistre la taille de la matrice
    for k in range(n):
        #print(f"k={k}")
        ip=0 #preparation indice du while
        while B[k,k]==0: #ajout pour éviter les matrices avec pivot nul, à améliorer
            ip+=1
            if k+ip<n:
                #print(f"B=\n{B}")
                #print(f"permutation entre ligne {k} et ligne {k+ip} à la {k+1}-ieme etape")
                B[k,:],B[k+ip,:]=copy.deepcopy(B[k+ip,:]),copy.deepcopy(B[k,:]) #inverse les lignes
                #print(f"B=\n{B}")
            else:
                raise Exception("matrice non trigonalisable") #quand on atteind la fin de la matrice sans trouver de pivot non nul, pas trigonalisable
                break
        for i in range(k+1,n):
            mik=-(B[i,k]/B[k,k])
            #print(f"Coefficient {mik}")
            B[i,k]=mik #stockage du pivot à la place du 0
            #print(f"Operation {A[i,:]}+{mik}*{A[k,:]}")
            #A[i,:]=A[i,:]+mik*A[k,:]
            #print(A[i,:])
            #print(f"Operation {B[i,:]}+{mik}*{B[k,:]}")
            B[i,(k+1):n]=B[i,(k+1):n]+B[i,k]*B[k,(k+1):n]
            #print(B[i,:])
    return B


'''
def inversion(T0):
    """
    fonction inutile en fait
    """
    Tsup=copy.deepcopy(T0)
    Tsup=Tsup.astype(float)
    n=np.shape(Tsup)[0]
    m=np.shape(Tsup)[1]
    Linf=n*[m*[1]]
    Tinf=np.array(Linf)
    for k in range(n):
        Tinf[k,:]=Tsup[n-1-k,:]
    Tinf=Tinf.astype(float)
    return Tinf
'''


### Exercice 1

print("\n\n")
print("Exercice 1:")
print("\n\n")

## Question 1

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
    #print(f"T=\n{T}")
    #print(f"b=\n{b}")
    for i in range(n-1,-1,-1):
        #print(f"\n \nEtape i={i+1}")
        if i<n-1: #quand on est sur la derniere ligne, on ne fait pas de soustraction
            #print("T ligne ",T[i,(i+1):n])
            #print("b colonne ",b[(i+1):n,0])
            #print(f"Operation b[{i+1}]= {b[i,0]} + {T[i,(i+1):n]}*{b[(i+1):n,0]}")
            b[i]-=float(sum(T[i,(i+1):n]*b[(i+1):n,0]))
        #print(f"Operation b[{i+1}]= {b[i,0]}/{T[i,i]}")
        b[i,0]=b[i,0]/T[i,i]
    return b

## Question 2

def descente(T0,b0): #OK
    """
    donne la solution du système triangulaire avec T0 matrice triangulaire inférieure
    et b0 vecteur colonne, fonctionne par descente
    """
    T=copy.deepcopy(T0) #crée un deepcopy de la matrice argument
    T=T.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(T)[0] #enregistre la taille du système
    b=copy.deepcopy(b0) #crée un deepcopy du vecteur argument
    b=b.astype(float) #convertit les éléments en float pour éviter les arrondis
    #print(f"T=\n{T}")
    #print(f"b=\n{b}")
    for i in range(0,n,1):
        #print(f"\n \nEtape i={i+1}")
        if i>0: #quand on est sur la premiere ligne, on ne fait pas de soustraction
            #print("T ligne ",T[i,(i+1):n])
            #print("b colonne ",b[(i+1):n,0])
            #print(f"Operation b[{i+1}]= {b[i,0]} + {T[i,(i+1):n]}*{b[(i+1):n,0]}")
            b[i]-=float(sum(T[i,0:i]*b[0:i,0]))
        #print(f"Operation b[{i+1}]= {b[i,0]}/{T[i,i]}")
        b[i,0]=b[i,0]/T[i,i]
    return b



### Tests avec matrices aléatoires

a1=matAleaN(5)  #matrice carree 5x5 aléatoire d'entiers
b1=np.array([5*[1]]) #vecteur ligne de taille 5 avec des 1
b1=b1.reshape(5,1) #vecteur colonne de taille 5
#print(f"a1=\n{a1}")
#print(f"b1=\n{b1}")
#a1[0,0]
#np.shape(a1)[0]
#print(np.linalg.eig(a1))
#print("methode1")
t1=Trig(a1) #methode 1 renvoie une matrice triangle sans les coefficients
#print(f"t1=\n{t1}")
#print("methode 2")
t12=Trig2(a1) #methode 2 renvoie une matrice triangle avec les coefs sur les 0
#print(f"t12=\n{t12}")
#print(f"t1=\n{t1}\n",f"t12=\n{t12}")

x1=remontee(t12,b1)
#print("t12=",t12,"\n","x1=",x1)
#print("verif:\n",np.dot(t1, x1))
print("\n\n")

'''
for u in range(10-1,-1,-1):
    print(u)
'''

### Exercice 2

print("\n\n")
print("Exercice 2:")
print("\n\n")

## Question 1
'''
Ecrire une fonction nommée Gauss1(A) qui prend en argument la matrice A, et retourne
T la matrice triangulaire produite par la méthode de Gauss (phase de descente seule)
'''

def Gauss1(A0):
    """
    methode 2 renvoie une matrice triangle avec les coefs à la place des 0
    """
    #A=copy.deepcopy(A0)
    #A=A.astype(float)
    B=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    B=B.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(B)[0] #enregistre la taille de la matrice
    for k in range(n):
        #print(f"k={k}")
        ip=0 #preparation indice du while
        while B[k,k]==0: #ajout pour éviter les matrices avec pivot nul, à améliorer
            raise Exception("Pivot nul, pas de decomposition LU possible")
            ip+=1
            if k+ip<n:
                #print(f"B=\n{B}")
                #print(f"permutation entre ligne {k} et ligne {k+ip} à la {k+1}-ieme etape")
                B[k,:],B[k+ip,:]=copy.deepcopy(B[k+ip,:]),copy.deepcopy(B[k,:]) #inverse les lignes
                #print(f"B=\n{B}")
            else:
                raise Exception("matrice non trigonalisable") #quand on atteind la fin de la matrice sans trouver de pivot non nul, pas trigonalisable
                break
        for i in range(k+1,n):
            mik=-(B[i,k]/B[k,k])
            #print(f"Coefficient {mik}")
            B[i,k]=mik #stockage du pivot à la place du 0
            #print(f"Operation {A[i,:]}+{mik}*{A[k,:]}")
            #A[i,:]=A[i,:]+mik*A[k,:]
            #print(A[i,:])
            #print(f"Operation {B[i,:]}+{mik}*{B[k,:]}")
            B[i,(k+1):n]=B[i,(k+1):n]+B[i,k]*B[k,(k+1):n]
            #print(B[i,:])
    return B

## Question 2
'''
Ecrire une fonction similaire nommée Gauss2(A,b) qui prend en argument la matrice
A, le vecteur b et retourne le couple [T, x] où T est la matrice triangulaire produite par la
méthode de Gauss et x est le vecteur solution (penser à utiliser la fonction remontee).
Tester sur les matrices des exercices de la feuille de TD2, puis sur des matrices aléatoires
de votre choix.
'''

def Gauss2(A0,b0):
    """
    Resoud le systeme avec Ax=b en renvoyant T et x
    """
    #A=copy.deepcopy(A0)
    #A=A.astype(float)
    B=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    B=B.astype(float) #convertit les éléments en float pour éviter les arrondis
    b=copy.deepcopy(b0) #crée un deepcopy du vecteur argument
    b=b.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(B)[0] #enregistre la taille de la matrice
    for k in range(n):
        #print(f"k={k}")
        ip=0 #preparation indice du while
        while B[k,k]==0: #ajout pour éviter les matrices avec pivot nul, à améliorer
            ip+=1
            if k+ip<n:
                #print(f"B=\n{B}")
                #print(f"permutation entre ligne {k} et ligne {k+ip} à la {k+1}-ieme etape")
                B[k,:],B[k+ip,:]=copy.deepcopy(B[k+ip,:]),copy.deepcopy(B[k,:]) #inverse les lignes
                b[k,:],b[k+ip,:]=copy.deepcopy(b[k+ip,:]),copy.deepcopy(b[k,:])
                #print(f"B=\n{B}")
            else:
                raise Exception("matrice non trigonalisable") #quand on atteind la fin de la matrice sans trouver de pivot non nul, pas trigonalisable
                break
        for i in range(k+1,n):
            mik=-(B[i,k]/B[k,k])
            #print(f"Coefficient {mik}")
            B[i,k]=mik #stockage du pivot à la place du 0
            #print(f"Operation {A[i,:]}+{mik}*{A[k,:]}")
            #A[i,:]=A[i,:]+mik*A[k,:]
            #print(A[i,:])
            #print(f"Operation {B[i,:]}+{mik}*{B[k,:]}")
            B[i,(k+1):n]=B[i,(k+1):n]+B[i,k]*B[k,(k+1):n]
            b[i,:]=b[i,:]+B[i,k]*b[k,:]
            #print(B[i,:])
    #print("verif:",np.dot(A0, remontee(B, b)))
    return [B,remontee(B, b)]




### Exemple Commun TP

print("\n\n")
print("Exemples du TP:")
print("\n\n")

##remontee
a3=np.array([[1,2,3],[0,1,2],[0,0,1]])
b3=np.array([1,3,5]) #vecteur ligne de taille 5 avec des 1
b3=b3.reshape(3,1) #vecteur colonne de taille 5
print("a3=\n",a3)
print("b3=\n",b3)
x3=remontee(a3, b3)
print("x3=\n",x3)
print("verif: Ax=\n",np.dot(a3, x3))
print("\n\n")


## Matrice grande randomisée
n=200
an=n*np.random.rand(n,n)
tn=np.triu(an)
bn=np.random.rand(n,1)
xn=remontee(tn, bn)
#print("tn=\n",tn)
#print("bn=\n",bn)
#print("xn=\n",xn)
#print("verif: Tx-b=\n",np.dot(tn, xn)-bn)


##descente
a4=np.array([[1,0,0],[2,1,0],[3,2,1]])
b4=np.array([1,3,5]) #vecteur ligne de taille 5 avec des 1
b4=b4.reshape(3,1) #vecteur colonne de taille 5
print("a4=\n",a4)
print("b4=\n",b4)
x4=descente(a4, b4)
print("x4=\n",x4)
print("verif: Ax=\n",np.dot(a3, x3))






### Exercice 3

print("\n\n")
print("Exercice 3:")
print("\n\n")

## Question 1

def LU1(A0):
    A=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    A=A.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(A)[0] #enregistre la taille de la matrice
    T=Gauss1(A)
    Id=np.eye(n)
    D=np.diag(np.diag(T))
    L=-np.tril(T)+D+Id
    U=np.triu(T)
    return [L,U]

## Question 2 et 3


def band(n):
    a=n*[2]
    b=(n-1)*[1]
    return np.diag(a)+np.diag(b,-1)+np.diag(b,1)

a5=band(6)
print(a5)

l5=LU1(a5)[0]
u5=LU1(a5)[1]
print("l5=\n",l5)
print("u5=\n",u5)
print("verif: l5*u5=\n",np.dot(l5,u5))

b5=np.array(6*[1]) #vecteur ligne de taille 6 avec des 1
b5=b5.reshape(6,1) #vecteur colonne de taille 6

y5=descente(l5, b5)
x5=remontee(u5, y5)

x53=Gauss2(a5, b5)[1]

print("x5=\n",x5)
print("x53=\n",x53)
print("diff=\n",x53-x5)

## Question 4

def LU2(A0):
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



print(a5)

l52=LU2(a5)[0]
u52=LU2(a5)[1]
print("l52=\n",l52)
print("u52=\n",u52)
print("verif: l52*u52=\n",np.dot(l52,u52))

y52=descente(l52, b5)
x52=remontee(u52, y52)

print("x5=\n",x52)
print("x53=\n",x53)
print("diff=\n",x53-x52)

## Question 5

def LDR(A0):
    L,U=LU1(A0)[0],LU2(A0)[1]
    D=np.diag(np.diag(U))
    D1=np.diag(1/np.diag(U))
    R=np.dot(D1,U)
    return [L,D,R]

def LDR2(A0):
    L,U=LU2(A0)[0],LU2(A0)[1]
    D=np.diag(np.diag(U))
    D1=np.diag(1/np.diag(U))
    R=np.dot(D1,U)
    return [L,D,R]


###Exemples pour PLU

def Permute(A0):
    """
    renvoie matrice P pour PLU
    """
    B=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    B=B.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(B)[0] #enregistre la taille de la matrice
    P=np.eye(n)
    for k in range(n):
        ip=0 #preparation indice du while
        while B[k,k]==0: #ajout pour éviter les matrices avec pivot nul, à améliorer
            ip+=1
            if k+ip<n:
                B[k,:],B[k+ip,:]=copy.deepcopy(B[k+ip,:]),copy.deepcopy(B[k,:]) #inverse les lignes
                P[k,:],P[k+ip,:]=copy.deepcopy(P[k+ip,:]),copy.deepcopy(P[k,:])
            else:
                raise Exception("matrice non trigonalisable") #quand on atteind la fin de la matrice sans trouver de pivot non nul, pas trigonalisable
                break
        for i in range(k+1,n):
            mik=-(B[i,k]/B[k,k])
            B[i,k]=mik #stockage du pivot à la place du 0
            B[i,(k+1):n]=B[i,(k+1):n]+B[i,k]*B[k,(k+1):n]
    return [P,B]


def PLU1(A0):
    A=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    A=A.astype(float) #convertit les éléments en float pour éviter les arrondis
    n=np.shape(A)[0] #enregistre la taille de la matrice
    Per=Permute(A)
    P=Per[0]
    T=Per[1]
    #T=Gauss1(np.dot(P,A))
    Id=np.eye(n)
    D=np.diag(np.diag(T))
    L=-np.tril(T)+D+Id
    U=np.triu(T)
    return [P,L,U]

a8=np.array([[2,1,1],[2,1,2],[1,1,2]])
a8=np.array([[2,4,-4,1],[3,6,1,-2],[-1,-2,2,3],[1,1,-4,1]])
b8=np.array([1,3,5]) #vecteur ligne de taille 5 avec des 1
b8=b8.reshape(3,1) #vecteur colonne de taille 5
print("a8=\n",a8)
print("b8=\n",b8)
p8=PLU1(a8)[0]
l8=PLU1(a8)[1]
u8=PLU1(a8)[2]
print("p8=\n",p8)
print("l8=\n",l8)
print("u8=\n",u8)
print(f"verif: l8*u8=\n{l8}\n*{u8}=\n",np.dot(l8,u8))
print(f"verif: p8*a8=\n{p8}\n*{a8}=\n",np.dot(p8,a8))


###Exemple du cours

print("\n\n")
print("Exemple du Cours:")
print("\n\n")
a2=np.array([[2,4,-4,1],[-1,1,2,3],[3,6,1,-2],[1,1,-4,1]])
b2=np.array([0,4,-7,2]) #vecteur ligne 
b2=b2.reshape(4,1) #vecteur colonne
print("a2=\n",a2)
print("b2=\n",b2)
#print(a2[1,:])
#np.shape(a2)
#t2=Trig(a2)
#t22=Trig2(a2)
#print(f"t2=\n{t2}\n",f"t22=\n{t22}")
#x2=remontee(t22,b2)
#print("t22=",t22,"\n","x2=",x2)


#print("T=\n",Gauss2(a2, b2)[0],"\nx=\n",Gauss2(a2, b2)[1])
#print("verif: Ax=\n",np.dot(a2, Gauss2(a2, b2)[1]))

l2=LU1(a2)[0]
u2=LU1(a2)[1]
print("l2=\n",l2)
print("u2=\n",u2)
l21=LDR(a2)[0]
d21=LDR(a2)[1]
r21=LDR(a2)[2]
print(f"verif: l2*d2*r2=\n{l21}\n*{d21}\n*{r21}\n",np.dot(np.dot(l21,d21),r21))

'''
%timeit LU1(an)
%timeit LU2(an)
'''

#print("\n\n")


### Exercice 4

## Question 1


def band2(n,d=2):
    if d<=n:
        a=np.zeros([n,n])-np.diag(n*[d])
        for i in range(d):
            a+=np.diag((n-i)*[d-i],-i)+np.diag((n-i)*[d-i],i)
    else:
        raise Exception("bande trop importante")    
    return a

a5=band2(10,10)
print(a5)



def CHdec1(A0):
    A=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    A=A.astype(float) #convertit les éléments en float pour éviter les arrondis
    At=copy.deepcopy(A0)
    At=At.astype(float) #convertit les éléments en float pour éviter les arrondis
    At.transpose()
    n=np.shape(A)[0] #enregistre la taille de la matrice
    #if At!=A:
    #    raise Exception("Matrice non Symétrique")
    if False:
        print("filler")
    else:
        B=np.tril(A)
        for j in range(n):
            B[j,j]-=sum(B[j,:j]**2)
            B[j,j]=np.sqrt(B[j,j])
            for i in range(j+1,n):
                B[i,j]-=sum(B[i,:j]*B[j,:j])
                B[i,j]=B[i,j]/B[j,j]
    return B


b5=CHdec1(a5)
print(f"b5=\n{b5}")
b55=copy.deepcopy(b5)
b55=b55.transpose()
print(f"verif: b5*b5^T=\n",np.dot(b5,b55))

def CHdec2(A0):
    A=copy.deepcopy(A0) #crée un deepcopy de la matrice argument
    A=A.astype(float) #convertit les éléments en float pour éviter les arrondis
    At=copy.deepcopy(A0)
    At.transpose()
    ldr=LDR(A)
    return np.dot(ldr[0],np.sqrt(ldr[1]))

b5=CHdec2(a5)
print(f"b5=\n{b5}")
b55=copy.deepcopy(b5)
b55=b55.transpose()
print(f"verif: b5*b5^T=\n",np.dot(b5,b55))


n=6
an0=np.sqrt(n)*np.random.rand(n,n)
an=np.dot(an0.transpose(),an0)
print("an=\n",an)
bn=CHdec1(an)
print(f"bn=\n{bn}")
bnn=copy.deepcopy(bn)
bnn=bnn.transpose()
print(f"verif: bn*bn^T=\n",np.dot(bn,bnn))
print("nb d'erreurs : ",n**2-sum(sum(np.isclose(an, np.dot(bn,bnn)))))

'''
%timeit CHdec1(a5)
%timeit CHdec2(a5)
'''





