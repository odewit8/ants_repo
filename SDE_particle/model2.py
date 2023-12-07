import numpy as np
from scipy.special import kvp

def eTh(th):
    return np.array([np.cos(th),np.sin(th)])

def nTh(th):
    return np.array([-np.sin(th),np.cos(th)])

def give_dC(X,Y,L,alpha):
    B = (X - Y[:,None] + L/2) % L - (L/2)
    for i in range(B.shape[0]):
        B[i,i,0] = 0
        B[i,i,1] = 0
    E = np.sqrt(B[:,:,0]**2+B[:,:,1]**2)
    C = np.divide(1,E,out=np.zeros_like(E), where=E!=0)
    C[C==np.inf] = 0
    A = B*C[:,:,None]
    D = -kvp(1,np.sqrt(alpha)*E)
    D[D==np.inf] = 0
    return np.sum(A*D[:,:,None],axis=1)

def update_X_Th(X,Xprev,Th,pdC,N,L,alpha,lmb,DTh,DT,v0,gamma,dt,step):
    lkXs = (X+lmb*np.transpose(eTh(Th))) % L
    a = (gamma/N)*np.sum(np.transpose(nTh(Th))*give_dC(lkXs,Xprev,L,alpha),axis=1)*dt
    Th1 = Th + a/(1+np.abs(a)) + gamma*np.sum(np.transpose(nTh(Th))*pdC(t=step,Xs=lkXs),axis=1)*dt + np.sqrt(2*dt*DTh)*np.random.normal(size=N)
    Th1 = np.mod(Th1,2*np.pi)
    X1 = X + v0*np.transpose(eTh(Th1))*dt + np.sqrt(2*dt*DT)*np.random.normal(size=(N,2))
    X1 = X1 % L
    return [X1, Th1]

