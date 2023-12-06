import numpy as np

def init_X_Th(N,L,init,seed):
    np.random.seed(seed)
    if init == "circle":
        rs = np.random.uniform(0,0.3*L,N)
        phis = np.random.uniform(0,2*np.pi,N)
        X = np.zeros((N,2))
        X[:,0] = rs*np.cos(phis) + 0.5*L
        X[:,1] = rs*np.sin(phis) + 0.5*L
        Th = np.random.uniform(0,2*np.pi,N)
    elif init == "square":
        X = np.random.uniform(0.0,L,(N,2))
        Th = np.random.uniform(0,2*np.pi,N)
    elif init == "sunflower":
        ri = L*0.22
        X = np.zeros((N, 2))
        b = np.round(2*np.pi)
        phi = (np.sqrt(5)+1)/2
        def radius(k,N,b):
            if k > N-b:
                r = ri
            else:
                r = ri*np.sqrt(k-1/2)/np.sqrt(N-(b+1)/2)
            return r
        for k in range(1,N+1):
            r = radius(k,N,b)
            theta = 2*np.pi*k/(phi**2)
            X[k-1,:] = np.array([r*np.cos(theta)+L/2,r*np.sin(theta)+L/2])
        Th = np.random.uniform(0,2*np.pi,N)
    elif init == "strip":
        X = np.zeros((N,2))
        X[:,0] = np.random.uniform(0,L,N)
        X[:,1] = np.random.uniform(0.4*L,0.6*L,N)
        Th = np.random.uniform(0,2*np.pi,N)
    elif init == "polarlane":
        X = np.zeros((N,2))
        X[:,0] = np.random.uniform(0,L,N)
        X[:,1] = np.random.uniform(0.4*L,0.6*L,N)
        Th = np.random.randint(0,2,N)*np.pi
    elif init == "polarline":
        X = np.zeros((N,2))
        X[:,0] = np.random.uniform(0,L,N)
        X[:,1] = 0.5*np.ones(N)
        Th = np.random.randint(0,2,N)*np.pi
    elif init == "polarline2":
        X = np.zeros((N,2))
        X[:,0] = np.linspace(0,1,num=N)
        X[:,1] = 0.5*np.ones(N)
        Th = np.resize([0,1], N)*np.pi
    elif init == "polar3":
        X = np.zeros((N,2))
        n = int(np.sqrt(N))
        xs = np.linspace(0,1,num=n,endpoint=False)
        ys0 = np.linspace(0,1,num=n)
        ys = 0.35+0.3*(ys0 + np.sin(2*np.pi*ys0)/(2*np.pi))
        for i in range(n):
            X[n*i:n*(i+1),0] = xs
            X[n*i:n*(i+1),1] = ys[i]
        Th = np.random.randint(0,2,N)*np.pi
    elif init == "polar4":
        X = np.zeros((N,2))
        n = int(np.sqrt(N))
        xs = np.linspace(0,1,num=n,endpoint=False)
        ys0 = np.linspace(0,1,num=n)
        ys = 0.30+0.4*(ys0 + np.sin(2*np.pi*ys0)/(2*np.pi))
        for i in range(n):
            X[n*i:n*(i+1),0] = xs
            X[n*i:n*(i+1),1] = ys[i]
        Th = np.zeros(N)
    elif init == "square_0p":
        X = np.random.uniform(0.0,L,(N,2))
        Th = np.resize([0,1], N)*np.pi
    elif init == "square_0p2":
        X = np.zeros((N,2))
        X[:,0] = np.linspace(0,1,num=N,endpoint=False)
        X[:,1] = 0.5
        Th = np.resize([0,1], N)*np.pi
    elif init == "square_1p":
        X = np.zeros((N,2))
        X[:,0] = ys = np.linspace(0,1,num=N,endpoint=False)
        X[:,1] = 0.5
        Th = np.ones(N)*np.pi
    return X, Th