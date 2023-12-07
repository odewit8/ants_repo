import matplotlib.pyplot as plt
from fenics import *
import numpy as np
plt.rc("text", usetex=False) 
plt.rc("mathtext", fontset="cm")
plt.rc("font", family="serif", serif="cmr10", size=12)
plt.rc("axes.formatter", use_mathtext = True)

Nx = 64
g = 50.0
D_T = 0.1
D_R = 1.0
v0 = 5.0
a = 1.0
n = 1.0
D = 1.0
Ll = 0.1

class PeriodicBoundaryX(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0],0) or near(x[0],1)) and (on_boundary))

    def map(self, x, y):
        y[0] = x[0] - 1.0

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
                return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


pbX = PeriodicBoundaryX()
pb = PeriodicBoundary()

mesh = UnitSquareMesh(Nx, Nx)
mesh1D = UnitIntervalMesh(Nx)
V1D = FunctionSpace(mesh1D, 'CG', 1, constrained_domain=pbX)
Q = FiniteElement("CG", triangle, 1)
LM = FiniteElement("Real", triangle, 0)
V = FunctionSpace(mesh, Q, constrained_domain=pb)
M = FunctionSpace(mesh, MixedElement([Q, LM]), constrained_domain=pb)

dx = Measure('dx', mesh)
dx1 = Measure('dx',mesh1D)

dxc = Expression("sin(2*pi*x[0])",degree=2)
costh = Expression("cos(2*pi*x[1])",degree=2)
sinth = Expression("sin(2*pi*x[1])",degree=2)
xl = Expression("x[0]+Ll*cos(2*pi*x[1])",Ll=Ll,degree=2)
dxcl = Expression("sin(2*pi*xl)",xl=xl,degree=2)


res_tot = 50
num_loop = 0
w = Function(M)
while res_tot > 1e-13:
    f, lm = split(w)
    wt = TestFunction(M)
    v, lmt = split(wt)
    mc = 1.0
    wt = TrialFunction(M)

    #for Pe=g=10 used 1e5 for LM
    lbfunc = Function(M)
    lbfunc.interpolate(Expression(("0.0","-1e5"),degree=2))
    ubfunc = Function(M)
    ubfunc.interpolate(Expression(("1e4","1e5"),degree=2))

    B = D_T*inner(grad(f),grad(v))*dx - \
        v0*costh*f*grad(v)[0]*dx + \
        g*sinth*dxcl*f*grad(v)[1]*dx + \
        lm*v*dx + lmt*f*dx - lmt*mc*dx


    J  = derivative(B, w, wt)
    problem = NonlinearVariationalProblem(B,w,J=J)
    problem.set_bounds(lbfunc,ubfunc)
    solver  = NonlinearVariationalSolver(problem)
    snes_solver_parameters = {"nonlinear_solver": "snes",
                            "snes_solver": {"linear_solver": "lu",
                                            #"absolute_tolerance": 1e-6,
                                            "maximum_iterations": 100
                            }}
    solver.parameters.update(snes_solver_parameters)
    solver.solve()

    f,lm=w.split()

    print("mass  = {:.2e}".format(assemble(f*dx)))
    print(lm((0,0)))

    rho_num=128
    thetas = np.linspace(0,0.99999,rho_num)
    xs = np.linspace(0.0,0.99999,rho_num)
    rhos = np.zeros(rho_num)

    for i in range(len(xs)):
        fs = [f((xs[i],j)) for j in thetas]
        rhos[i] = np.mean(fs)

    class rho_inter_1D(UserExpression):
        def eval(self, values, x):
            values[0] = rhos[int(np.floor((rho_num-1)*x[0]))]
        def value_shape(self):
            return ()

    rho1D = rho_inter_1D()
    rho1D = project(rho1D,V1D)  

    c = Function(V1D)
    wc = TestFunction(V1D)
    wct = TrialFunction(V1D)       

    Bc = D*grad(c)[0]*grad(wc)[0]*dx1+a*c*wc*dx1-n*rho1D*wc*dx1

    Jc  = derivative(Bc, c, wct) 

    problemc = NonlinearVariationalProblem(Bc,c,J=Jc)
    solverc  = NonlinearVariationalSolver(problemc)
    prmc = solverc.parameters
    prmc['newton_solver']['absolute_tolerance'] = 1e-12
    prmc['newton_solver']['maximum_iterations'] = 100
    solverc.solve()

    class c_inter_2D2(UserExpression):
        def eval(self, values, x):
            t = np.mod(x[0]+Ll*cos(2*pi*x[1]),1.0)
            values[0] = c((t))
        def value_shape(self):
            return ()

    c2D = c_inter_2D2()
    c2D = interpolate(c2D,V)
    dxcl = grad(c2D)[0]

    res1 = assemble(B)
    res1max = max(abs(res1.max()),abs(res1.min()))
    resc = assemble(Bc)
    rescmax = max(abs(resc.max()),abs(resc.min()))
    print("tot res = {:.4e}".format(res1max+rescmax))

    plot_num = 200
    plot_thetas = np.linspace(0.0,0.99999,plot_num)
    plot_xs = np.linspace(0.0,0.99999,plot_num)
    xx, yy = np.meshgrid(plot_xs,plot_thetas)
    fs_end = np.zeros((plot_num,plot_num))
    for i in range(plot_num):
        for j in range(plot_num):
            fs_end[j,i]=f((plot_xs[i],np.mod(plot_thetas[j]+0.5,1)))

    if num_loop % 2 == 0:
        plt.contourf(xx,yy,fs_end,levels=20,extend="both")
        plt.yticks(ticks=np.linspace(0,0.99999,5),labels=[r"$-\pi$",r"$-\frac{1}{2}\pi$",r"$0\pi$",r"$\frac{1}{2}\pi$",r"$\pi$"],fontsize=14)
        plt.xticks(ticks=np.linspace(0,0.99999,5),labels=np.linspace(0,1.0,5),fontsize=14)

        cb = plt.colorbar(shrink=0.8)
        cb.ax.set_title(r'$f$',fontsize=18)

        plt.ylabel(r'$\theta$',fontsize=18)
        plt.xlabel(r'$x$',fontsize=18)
        plt.savefig("fxth_D_T={:.2f}_D_R={:.2f}_v_0={:.2f}_gam={:.2f}_lam={:.2f}.eps".format(D_T,D_R,v0,g,Ll))
        plt.show()
        plt.close()
        plot(rho1D)
        plt.title(r'$\rho(x)$')
        plt.xlabel(r'$x$')
        plt.savefig("rho_D_T={:.2f}_D_R={:.2f}_v_0={:.2f}_gam={:.2f}_lam={:.2f}.eps".format(D_T,D_R,v0,g,Ll))
        plt.show()
        plt.close()


    res_tot = res1max+rescmax
    num_loop += 1