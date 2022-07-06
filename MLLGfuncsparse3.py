from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
# from fenics import *
# from fenics_rhs import FenicsRHS
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import *#gmres,spsolve
#from bempp.api.linalg.iterative_solvers import *
from bempp.api.external.fenics import FenicsOperator
#from bempp.api import as_matrix
from maxwellrt0 import *
from bempp.api.operators.boundary import maxwell
import timeit
import sys
from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
import scipy

sys.setrecursionlimit(100000)
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()


def MLLGfuncsparse(T, N, tau, h, m, rho, L, tolgmres, eps, mu, sig, theta, alpha, Ce,thick):  # ,J,m0,E0,H0):
    print('h is ', h, ' N is', N, 'ich bin 3')
    
    onof= 1.0 # with onof= 0.0 you can simulate the uncoupled system. 
    refadd=2.0 # no exact solution given. with refadd=0.0 you can compare the approximation to a predefined solution given via dtMex (time derivative of exact solution) and Mex (exact solution) 
    ps=1 # degree of FEM and BEM spaces. (only ps=1 is implemented in BEMPP) 
    
    ### plots
    vtkfilem = File('plots/mfield.pvd') 
    vtkfileE = File('plots/Efield.pvd')
    vtkfileH = File('plots/Hfield.pvd')
    vtkfilePhi = File('plots/phifield.pvd')
    vtkfilePsi = File('plots/psifield.pvd')
    
      
    
    # matrices for discretization and input data 
    (Lhs,rhsLLGH, M0,mj,Ej,Hj,JccE,JE,nBEM,nFEM,nMAG,X,trace_space,VV,V3,wei,boundary_rhs,dtmfunc) =assign_data_and_systems( tau, h, m, eps, mu, sig,rho,L,N,thick)
    # external magnetic field
    Hex = project(Expression(['0.0', '-10.0', '0.0'], degree=1), V3)
    
    # Coefficient vector [E,H,phi,psi]
    sol = np.concatenate([Ej.vector()[:], Hj.vector()[:], np.zeros(nBEM),np.zeros(nBEM)])
    randkoeff = np.zeros([int(N + 1), 2 * nBEM])  # storage variable for boundary coefficients
    dtmpsiko = np.zeros([int(N + 1), 2 * nBEM])
    mkoeff = np.zeros([int(N + 1), nMAG])
    Ekoeff = np.zeros([int(N + 1), nFEM])
    Hkoeff = np.zeros([int(N + 1), nFEM])

    randkoeff[0, :] = np.zeros(2*nBEM)
    Ekoeff[0, :] = np.real(Ej.vector())
    Hkoeff[0, :] = np.real(Hj.vector())
    mkoeff[0, :] = np.real(mj.vector())
    
    #variables for plot 
    mplot= interpolate(Expression(['0.00', '0.0', '0.0'], degree=ps), V3)
    Hplot= interpolate(Expression(['0.00', '0.0', '0.0'], degree=ps), X)
    Eplot= interpolate(Expression(['0.00', '0.0', '0.0'], degree=ps), X)
    #Phiplot = bempp.api.GridFunction(trace_space, coefficients=psikoeff[1,:])
    #Psiplot = bempp.api.GridFunction(trace_space, coefficients=psikoeff[1,:])
    
    
    ########## Time Stepping ##################################################################

    for j in range(0, int(N)):  # time stepping: update from timestep t_j to j+1

        ##### LLG update #########################################################################

        start6 = timeit.default_timer()
        
        #exact solution
        #dtMex = interpolate(Expression(dtexSolutionCode,multF=3,tend=10*T,t=(j)*tau,degree=1),V3)
        #Mex = interpolate(Expression(exSolutionCode,multF=3,tend=10*T,t=(j+1)*tau,degree=1),V3)
        
        (v, lam) = TrialFunctions(VV)
        (phi, mue) = TestFunctions(VV)
        lhs1 = ((alpha * inner(v, phi) + inner(cross(mj, v), phi) + Ce * tau * theta * inner(nabla_grad(v), nabla_grad(phi))) * dx + inner(dot(phi, mj), lam) * dx + inner(dot(v, mj), mue) * dx)
        rhs1 = (-Ce * inner(nabla_grad(mj), nabla_grad(phi)) )*dx #+(onof*inner(Hj, phi) + onof*inner(Hex, phi))*dx
        
        if (refadd<1.0):
            rhs1 = rhs1 +(inner(alpha*dtMex + cross(Mex,dtMex), phi) +Ce*inner(nabla_grad(Mex),nabla_grad(phi)))*dx
        if (onof >0.0 ):
            #lhs1= lhs1
            rhs1 = rhs1 +(onof*inner(Hj, phi) + onof*inner(Hex, phi))*dx
        
        # compute solution
        vlam = Function(VV)
        solve(lhs1 == rhs1, vlam, solver_parameters={"linear_solver": "gmres"},
              form_compiler_parameters={"optimize": True})

        # update magnetization
        (v, lam) = vlam.split(deepcopy=True)
        mj.vector()[:] = mj.vector()[:] + tau * v.vector()[:]
        mj = mj / sqrt(dot(mj, mj))
        mj = project(mj, V3, solver_type='cg')

        stop6 = timeit.default_timer()
        print(' Time for LLG step                 ', j, ': ', stop6 - start6)

        ###### Maxwell update ###################################################################

        start4 = timeit.default_timer()
        dtmpsiko[:, :] = randkoeff[:, :]  # compute \partial_t^m phi,  \partial_t^m psi,
        for k in range(0, m):
            for r in range(0, j +1):
                dtmpsiko[j + 1 - r, :] = (dtmpsiko[j + 1 - r, :] - dtmpsiko[j + 1 - r - 1, :]) / tau
        
        boundary_rhs[0].coefficients[:] = np.zeros(nBEM)
        boundary_rhs[1].coefficients[:] = np.zeros(nBEM)

        for kk in range(1, j +2):  # Convolution, start with index 1, because psiko(0)=0 by definition
            dtmfunc[0].coefficients[:] = -dtmpsiko[kk,nBEM:]
            dtmfunc[1].coefficients[:] = np.sqrt(mu*eps)**(-1) *dtmpsiko[kk,:nBEM]
            boundary_rhs += rho ** (-(j + 1 - kk)) / L *np.real(wei[0] *dtmfunc)
            for ell in range(1, int(np.ceil(L / 2) - 1) + 1):  # it is wei(L-d)=complconj(wei(d))
                boundary_rhs += rho ** (-(j + 1 - kk)) / L * np.real( 2 * np.exp(-2.0 * np.pi * 1j * (j + 1 - kk) * ell / L) * wei[ell] *dtmfunc)
            if not (L % 2):
                boundary_rhs += rho ** (-(j + 1 - kk)) / L * np.real( (-1) ** (j + 1 - kk) * wei[int(L / 2)] * dtmfunc)
        boundary_rhs[1].coefficients[:] = np.real(np.sqrt(mu*eps)*boundary_rhs[1].coefficients)

        ##### Right hand side  #####
        t=(j+1)*tau
        Rhs = np.concatenate([eps / tau * M0 * Ej.vector()[:] +1.0/3.0*t**3*M0.dot(JccE.vector())+t*(sig*t+2.0)*M0.dot(JE.vector()),#- ((j+1)*tau )**2 * M0 * J.vector(),
                              mu / tau * M0 * Hj.vector()[:] - onof * mu * rhsLLGH.transpose() * (v.vector()[:]),
                              - 1.0 * np.real(boundary_rhs[0].projections(wei[0].dual_to_range_spaces[0])), #das projections projeziert nicht, sonern macht das richtige, mit der wform multiplizieren
                              - 1.0 * np.real(boundary_rhs[1].projections(wei[0].dual_to_range_spaces[1]))])  # - stimmt, da drei minus: unser B=-B, auf andere seite, und -1 durchmultipliziert
#-dtMex.vector() 
        stop4 = timeit.default_timer()
        print(' Time for Convolution in time step ', j, ': ', stop4 - start4)


        # Solution of Lhs=Rhs with gmres
        start5 = timeit.default_timer()
        it_count = 0
        def count_iterations(x):
            nonlocal it_count
            it_count += 1
        
        
        sol = scipy.linalg.solve(Lhs, Rhs)#, tol=tolgmres,callback=count_iterations, x0=sol)
        info=0
        #sol, info = gmres(Lhs, Rhs, tol=tolgmres, callback=count_iterations,  x0=sol)
        stop5 = timeit.default_timer()
        print(' Time for gmres                    ', j, ': ', stop5 - start5)
        if (info > 0):
            print("Failed to converge after " + str(info) + " iterations")
        else:
            print("Solved system " + str(j) + " in " + str(it_count) + " iterations." )

        

        #### Storage and end of loop ###################
        Ej.vector()[:] = sol[:nFEM]  # coefficients for next timestep
        Hj.vector()[:] = sol[nFEM:2*nFEM]

        mkoeff[j+1, :] = np.real(mj.vector()[:]) #storage
        Ekoeff[j+1, :] = np.real(Ej.vector()[:])
        Hkoeff[j+1, :] = np.real(Hj.vector()[:])
        randkoeff[j + 1, :] = np.real(sol[2 * nFEM:])

        mplot.rename("m", "m") #see the QA reported below.
        mplot.vector()[:]= mkoeff[j+1,:]
        vtkfilem << mplot, j+1
        Eplot.rename("E", "E") #see the QA reported below.
        Eplot.vector()[:]= Ekoeff[j+1,:]
        vtkfileE << Eplot, j+1
        
        Hplot.rename("H", "H") #see the QA reported below.
        Hplot.vector()[:]= Hkoeff[j+1,:]
        vtkfileH << Hplot, j+1
        
        #Phiplot = bempp.api.GridFunction(trace_space, coefficients=phi12ko[:])
        #vtkfilePhi << Phiplot, j+1
        #Phiplot.plot()
        #from exportfun import export
        #export('grid_functionPhi.msh', grid_function=Phiplot, write_binary=False)
        #Psiplot = bempp.api.GridFunction(trace_space, coefficients=-psiko[:])
        #export('grid_functionPsi.msh', grid_function=Psiplot, write_binary=False)
        
    return (mkoeff, Ekoeff, Hkoeff, randkoeff[:, :nBEM], randkoeff[:, nBEM:2 * nBEM])

def assign_data_and_systems( tau, h, m, eps, mu, sig,rho,L,N,thick):
    start = timeit.default_timer()
    meshh = BoxMesh(Point(0,0,0),Point(thick,thick,1),int(h),int(h),int(h/thick))#UnitCubeMesh(h, h, h)

    def dlt(z):
        # BDF1
        return 1.0 - z
        # BDF2
        # return 1.0-z+0.5*(1.0-z)**2
        
    # approximaion spaces
    # m   = Lagrange1
    # E,H = N1curl
    # gamma_TE, gamma_TH = trace_space = rwg
    # phi ~ gamma_TH = rwg
    # psi ~ -gamma_TE = RWG

    Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), 1, dim=3)
    V3 = FunctionSpace(meshh, Pr3)
    Pr = FiniteElement('P', meshh.ufl_cell(), 1)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(meshh, element)

    Xr = FiniteElement("N1curl", meshh.ufl_cell(), 1)
    X = FunctionSpace(meshh, Xr)

    trace_space, trace_matrix = nc1_tangential_trace(X)  # trace space and restriction matrix



    nBEM = trace_space.global_dof_count  # DOFs
    nFEM = X.dim()
    nMAG = V3.dim()

    # initial data and input data 
    (mj,Ej,Hj,JccE,JE)= initialdata(X,V3)
    
    # left hand side matrix and matrices to build right hand side 
    (Lhs,M0,rhsLLGH,boundary_rhs,dtmfunc)=assign_LHS(X,V3,mu,trace_matrix,dlt,m,trace_space,eps,tau,sig,nFEM,nBEM)

    stop = timeit.default_timer()

    # Convolution quadrature weights
    wei= assign_CQweights(L,tau,m,mu,eps,trace_space.grid,dlt,rho,N,stop,start)

    return (Lhs,rhsLLGH, M0,mj,Ej,Hj,JccE,JE,nBEM,nFEM,nMAG,X,trace_space,VV,V3,wei,boundary_rhs,dtmfunc)

def initialdata(X,V3):
    # Initial Data and Input functions
    class MyExpression1(UserExpression):
        def eval(self, value, x):
            sqnx = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)
            A = (1 - 2 * sqnx ** 0.5) ** 4 / 4
            if sqnx <= 0.25:
                value[0] = 2 * A * (x[0] - 0.5) / (A * A + sqnx)
                value[1] = 2 * A * (x[1] - 0.5) / (A * A + sqnx)
                value[2] = (A * A - sqnx) / (A * A + sqnx)
            else:
                value[0] = 0
                value[1] = 0
                value[2] = -1

        def value_shape(self):
            return (3,)

    class MyExpression2(UserExpression):
        def eval(self, value, x):
            sqnx = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)+ (x[2] - 0.5) * (x[2] - 0.5)
            #A = (1 - 2 * sqnx ** 0.5) ** 4 / 4
            r = 3
            if sqnx <= r:
                value[0] = -(x[1] - 0.5) * sqnx * 100
                value[1] = +(x[0] - 0.5) * sqnx * 100
                value[2] = 100
            else:
                value[0] = 0
                value[1] = 0
                value[2] = 0

        def value_shape(self):
            return (3,)

    class MyExpression3(UserExpression):
        def eval(self, value, x):
            sqnx = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)+ (x[2] - 0.5) * (x[2] - 0.5)
            #A = (1 - 2 * sqnx ** 0.5) ** 4 / 4
            r = 0.25**2
            if sqnx <= r:
                value[0] = 100*np.abs(r-sqnx)/r
                value[1] = 0
                value[2] = 0
            else:
                value[0] = 0
                value[1] = 0
                value[2] = 0

        def value_shape(self):
            return (3,)

    t0 = -0.25
    t = 0.125

    def phiji(s, r):
        s = 2 ** 4 * 16 / r ** 8 * s ** 4 * (
                    r - s) ** 4  # 16/r**4*s**2*(r-s)**2 # #np.exp(-r ** 2 / 4 / s / (r - s) + 1)1-cos(s/r*np.pi)  #1.0-np.abs(s/r) #
        return s

    def dsphiji(s, r):
        s = 2 ** 4 * 2 * 32 / r ** 8 * s ** 3 * (r - s) ** 3 * (
                    r - 2 * s)  # np.exp(-r ** 2 / 4 / s / (r - s) + 1) *r**2/4*(r-2*s)/(s*(r-s))**2
        return s

    class MyExpression4(UserExpression):
        def eval(self, value, x):
            sqnx = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) + (x[2] - 0.5) * (x[2] - 0.5)
            # A = (1 - 2 * sqnx ** 0.5) ** 4 / 4
            s = sqnx - t - t0
            r = 0.25
            if (s <= r - 0.0000) and (0.0000 <= s):
                value[0] = 0.0 * dsphiji(s, r) * (4 * (x[1] - 0.5) ** 2 + 4 * (x[2] - 0.5) ** 2 - 1) + (
                            4.0) * phiji(s, r)
                value[1] = +0.0 * 4 * dsphiji(s, r) * (x[0] - 0.5) * (x[1] - 0.5)
                value[2] = +0.0 * 4 * dsphiji(s, r) * (x[0] - 0.5) * (x[2] - 0.5)
            else:
                value[0] = 0.0
                value[1] = 0.0
                value[2] = 0.0

        def value_shape(self):
            return (3,)
    class MyExpression5(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])**2*sin(x[1])**2*sin(x[2])**2
            value[1] = 0.0
            value[2] = 0.0

        def value_shape(self):
            return (3,)
    
    class MyExpressionccE(UserExpression):
        def eval(self, value, x):
            s1=sin(np.pi*x[0])
            s2=sin(np.pi*x[1])
            s3=sin(np.pi*x[2])
            s1s=s1**2
            s2s=s2**2
            s3s=s3**2
            c1=cos(np.pi*x[0])
            c2=cos(np.pi*x[1])
            c3=cos(np.pi*x[2])
            pis=np.pi**2
            value[0] = -pis*2*s1s*((c2**2-s2s)*s3s+s2s*(c3**2-s3s))
            value[1] = pis*4*s1*c1*s2*c2*s3s
            value[2] = pis*4*s1*c1*s3*c3*s2s
        def value_shape(self):
            return (3,) 
    class MyExpressionE(UserExpression):
        def eval(self, value, x):
            s1=sin(np.pi*x[0])
            s2=sin(np.pi*x[1])
            s3=sin(np.pi*x[2])
            s1s=s1**2
            s2s=s2**2
            s3s=s3**2
            value[0] = s1s*s2s*s3s
            value[1] = 0
            value[2] = 0 
        def value_shape(self):
            return (3,)             
    #minit = MyExpression1(degree=1)
    #mj= project(minit, X)
    
    mj = interpolate(Expression(['1.0', '0.0', '0.0'], degree=1), V3)
    #mj = mj / sqrt(dot(mj, mj))
    #mj = project(mj, V3, solver_type='cg')
    Ej = project(Expression(['0.00', '0.0', '0.0'], degree=1), X)#interpolate(Jinit, X)#
    Hj = project(Expression(['0.0', '-10.0', '0.0'], degree=1), X)#
    #JccE = MyExpressionccE(degree=1)
    JccE = project(Expression(('0.0', '0.0', '0.0'), degree=1), X)#interpolate(JccE, X)
    #dolfin.plot(J)
    #plt.show()
    #JE = MyExpressionE(degree=1)
    JE=project(Expression(('0.0', '0.0', '0.0'), degree=1), X)# interpolate(JE, X)#project(Expression(['0.00','000.0','00.0'],degree=1),X)
    
    class MyExpressionM(UserExpression):
        def eval(self, value, x):
            #value[0] = 0.0
            #value[1] = -1.0
            #value[2] = 0.0
            #if np.abs(x[2]-0.5)<0.25:# and (x[1]-0.05)**2+(x[0]-0.05)**2< 0.049**2:
            x=x[2]
            #x= (1-np.cos(np.pi*x))/2
            s3=np.sin(2*np.pi*x)
            c3=np.cos(2*np.pi*x)
            value[0] = s3
            value[1] = c3
            value[2] = 0.0

        def value_shape(self):
                return (3,)
    mj = interpolate(MyExpressionM(degree=3), V3)
    Hj = project(Expression(['0.0', '00.0', '0.0'], degree=1), X)
    
    return(mj,Ej,Hj,JccE,JE)

def assign_LHS(X,V3,mu,trace_matrix,dlt,m,trace_space,eps,tau,sig,nFEM,nBEM):
    bc_space = bempp.api.function_space(trace_space.grid, "BC", 0)  # domain spaces
    rwg_space = bempp.api.function_space(trace_space.grid, "RWG", 0)
    snc_space = bempp.api.function_space(trace_space.grid, "SNC", 0)  # dual to range spaces
    rbc_space = bempp.api.function_space(trace_space.grid, "RBC", 0)
    #brt_space = bempp.api.function_space(trace_space.grid, "RT", 0)

    boundary_rhs=[bempp.api.GridFunction(bc_space, coefficients=np.ones(nBEM)),bempp.api.GridFunction(bc_space, coefficients=np.ones(nBEM))]
    dtmfunc=[bempp.api.GridFunction(rwg_space, coefficients=np.zeros(nBEM)),bempp.api.GridFunction(rwg_space, coefficients=np.zeros(nBEM))]
    Ulitt = TrialFunction(X)
    phi = TestFunction(V3)

    # interior operators and mass matrices
    # LLG Part

    rhsLLGH = FenicsOperator((+inner(phi, Ulitt)) * dx).weak_form()

    # Maxwell part
    Uli = TrialFunction(X)
    UliT = TestFunction(X)
    M0 = FenicsOperator((inner(Uli, UliT)) * dx).weak_form()
    D = FenicsOperator(0.5 * inner(Uli, curl(UliT)) * dx + 0.5 * inner(curl(Uli), UliT) * dx).weak_form()

    trace_op = aslinearoperator(trace_matrix)  # trace operator

    cald =  maxwell.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(0) / tau, space_type="all_rwg").weak_form()  # calderon operator
    caldfac= 1.0 / mu /(dlt(0)) ** (m) 

    #mass2 = bempp.api.operators.boundary.sparse.identity(brt_space, bc_space, snc_space).weak_form()
    #mass1 = bempp.api.operators.boundary.sparse.identity(brt_space, rwg_space, rbc_space).weak_form()
    massbd = bempp.api.operators.boundary.sparse.identity(rwg_space, bc_space, snc_space).weak_form()
    # Definition coupled 4x4 matrix
    blocke1 = np.ndarray([4, 4], dtype=np.object)
    blocke1[0, 0] = (eps / tau + sig) * M0
    blocke1[0, 2] = -(0.5 / mu) * trace_op.adjoint() * massbd.transpose()
    blocke1[0, 1] = -1.0 * D
    blocke1[1, 2] = np.zeros((nFEM, nBEM))
    blocke1[1, 0] = 1.0 * D
    blocke1[0, 3] = np.zeros((nFEM, nBEM))
    blocke1[1, 1] = mu / tau * M0
    blocke1[1, 3] = -0.5 * trace_op.adjoint() * massbd.transpose()

    blocke1[2, 0] = 0.5 / mu * massbd * trace_op
    blocke1[2, 2] = 1.0 / mu * np.sqrt(mu / eps) * caldfac*cald[0, 1]  # Calderon* \partial_t^m phiko =
    blocke1[3, 0] = np.zeros((nBEM, nFEM))
    blocke1[2, 3] = -caldfac*cald[0, 0]
    blocke1[2, 1] = np.zeros((nBEM, nFEM))
    blocke1[3, 2] = caldfac*cald[1, 1]
    blocke1[3, 1] = 0.5 * massbd * trace_op
    blocke1[3, 3] = -mu * np.sqrt(eps / mu) *caldfac* cald[1, 0]

    Lhs = assmatrix(BlockedDiscreteOperator(np.array(blocke1)))
    return(Lhs,M0,rhsLLGH,boundary_rhs,dtmfunc)

def assign_CQweights(L,tau,m,mu,eps,trace_spacegrid,dlt,rho,N,stop,start):
    start2 = timeit.default_timer()
    # Definition of Convolution Quadrature weights
    storblock = np.ndarray([2, 2], dtype=np.object)  # dummy variable
    wei = np.ndarray([int(np.floor(L/2)+1)], dtype=np.object)  # dummy array of B(zeta_l)(zeta_l)**(-m)

    for ell in range(0, int(np.floor(L/2)+1)):  # CF Lubich 1993 On the multistep time discretization of linearinitial-boundary value problemsand their boundary integral equations, Formula (3.10)
        wei[ell]= (dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau) ** (-m) * 1.0 / mu *maxwell.multitrace_operator(trace_spacegrid, 1j * sqrt(mu * eps) * dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau, space_type="all_rwg")#.weak_form()
        #storblock[0, 0] = 1.0 / mu * np.sqrt(mu / eps) * cald[0, 1]
        #storblock[0, 1] = -cald[0, 0]
        #storblock[1, 0] = cald[1, 1]
        #storblock[1, 1] = -mu * np.sqrt(eps / mu) * cald[1, 0]
        #cald= bempp.api.BlockedDiscreteOperator(np.array(storblock))
    stop2 = timeit.default_timer()
    print('Time for initial data and LHS: ', stop - start, ' Time for Calderon evaluation: ', stop2 - start2)

    return wei

def assmatrix(operator):
    from numpy import eye
    cols = operator.shape[1]
    return operator @ eye(cols)
