from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
# from fenics import *
# from fenics_rhs import FenicsRHS
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import gmres
#from bempp.api.linalg.iterative_solvers import gmres
from bempp.api.fenics_interface import FenicsOperator
from bempp.api import as_matrix
from haha import *
from bempp.api.operators.boundary import maxwell
import operatorn
import timeit
import sys

sys.setrecursionlimit(100000)
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()


def MLLGfuncsparse2(T, N, tau, h, m, rho, L, tolgmres, eps, mu, sig, theta, alpha, Ce):  # ,J,m0,E0,H0):
    print('h is ', h, ' N is', N)

    (Lhs,rhsLLGH, M0,mj,Ej,Hj,J,nBEM,nFEM,nMAG,trace_space,VV,V3,CQweights)=assLHS( tau, h, m, eps, mu, sig,rho,L,N)


    # Coefficient vector [E,H,phi,psi]
    sol = np.concatenate([Ej.vector(), Hj.vector(), np.zeros(nBEM),np.zeros(nBEM)])
    randkoeff = np.zeros([int(N + 1), 2 * nBEM])  # storage variable for boundary coefficients
    dtmpsiko = np.zeros([int(N + 1), 2 * nBEM])
    mkoeff = np.zeros([int(N + 1), nMAG])
    Ekoeff = np.zeros([int(N + 1), nFEM])
    Hkoeff = np.zeros([int(N + 1), nFEM])

    randkoeff[0, :] = np.zeros(2*nBEM)
    Ekoeff[0, :] = np.real(Ej.vector())
    Hkoeff[0, :] = np.real(Hj.vector())
    mkoeff[0, :] = np.real(mj.vector())

    ########## Time Stepping ##################################################################

    for j in range(0, int(N)):  # time stepping: update from timestep t_j to j+1

        ##### LLG update #########################################################################

        start6 = timeit.default_timer()

        (v, lam) = TrialFunctions(VV)
        (phi, mue) = TestFunctions(VV)
        lhs1 = ((alpha * inner(v, phi) + inner(cross(mj, v), phi) + Ce * tau * theta * inner(nabla_grad(v), nabla_grad(phi))) * dx + inner(dot(phi, mj), lam) * dx + inner(dot(v, mj), mue) * dx)
        rhs1 = (-Ce * inner(nabla_grad(mj), nabla_grad(phi)) + 1.0 * inner(Hj, phi)) * dx

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

        boundary_rhs = np.zeros(2 * nBEM)

        for kk in range(1, j +2):  # Convolution, start bei 1, da psiko(0)=0
            boundary_rhs[:] += np.real(CQweights[j + 1 - kk].dot(dtmpsiko[kk, :]))

        ##### Right hand side  #####
        Rhs = np.concatenate([eps / tau * M0 * Ej.vector() - (1 - (j / N)) * M0 * J.vector(),
                              mu / tau * M0 * Hj.vector() - 1.0 * mu * rhsLLGH.transpose() * v.vector(),
                              np.zeros(2 * nBEM) - 1.0 * boundary_rhs])  # - stimmt, da drei minus: unser B=-B, auf andere seite, und -1 durchmultipliziert

        stop4 = timeit.default_timer()
        print(' Time for Convolution in time step ', j, ': ', stop4 - start4)


        # Solution of Lhs=Rhs with gmres
        start5 = timeit.default_timer()
        it_count = 0
        def count_iterations(x):
            nonlocal it_count
            it_count += 1

        sol, info = gmres(Lhs, Rhs, tol=tolgmres, callback=count_iterations, x0=sol)

        if (info > 0):
            print("Failed to converge after " + str(info) + " iterations")
        else:
            print("Solved system " + str(j) + " in " + str(it_count) + " iterations." )

        stop5 = timeit.default_timer()
        print(' Time for gmres                    ', j, ': ', stop5 - start5)

        #### Storage and end of loop ###################
        Ej.vector()[:] = sol[:nFEM]  # coefficients for next timestep
        Hj.vector()[:] = sol[nFEM:2*nFEM]

        mkoeff[j+1, :] = np.real(mj.vector()[:]) #storage
        Ekoeff[j+1, :] = np.real(Ej.vector()[:])
        Hkoeff[j+1, :] = np.real(Hj.vector()[:])
        randkoeff[j + 1, :] = np.real(sol[2 * nFEM:])

    return (mkoeff, Ekoeff, Hkoeff, randkoeff[:, :nBEM], randkoeff[:, nBEM:2 * nBEM])

def assLHS( tau, h, m, eps, mu, sig,rho,L,N):
    start = timeit.default_timer()
    meshh = UnitCubeMesh(h, h, h)

    def dlt(z):
        # BDF1
        return 1.0 - z
        # BDF2
        # return 1.0-z+0.5*(1.0-z)**2
    # approximaion spaces
    # m   = Lagrange1
    # E,H = N1curl
    # gamma_TE, gamma_TH = trace_space = RT
    # phi ~ gamma_TH = BC
    # psi ~ -gamma_TE = RWG

    Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), 1, dim=3)
    V3 = FunctionSpace(meshh, Pr3)
    Pr = FiniteElement('P', meshh.ufl_cell(), 1)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(meshh, element)

    Xr = FiniteElement("N1curl", meshh.ufl_cell(), 1)
    X = FunctionSpace(meshh, Xr)

    trace_space, trace_matrix = n1curl_to_rt0_tangential_trace(X)  # trace space and restriction matrix


    nBEM = trace_space.global_dof_count  # DOFs
    nFEM = X.dim()
    nMAG = V3.dim()


    (mj,Ej,Hj,J)= initialdata(X,V3)

    (Lhs,M0,rhsLLGH)=assLHS2(X,V3,mu,trace_matrix,dlt,m,trace_space,eps,tau,sig,nFEM,nBEM)

    stop = timeit.default_timer()


    CQweights= assCQw(L,tau,m,mu,eps,trace_space.grid,dlt,rho,N,stop,start)

    return (Lhs,rhsLLGH, M0,mj,Ej,Hj,J,nBEM,nFEM,nMAG,trace_space,VV,V3,CQweights)

def initialdata(X,V3):
    # Initial Data and Input functions
    class MyExpression1(Expression):
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

    class MyExpression2(Expression):
        def eval(self, value, x):
            sqnx = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)
            A = (1 - 2 * sqnx ** 0.5) ** 4 / 4
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


    #minit = MyExpression1(degree=1)
    #mj= project(minit, X)
    mj = interpolate(Expression(['1.0', '0.0', '0.0'], degree=1), V3)
    mj = mj / sqrt(dot(mj, mj))
    mj = project(mj, V3, solver_type='cg')
    Ej = project(Expression(['0.00', '0.0', '0.0'], degree=1), X)
    Hj = project(Expression(['0.0', '0.0', '0.0'], degree=1), X)
    Jinit = MyExpression2(degree=1)
    J = project(Jinit, X)
    #J=project(Expression(['100.00','000.0','00.0'],degree=1),X)
    return(mj,Ej,Hj,J)

def assLHS2(X,V3,mu,trace_matrix,dlt,m,trace_space,eps,tau,sig,nFEM,nBEM):
    bc_space = bempp.api.function_space(trace_space.grid, "BC", 0)  # domain spaces
    rwg_space = bempp.api.function_space(trace_space.grid, "B-RWG", 0)
    snc_space = bempp.api.function_space(trace_space.grid, "B-SNC", 0)  # dual to range spaces
    rbc_space = bempp.api.function_space(trace_space.grid, "RBC", 0)
    brt_space = bempp.api.function_space(trace_space.grid, "B-RT", 0)

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

    cald = 1.0 / mu /(dlt(0)) ** (m) * operatorn.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(0) / tau).weak_form()  # calderon operator

    mass2 = bempp.api.operators.boundary.sparse.identity(brt_space, bc_space, snc_space).weak_form()
    mass1 = bempp.api.operators.boundary.sparse.identity(brt_space, rwg_space, rbc_space).weak_form()

    # Definition coupled 4x4 matrix
    blocke1 = np.ndarray([4, 4], dtype=np.object)
    blocke1[0, 0] = (eps / tau + sig) * M0
    blocke1[0, 2] = -(0.5 / mu) * trace_op.adjoint() * mass1.transpose()
    blocke1[0, 1] = -1.0 * D
    blocke1[1, 2] = np.zeros((nFEM, nBEM))
    blocke1[1, 0] = 1.0 * D
    blocke1[0, 3] = np.zeros((nFEM, nBEM))
    blocke1[1, 1] = mu / tau * M0
    blocke1[1, 3] = -0.5 * trace_op.adjoint() * mass2.transpose()

    blocke1[2, 0] = 0.5 / mu * mass1 * trace_op
    blocke1[2, 2] = 1.0 / mu * np.sqrt(mu / eps) * cald[0, 1]  # Calderon* \partial_t^m phiko =
    blocke1[3, 0] = np.zeros((nBEM, nFEM))
    blocke1[2, 3] = -cald[0, 0]
    blocke1[2, 1] = np.zeros((nBEM, nFEM))
    blocke1[3, 2] = cald[1, 1]
    blocke1[3, 1] = 0.5 * mass2 * trace_op
    blocke1[3, 3] = -mu * np.sqrt(eps / mu) * cald[1, 0]

    Lhs = bempp.api.BlockedDiscreteOperator(np.array(blocke1))
    return(Lhs,M0,rhsLLGH)

def assCQw(L,tau,m,mu,eps,trace_spacegrid,dlt,rho,N,stop,start):
    start2 = timeit.default_timer()
    # Definition of Convolution Quadrature weights
    storblock = np.ndarray([2, 2], dtype=np.object)  # dummy variable
    wei = np.ndarray([int(np.floor(L/2)+1)], dtype=np.object)  # dummy array of B(zeta_l)(zeta_l)**(-m)
    CQweights = np.ndarray([int(N + 1)], dtype=np.object)  # array of the weights CQweights[n]~B_n

    for ell in range(0, int(np.floor(L/2)+1)):  # CF Lubich 1993 On the multistep time discretization of linearinitial-boundary value problemsand their boundary integral equations, Formula (3.10)
        cald = (dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau) ** (-m) * 1.0 / mu *maxwell.multitrace_operator(trace_spacegrid, 1j * sqrt(mu * eps) * dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau).weak_form()
        storblock[0, 0] = 1.0 / mu * np.sqrt(mu / eps) * cald[0, 1]
        storblock[0, 1] = -cald[0, 0]
        storblock[1, 0] = cald[1, 1]
        storblock[1, 1] = -mu * np.sqrt(eps / mu) * cald[1, 0]
        wei[ell] = bempp.api.BlockedDiscreteOperator(np.array(storblock))
    stop2 = timeit.default_timer()
    print('Time for initial data and LHS: ', stop - start, ' Time for Calderon evaluation: ', stop2 - start2)

    start3 = timeit.default_timer()
    for n in range(0, int(N + 1)):
        CQweights[n] = wei[0]  # Fourier Transform
        for ell in range(1, int(np.ceil(L/2)-1)+1): # it is wei(L-d)=complconj(wei(d))
            CQweights[n] = CQweights[n] + 2*np.exp(-2.0 * np.pi * 1j * n * ell / L) * wei[ell]
        if not (L%2):
            CQweights[n] = CQweights[n] + (-1)**n * wei[int(L/2)]  #later only realpart is used
        CQweights[n] = rho ** (-n) / L * CQweights[n]

    stop3 = timeit.default_timer()
    print(' Time for FT: ', stop3 - start3)

    return CQweights
