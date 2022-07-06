from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
from fenics import *
# from fenics_rhs import FenicsRHS
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import gmres
from bempp.api.external.fenics import FenicsOperator
#from bempp.api import as_matrix
from maxwellrt0 import * 
from bempp.api.operators.boundary import maxwell
import timeit
#from MLLGfuncsparse2 import *
#from MLLGfuncsparse import *
from MLLGfuncsparse3 import *
#from Merrortest import *

set_log_level(31)


h0 = 1
N0 = 20.0
# mult=np.asarray([1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,512.0])
mult = np.asarray([1.0])#, 2.0, 4.0, 8.0, 16.0])  # ,32.0])#,64.0])#,128.0])
# mult = np.asarray([1.0, 2.0,4.0,8.0,16.0,32.0,64.0,128.0])

#hmult=np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
hmult=np.asarray([1,2,3,4,5,8,10,12,14,16,18])
hmult=np.asarray([1,2,3,4,5,8,10,12])#
#hmult=np.asarray([1,2,4])#,4,6
#hmult = np.asarray([1,2])
refmult = 1.0
hrefadd = 0.0
Nref = np.amax(mult) * refmult * N0
href = np.amax(hmult)  * h0+hrefadd
tolgmres = 10 ** (-8)  # tolerance gmres

Nvec = N0 * mult
hvec = h0 * hmult
m = 0  # CQ approximation parameters
rho = tolgmres ** (0.5 * np.power(Nvec, -1))
L = 1 * Nvec

rhoref = tolgmres ** (0.5 * (Nref ** (-1)))
Lref = 1 * Nref

T = 0.5
#T=0.0625
#T=1.0
theta = 1.0

eps = 1.0
mu = 1.0
sig = 1.3
alpha = 1.4
Ce = 1.5

tauref = float(T) / Nref
tau = np.zeros(len(mult))

meshh = np.ndarray([len(hmult)], dtype=np.object)
X = np.ndarray([len(hmult)], dtype=np.object)
V3 = np.ndarray([len(hmult)], dtype=np.object)
rwg_space = np.ndarray([len(hmult)], dtype=np.object)
nMAG = np.zeros(len(hmult))
hlg= np.zeros(len(hmult))
nFEM = np.zeros(len(hmult))
nBEM = np.zeros(len(hmult))
for k in range(0, len(hmult)):
    meshh[k] = UnitCubeMesh(int(hvec[k]), int(hvec[k]), int(hvec[k]))
    hlg[k]= meshh[k].hmax()
    Xr = FiniteElement("N1curl", meshh[k].ufl_cell(), 1)
    X[k] = FunctionSpace(meshh[k], Xr)
    Pr3 = VectorElement('Lagrange', meshh[k].ufl_cell(), 1, dim=3);
    V3[k] = FunctionSpace(meshh[k], Pr3)
    trace_space, trace_matrix = nc1_tangential_trace(X[k])
    rwg_space[k] = bempp.api.function_space(trace_space.grid, "RWG", 0)
    nMAG[k] = V3[k].dim()
    nFEM[k] = X[k].dim()
    nBEM[k] = rwg_space[k].global_dof_count
    # print(meshh[k].hmax())
meshhref = UnitCubeMesh(int(href), int(href), int(href))

Pr3 = VectorElement('Lagrange', meshhref.ufl_cell(), 1, dim=3);
V3ref = FunctionSpace(meshhref, Pr3)

Xr = FiniteElement("N1curl", meshhref.ufl_cell(), 1)
Xref = FunctionSpace(meshhref, Xr)
dd2 = project(Expression(['0.00', '0.0', '3.0'], degree=1), Xref)
trace_space, trace_matrix = nc1_tangential_trace(Xref)
rwg_spaceref = bempp.api.function_space(trace_space.grid, "RWG", 0)
if hrefadd>0:
    [mref, Eref, Href, phiref, psiref] = MLLGfuncsparse(T, Nref, tauref, int(href), m, rhoref, Lref, tolgmres, eps, mu, sig,
                                              theta, alpha, Ce)  # ,J,m0,E0,H0)
else:
    exSolutionCode = ("-1*(pow(x[0],3)-0.75*x[0])*sin(multF*M_PI*t/tend)","sqrt(1-1*pow(pow(x[0],3)-0.75*x[0],2))","-1*(pow(x[0],3)-0.75*x[0])*cos(multF*M_PI*t/tend)")
    dtexSolutionCode = ("-1*(pow(x[0],3)-0.75*x[0])*cos(multF*M_PI*t/tend)*multF*M_PI/tend","0","1*(pow(x[0],3)-0.75*x[0])*sin(multF*M_PI*t/tend)*multF*M_PI/tend")
err = np.zeros([len(mult), len(hmult)])  # range(0,len(mult))
errmax = np.zeros([len(mult), len(hmult)])
errE = np.zeros([len(mult), len(hmult)])  # range(0,len(mult))
errmaxE = np.zeros([len(mult), len(hmult)])
errH = np.zeros([len(mult), len(hmult)])  # range(0,len(mult))
errmaxH = np.zeros([len(mult), len(hmult)])
errPhi = np.zeros([len(mult), len(hmult)])  # range(0,len(mult))
errmaxPhi = np.zeros([len(mult), len(hmult)])
errPsi = np.zeros([len(mult), len(hmult)])  # range(0,len(mult))
errmaxPsi = np.zeros([len(mult), len(hmult)])
test = np.zeros([len(mult), len(hmult)])
for i in range(0, len(mult)):
    tau[i] = tauref * refmult * np.amax(mult) / mult[i]
    err[i] = 0.0
    errmax[i] = 0.0
    test[i] = 0.0

for i in range(0, len(mult)):
    errim=np.zeros(int(Nvec[i]+1))
    erriE=np.zeros(int(Nvec[i]+1))
    erriH=np.zeros(int(Nvec[i]+1))
    erriphi=np.zeros(int(Nvec[i]+1))
    erripsi=np.zeros(int(Nvec[i]+1))
    for k in range(0, len(hmult)):
        mreffkt = Function(V3ref)
        dd=Function(V3ref)
        Ereffkt = Function(Xref)
        Hreffkt = Function(Xref)
        maprfkt = Function(V3[k])
        Eaprfkt = Function(X[k])
        Haprfkt = Function(X[k])
        [mapr, Eapr, Hapr, phiapr, psiapr] = MLLGfuncsparse(T, Nvec[i], tau[i], int(hvec[k]), m, rho[i], L[i], tolgmres, eps,
                                                      mu, sig, theta, alpha, Ce)  # ,J,m0,E0,H0)
        for j in range(0, int(Nvec[i]) + 1):
            if hrefadd>0:
                mreffkt.vector()[:] = mref[int(j * np.amax(mult) * refmult / mult[i]), :]
                Ereffkt.vector()[:] = Eref[int(j * np.amax(mult) * refmult / mult[i]), :]
                Hreffkt.vector()[:] = Href[int(j * np.amax(mult) * refmult / mult[i]), :]
                #Phiref = bempp.api.GridFunction(rwg_spaceref,
                #                                coefficients=phiref[int(j * np.amax(mult) * refmult / mult[i]), :])
                #Psiref = bempp.api.GridFunction(rwg_spaceref,
                #                                coefficients=psiref[int(j * np.amax(mult) * refmult / mult[i]), :])
                maprfkt.vector()[:] = mapr[j, :]
                Eaprfkt.vector()[:] = Eapr[j, :]
                Haprfkt.vector()[:] = Hapr[j, :]
                #Phiapr = bempp.api.GridFunction(rwg_space[k], coefficients=phiapr[j, :])
                #Psiapr = bempp.api.GridFunction(rwg_space[k], coefficients=psiapr[j, :])

                # if (k == len(hmult) - 1) and (hrefmult == 1):
                #     Phiapr = bempp.api.GridFunction(rwg_spaceref, coefficients=phiapr[j, :])
                #     Psiapr = bempp.api.GridFunction(rwg_spaceref, coefficients=psiapr[j, :])
                #     ddPhi = Phiapr - Phiref
                #     ddPsi = Psiapr - Psiref
                #     errPhi[i, k] = errPhi[i, k] + tau[i] * ddPhi.l2_norm() ** 2
                #     errPsi[i, k] = errPsi[i, k] + tau[i] * ddPsi.l2_norm() ** 2
                #     errmaxPhi[i, k] = np.amax([errmaxPhi[i, k], ddPhi.l2_norm()])
                #     errmaxPsi[i, k] = np.amax([errmaxPsi[i, k], ddPsi.l2_norm()])
                #     # print('L2-norm:', ddPhi.l2_norm(), 'Errornorm:',
                #     #      np.amax(np.abs(phiapr[j, :] - phiref[int(j * np.amax(mult) * refmult / mult[i])])))
                #     # print("Phi:",j * np.amax(mult) * refmult / mult[i], " is ",ddPhi.l2_norm())
                #     # print("Psi:", j," is ",ddPsi.l2_norm())
                # else:
                #     errPhi[i, k] = 2.0
                #     errmaxPhi[i, k] = 2.0
                #     errPsi[i, k] = 2.0
                #     errmaxPsi[i, k] = 2.0  # np.amax([errmaxPhi[i, k], dd.l2_norm()])
                hilfe= interpolate(maprfkt,V3ref)
                dd.vector()[:] = mreffkt.vector()[:] - hilfe.vector()[:]
                errmax[i, k] = np.amax([errmax[i, k], norm(dd, 'H1')])#np.amax([errmax[i, k], errornorm(Ereffkt, Eaprfkt)])#
                #print("m", j," in H1: ",norm(dd, 'H1') , j," in L2: ",norm(dd, 'L2')  )
                hilfe2= interpolate(Eaprfkt,Xref)
                dd2.vector()[:] = Ereffkt.vector()[:] -  hilfe2.vector()[:]
                errmaxE[i, k] = np.amax([errmaxE[i, k], norm(dd2, 'Hcurl')])
                #print("E", j, " in Hcurl: ", norm(dd2, 'Hcurl'), j, " in L2: ", norm(dd2, 'L2'))
                # print('L2-norm:',norm(dd2,'l2'),'Errornorm:' ,errornorm(Ereffkt, Eaprfkt))
                #errmaxE[i, k] = np.amax([errmaxE[i, k], errornorm(Ereffkt, Eaprfkt)])
                # errmaxH[i, k] = np.amax([errmaxH[i, k], errornorm(Hreffkt, Haprfkt)])
                hilfe2 = interpolate(Haprfkt, Xref)
                dd2.vector()[:] = Hreffkt.vector()[:] - hilfe2.vector()[:]
                errmaxH[i, k] = np.amax([errmaxH[i, k], norm(dd2, 'Hcurl')])
                #print("H", j, " in Hcurl: ", norm(dd2, 'Hcurl'), j, " in L2: ", norm(dd2, 'L2'))
                # print(np.amax(np.abs(Hreffkt.vector()[:])))
                # print(np.amax(np.abs(Haprfkt.vector()[:])))
            else:
                class MyExpressionE(UserExpression):
                    def eval(self, value, x):
                        s1=sin(np.pi*x[0])
                        s2=sin(np.pi*x[1])
                        s3=sin(np.pi*x[2])
                        s1s=s1**2
                        s2s=s2**2
                        s3s=s3**2
                        value[0] = s1s*s2s*s3s    * (j*tauref)**(2)
                        value[1] = 0
                        value[2] = 0 
                    def value_shape(self):
                        return (3,)
                class MyExpressionH(UserExpression):
                    def eval(self, value, x):
                        s1=sin(np.pi*x[0])
                        s2=sin(np.pi*x[1])
                        s3=sin(np.pi*x[2])
                        s1s=s1**2
                        s2s=s2**2
                        s3s=s3**2
                        value[0] = 0
                        value[1] =-2*np.pi*s1s*s2s*s3*cos(np.pi*x[2])  *(j*tauref)**(3)/3 /mu
                        value[2] = 2*np.pi*s1s*s2*cos(np.pi*x[1])*s3s  *(j*tauref)**(3)/3 /mu
                    def value_shape(self):
                        return (3,)    
                maprfkt.vector()[:] = mapr[j, :]
                Eaprfkt.vector()[:] = Eapr[j, :]
                Haprfkt.vector()[:] = Hapr[j, :] 
                Phiapr = bempp.api.GridFunction(rwg_space[k], coefficients=phiapr[j,:])
                Psiapr = bempp.api.GridFunction(rwg_space[k], coefficients=psiapr[j,:])
                #error Magnetization 
                errim[j]=errornorm(Expression(exSolutionCode,degree=1,t=j*tauref,tend=10*T,multF=3),maprfkt, 'H1')
                errmax[i,k] = np.amax([errmax[i,k], errim[j]])
                #errmax[i, k] = np.amax([errmax[i, k], errornorm(MyExpressionE(degree=1),Eaprfkt,'L2')])
                #error E and H
                erriE[j]=errornorm(MyExpressionE(degree=1),Eaprfkt,'Hcurl')
                errmaxE[i, k] = np.amax([errmaxE[i, k], erriE[j] ])
                erriH[j]=errornorm(MyExpressionH(degree=1),Haprfkt,'Hcurl')
                errmaxH[i, k] = np.amax([errmaxH[i, k], erriH[j]])
                #error Phi and Psi
                erriphi[j]=Phiapr.l2_norm()#errornorm(,Phiapr)
                errmaxPhi[i, k] = np.amax([errmaxPhi[i, k], erriphi[j] ]) 
                erripsi[j]=Psiapr.l2_norm()#errornorm(,Psiapr)      
                errmaxPsi[i, k]= np.amax([errmaxPsi[i, k], erripsi[j]])
        plt.plot(errim)
        plt.show()
        plt.plot(erriE)
        plt.plot(erriH)
        plt.show()
        plt.plot(erriphi)
        plt.plot(erripsi)
        plt.show()
print('Results for m')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        err[i, k] = sqrt(err[i, k])
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmax[i, k] / errmax[i, k - 1]) / np.log(hlg[k] / hlg[k - 1]))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmax[i, k] / errmax[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nMAG[k], meshh[k].hmax(), err[ i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tau[i], nMAG[k], meshh[k].hmax(), errmax[i, k], dd)
print('Results for E')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        errE[i, k] = sqrt(errE[i, k])
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmaxE[i, k] / errmaxE[i, k - 1]) / np.log(hlg[k] / hlg[k - 1]))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxE[i, k] / errmaxE[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nFEM[k], meshh[k].hmax(), errE[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tau[i], nFEM[k], meshh[k].hmax(), errmaxE[i, k], dd)
print('Results for H')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        errH[i, k] = sqrt(errH[i, k])
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str( np.log(errmaxH[i, k] / errmaxH[i, k - 1]) / np.log(hlg[k] / hlg[k - 1]))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxH[i, k] / errmaxH[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        # print(tau[i], nFEM[k], meshh[k].hmax(), errH[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tau[i], nFEM[k], meshh[k].hmax(), errmaxH[i, k], dd)
print('Results for phi')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        errPhi[i, k] = sqrt(errPhi[i, k])
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str(
                3.0 * np.log(errmaxPhi[i, k] / errmaxPhi[i, k - 1]) / np.log((nBEM[k]) / (nBEM[k - 1])))#

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxPhi[i, k] / errmaxPhi[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        #print(tau[i], nBEM[k], meshh[k].hmax(), errPhi[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tau[i], nBEM[k], meshh[k].hmax(), errmaxPhi[i, k], dd)
        # print (tau[i],nMAG[k],test[i,k])
print('Results for psi')
for i in range(0, len(mult)):
    for k in range(0, len(hmult)):
        errPsi[i, k] = sqrt(errPsi[i, k])
        dd = ' '
        if (k > 0):
            dd = dd + 'EOCh:' + str(
               3.0 * np.log(errmaxPsi[i, k] / errmaxPsi[i, k - 1]) / np.log((nBEM[k]) / (nBEM[k - 1])))

        if (i > 0):
            dd = dd + '  EOCtau:' + str(np.log(errmaxPsi[i, k] / errmaxPsi[i - 1, k]) / np.log(tau[i - 1] / tau[i]))
        #print(tau[i], nBEM[k], meshh[k].hmax(), errPhi[i, k])  # 'EOCtau:',  np.log(errmax[i,k]/errmax[i-1,k])/np.log(tau[k]/tau[k-1]) , 'EOCh:', 3.0* np.log(errmax[i,k]/errmax[i,k-1])/np.log(nMAG[k]/nMAG[k-1]))
        print(tau[i], nBEM[k], meshh[k].hmax(), errmaxPsi[i, k], dd)
        # print (tau[i],nMAG[k],test[i,k])

if 1:
    err = errmax.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error')
    plt.legend((r'$err(\tau)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    #plt.savefig('books_read.png')
    plt.show()
    err = errmaxE.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error')
    plt.legend((r'$err(\tau)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    #plt.savefig('books_read.png')
    plt.show()
    err = errmaxH.T;
    x = hlg   # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]   # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('mesh size')
    plt.ylabel('Error')
    plt.legend((r'$err(\tau)$', r'$O(h)$'), loc='lower right')
    plt.tight_layout()
    #plt.savefig('books_read.png')
    plt.show()
    err = errmaxPhi.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error')
    plt.legend((r'$err(\tau)$', r'$O(\tau)$'), loc='lower right')
    plt.tight_layout()
    plt.savefig('books_read.png')
    plt.show()
    err = errmaxPsi.T;
    x = hlg  # 0.25 * np.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    od1 = err[0] * 1.2 * hlg / hlg[0]  # *np.asarray([1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625])
    plt.loglog(x, err, marker='o')
    plt.loglog(x, od1, ls='--')
    plt.xlabel('Time step size')
    plt.ylabel('Error')
    plt.legend((r'$err(\tau)$', r'$O(\tau)$'), loc='lower right')
    plt.tight_layout()
##   plt.savefig('books_read.png')
    plt.show()

'''for j in range(0,int(Nref/refmult)):
    print j*tauref

        if np.remainder(float(j+1)/refmult*mult[i],np.amax(mult)) == 0: 
            print(float(j+1)/refmult*tau[i],mult[i])
            #print(tau[i])

            dd=project(m0[i]-mref,V3)
            err[i]= err[i]+tau[i]* norm(dd,'l2')**2 
            errmax[i]= np.amax([errmax[i],norm(dd,'l2')])
            test[i]=test[i]+1
for i in range(0,len(mult)):
        err[i]=sqrt(err[i])
        print (tau[i],err[i])
        print (tau[i],errmax[i])
        print (tau[i],test[i])

'''
