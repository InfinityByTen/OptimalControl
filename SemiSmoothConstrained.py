from dolfin import *
from scipy.sparse import csr_matrix, identity,bmat,hstack,vstack,spdiags, eye, block_diag
from scipy.sparse.linalg import spsolve,cg,minres,inv,LinearOperator,gmres
from mshr import *
import numpy as np
import time

from SemiSmoothNewton import SemiSmoothUnbounded, SemiSmoothBoxed
import ProblemSetup as PS

maxit = 9      # max iterations in semismooth Newton method

N     = 64 # number of grid points per dimension

# upper bound
ua = 1
# lower bound
ub = 12

# ld = 1e-6
Constrained = True

Plot = True

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def RemoveZeroRows(A):
    A_mat_csr = csr_matrix(A)
    num_nonzeros = np.diff(A_mat_csr.indptr)
    return A_mat_csr[num_nonzeros != 0]

def Projection_Ua_Ub(vec):
    return np.array(map(lambda x: ua if x< ua else ub if x >ub else x, vec))

def CheckResult(u, yn, pn):
    y = spsolve(OCprob.L, OCprob.B.dot(u))
    p = spsolve(OCprob.LT, (OCprob.M.dot(y)- OCprob.vec))
    # fullprint(p -pn)
    uex = Projection_Ua_Ub(-(1/OCprob.ld)*p )
    uerr = np.linalg.norm(uex - u)
    unorm = np.linalg.norm(u)
    yerr = np.linalg.norm(y - yn)
    perr = np.linalg.norm(p - pn)
    OCprob.ufun2.vector().set_local(uex)
    plot(OCprob.ufun2,title='control_uex')
    print uerr/unorm, yerr, perr

OCprob = PS.OptimalControlProblem(N)

# fullprint(OCprob.L.todense())
# fullprint(OCprob.LT.todense())
n = OCprob.n
m = OCprob.m

ans = SemiSmoothUnbounded(OCprob)

if Plot:
    un = ans[n:m+n:1]

    OCprob.ufun.vector().set_local(un)
    UnbddControlPlot = plot(OCprob.ufun,title='Unbounded Control', backend = "vtk")


# if Plot:
#     yn = ans[:n:1] 
#     OCprob.yfun.vector() [:] = yn
#     UnbddStatePlot = plot(OCprob.yfun, title="State")
#     UnbddErrorPlot = plot(OCprob.yfun-OCprob.yd,title="Error_un")

pn = ans[m+n:m+2*n:1]
mupun = ans[n:m+n:1]
# mupun = spsolve(OCprob.ld*OCprob.D,-OCprob.B.transpose().dot(pn))
# print ans[n:m+n:1] - mupun
# mun = -(inv(ld*D).dot(B.transpose().dot(pn)) + un)
# print mu - mun
# print mu, un
################################################################################
if Constrained:


    Plot = True

    # active sets for upper, lower bound
    Ap = np.zeros(m)
    Am = np.zeros(m)
    Ap_old = Ap
    Am_old = Am

    # set initial active sets
    Ap = ((mupun) > ub).astype(float)
    Am = ((mupun) < ua).astype(float)
    # print Ap, Am

    Ap_old,Am_old = Ap,Am

    Ix = identity(m,format='csr')

    # semi-smooth iteration
    for it in xrange(maxit):

        dx, Asp, Asm = SemiSmoothBoxed(Am, Ap, ua, ub, OCprob)
        print "Just after solve"


        Pa = RemoveZeroRows(Asp+Asm)
        # un  =dx[n:n+m:1]
        mua = Pa.transpose().dot(dx[2*n+m::1])
        # print mua.shape
        mun = mua + dx[n:n+m:1]
        # mun = dx[2*n+m::1] + dx[n:n+m:1]

        # update active sets
        Ap = ( mun >  ub).astype(float)
        Am = ( mun < ua).astype(float)
        # print Ap
        # print Am

        change = (Ap-Ap_old)+(Am-Am_old)
        update = len(change[change.nonzero()])
        print 'Iteration %d: %d points changed in active set' % (it+1,update)
        if update == 0: 
            break

        Ap_old,Am_old = Ap,Am


    yn = dx[0:n:1]
    if Plot:
        # yfun = Function(OCprob.Y)
        # yfun.vector()[:] = yn
        OCprob.yfun.vector() [:] = yn
        plot(OCprob.yfun-OCprob.yd,title="Error")

    un = dx[n:n+m:1]
    pn = dx[n+m:2*n+m:1]

    if Plot:
        # ufun = Function(OCprob.U)
        # ufun.vector().set_local(un)
        OCprob.ufun.vector().set_local(un)
        plot(OCprob.ufun,title='Box Control')

    CheckResult( un, yn, pn )

interactive()
