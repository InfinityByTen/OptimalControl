import pyamg
import numpy as np
from scipy.sparse.linalg import spsolve, LinearOperator, minres, eigs, inv
from scipy.sparse import bmat, spdiags, block_diag

def is_diagonal_matrix(A):
    '''Checks the sparse matrix A is really just a diagonal matrix.
    '''
    offsets = A.todia().offsets
    return len(offsets) == 1 and offsets[0] == 0

def GetSplit(v, n, m):
    xy = v[:n:1]
    xu = v[n:m+n:1]
    xp = v[m+n:m+2*n:1]
    return [xy,xu,xp]


def BlockDiagonal(LinSys,v):
    n = LinSys.n
    m = LinSys.m

    [xy,xu,xp] = GetSplit(v,n,m)

    Mcsr = LinSys.M.tocsr()
    ml = pyamg.ruge_stuben_solver(Mcsr)
    by = ml.solve(xy, tol=1e-10,maxiter=3)
    # by = spsolve(M,xy)

    # PyAMG fails if the input matrix is diagonal (such as the case for
    # discontinuous Galerkin input). Intercept this case.
    # if is_diagonal_matrix(LinSys.D):
    #     bu = spsolve(LinSys.ld * LinSys.D.diagonal(),xu)
    # else:
    ml = pyamg.ruge_stuben_solver(LinSys.ld*LinSys.D)
    bu = ml.solve(xu, tol=1e-10, maxiter=3)

    ml = pyamg.ruge_stuben_solver(LinSys.LT)
    bp_t = ml.solve(xp, tol=1e-10,maxiter=3)
    ml = pyamg.ruge_stuben_solver(LinSys.L)
    bp = ml.solve(LinSys.M.dot(bp_t), tol=1e-10,maxiter=3)
    # bp = spsolve( L, M.dot(spsolve(LT,xp)))

    return np.hstack([by, bu, bp])

def Constraint(LinSys,v):
    n = LinSys.n
    m = LinSys.m

    [xy,xu,xp] = GetSplit(v,n,m)

    BTcsr = LinSys.B.transpose().tocsr()
    # bp = spsolve(BTcsr,-xu)
    # Mcsr = M.tocsr()
    ml = pyamg.ruge_stuben_solver(BTcsr)
    bp = ml.solve(-xu, tol=1e-10,maxiter=3)

    # by = spsolve(LinSys.ld*LinSys.LT, LinSys.B.dot( spsolve(LinSys.L, xy - LinSys.LT.dot(bp)) ) )
    ml = pyamg.ruge_stuben_solver(LinSys.L)
    by_t = ml.solve(xy - LinSys.LT.dot(bp), tol=1e-10,maxiter=3)
    ml = pyamg.ruge_stuben_solver(LinSys.ld*LinSys.LT)
    by = ml.solve(LinSys.B.dot(by_t), tol=1e-10,maxiter=3)

    # bu = spsolve(LinSys.B, LinSys.L.dot(by) - xp )
    Bcsr = LinSys.B.tocsr()
    ml = pyamg.ruge_stuben_solver(Bcsr)
    bu = ml.solve( LinSys.L.dot(by) -xp, tol=1e-10,maxiter=3)

    return np.hstack([by, bu, bp])



def BlockDiagonalOperator(LinSys,v):
    n = LinSys.n
    m = LinSys.m

    [xy,xu,xp] = GetSplit(v,n,m)

    LMcsr = (LinSys.L+LinSys.M).tocsr()
    ml = pyamg.ruge_stuben_solver(LMcsr)
    by = ml.solve(xy, tol=1e-10,maxiter=2)
    # by = spsolve(L+M,xy)

    # PyAMG fails if the input matrix is diagonal (such as the case for
    # discontinuous Galerkin input). Intercept this case.

    # KryPy fails in this solve, outputs a matrix instead of a vector.
    # Check needed.
    # print "xu has shape",xu.shape
    if is_diagonal_matrix(LinSys.D):
        bu = xu / (LinSys.D.diagonal())
    else:
        ml = pyamg.ruge_stuben_solver(LinSys.D)
        bu = ml.solve(xu, tol=1e-10, maxiter=2)
    # bu = spsolve(ld*D,xu)
    # print "bu has shape",bu.shape

    LMcsr = (LinSys.LT+LinSys.M).tocsr()
    ml = pyamg.ruge_stuben_solver(LMcsr)
    bp = ml.solve(xp, tol=1e-10,maxiter=2)
    # bp = spsolve(LT,xp)

    # print by.shape, bu.shape, bp.shape

    return np.hstack([by, bu, bp])

def IndefiniteHerzog(LinSys,v, anorm, ksquare):
    n = LinSys.n
    m = LinSys.m

    sigma = 0.9/anorm
    tau = 1.2/ksquare

    [xy,xu,xp] = GetSplit(v,n,m)

    byPrime = spsolve((1/sigma)*(LinSys.L + LinSys.M), xy)
    buPrime = spsolve((1/sigma)*LinSys.D, xu)

    xpPrime = xp - (-LinSys.L.dot(xy) + LinSys.B.dot(xu))
    bp = spsolve(-(sigma/tau)*(LinSys.L + LinSys.M), xpPrime)

    xyPrime = xy - (-LinSys.LT.dot(bp))
    xuPrime = xu - (LinSys.B.transpose().dot(bp))

    by = spsolve((1/sigma)*(LinSys.LT + LinSys.M), xyPrime)
    bu = spsolve((1/sigma)* LinSys.D, xuPrime)

    return np.hstack([by, bu, bp])

def BlockDiagonalOperator2(LinSys,v, ac, Pa):
    n = LinSys.n
    m = LinSys.m

    xy = v[:n:1]
    xu = v[n:m+n:1]
    xp = v[m+n:m+2*n:1]
    xm = v[2*n+m:2*n+m+ac:1]

    LMcsr = (LinSys.L + LinSys.M).tocsr()
    ml = pyamg.ruge_stuben_solver(LMcsr)
    by = ml.solve(xy, tol=1e-10,maxiter=3)
    # by = spsolve((LinSys.L+ LinSys.M),xy)

    if is_diagonal_matrix(LinSys.D):
        bu = xu / (LinSys.D.diagonal())
    else:
        ml = pyamg.ruge_stuben_solver(LinSys.D)
        bu = ml.solve(xu, tol=1e-10, maxiter=3)
    # bu = spsolve(LinSys.D,xu)

    LMcsr = (LinSys.LT+LinSys.M).tocsr()
    ml = pyamg.ruge_stuben_solver(LMcsr)
    bp = ml.solve(xp, tol=1e-10,maxiter=3)
    # bp = spsolve((LinSys.LT+LinSys.M),xp)
    
    Aux = bmat([[LinSys.D, Pa.transpose()],[Pa, None]], format='csr')
    vec_aux = -np.hstack([np.zeros(m), xm ])
    bm = spsolve(Aux, vec_aux)[m:m+ac:1]

    # Paux = bmat([[spdiags(LinSys.D.diagonal(),0,m,m), Pa.transpose()],[Pa, None]])
    # bm0 = spsolve(Paux,vec_aux)
    # bm = minres(Aux, vec_aux,x0=bm0, M=Paux)[m:m+ac:1]
    return(np.hstack([by, bu, bp, bm]))

def BlockDiagonal2(LinSys,v, ac, Pa):

    n = LinSys.n
    m = LinSys.m

    xy = v[:n:1]
    xu = v[n:m+n:1]
    xp = v[m+n:m+2*n:1]
    xm = v[2*n+m:2*n+m+ac:1]

    # by = spsolve(LinSys.M,xy)
    Mcsr = (LinSys.M).tocsr()
    ml = pyamg.ruge_stuben_solver(Mcsr)
    by = ml.solve(xy, tol=1e-10,maxiter=3)

    # bu = spsolve(LinSys.ld*LinSys.D, xu)
    ml = pyamg.ruge_stuben_solver(LinSys.ld * LinSys.D)
    bu = ml.solve(xu, tol=1e-10, maxiter=3)

    # bp = spsolve( LinSys.L, LinSys.M.dot(spsolve(LinSys.LT,xp)))
    ml = pyamg.ruge_stuben_solver(LinSys.LT)
    bp_t = ml.solve(xp, tol=1e-10,maxiter=3)
    ml = pyamg.ruge_stuben_solver(LinSys.L)
    bp = ml.solve(LinSys.M.dot(bp_t), tol=1e-10,maxiter=3)

    Aux = bmat([[LinSys.D, Pa.transpose()],[Pa, None]], format='csr')
    vec_aux = -np.hstack([np.zeros(m), xm ])
    bm = spsolve(Aux, vec_aux)[m:m+ac:1]
    # bm = spsolve(Pa*LinSys.Dinv*Pa.transpose(), xm)


    return(np.hstack([by, bu, bp, bm]))

# def PPCGsub(LinSys, xm, ac):
#     n = LinSys.n
#     m = LinSys.m

#     Dinv = spdiags(1.0 / LinSys.D.diagonal()0,m,m)

#     def PC(v):
#         v1 = v[:m:1]
#         v2 = v[m:m+ac:1]
#         pv2 = spsolve(Pa * Dinv * Pa.transpose(), Pa * Dinv.dot(v1) - v2)
#         pv1 = Dinv.dot(v1 - Pa.transpose().dot(pv2))
#         return np.hstack([pv1, pv2])

#     x0 = spsolve()
#     Aux = bmat([[LinSys.D, Pa.transpose()],[Pa, None]], format='csr')

#     ans = cgs(sys.A,sys.b,M=PC, maxiter = 500, tol=1e-10)[0]

