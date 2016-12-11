from scipy.sparse import csr_matrix, identity, bmat, hstack, vstack, spdiags, \
        eye, block_diag, spmatrix
from scipy.sparse.linalg import spsolve, minres, LinearOperator, gmres, inv, cgs, cg
import numpy as np
import time
import Preconditioners as PreCon
import krypy.linsys as kp
from LinSys import LinearSystem
import Solver 

# import ProblemSetup as PS

def is_diagonal_matrix(A):
    '''Checks the sparse matrix A is really just a diagonal matrix.
    '''
    offsets = A.todia().offsets
    return len(offsets) == 1 and offsets[0] == 0


def RemoveZeroRows(A):
    A_mat_csr = csr_matrix(A)
    num_nonzeros = np.diff(A_mat_csr.indptr)
    return A_mat_csr[num_nonzeros != 0]


def SemiSmoothUnbounded(Prob):

    n = Prob.n
    m = Prob.m
    ans = Solver.DirectSolver(Prob)
    # ansBDO = Solver.BlockDiagOperator(Prob)
    # ans = Solver.BlockDiag(Prob)
    # ansCON = Solver.Constraint(Prob)
    # ans = Solver.Indefinite(Prob) # Needs BP cg       Scaling available 
    # ans = Solver.BPSchur(Prob)    # Needs BP cg       Scaling? 
    # ans = Solver.BPPlus(Prob)     # Needs BP Minres   no scaling available
    # err = np.linalg.norm(ans[n:m+n:1] - ansCON[n:m+n:1])
    # print "error norm Direct-CON", err

    return ans


###############################################################################


def SemiSmoothBoxed(Am, Ap, ua, ub, Prob):

    m = Prob.m
    n = Prob.n

    Asp  = spdiags(Ap,0,m,m)
    Asm  = spdiags(Am,0,m,m)
    As = Asp +Asm
    Pa = RemoveZeroRows(As)

    muvec  = ub*Ap+ua*Am
    # print type(muvec)
    # print type(muvec[np.nonzero(muvec)])
    # arr = RemoveZeroRows(ub*Ap+ua*Am)
    # print "Test lagrange multiplier size ", arr.shape
    arr2 = muvec[np.nonzero(muvec)]
    print "Test lagrange multiplier size nonzero", arr2.shape
    ac = arr2.shape[0]

    Direct = False
    KrylovBD = True
    KrylovBDO = False

    # SystemC = LinearSystem(Prob, Direct=Direct,\
    #                             Constrained=True, Pa=Pa, arr2= arr2)
    # C = SystemC.A
    # b = SystemC.b

    if Direct is True:
        directStart = time.time()
        dx = Solver.DirectSolver(Prob, Constrained=True, Pa=Pa, arr2=arr2)
        # dx = Solver.DirectSolver(Prob, Constrained=True, Pa=As, arr2=muvec)
        directEnd = time.time()
        print "Direct solve complete, Time elapsed:", directEnd - directStart

    sys = LinearSystem(Prob,Direct = False, Constrained=True, 
         Pa = Pa, arr2 = arr2)

    if KrylovBDO is True:
        def pcc(v):
            '''
            Block Diagonal preconditioner Roland Herzog Style.
            Linear System formulation also borrowed from there.
            '''
            return PreCon.BlockDiagonalOperator2(Prob,v,ac, Pa)

        print "Will form a linear operator for preconditioner"
        linOpStart = time.time()
        PC = LinearOperator((2*n+m+ac,2*n+m+ac), matvec = pcc)
        linOpEnd = time.time()
        print "Linear operator build time =", linOpEnd - linOpStart

        print "Let's start minres for BDO operator, finally"
        minresStart = time.time()
        PI = identity(m) - As
        dx = minres(sys.A,sys.b,M=PC, maxiter = 500, show=True, tol=1e-6)[0]
        minresEnd = time.time()
        print "minres Time  = ", minresEnd - minresStart

    if KrylovBD is True:
        def pcc(v):
            '''
            Implementing Wathen's Block Diagonal preconditioner. 
            Linear system Roland Herzog type.
            '''
            return PreCon.BlockDiagonal2(Prob,v,ac, Pa)

        print "Will form a linear operator for preconditioner"
        linOpStart = time.time()
        PC = LinearOperator((2*n+m+ac,2*n+m+ac), matvec = pcc)
        linOpEnd = time.time()
        print "Linear operator build time =", linOpEnd - linOpStart

        print "Let's start minres for BD operator, finally"
        minresStart = time.time()
        PI = identity(m) - As
        dx = minres(sys.A,sys.b,M=PC, maxiter = 500, show=True, tol=1e-12)[0]
        minresEnd = time.time()
        print "minres Time  = ", minresEnd - minresStart


    return dx, Asp, Asm