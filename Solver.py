import numpy as np
from scipy.sparse.linalg import minres, cgs, spsolve, LinearOperator, eigs, inv
from scipy.sparse import bmat
import Preconditioners as PreCon
from LinSys import LinearSystem
import time
from _cg_inner import cgIP


# Preoncditioner Possibilities:
#                      1)   Constraint (Restricted to Square type?):: uses PPCG :: (can work with with Minres?)
#                      2)   Block Diagonal Operator (Herzog):: uses Precon Minres
#                      3)   Block Diagonal Wathen:: uses Precon Minres
#                      4)   Indefinite Schroebel:: uses BP CG
#                      5)   Bramble Pasciack Schur:: uses BP CG
#                      6)   Bramble Pasciack Plus:: uses BP Minres

# Preconditioners = {"Constraint":Constraint,                     #1
#                    "Block Diagonal Operator":BlockDiagOperator, #2
#                    "Block Diagonal":BlockDiag,                  #3
#                    "Indefinite": Indefinite,                    #4
#                    "Bramble Pasciack":BPSchur,                  #5
#                    "Bramble Pasciack Plus":BPPlus}              #6

num_iters = 0

def getLinearSystem(Prob, Direct=False, Constrained=False, 
                 Pa = None, arr2 = None):

    System = LinearSystem(Prob, Direct = Direct, Constrained = Constrained,\
                          Pa = Pa, arr2 = arr2)
    return System.A , System.b

def DirectSolver(Prob, Constrained=False, Pa=None, arr2=None):
    A,b = getLinearSystem(Prob, Direct=True, Constrained=Constrained, Pa=Pa,\
                            arr2= arr2)
    DStart = time.time()
    ans = spsolve(A,b)
    DEnd = time.time()
    print "Solve Time  = ", DEnd - DStart

    return ans

def Constraint(Prob):

    if Prob.B.shape != Prob.M.shape:
        raise ValueError("B block should be square matrix type for this\
                         preconditioner to work")

    A,b = getLinearSystem(Prob)

    def pc(v):
        return PreCon.Constraint(Prob,v)

    PC = LinearOperator(A.shape, matvec = pc)
    
    def callback(xk):
                global num_iters
                num_iters += 1
                print num_iters

    print "Let's start PPCG, finally"
    PPCGStart = time.time()
    x0 = PreCon.Constraint(Prob,b)
    # print np.linalg.norm(x0[n+m::1])
    # print x0[n+m::1]
    ans = cgs(A,b,x0 = x0, M=PC,maxiter=500, tol=1e-6,\
             callback = callback)[0]
    PPCGEnd = time.time()
    print "PPCG (cgs) Time  = ", PPCGEnd - PPCGStart

    return ans

def BlockDiagOperator(Prob):

    A,b = getLinearSystem(Prob)

    def pc(v):
        return PreCon.BlockDiagonalOperator(Prob,v)
    PC = LinearOperator(A.shape, matvec = pc)

    print "Let's start minres, finally"
    minresStart = time.time()
    # '''KryPy Run, fires up error with a matrix output for a preconditioner
    #     block solve instead of a vector. Intercept Here. Check Preconitioners.py.
    # '''
    # System = kp.LinearSystem(A=H, b=b,Minv=PreCon_lo)
    # ans = kp.Minres(System,maxiter=500, tol=1e-6 )
    
    '''SciPy run. Works smooth.
    '''
    ans = minres(A,b,M=PC,maxiter=500, show=True, tol=1e-6)[0]
    minresEnd = time.time()
    print "minres Time  = ", minresEnd - minresStart
    # print ans
    return ans

def BlockDiag(Prob):

    A,b = getLinearSystem(Prob)

    def pc(v):
        return PreCon.BlockDiagonal(Prob,v)
    PC = LinearOperator(A.shape, matvec = pc)

    print "Let's start minres, finally"
    minresStart = time.time()
    # '''KryPy Run, fires up error with a matrix output for a preconditioner
    #     block solve instead of a vector. Intercept Here. Check Preconitioners.py.
    # '''
    # System = kp.LinearSystem(A=H, b=b,Minv=PreCon_lo)
    # ans = kp.Minres(System,maxiter=500, tol=1e-6 )
    
    '''SciPy run. Works smooth.
    '''
    ans = minres(A,b,M=PC,maxiter=500, show=True, tol=1e-6)[0]
    minresEnd = time.time()
    print "minres Time  = ", minresEnd - minresStart
    # print ans
    return ans

def Indefinite(Prob):
    A,b = getLinearSystem(Prob)

    Eig = eigs(A = bmat([[Prob.M, None], [None,Prob.ld*Prob.D]]),\
     M = bmat([[Prob.M + Prob.L, None],[None,Prob.D]]))[0]
    anorm = np.absolute(Eig.max())

    Eig = eigs(A = Prob.M + Prob.L, \
    M = bmat([[-Prob.L, Prob.B]]) *\
     inv(bmat([[Prob.M, None], [None,Prob.ld*Prob.D]]))*\
      bmat([[-Prob.LT], [Prob.B.transpose()]]))[0]
    ksquare = np.absolute(Eig.max())

    def pc(v):
        return PreCon.IndefiniteHerzog(Prob,v, anorm, ksquare)

    PC = LinearOperator(A.shape, matvec = pc)

    def InnerProduct(r,z):
        return r - A * z

    def callback(xk):
                global num_iters
                num_iters += 1
                print num_iters

    print "Let's start minres, finally"
    BPCGStart = time.time()
    ans,status = cgIP(A,b,M=PC,maxiter=500, tol=1e-6,H=InnerProduct, callback = callback)
    print "Status: ", status
    BPCGEnd = time.time()
    print "BPCG Time  = ", BPCGEnd - BPCGStart
    # print ans
    return ans

def BPSchur(Prob):
    pass

def BPPlus(Prob):
    pass
