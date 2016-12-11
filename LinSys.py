from scipy.sparse import block_diag, bmat, csr_matrix, identity
from scipy.sparse.linalg import LinearOperator
import time
import numpy as np


class LinearSystem(object):
    """docstring for LinearSystem"""
    def __init__(self, OCProblem,
                 Direct = False, Constrained=False, 
                 Pa = None, arr2 = None):

        consTimeStart = time.time()

        m = OCProblem.m
        n = OCProblem.n

        if Constrained:
            if Pa is None or arr2 is None:
                raise TypeError("Constrained case needs an active set")

            if Direct:
                self.DirectBoxed(OCProblem, Pa)
            else:
                ac = arr2.shape[0]
                self.IterativeBoxed(OCProblem, m, n, Pa, ac)
            self.b  = np.hstack([OCProblem.vec.array(), 
                                OCProblem.vec2, 
                                OCProblem.vec1,
                                arr2])
            # self.b  = np.hstack([OCProblem.vec.array(), 
            #                     OCProblem.ld*OCProblem.D.dot(arr2), 
            #                     OCProblem.vec2])

        else:
            if Direct:
                self.DirectUnbounded(OCProblem)
            else:
                self.IterativeUnbounded(OCProblem, m, n)

            self.b = np.hstack([OCProblem.vec.array(), OCProblem.vec2,\
                                 OCProblem.vec1])
        consTimeEnd = time.time()
        print "construction Time = ", consTimeEnd - consTimeStart

        print self.A.shape
        # print type(A)

        return

    def DirectUnbounded(self, Problem):

        self.A = bmat( [[Problem.M, None, -Problem.LT ], \
                        [None, Problem.ld*Problem.D, Problem.B.transpose() ],\
                        [ -Problem.L, Problem.B, None] ],format='csr')
        return

    def IterativeUnbounded(self, Problem, m, n):

        Ablock = block_diag((Problem.M, Problem.ld*Problem.D)).tocsr()
        Bblock = bmat([[-Problem.L, Problem.B ]],format='csr')
        BTblock = bmat([[-Problem.LT],[Problem.B.transpose()]], format='csr')

        def mv(v):
            xy = v[:n:1]
            xu = v[n:m+n:1]
            xp = v[m+n:m+2*n:1]

            bx = Ablock.dot(np.hstack([xy, xu])) + BTblock.dot(xp)
            bp = Bblock.dot(np.hstack([xy, xu]))
            return np.hstack([bx,bp])

        self.A = LinearOperator( (2*n+m,2*n+m) , matvec=mv )
        return

    def DirectBoxed(self, Problem, Pa):
        self.A  = bmat([[Problem.M,None,-Problem.LT,None],\
                    [None,Problem.ld*Problem.D,Problem.B.transpose(),Pa.transpose()],\
                    [-Problem.L,Problem.B,None,None],\
                    [None,Pa,None,None]],format='csr')
        # PI = identity(Pa.shape[0]) - Pa
        # self.A  = bmat([[Problem.M,None,-Problem.LT],\
        #             [None,Problem.ld*Problem.D,PI.dot(Problem.B.transpose())],\
        #             [-Problem.L,Problem.B,None] ],format='csr')
        return

    def IterativeBoxed(self, Problem, m ,n, Pa, ac):

        Ablock = block_diag((Problem.M, Problem.ld*Problem.D)).tocsr()
        Bblock = bmat([[-Problem.L, Problem.B ],[None, Pa]],format='csr')
        BTblock = bmat([[-Problem.LT, None],[Problem.B.transpose(), Pa.transpose()]],\
                        format='csr')

        def mvc(v):
            xy = v[:n:1]
            xu = v[n:m+n:1]
            xp = v[m+n:m+2*n:1]
            xm = v[2*n+m::1]
            bx = Ablock.dot(np.hstack([xy, xu])) +\
                 BTblock.dot(np.hstack([xp, xm]))
            bp = Bblock.dot(np.hstack([xy, xu]))
            return np.hstack([bx,bp])

        self.A = LinearOperator((2*n+m+ac,2*n+m+ac), matvec = mvc)

        return