from dolfin import *
from mshr import *
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

# N     = 64 # number of grid points per dimension

parameters["linear_algebra_backend"] = 'PETSc'

def mat2sp (A):
    A_mat = as_backend_type(A).mat()
    A_mat_csr = csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
    return A_mat_csr

class OptimalControlProblem(object):
    '''
        OptimalControlProblem takes a mesh size to construct a hard-coded 
        PDE constrained Optimal Control Problem. The user needs to specifiy 
        the following parameters

        ld:       lambda, the regularisation parameter
        yd:       The desired state y_d
        Y:        The State Space discretisation
        P:        The Adjoint Space discretisation (should be the same as for Y)
        U:        The Control Space discrettisation
        oprtator: The Ufl expression for the PDE

        The OptimalControlProblem object has all the matrix blocks that shall be
        needed to construct a linear system of interest
    '''
    def __init__(self, N):

        parameters["linear_algebra_backend"] = 'PETSc'

        self.ld = 1e-3
        mesh = UnitSquareMesh(N,N)

        # self.yd = Expression('-x[0]*exp(-1*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)))')
        # self.yd = Expression('exp(-64*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)))')
        self.yd = Expression(" sin(pi*x[0])*sin(pi*x[1])")
        # plot(self.yd,mesh=mesh,title='Desired State') #plot(mesh)

        Y = FunctionSpace(mesh,"CG", 1)     # The state space
        P = FunctionSpace(mesh,"CG", 1)     # The adjoint space
        U = FunctionSpace(mesh,"CG", 1)     # The control Space

        y = TrialFunction(Y)
        v = TestFunction(Y)
        w = TestFunction(P)
        u = TrialFunction(U)
        zeta = TestFunction(U)
        self.yfun = Function(Y)
        self.ufun = Function(U)
        self.ufun2 = Function(U)

        self.n = Y.dim()
        self.m = U.dim()

        bc_op = DirichletBC(Y, 0, "on_boundary")
        bc_ad = DirichletBC(P, 0, "on_boundary")

        # alpha = Expression(" sin(pi * x[0])")
        operator = inner( grad(y), grad(v) ) * dx;

        # matdummy, random = assemble_system( operator,Constant(0)*v*dx, bcs=bc_op)
        matdummy = assemble(operator)
        bc_op.apply(matdummy)
        self.L = mat2sp(matdummy)
        # print random.array()

        matdummy = assemble(adjoint(operator))
        bc_ad.apply(matdummy)
        self.LT = mat2sp( matdummy )
        # self.LT = self.L.transpose()

        M = assemble( y*w*dx) 
        # bc_m.apply(M)
        self.M = mat2sp(M)
        self.D = mat2sp( assemble( u*zeta*dx ) )
        # self.Dinv = inv(self.D)
        self.vec = assemble(self.yd*v*dx)
        # vec2 = random.vector()
        self.vec1 = np.zeros(self.n)
        self.vec2 = np.zeros(self.m)

        # matdummy = assemble(u*v*dx)
        # bc_b.apply(matdummy)
        # self.B = mat2sp(matdummy)
        self.B = mat2sp(assemble(u*v*dx))

        print type(self.B)
        print U.dim()
        print Y.dim()
        print self.L.shape
        print self.LT.shape
        print self.D.shape
        print self.B.shape
        print self.M.shape

        return