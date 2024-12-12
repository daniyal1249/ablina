import random
from numbers import Real

from sympy import *
from vector import *
from parser import to_ns_matrix

class VectorSpace:
    def _new__(cls, field, n, constraints=None, ns_matrix=None, rs_matrix=None):
        # Verify if constraints satisfy vector space properties
        if ns_matrix is None and rs_matrix is None:
            if constraints is None:
                constraints = set()
            if not is_vectorspace(field, n, constraints):
                raise Exception  # add error msg
        return super()._new__(cls)
    
    def __init__(self, field, n, constraints=None, ns_matrix=None, rs_matrix=None):
        # Initialize ns_matrix
        if ns_matrix is None:
            ns_matrix = rs_to_ns(rs_matrix) if rs_matrix is not None else to_ns_matrix(n, constraints)
        else:
            ns_matrix, _ = Matrix(ns_matrix).rref()

        # Initialize rs_matrix
        if rs_matrix is None:
            rs_matrix = ns_to_rs(ns_matrix)
        else:
            rs_matrix, _ = Matrix(rs_matrix).rref()

        self._field = field
        self._n = n
        self._constraints = constraints
        self._ns_matrix = ns_matrix
        self._rs_matrix = rs_matrix
        self._dim = rs_matrix.rank()

    @property
    def field(self):
        return self._field
    
    @property
    def n(self):
        return self._n
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def dim(self):
        return self._dim

    def __contains__(self, vec):
        if not isinstance(vec, Vector):
            return False
        if self.field != vec.field or self.n != len(vec):
            return False
        return Matrix(self._ns_matrix @ vec).is_zero_matrix  # make more efficent

    def __add__(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError

    def __and__(self, vs2):
        return self.intersection(vs2)
    
    def vector(self, std=1):
        size = self._rs_matrix.rows
        weights = [round(random.gauss(0, std)) for _ in range(size)]
        if isinstance(self.field, Real):
            return R(*weights) @ self._rs_matrix
        return C(*weights) @ self._rs_matrix
    
    def complement(self):
        constraints = {f'complement({str(self.constraints)[1:-1]})'}
        return VectorSpace(self.field, self.n, constraints, ns_matrix=self._rs_matrix, 
                           rs_matrix=self._ns_matrix)
    
    def intersection(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError
        if self.field != vs2.field or self.n != vs2.n:
            raise ValueError
        
        return VectorSpace(self.field, self.n, self.constraints | vs2.constraints)
    
def ns_to_rs(matrix):
    matrix = Matrix(matrix)
    ns_basis = matrix.nullspace()
    rs_matrix, _ = Matrix([vec.T for vec in ns_basis]).rref()
    return rs_matrix

def rs_to_ns(matrix):
    matrix = Matrix(matrix)
    rs_basis = matrix.rowspace()
    ns_basis = Matrix(rs_basis).nullspace()
    ns_matrix, _ = Matrix.hstack(*ns_basis).rref()
    return ns_matrix

def is_vectorspace(field, n, constraints):
    pass

def is_subspace(vs1, vs2):
    if not isinstance(vs1, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs1)._name__} instead.')
    if not isinstance(vs2, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs2)._name__} instead.')
    

def span(*vectors):
    for vec in vectors:
        if not isinstance(vec, Vector):
            raise TypeError(f'Expected vectors, got {type(vec)._name__} instead.')

    field, n = vectors[0].field, len(vectors[0])
    for vec in vectors:
        if field != vec.field or n != len(vec):
            raise ValueError('Vectors must be elements of the same subspace.')
        
    constraints = {f'span{vectors}'}
    return VectorSpace(field, n, constraints, rs_matrix=vectors)

def columnspace(matrix, field=Real):
    constraints = {f'col({matrix})'}

def rowspace(matrix, field=Real):
    constraints = {f'row({matrix})'}
    return VectorSpace(field)

def nullspace(matrix, field=Real):
    constraints = {f'null({matrix})'}

def left_nullspace(matrix, field=Real):
    matrix = Matrix(matrix).T
    return nullspace(matrix, field)

# x, y, z = symbols('x y z')
# vs = VectorSpace(field=Complex, n=3, constraints={'x0 = x1 = 10*x2'})
# print(C(10*x, 10*x, x) in vs)
# print(vs.vector())