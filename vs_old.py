from numbers import Real, Complex
from random import gauss
from math_set import MathematicalSet

from sympy import *
from parser import *
from utils import *

class NotAVectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class IsomorphismError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class VectorSpace:
    def __new__(cls, field, vector_cls, isomorphism, shape, zero=None, add=None, mul=None, 
                constraints=None, *, ns_matrix=None, rs_matrix=None):
        if field not in (Real, Complex):
            raise TypeError('Field must be either Real or Complex.')
        if not isinstance(vector_cls, type):
            raise TypeError('Vector class must be a class.')

        if callable(isomorphism):
            if not of_arity(isomorphism, 1):
                raise IsomorphismError('The isomorphism must be able to accept a single argument.')
        elif isinstance(isomorphism, tuple) and len(isomorphism) == 2 and all(callable(i) for i in isomorphism):
            if not all(of_arity(i, 1) for i in isomorphism):
                raise IsomorphismError('The isomorphisms must be able to accept a single argument.')
        else:
            raise IsomorphismError('Isomorphism must be a callable or a 2-tuple of callables.')
        
        if not isinstance(shape, int) and not (isinstance(shape, tuple) and shape and 
                                               all(isinstance(i, int) for i in shape)):
            raise TypeError('Vector shape must be an integer or a non-empty tuple of integers.')
        
        if zero is not None and not isinstance(zero, vector_cls):  # check if zero vector has the right shape
            raise TypeError('Zero vector must be an instance of the vector class.')
        if add is not None and not (callable(add) and of_arity(add, 2)):
            raise TypeError('Add function must be a callable that can accept two arguments.')
        if mul is not None and not (callable(mul) and of_arity(mul, 2)):
            raise TypeError('Mul function must be a callable that can accept two arguments.')
        
        # Verify if constraints satisfy vector space properties
        if ns_matrix is None and rs_matrix is None:
            if constraints is None:
                constraints = set()
            if not is_vectorspace(prod(tuple(shape)), constraints):
                raise NotAVectorSpaceError()  # add error msg
        return super().__new__(cls)

    def __init__(self, field, vector_cls, isomorphism, shape, zero=None, add=None, mul=None, 
                 constraints=None, *, ns_matrix=None, rs_matrix=None):

        if isinstance(isomorphism, tuple):
            to_array = lambda vec: Array(isomorphism[0](vec))
            from_array = lambda arr: isomorphism[1](arr.tolist())
        else:
            to_array = lambda vec: Array(isomorphism(vec))
            from_array = lambda arr: arr

        if isinstance(shape, int):
            shape = tuple(shape)

        if zero is None:
            zero = zeros(prod(shape), 1)
        else:
            zero = self._to_euclidean(zero)
        if add is None:
            add = lambda vec1, vec2: vec1 + vec2
        if mul is None:
            mul = lambda vec, scalar: scalar * vec
        if constraints is None:
            constraints = set()

        # Initialize ns_matrix
        if ns_matrix is None:
            if rs_matrix is None:
                ns_matrix = to_ns_matrix(prod(shape), constraints)
            else:
                ns_matrix = rs_to_ns(rs_matrix)
        else:
            ns_matrix, _ = Matrix(ns_matrix).rref()

        # Initialize rs_matrix
        if rs_matrix is None:
            rs_matrix = ns_to_rs(ns_matrix)
        else:
            rs_matrix, _ = Matrix(rs_matrix).rref()

        self._field = field
        self._vector_cls = vector_cls
        self._isomorphism = isomorphism
        self._to_array = to_array
        self._from_array = from_array
        self._shape = shape
        self._zero = zero
        self._add = add
        self._mul = mul
        self._constraints = constraints
        self._ns_matrix = ns_matrix
        self._rs_matrix = rs_matrix

        self._basis = [Matrix(row) for row in self._rs_matrix.tolist() if any(row)]
        self._dim = len(self._basis)

    @property
    def field(self):
        return self._field
    
    @property
    def vector_cls(self):
        return self._vector_cls
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def zero(self):
        return self._from_euclidean(self._zero)
    
    @property
    def add(self):
        return self._add
    
    @property
    def mul(self):
        return self._mul
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def basis(self):
        return [self._from_euclidean(vec) for vec in self._basis]
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def ambient_space(self):
        # Return the vector space without any constraints
        return VectorSpace(self.field, self.vector_cls, self._isomorphism, self.shape, 
                           self.zero, self.add, self.mul)
    
    def __repr__(self):
        raise NotImplementedError

    def __contains__(self, vec):
        return (isinstance(vec, self.vector_cls) and (self._to_array(vec).shape == self.shape) and 
                (self._ns_matrix @ self._to_euclidean(vec) == self._zero))

    def __eq__(self, vs2):
        pass

    def __add__(self, vs2):
        return self.sum(vs2)

    def __and__(self, vs2):
        return self.intersection(vs2)

    def vector(self, std=1):
        size = self._rs_matrix.rows
        weights = [round(gauss(0, std))] * size
        vec = (Matrix(weights) @ self._rs_matrix)
        return self._from_euclidean(vec)  # check if transpose is needed

    def complement(self):
        constraints = {f'complement({', '.join(self.constraints)})'}
        return VectorSpace(self.field, self.vector_cls, self._isomorphism, self.shape, 
                           self.zero, self.add, self.mul, constraints, 
                           ns_matrix=self._rs_matrix, rs_matrix=self._ns_matrix)

    def sum(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError()
        if self.ambient_space != vs2.ambient_space:
            raise ValueError()
        
        rs_matrix = Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix, _ = rs_matrix.rref()
        constraints = self.constraints.intersection(vs2.constraints)  # need to fix
        return VectorSpace(self.field, self.vector_cls, self._isomorphism, self.shape, 
                           self.zero, self.add, self.mul, constraints, rs_matrix=rs_matrix)

    def intersection(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError()
        if self.ambient_space != vs2.ambient_space:
            raise ValueError()
        
        ns_matrix = Matrix.vstack(self._ns_matrix, vs2._ns_matrix)
        ns_matrix, _ = ns_matrix.rref()
        constraints = self.constraints.union(vs2.constraints)
        return VectorSpace(self.field, self.vector_cls, self._isomorphism, self.shape, 
                           self.zero, self.add, self.mul, constraints, ns_matrix=ns_matrix)
    
    def subspace(self, ambient_space, constraints):
        if not isinstance(ambient_space, VectorSpace):
            raise TypeError('Ambient space must be a vector space.')
        # Create new vector space using these constraints and do intersection
        # Translate constraints to ns_matrix and combine
        return VectorSpace()
    
    def span(self, *vectors):
        pass

    def is_independent(self, *vectors):
        pass

    def _to_euclidean(self, vec):
        arr = self._to_array(vec)
        vec = arr.reshape(len(arr), 1)
        return vec.tomatrix()
    
    def _from_euclidean(self, vec):
        arr = Array(vec, self.shape)
        return self._from_array(arr)

    @classmethod
    def euclidean(cls, field, n, constraints=None, *, ns_matrix=None, rs_matrix=None):
        isomorphism = lambda x: x
        return VectorSpace(field, list, isomorphism, n, [0] * n, constraints=constraints, 
                           ns_matrix=ns_matrix, rs_matrix=rs_matrix)

    @classmethod
    def matrix(cls, field, shape, constraints=None, *, ns_matrix=None, rs_matrix=None):
        isomorphism = (lambda x: x.tolist(), lambda x: Matrix(x))
        return VectorSpace(field, Matrix, isomorphism, shape, zeros(*shape), constraints=constraints, 
                           ns_matrix=ns_matrix, rs_matrix=rs_matrix)

    @classmethod
    def polynomial(cls, field, max_degree, constraints=None, *, ns_matrix=None, rs_matrix=None):
        isomorphism = ()
        return VectorSpace(field, Poly, isomorphism, max_degree + 1, None, constraints=constraints, 
                           ns_matrix=ns_matrix, rs_matrix=rs_matrix)


def is_vectorspace(n, constraints):
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))

    for expr in exprs:
        expr = parse_expression(n, expr)
        if not is_linear(expr):
            return False
        # Check for nonzero constant terms
        if any(term.is_constant() and term != 0 for term in expr.args):
            return False
    return True

def is_subspace(vs1, vs2):
    if not isinstance(vs1, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs1).__name__} instead.')
    if not isinstance(vs2, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs2).__name__} instead.')
    if vs1.ambient_space != vs2.ambient_space:
        return False
    
    for row in vs1._rs_matrix.tolist():
        vec = Matrix(row).T
        if not (vs2._ns_matrix @ vec == vs1._zero):
            return False
    return True

def columnspace(matrix, field=Real):
    matrix = Matrix(matrix)
    n = matrix.rows
    constraints = {f'col({matrix})'}
    return VectorSpace.euclidean(field, n, constraints, rs_matrix=matrix.T)

def rowspace(matrix, field=Real):
    matrix = Matrix(matrix)
    n = matrix.cols
    constraints = {f'row({matrix})'}
    return VectorSpace.euclidean(field, n, constraints, rs_matrix=matrix)

def nullspace(matrix, field=Real):
    matrix = Matrix(matrix)
    n = matrix.cols
    constraints = {f'null({matrix})'}
    return VectorSpace.euclidean(field, n, constraints, ns_matrix=matrix)

def left_nullspace(matrix, field=Real):
    matrix = Matrix(matrix).T
    return nullspace(matrix, field)