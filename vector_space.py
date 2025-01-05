from numbers import Real, Complex
from random import gauss
import sympy as sp

from math_set import *
from parser import *
from utils import *
from vs_utils import *
from operation import VectorAdd, ScalarMul

class VectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class NotAVectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class StandardFn:
    def __init__(self, field, n, constraints=None, *, ns_matrix=None, rs_matrix=None):
        if field not in (Real, Complex):
            raise TypeError('Field must be either Real or Complex.')

        # Verify whether constraints satisfy the vector space properties
        constraints = constraints if constraints else []
        if ns_matrix is None and rs_matrix is None:
            if not is_vectorspace(n, constraints):
                raise NotAVectorSpaceError()  # add error msg

        ns, rs = StandardFn._init_matrices(n, constraints, ns_matrix, rs_matrix)

        self._field = field
        self._n = n
        self._constraints = constraints
        self._ns_matrix = ns
        self._rs_matrix = rs

    @staticmethod
    def _init_matrices(n, constraints, ns_matrix, rs_matrix):
        # Initialize ns_matrix
        if ns_matrix is None:
            if rs_matrix is None:
                ns_matrix = to_ns_matrix(n, constraints)
            else:
                ns_matrix = rs_to_ns(rs_matrix)
        else:
            ns_matrix, _ = sp.Matrix(ns_matrix).rref()

        # Initialize rs_matrix
        if rs_matrix is None:
            rs_matrix = ns_to_rs(ns_matrix)
        else:
            rs_matrix, _ = sp.Matrix(rs_matrix).rref()

        return ns_matrix, rs_matrix

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
    def basis(self):
        return [row for row in self._rs_matrix.tolist() if any(row)]
    
    @property
    def dim(self):
        return len(self.basis)
    
    def __eq__(self, vs2):
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)
    
    def __contains__(self, vec):
        try:
            if self.field is Real:
                if not all(is_real(coord) for coord in vec):
                    return False
            elif not all(is_complex(coord) for coord in vec):
                return False
            
            # Check if vec satisfies vector space constraints
            vec = sp.Matrix(vec)
            return bool((self._ns_matrix @ vec).is_zero_matrix)
        except Exception:
            return False
    
    def vector(self, std=1):
        size = self._rs_matrix.rows
        weights = [round(gauss(0, std)) for _ in range(size)]
        vec = sp.Matrix([weights]) @ self._rs_matrix
        return vec.flat()  # return list
    
    def to_coordinate(self, vector, basis=None):
        if basis is None:
            basis = self.basis

    def is_subspace(self, vs2):
        '''
        Returns True if self is a subspace of vs2, otherwise False.
        '''
        if not self.share_ambient_space(vs2):
            return False
        for i in range(self._rs_matrix.rows):
            vec = self._rs_matrix.row(i).T
            if not (vs2._ns_matrix @ vec).is_zero_matrix:
                return False
        return True
    
    def is_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows

    def share_ambient_space(self, vs2):
        if type(self) is not type(vs2):
            return False
        return self.field is vs2.field and self.n == vs2.n


class Fn(StandardFn):
    def __init__(self, field, n, constraints=None, add=None, mul=None, 
                 *, isomorphism=None, ns_matrix=None, rs_matrix=None):
        
        if isomorphism is not None:
            if not (isinstance(isomorphism, tuple) and len(isomorphism) == 2):
                raise TypeError('Isomorphism must be a 2-tuple of callables.')
        
        add, mul, iso = Fn._init_operations(field, n, add, mul, isomorphism)

        self._to_standard, self._from_standard = iso
        mapped_constraints = map_constraints(self._to_standard, constraints)
        super().__init__(field, n, mapped_constraints, ns_matrix=ns_matrix, 
                         rs_matrix=rs_matrix)
        
        self._add = VectorAdd(field, n, add)
        self._mul = ScalarMul(field, n, mul)
        # Reassign constraints
        self._constraints = constraints

    @staticmethod
    def _init_operations(field, n, add, mul, isomorphism):
        # For efficiency
        if add is None and mul is None:
            isomorphism = (lambda vec: vec, lambda vec: vec)

        if add is None:
            add = lambda vec1, vec2: [i + j for i, j in zip(vec1, vec2)]
        if mul is None:
            mul = lambda scalar, vec: [scalar * i for i in vec]
        if not isomorphism:
            isomorphism = standard_isomorphism(field, n, add, mul)

        return add, mul, isomorphism

    @property
    def add(self):
        return self._add
    
    @property
    def mul(self):
        return self._mul
    
    @property
    def basis(self):
        return [self._from_standard(vec) for vec in super().basis]

    def __contains__(self, vec):
        standard_vec = self._to_standard(vec)
        return super().__contains__(standard_vec)
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)
    
    def vector(self, std=1):
        standard_vec = super().vector(std)
        return self._from_standard(standard_vec)
    
    def complement(self):
        constraints = [f'complement({', '.join(self.constraints)})']
        return Fn(self.field, self.n, constraints, self.add, self.mul, 
                  isomorphism=(self._to_standard, self._from_standard), 
                  ns_matrix=self._rs_matrix, rs_matrix=self._ns_matrix)
    
    def sum(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        
        rs_matrix = sp.Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix, _ = rs_matrix.rref()
        constraints = self.constraints  # need to fix
        return Fn(self.field, self.n, constraints, self.add, self.mul, 
                  isomorphism=(self._to_standard, self._from_standard), 
                  rs_matrix=rs_matrix)
    
    def intersection(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        
        ns_matrix = sp.Matrix.vstack(self._ns_matrix, vs2._ns_matrix)
        ns_matrix, _ = ns_matrix.rref()
        constraints = self.constraints + vs2.constraints
        return Fn(self.field, self.n, constraints, self.add, self.mul, 
                  isomorphism=(self._to_standard, self._from_standard), 
                  ns_matrix=ns_matrix)
    
    def span(self, *vectors):
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        standard_vecs = [self._to_standard(vec) for vec in vectors]
        constraints = [f'span{vectors}']
        return Fn(self.field, self.n, constraints, self.add, self.mul, 
                  isomorphism=(self._to_standard, self._from_standard), 
                  rs_matrix=standard_vecs)
    
    def is_independent(self, *vectors):
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        vectors = [self._to_standard(vec) for vec in vectors]
        return super().is_independent(vectors)
    
    def share_ambient_space(self, vs2):
        # if not super().share_ambient_space(vs2):
        #     return False
        # return self.add == vs2.add and self.mul == vs2.mul
        return True


class VectorSpace:
    def __init__(self, vectors, fn, isomorphism):
        if not isinstance(vectors, MathematicalSet):
            raise TypeError('vectors must be a MathematicalSet.')
        if not isinstance(fn, Fn):
            raise TypeError('fn must be of type Fn.')
        iso = VectorSpace._check_isomorphism(isomorphism)
        
        self._vectors = vectors
        self._fn = fn
        self._to_fn, self._from_fn = iso

    @staticmethod
    def _check_isomorphism(iso):
        if (isinstance(iso, tuple) and len(iso) == 2 and 
            all(of_arity(i, 1) for i in iso)):
            return iso
        
        if of_arity(iso, 1):
            return (iso, lambda vec: vec)
        else:
            raise TypeError('isomorphism must be a callable or a 2-tuple '
                            'of callables.')
        
    @property
    def vectors(self):
        return self._vectors
    
    @property
    def field(self):
        return self._fn.field
    
    @property
    def add_id(self):
        pass
    
    @property
    def add_inv(self):
        pass
    
    @property
    def mul_id(self):
        pass
    
    @property
    def basis(self):
        return [self._from_fn(vec) for vec in self._fn.basis]
    
    @property
    def dim(self):
        return self._fn.dim
    
    @property
    def set(self):
        return Set(self.vectors.cls, lambda vec: vec in self)
    
    def __contains__(self, vec):
        if vec not in self.vectors:
            return False
        return self._to_fn(vec) in self._fn
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)
    
    def vector(self, std=1):
        fn_vector = self._fn.vector(std)
        return self._from_fn(fn_vector)
    
    def complement(self):
        fn = self._fn.complement()
        return VectorSpace(self.vectors, fn, (self._to_fn, self._from_fn))
    
    def sum(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        fn = self._fn.sum(vs2._fn)
        return VectorSpace(self.vectors, fn, (self._to_fn, self._from_fn))
    
    def intersection(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        fn = self._fn.intersection(vs2._fn)
        return VectorSpace(self.vectors, fn, (self._to_fn, self._from_fn))
    
    def span(self, *vectors):
        if not all(vec in self.vectors for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vectors = [self._to_fn(vec) for vec in vectors]
        fn = self._fn.span(*fn_vectors)
        return VectorSpace(self.vectors, fn, (self._to_fn, self._from_fn))
    
    def to_coordinate(self, vector, basis=None):
        if basis is None:
            basis = self.basis
    
    def is_subspace(self, vs2):
        '''
        Returns True if self is a subspace of vs2, otherwise False.
        '''
        if not self.share_ambient_space(vs2):
            return False
        return self._fn.is_subspace(vs2._fn)
    
    def is_independent(self, *vectors):
        if not all(vec in self.vectors for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vectors = [self._to_fn(vec) for vec in vectors]
        return self._fn.is_independent(*fn_vectors)
    
    def share_ambient_space(self, vs2):
        # if self.vectors is not vs2.vectors:
        #     return False
        # return self._fn.share_ambient_space(vs2._fn)
        return True

    @classmethod
    def fn(cls, field, n, constraints=None, add=None, mul=None, 
           *, ns_matrix=None, rs_matrix=None):
        def pred(vec):
            try:
                return sp.Matrix(vec).shape == (n, 1)
            except Exception:
                return False

        vectors = Set(object, pred)
        fn = Fn(field, n, constraints, add, mul, ns_matrix=ns_matrix, 
                rs_matrix=rs_matrix)
        return cls(vectors, fn, lambda vec: vec)

    @classmethod
    def matrix(cls, field, shape, constraints=None, add=None, mul=None):
        def to_fn(mat):
            return mat.flat()
        def from_fn(vec):
            return sp.Matrix(*shape, vec)
        
        vectors = Set(sp.Matrix, lambda mat: mat.shape == shape)
        fn = Fn(field, sp.prod(shape), constraints, add, mul)
        return cls(vectors, fn, (to_fn, from_fn))

    @classmethod
    def poly(cls, field, max_degree, constraints=None, add=None, mul=None):
        def to_fn(poly):
            coeffs = poly.all_coeffs()
            degree_diff = max_degree - len(coeffs) + 1
            return ([0] * degree_diff) + coeffs
        def from_fn(vec):
            x = sp.symbols('x')
            return sp.Poly.from_list(vec, x)

        vectors = Set(sp.Poly, lambda poly: sp.degree(poly) <= max_degree)
        fn = Fn(field, max_degree + 1, constraints, add, mul)
        return cls(vectors, fn, (to_fn, from_fn))
    
    @classmethod
    def hom(cls, vs1, vs2):
        if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
            raise TypeError()
        if vs1.field is not vs2.field:
            raise VectorSpaceError()
        return cls.matrix(vs1.field, (vs2.dim, vs1.dim))


def is_vectorspace(n, constraints):
    '''
    Returns True if F^n forms a vector space under the given constraints with 
    standard operations, otherwise False.
    '''
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))

    allowed_vars = sp.symbols(f'v:{n}')
    for expr in exprs:
        expr = sympify(expr, allowed_vars)
        if not is_linear(expr):
            return False
        
        # Check for nonzero constant terms
        if any(term.is_constant() and term != 0 for term in expr.args):
            return False
    return True

def columnspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.rows
    constraints = [f'col({matrix.tolist()})']
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix.T)

def rowspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.cols
    constraints = [f'row({matrix.tolist()})']
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix)

def nullspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.cols
    constraints = [f'null({matrix.tolist()})']
    return VectorSpace.fn(field, n, constraints, ns_matrix=matrix)

def left_nullspace(matrix, field=Real):
    matrix = sp.Matrix(matrix).T
    return nullspace(matrix, field)

# Aliases
image = columnspace
kernel = nullspace


# to_iso = lambda vec: [sp.log(i) for i in vec]
# from_iso = lambda vec: [sp.exp(i) for i in vec]

# vectors = Set(sp.Poly, lambda poly: sp.degree(poly) <= 3)
# vs1 = VectorSpace.poly(field=Real, max_degree=3, constraints=[])
# vs2 = VectorSpace.poly(field=Real, max_degree=3, constraints=['v0==0'])
# x = sp.symbols('x', real=True)
# print(vs1.is_subspace(vs2))