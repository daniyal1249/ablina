from numbers import Real, Complex
from random import gauss
import sympy as sp

from linear_map import Isomorphism, IsomorphismError
from vs_base import *
from math_set import *
from vs_utils import *

class Fn(StandardFn):
    def __init__(self, field, n, constraints=None, add=None, mul=None, 
                 *, isomorphism=None, ns_matrix=None, rs_matrix=None):
        
        if isomorphism is not None:
            if not (isinstance(isomorphism, tuple) and len(isomorphism) == 2):
                raise TypeError('Isomorphism must be a 2-tuple of callables.')
        
        self._add, self._mul, iso = Fn._init_operations(field, n, add, mul, isomorphism)

        self._to_standard, self._from_standard = iso
        mapped_constraints = map_constraints(self._to_standard, constraints)
        super().__init__(field, n, mapped_constraints, ns_matrix=ns_matrix, 
                         rs_matrix=rs_matrix)
        
        # Reassign constraints correctly
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
        basis = super().basis
        return [self._from_standard(vec) for vec in basis]

    def __contains__(self, vec):
        standard_vec = self._to_standard(vec)
        return super().__contains__(standard_vec)
    
    def vector(self, std=1):
        standard_vec = super().vector(std)
        return self._from_standard(standard_vec)
    
    def span(self, *vectors):
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        constraints = [f'span{vectors}']
        vectors = [self._to_standard(vec) for vec in vectors]
        return Fn(self.field, self.n, constraints, self.add, self.mul, 
                  isomorphism=(self._to_standard, self._from_standard), 
                  rs_matrix=vectors)
    
    def is_independent(self, *vectors):
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        vectors = [self._to_standard(vec) for vec in vectors]
        return super().is_independent(vectors)
    
    def share_ambient_space(self, vs2):
        if not super().share_ambient_space(vs2):
            return False
        return self.add == vs2.add and self.mul == vs2.mul
        

class VectorSpace(Fn):
    def __init__(self, vectors, isomorphism):
        if not isinstance(vectors, MathematicalSet):
            raise TypeError('vectors must be a MathematicalSet.')
        iso = VectorSpace._check_isomorphism(vectors, isomorphism)

        # Call super init
        to_fn, from_fn = iso
        fn = to_fn.codomain
        args = (fn.field, fn.n, fn.constraints, fn.add, fn.mul)
        kwargs = {'isomorphism': (fn._to_standard, fn._from_standard), 
                  'ns_matrix': fn._ns_matrix, 'rs_matrix': fn._rs_matrix}
        super().__init__(*args, **kwargs)

        self._to_fn, self._from_fn = to_fn.mapping, from_fn.mapping
        fn_to_std, std_to_fn = self._to_standard, self._from_standard
        self._to_standard = lambda vec: fn_to_std(self._to_fn(vec))
        self._from_standard = lambda vec: self._from_fn(std_to_fn(vec))
        self._vectors = vectors

    @staticmethod
    def _check_isomorphism(vectors, iso):
        if isinstance(iso, Isomorphism):
            if not (iso.domain is vectors and isinstance(iso.codomain, Fn)):
                raise IsomorphismError()
            from_fn = Isomorphism(iso.codomain, iso.domain, lambda vec: vec)
            iso = (iso, from_fn)

        elif (isinstance(iso, tuple) and len(iso) == 2 and 
              all(isinstance(i, Isomorphism) for i in iso)):
            to_fn, from_fn = iso
            if not (to_fn.domain is vectors and isinstance(to_fn.codomain, Fn)):
                raise IsomorphismError()
            if not (from_fn.codomain is vectors and isinstance(from_fn.domain, Fn)):
                raise IsomorphismError()
        else:
            raise TypeError('isomorphism must be an Isomorphism or a 2-tuple '
                            'of Isomorphisms.')
        return iso
    
    @property
    def vectors(self):
        return self._vectors
    
    def __contains__(self, vec):
        if vec not in self.vectors:
            return False
        return super().__contains__(vec)

    # def span(self, *vectors):
    #     # Need to rework
    #     fn = super().span(vectors)
    #     pred = lambda vec: self._to_fn(vec) in fn  # modify pred
    #     new_vectors = self.vectors.add_predicates(pred)
    #     to_fn = Isomorphism(new_vectors, fn, self._to_fn)
    #     from_fn = Isomorphism(fn, new_vectors, self._from_fn)
    #     return VectorSpace(new_vectors, (to_fn, from_fn))

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
        to_fn = Isomorphism(vectors, fn, lambda vec: vec)
        return cls(vectors, to_fn)

    @classmethod
    def matrix(cls, field, shape, constraints=None, add=None, mul=None):
        def to_fn_mapping(mat):
            return mat.flat()
        def from_fn_mapping(vec):
            return sp.Matrix(*shape, vec)
        
        vectors = Set(sp.Matrix, lambda mat: mat.shape == shape)
        fn = Fn(field, sp.prod(shape), constraints, add, mul)
        to_fn = Isomorphism(vectors, fn, to_fn_mapping)
        from_fn = Isomorphism(fn, vectors, from_fn_mapping)
        return cls(vectors, (to_fn, from_fn))

    @classmethod
    def polynomial(cls, field, max_degree, constraints=None, add=None, mul=None):
        def to_fn_mapping(poly):
            coeffs = poly.all_coeffs()
            degree_diff = max_degree - len(coeffs) + 1
            return ([0] * degree_diff) + coeffs
        def from_fn_mapping(vec):
            x = sp.symbols('x')
            return sp.Poly.from_list(vec, x)

        vectors = Set(sp.Poly, lambda poly: sp.degree(poly) <= max_degree)
        fn = Fn(field, max_degree + 1, constraints, add, mul)
        to_fn = Isomorphism(vectors, fn, to_fn_mapping)
        from_fn = Isomorphism(fn, vectors, from_fn_mapping)
        return cls(vectors, (to_fn, from_fn))


def columnspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.rows
    constraints = [f'col({matrix})']
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix.T)

def rowspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.cols
    constraints = [f'row({matrix})']
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix)

def nullspace(matrix, field=Real):
    matrix = sp.Matrix(matrix)
    n = matrix.cols
    constraints = [f'null({matrix})']
    return VectorSpace.fn(field, n, constraints, ns_matrix=matrix)

def left_nullspace(matrix, field=Real):
    matrix = sp.Matrix(matrix).T
    return nullspace(matrix, field)


# to_iso = lambda vec: [sp.log(i) for i in vec]
# from_iso = lambda vec: [sp.exp(i) for i in vec]

# vs1 = VectorSpace.polynomial(field=Real, max_degree=2)
# x = sp.symbols('x', real=True)
# print(vs1.basis)