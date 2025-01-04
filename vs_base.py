from numbers import Real, Complex
from random import gauss
import sympy as sp

from parser import *
from utils import *

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
    
    # def __repr__(self):
    #     return (f'StandardFn(field={self.field.__name__}, n={self.n}, '
    #             f'constraints={self.constraints})')

    def __contains__(self, vec):
        try:
            if self.field is Real:
                if not all(is_real(coord) for coord in vec):
                    return False
            elif not all(is_complex(coord) for coord in vec):
                return False
              
            # Check if vec satisfies vector space constraints
            shape = (self.n, 1)
            vec = sp.Matrix(*shape, vec)
            return bool((self._ns_matrix @ vec).is_zero_matrix)
        except Exception:
            return False
        
    def __eq__(self, vs2):
        return self.is_subspace(vs2) and vs2.is_subspace(self)
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)
    
    def vector(self, std=1):
        size = self._rs_matrix.rows
        weights = [round(gauss(0, std))] * size
        vec = sp.Matrix([weights]) @ self._rs_matrix
        return vec.tolist()  # return list

    def complement(self):
        constraints = [f'complement({', '.join(self.constraints)})']
        return StandardFn(self.field, self.n, constraints, 
                          ns_matrix=self._rs_matrix, 
                          rs_matrix=self._ns_matrix)

    def sum(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        
        rs_matrix = sp.Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix, _ = rs_matrix.rref()
        constraints = self.constraints  # need to fix
        return StandardFn(self.field, self.n, constraints, rs_matrix=rs_matrix)

    def intersection(self, vs2):
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        
        ns_matrix = sp.Matrix.vstack(self._ns_matrix, vs2._ns_matrix)
        ns_matrix, _ = ns_matrix.rref()
        constraints = self.constraints + vs2.constraints
        return StandardFn(self.field, self.n, constraints, ns_matrix=ns_matrix)

    def is_subspace(self, vs2):
        '''
        Returns True if self is a subspace of vs2, otherwise False.
        '''
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        
        for i in range(self._rs_matrix.rows):
            vec = self._rs_matrix.row(i).T
            if not (vs2._ns_matrix @ vec).is_zero_matrix:
                return False
        return True

    def span(self, *vectors):
        # include type check
        constraints = [f'span{vectors}']
        return StandardFn(self.field, self.n, constraints, rs_matrix=vectors)
        
    def is_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows

    def share_ambient_space(self, vs2):
        if type(self) is not type(vs2):
            raise TypeError()
        return self.field is vs2.field and self.n == vs2.n
    
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