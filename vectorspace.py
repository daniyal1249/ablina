from numbers import Real, Complex
import numpy as np
from vector import Vector
from algebra_parser import to_matrix

class VectorSpace:
    def __new__(cls, field, n, constraints=None, ns_matrix=None, rs_matrix=None):
        if ns_matrix is None and rs_matrix is None:
            if constraints is None:
                constraints = set()
            if not is_vectorspace(field, n, constraints):
                raise Exception  # add error msg
        return super().__new__(cls)
    
    def __init__(self, field, n, constraints=None, ns_matrix=None, rs_matrix=None):

        self.__field = field
        self.__n = n
        self.__constraints = constraints
        self.__ns_matrix = ns_matrix
        self.__rs_matrix = rs_matrix

    @property
    def field(self):
        return self.__field
    
    @property
    def n(self):
        return self.__n
    
    @property
    def constraints(self):
        return self.__constraints
    
    @property
    def dim(self):
        pass

    def __contains__(self, vec):
        if not isinstance(vec, Vector):
            raise TypeError
        if self.field != vec.field or self.n != len(vec):
            raise ValueError

    def __add__(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError

    def __and__(self, vs2):
        return self.intersection(vs2)
    
    # def vector(self, std=1):
    #     size = len(self.__rs_matrix)
    #     weights = np.random.normal(loc=0, scale=std, size=size)
    
    def complement(self):
        constraints = {f'complement({str(self.constraints)[1:-1]})'}
        return VectorSpace(self.field, self.n, constraints, ns_matrix=self.__rs_matrix, 
                           rs_matrix=self.__ns_matrix)
    
    def intersection(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError
        if self.field != vs2.field or self.n != vs2.n:
            raise ValueError
        
        matrix = None
        return VectorSpace(self.field, self.n, self.constraints | vs2.constraints)

def is_vectorspace(field, n, constraints):
    pass

def is_subspace(vs1, vs2):
    if not isinstance(vs1, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs1).__name__} instead.')
    if not isinstance(vs2, VectorSpace):
        raise TypeError(f'Expected vector spaces, got {type(vs2).__name__} instead.')
    

def span(*vectors):
    for vec in vectors:
        if not isinstance(vec, Vector):
            raise TypeError(f'Expected vectors, got {type(vec).__name__} instead.')

    field, n = vectors[0].field, len(vectors[0])
    for vec in vectors:
        if field != vec.field or n != len(vec):
            raise ValueError('Vectors must be elements of the same subspace.')
        
    constraints = {f'span{vectors}'}
    return VectorSpace(field, n, constraints, rs_matrix=vectors)

def columnspace(matrix, field=Real):
    pass

def rowspace(matrix, field=Real):
    return VectorSpace(field, )

def nullspace(matrix, field=Real):
    pass