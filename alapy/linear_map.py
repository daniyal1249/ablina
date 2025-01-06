import sympy as sp

from alapy.math_set import *
from alapy.vector_space import VectorSpace
from alapy.utils import of_arity

class LinearMapError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class IsomorphismError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class LinearMap:  # implement matrix representation
    def __init__(self, domain, codomain, mapping=None, matrix=None, name=None):
        if not isinstance(domain, VectorSpace):
            raise TypeError('The domain must be a VectorSpace.')
        if not isinstance(codomain, VectorSpace):
            raise TypeError('The codomain must be a VectorSpace.')
        if mapping is None and matrix is None:
            raise LinearMapError('Either a matrix or mapping must be provided.')
        if domain.field is not codomain.field:
            raise LinearMapError()
        
        if mapping is None:
            mapping = LinearMap._from_matrix(matrix)
        if matrix is None:
            matrix = LinearMap._to_matrix(mapping)
        else:
            matrix = sp.Matrix(matrix)
        
        self._domain = domain
        self._codomain = codomain
        self._mapping = mapping
        self._matrix = matrix
        if name is not None:
            self.__name__ = name

    @staticmethod
    def _to_matrix(mapping):
        pass

    @staticmethod
    def _from_matrix(matrix):
        pass

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    @property
    def mapping(self):
        return self._mapping
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def field(self):
        return self.domain.field
    
    def __repr__(self):
        return (f'LinearMap(domain={self.domain}, codomain={self.codomain}, '
                f'mapping={self.mapping.__name__})')
    
    def __str__(self):
        if hasattr(self, '__name__'):
            return f'{self.__name__}: {self.domain} -> {self.codomain}'
        else:
            return self.__repr__()

    def __eq__(self, map2):
        return vars(self) == vars(map2)
    
    def __add__(self, map2):
        mapping = lambda vec: self.mapping(vec) + map2.mapping(vec)
        matrix = self.matrix + map2.matrix
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __rmul__(self, scalar):
        if not isinstance(scalar, self.field):
            raise TypeError()
        mapping = lambda vec: scalar * self.mapping(vec)
        matrix = scalar * self.matrix
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __call__(self, vec):
        if vec not in self.domain:
            raise TypeError(f'{vec} is not an element of the domain.')
        return self.mapping(vec)
    
    def composition(self, map2):
        if self.domain != map2.codomain:
            raise LinearMapError('The linear maps are not compatible.')
        
        mapping = lambda vec: self.mapping(map2.mapping(vec))
        matrix = self.matrix @ map2.matrix  # check
        if hasattr(self, '__name__') and hasattr(map2, '__name__'):
            name = f'{self.__name__} o {map2.__name__}'
        else:
            name = None
        return LinearMap(map2.domain, self.codomain, mapping, matrix, name)


class Isomorphism(LinearMap):
    def __repr__(self):
        return (f'Isomorphism(domain={self.domain}, codomain={self.codomain}, '
                f'mapping={self.mapping.__name__})')

class IdentityMap(Isomorphism):
    def __init__(self, vectors):
        super().__init__(vectors, vectors, lambda vec: vec)
