import sympy as sp

from alapy.utils import is_invertible, of_arity
from alapy.vectorspace import VectorSpace


class LinearMapError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class IsomorphismError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class LinearMap:
    def __init__(self, domain, codomain, mapping=None, matrix=None, name=None):
        if not isinstance(domain, VectorSpace):
            raise TypeError('The domain must be a VectorSpace.')
        if not isinstance(codomain, VectorSpace):
            raise TypeError('The codomain must be a VectorSpace.')
        if mapping is None and matrix is None:
            raise LinearMapError('Either a matrix or mapping must be provided.')
        if domain.field is not codomain.field:
            raise LinearMapError('The domain and codomain must be vector spaces '
                                 'over the same field.')
        
        if mapping is None:
            mapping = LinearMap._from_matrix(domain, codomain, matrix)
        elif not of_arity(mapping, 1):
            raise TypeError('Mapping must be a function of arity 1.')
        if matrix is None:
            matrix = LinearMap._to_matrix(domain, codomain, mapping)
        else:
            matrix = sp.Matrix(matrix)
        
        self._domain = domain
        self._codomain = codomain
        self._mapping = mapping
        self._matrix = matrix
        if name is not None:
            self.__name__ = name

    @staticmethod
    def _to_matrix(domain, codomain, mapping):
        matrix = []
        for vec in domain.basis:
            mapped_vec = mapping(vec)
            coord_vec = codomain.to_coordinate(mapped_vec)
            matrix.extend(coord_vec)
        return sp.Matrix(domain.dim, codomain.dim, matrix).T

    @staticmethod
    def _from_matrix(domain, codomain, matrix):
        matrix = sp.Matrix(matrix)
        def to_coord(vec): sp.Matrix(domain.to_coordinate(vec))
        def from_coord(vec): codomain.from_coordinate(vec.flat())
        return lambda vec: from_coord(matrix @ to_coord(vec))

    @property
    def field(self):
        return self.domain.field

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
    def rank(self):
        return self.matrix.rank()
    
    @property
    def nullity(self):
        return self.matrix.cols - self.rank
    
    def __repr__(self):
        return (f'LinearMap(domain={self.domain}, '
                f'codomain={self.codomain}, '
                f'mapping={self.mapping.__name__}, '
                f'matrix={self.matrix})')
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, map2):
        return (self.domain == map2.domain 
                and self.codomain == map2.codomain 
                and self.matrix == map2.matrix)
    
    def __add__(self, map2):
        def mapping(vec): self.mapping(vec) + map2.mapping(vec)
        matrix = self.matrix + map2.matrix
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, self.field):
            raise TypeError('Scalar must be an element of the vector space field.')
        def mapping(vec): self.mapping(vec) * scalar
        matrix = self.matrix * scalar
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __rmul__(self, scalar):
        if not isinstance(scalar, self.field):
            raise TypeError('Scalar must be an element of the vector space field.')
        def mapping(vec): scalar * self.mapping(vec)
        matrix = scalar * self.matrix
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __call__(self, vec):
        if vec not in self.domain:
            raise TypeError(f'{vec} is not an element of the domain.')
        return self.mapping(vec)
    
    def composition(self, map2):
        if self.domain != map2.codomain:
            raise LinearMapError('The linear maps are not compatible.')
        
        def mapping(vec): self.mapping(map2.mapping(vec))
        matrix = self.matrix @ map2.matrix
        if hasattr(self, '__name__') and hasattr(map2, '__name__'):
            name = f'{self.__name__} o {map2.__name__}'
        else:
            name = None
        return LinearMap(map2.domain, self.codomain, mapping, matrix, name)
    
    def range(self):
        basis = [vec.tolist() for vec in self.matrix.columnspace()]
        basis = [self.domain.from_coordinate(vec) for vec in basis]
        return self.domain.span(*basis)

    def nullspace(self):
        basis = [vec.tolist() for vec in self.matrix.nullspace()]
        basis = [self.domain.from_coordinate(vec) for vec in basis]
        return self.domain.span(*basis)
    
    def pseudoinverse(self):
        raise NotImplementedError()
    
    def adjoint(self):
        raise NotImplementedError()

    def is_injective(self):
        return self.matrix.cols == self.rank

    def is_surjective(self):
        return self.matrix.rows == self.rank
    
    def is_bijective(self):
        return is_invertible(self.matrix)

    # Aliases
    image = range
    kernel = nullspace


class Isomorphism(LinearMap):
    def __init__(self, domain, codomain, mapping=None, matrix=None, name=None):
        super().__init__(domain, codomain, mapping, matrix, name)

        if not self.is_bijective():
            raise IsomorphismError('Linear map is not invertible.')

    def __repr__(self):
        return super().__repr__().replace('LinearMap', 'Isomorphism')
    
    def inverse(self):
        matrix = self.matrix.inv()
        return Isomorphism(self.codomain, self.domain, matrix=matrix)


class IdentityMap(Isomorphism):
    def __init__(self, vectorspace, name=None):
        super().__init__(vectorspace, vectorspace, lambda vec: vec, name=name)