from math_set import *
from vs_base import StandardFn
from utils import of_arity

class LinearMapError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class IsomorphismError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class LinearMap:  # implement matrix representation
    def __init__(self, domain, codomain, mapping, name=None):
        if not isinstance(domain, (MathematicalSet, StandardFn)):
            raise TypeError('The domain must be a MathematicalSet or '
                            'VectorSpace.')
        if not isinstance(codomain, (MathematicalSet, StandardFn)):
            raise TypeError('The codomain must be a MathematicalSet or '
                            'VectorSpace.')
        if not of_arity(mapping, 1):
            raise TypeError('The mapping must be able to accept a single '
                            'positional argument.')

        self._domain = domain
        self._codomain = codomain
        self._mapping = mapping
        if name:
            self.__name__ = name

    @staticmethod
    def _as_matrix():
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
    
    def __call__(self, vec):
        if vec not in self.domain:
            raise TypeError(f'{vec} is not an element of the domain.')
        return self.mapping(vec)
    
    def composition(self, map2):
        if self.domain != map2.codomain:
            raise LinearMapError('The linear maps are not compatible.')
        
        mapping = lambda vec: self.mapping(map2.mapping(vec))
        if hasattr(self, '__name__') and hasattr(map2, '__name__'):
            name = f'{self.__name__} o {map2.__name__}'
        else:
            name = None
        return LinearMap(map2.domain, self.codomain, mapping, name)


class Isomorphism(LinearMap):
    def __repr__(self):
        return (f'Isomorphism(domain={self.domain}, codomain={self.codomain}, '
                f'mapping={self.mapping.__name__})')

class IdentityMap(Isomorphism):
    def __init__(self, vectors):
        super().__init__(vectors, vectors, lambda vec: vec)
