import sympy as sp

from alapy.vector_space import VectorSpace
from alapy.operation import InnerProduct

class InnerProductSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class InnerProductSpace(VectorSpace):
    def __init__(self, vectorspace, innerproduct=None):
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError('vectorspace must be of type VectorSpace.')
        
        super().__init__(vectorspace._vectors, vectorspace._fn, 
                         (vectorspace._to_fn, vectorspace._from_fn))
        
        self._innerproduct = self._init_innerproduct(innerproduct)

    def _init_innerproduct(self, ip):
        if ip is None:
            return super().dot
        return ip
    
    def dot(self, vec1, vec2):
        return self._innerproduct(vec1, vec2)

    def ortho_complement(self):
        return super().ortho_complement()
    
    def ortho_projection(self, vs2):
        pass
