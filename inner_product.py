import sympy as sp

from vector_space import VectorSpace

class InnerProductSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class InnerProductSpace(VectorSpace):
    def __init__(self, vectorspace, innerproduct):
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError('vectorspace must be of type VectorSpace.')
        
        super().__init__(vectorspace.vectors, vectorspace._fn, 
                         (vectorspace._to_fn, vectorspace._from_fn))
        
        self._innerproduct = innerproduct

    @property
    def innerproduct(self):
        return self._innerproduct