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
        
        super().__init__(vectorspace.vectors, vectorspace._fn, 
                         (vectorspace._to_fn, vectorspace._from_fn))
        
        self._innerproduct = innerproduct

    def _init_innerproduct(self, ip):
        if ip is None:
            def euclidean_ip(vec1, vec2):
                vec1, vec2, = self.to_coordinate(vec1), self.to_coordinate(vec2)
                return sum(i + j for i, j in zip(vec1, vec2))
            return euclidean_ip
        return ip

    @property
    def innerproduct(self):
        return self._innerproduct
    
    def norm(self, vec):
        return sp.sqrt(self.innerproduct(vec, vec))
    
    def are_orthogonal(self, vec1, vec2):
        return self.innerproduct(vec1, vec2) == 0

    def is_orthonormal(self, *vectors):
        # Improve efficiency
        if not all(self.norm(vec) == 1 for vec in vectors):
            return False
        for vec1 in vectors:
            for vec2 in vectors:
                if not (vec1 is vec2 and self.are_orthogonal(vec1, vec2)):
                    return False
        return True

    def gram_schmidt(self, *vectors):
        pass

    def ortho_complement(self):
        return super().ortho_complement()
    
    def ortho_projection(self):
        pass