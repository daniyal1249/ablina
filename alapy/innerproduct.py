import sympy as sp

from alapy.operations import InnerProduct
from alapy.vectorspace import VectorSpace


class InnerProductSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InnerProductSpace(VectorSpace):
    """
    """

    def __init__(self, vectorspace, innerproduct=None):
        """

        Parameters
        ----------
        vectorspace : VectorSpace
            x
        innerproduct : callable
            x

        Returns
        -------
        InnerProductSpace
            x
        """
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
        """
        The dot (inner) product between two vectors.

        Parameters
        ----------
        vec1, vec2
            The vectors in the inner product space.

        Returns
        -------
        float
            The result of the dot product between `vec1` and `vec2`.
        """
        return self._innerproduct(vec1, vec2)

    def ortho_complement(self):
        """

        Returns
        -------
        InnerProductSpace
            The orthogonal complement of `self`.
        """
        raise NotImplementedError()
    
    def ortho_projection(self, vs2):
        """

        Parameters
        ----------
        vs2 : InnerProductSpace
            x

        Returns
        -------
        InnerProductSpace
            x

        Raises
        ------
        InnerProductSpaceError
            If `self` and `vs2` do not share the same ambient space.
        """
        raise NotImplementedError()