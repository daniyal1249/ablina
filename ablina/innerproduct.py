import sympy as sp

from .operations import InnerProduct
from .vectorspace import VectorSpace


class InnerProductSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InnerProductSpace:
    """
    pass
    """

    def __init__(self, vectorspace, innerproduct=None):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            pass
        innerproduct : callable
            pass

        Returns
        -------
        InnerProductSpace
            pass
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError('vectorspace must be of type VectorSpace.')
        self._innerproduct = InnerProductSpace._init_innerproduct(innerproduct)

    @staticmethod
    def _init_innerproduct(ip):
        if ip is None:
            raise NotImplementedError()
        return ip
    
    def dot(self, vec1, vec2):
        """
        The dot product between two vectors.

        Parameters
        ----------
        vec1, vec2
            The vectors in the inner product space.

        Returns
        -------
        float
            The dot product between `vec1` and `vec2`.

        See Also
        --------
        VectorSpace.norm, VectorSpace.are_orthogonal

        Examples
        --------

        >>> vs = fn(Real, 3)
        >>> vs.dot([1, 2, 3], [4, 5, 6])
        32
        >>> vs.dot([1, 0, 1], [0, 1, 0])
        0
        """
        return self._innerproduct(vec1, vec2)

    def ortho_complement(self):
        """
        The orthogonal complement of an inner product space.

        Returns
        -------
        InnerProductSpace
            The orthogonal complement of `self`.

        See Also
        --------
        VectorSpace.ortho_projection, VectorSpace.dot

        Examples
        --------

        >>> vs = fn(Real, 3, constraints=['v0 == v1'])
        >>> vs.ortho_complement().basis
        [[1, -1, 0]]
        >>> vs.ortho_complement().ortho_complement() == vs
        True
        """
        raise NotImplementedError()
    
    def ortho_projection(self, vs2):
        """
        The orthogonal projection of `self` onto `vs2`.

        Parameters
        ----------
        vs2 : VectorSpace
            pass

        Returns
        -------
        InnerProductSpace
            pass

        Raises
        ------
        InnerProductSpaceError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.ortho_complement, VectorSpace.dot
        """
        raise NotImplementedError()