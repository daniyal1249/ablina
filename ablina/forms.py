import sympy as sp

from .utils import is_invertible, of_arity
from .vectorspace import VectorSpace


class FormError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class BilinearForm:
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the bilinear form is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : list of list or sympy.Matrix, optional
            The matrix representation of the bilinear form with respect 
            to the basis of the vector space.
        name : str, optional
            pass

        Returns
        -------
        BilinearForm
            pass

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError('vectorspace must be of type VectorSpace.')
        if mapping is None and matrix is None:
            raise FormError('Either a matrix or mapping must be provided.')
        
        if mapping is None:
            mapping = BilinearForm._from_matrix(vectorspace, matrix)
        elif not of_arity(mapping, 2):
            raise TypeError('Mapping must be a function of arity 2.')
        if matrix is None:
            matrix = BilinearForm._to_matrix(vectorspace, mapping)
        else:
            matrix = sp.Matrix(matrix)
        
        self._vectorspace = vectorspace
        self._mapping = mapping
        self._matrix = matrix
        if name is not None:
            self.__name__ = name

    @staticmethod
    def _to_matrix(vectorspace, mapping):
        # matrix = []
        # for vec in domain.basis:
        #     mapped_vec = mapping(vec)
        #     coord_vec = codomain.to_coordinate(mapped_vec)
        #     matrix.extend(coord_vec)
        # return sp.Matrix(domain.dim, codomain.dim, matrix).T
        pass

    @staticmethod
    def _from_matrix(vectorspace, matrix):
        # matrix = sp.Matrix(matrix)
        # def to_coord(vec): return sp.Matrix(domain.to_coordinate(vec))
        # def from_coord(vec): return codomain.from_coordinate(vec.flat())
        # return lambda vec: from_coord(matrix @ to_coord(vec))
        pass

    @property
    def vectorspace(self):
        """
        VectorSpace: The vector space the bilinear form is defined on.
        """
        return self._vectorspace
    
    @property
    def mapping(self):
        """
        callable: The function that maps vectors to scalars.
        """
        return self._mapping
    
    @property
    def matrix(self):
        """
        sympy.Matrix: The matrix representation of the bilinear form.
        """
        return self._matrix


class SesquilinearForm:
    pass


class HermitianForm(SesquilinearForm):
    pass


class InnerProduct:
    pass