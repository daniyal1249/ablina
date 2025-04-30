from numbers import Complex, Real

import sympy as sp

from .utils import is_invertible, of_arity
from .vectorspace import VectorSpace

# Note that methods/properties such as is_positive_definite 
# will return None if the matrix is symbolic


class FormError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InnerProductError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class SesquilinearForm:
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the form is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : list of list or sympy.Matrix, optional
            The matrix representation of the form with respect to the 
            basis of the vector space.
        name : str, optional
            pass

        Returns
        -------
        SesquilinearForm
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
            mapping = SesquilinearForm._from_matrix(vectorspace, matrix)
        elif not of_arity(mapping, 2):
            raise TypeError('Mapping must be a function of arity 2.')
        if matrix is None:
            matrix = SesquilinearForm._to_matrix(vectorspace, mapping)
        else:
            matrix = sp.Matrix(matrix)
        
        self._vectorspace = vectorspace
        self._mapping = mapping
        self._matrix = matrix
        if name is not None:
            self.__name__ = name

    @staticmethod
    def _to_matrix(vectorspace, mapping):
        basis = vectorspace.basis
        n = len(basis)
        return sp.Matrix(n, n, lambda i, j: mapping(basis[i], basis[j]))

    @staticmethod
    def _from_matrix(vectorspace, matrix):
        matrix = sp.Matrix(matrix)
        def to_coord(v): return sp.Matrix(vectorspace.to_coordinate(v))
        return lambda u, v: (to_coord(u).H @ matrix @ to_coord(v))[0]

    @property
    def vectorspace(self):
        """
        VectorSpace: The vector space the form is defined on.
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
        sympy.Matrix: The matrix representation of the form.
        """
        return self._matrix
    
    def __repr__(self):
        return (
            f'SesquilinearForm(vectorspace={self.vectorspace}, '
            f'mapping={self.mapping.__name__}, '
            f'matrix={self.matrix})'
            )
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, form2):
        if not isinstance(form2, SesquilinearForm):
            return NotImplemented
        return (
            self.vectorspace == form2.vectorspace 
            and self.matrix == form2.matrix
            )
    
    def __call__(self, vec1, vec2):
        if vec1 not in self.vectorspace or vec2 not in self.vectorspace:
            raise TypeError(f'Vectors must be elements of the vector space.')
        return self.mapping(vec1, vec2)

    def inertia(self):
        if self.vectorspace.field is Real:
            if not self.is_symmetric():
                raise FormError()
        elif not self.is_hermitian():
            raise FormError()
        tol = 1e-8
        eigenvals = self.matrix.evalf().eigenvals().items()
        p = sum(m for val, m in eigenvals if val >= tol)
        m = sum(m for val, m in eigenvals if val <= -tol)
        z = sum(m for val, m in eigenvals if abs(val) < tol)
        return p, m, z
    
    def signature(self):
        p, m, _ = self.inertia()
        return p - m

    def is_degenerate(self):
        return not is_invertible(self.matrix)
    
    def is_symmetric(self):
        return self.matrix.is_symmetric()

    def is_hermitian(self):
        if self.vectorspace.field is not Complex:
            raise FormError()
        return self.matrix.is_hermitian

    def is_positive_definite(self):
        if not self.matrix.is_hermitian:
            return False
        return self.matrix.is_positive_definite

    def is_negative_definite(self):
        if not self.matrix.is_hermitian:
            return False
        return self.matrix.is_negative_definite

    def is_positive_semidefinite(self):
        if not self.matrix.is_hermitian:
            return False
        return self.matrix.is_positive_semidefinite

    def is_negative_semidefinite(self):
        if not self.matrix.is_hermitian:
            return False
        return self.matrix.is_negative_semidefinite

    def is_indefinite(self):
        if not self.matrix.is_hermitian:
            return False
        return self.matrix.is_indefinite


class InnerProduct(SesquilinearForm):
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the inner product is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : list of list or sympy.Matrix, optional
            The matrix representation of the inner product with respect 
            to the basis of the vector space.
        name : str, optional
            pass

        Returns
        -------
        InnerProduct
            pass

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        InnerProductError
            If 
        """
        super().__init__(vectorspace, mapping, matrix, name)

        if not self.is_positive_definite():
            raise InnerProductError('Form is not positive definite.')


class QuadraticForm:
    pass
