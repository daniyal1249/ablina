from numbers import Real

import sympy as sp

from . import utils as u
from .vectorspace import VectorSpace


class LinearMapError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class LinearMap:
    """
    pass
    """
    
    def __init__(self, domain, codomain, mapping=None, matrix=None, name=None):
        """
        pass

        Parameters
        ----------
        domain : VectorSpace
            The domain of the linear map.
        codomain : VectorSpace
            The codomain of the linear map.
        mapping : callable, optional
            A function that takes a vector in the domain and returns a 
            vector in the codomain.
        matrix : list of list or sympy.Matrix, optional
            The matrix representation of the linear map with respect to 
            the basis vectors of the domain and codomain.
        name : str, optional
            pass

        Returns
        -------
        LinearMap
            pass

        Raises
        ------
        LinearMapError
            If neither the mapping nor the matrix is provided.
        LinearMapError
            If the field of the domain and codomain are not the same.
        """
        if not isinstance(domain, VectorSpace):
            raise TypeError('Domain must be a VectorSpace.')
        if not isinstance(codomain, VectorSpace):
            raise TypeError('Codomain must be a VectorSpace.')
        if mapping is None and matrix is None:
            raise LinearMapError('Either a matrix or mapping must be provided.')
        if domain.field is not codomain.field:
            raise LinearMapError(
                'Domain and codomain must be vector spaces over the same field.'
                )
        
        if mapping is None:
            mapping = LinearMap._from_matrix(domain, codomain, matrix)
        elif not u.of_arity(mapping, 1):
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
            matrix.append(coord_vec)
        return sp.Matrix(matrix).T

    @staticmethod
    def _from_matrix(domain, codomain, matrix):
        matrix = sp.Matrix(matrix)
        def to_coord(vec): return sp.Matrix(domain.to_coordinate(vec))
        def from_coord(vec): return codomain.from_coordinate(vec.flat())
        return lambda vec: from_coord(matrix @ to_coord(vec))

    @property
    def field(self):
        """
        {Real, Complex}: The field of the domain and codomain.
        """
        return self.domain.field

    @property
    def domain(self):
        """
        VectorSpace: The domain of the linear map.
        """
        return self._domain
    
    @property
    def codomain(self):
        """
        VectorSpace: The codomain of the linear map.
        """
        return self._codomain
    
    @property
    def mapping(self):
        """
        callable: The function that maps vectors from the domain to the codomain.
        """
        return self._mapping
    
    @property
    def matrix(self):
        """
        sympy.Matrix: The matrix representation of the linear map.
        """
        return self._matrix
    
    @property
    def rank(self):
        """
        int: The dimension of the image of the linear map.
        """
        return self.matrix.rank()
    
    @property
    def nullity(self):
        """
        int: The dimension of the kernel of the linear map.
        """
        return self.matrix.cols - self.rank
    
    def __repr__(self):
        return (
            f'LinearMap(domain={self.domain}, '
            f'codomain={self.codomain}, '
            f'mapping={self.mapping.__name__}, '
            f'matrix={self.matrix})'
            )
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, map2):
        if not isinstance(map2, LinearMap):
            return False
        if not (self.domain == map2.domain and self.codomain == map2.codomain):
            return False
        basis1, basis2 = map2.domain.basis, map2.codomain.basis
        matrix, _, _ = LinearMap.change_of_basis(self, basis1, basis2)
        return map2.matrix == matrix  # FIX: consider .equals()
    
    def __add__(self, map2):
        """
        The sum of two linear maps.

        Parameters
        ----------
        map2 : LinearMap
            The linear map being added.

        Returns
        -------
        LinearMap
            The sum of `self` and `map2`.

        Examples
        --------
        
        >>> R3 = fn(Real, 3)
        >>> def mapping1(vec): return [2*i for i in vec]
        >>> def mapping2(vec): return [3*i for i in vec]
        >>> map1 = LinearMap(R3, R3, mapping1)
        >>> map2 = LinearMap(R3, R3, mapping2)
        >>> map3 = map1 + map2
        >>> map3([1, 2, 3])
        [5, 10, 15]
        """
        # FIX: Add check to make sure the domains and codomains are equal
        def mapping(vec):
            vec1 = self.mapping(vec)
            vec2 = map2.mapping(vec)
            return self.codomain.add(vec1, vec2)
        matrix = self.matrix + map2.matrix
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __mul__(self, scalar):
        """
        The product of a linear map and scalar.

        Parameters
        ----------
        scalar : Real or Complex
            The scalar to multiply with.

        Returns
        -------
        LinearMap
            The product of `self` and `scalar`.

        Examples
        --------
        
        >>> R3 = fn(Real, 3)
        >>> def mapping(vec): return [2*i for i in vec]
        >>> map1 = LinearMap(R3, R3, mapping)
        >>> map2 = 3 * map1
        >>> map2([1, 2, 3])
        [6, 12, 18]
        """
        if not isinstance(scalar, self.field):
            raise TypeError('Scalar must be an element of the vector space field.')
        def mapping(vec):
            return self.codomain.mul(scalar, self.mapping(vec))
        matrix = self.matrix * scalar
        return LinearMap(self.domain, self.codomain, mapping, matrix)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __call__(self, vec):
        """
        Apply the linear map to a vector.

        Parameters
        ----------
        vec : object
            The vector to map.

        Returns
        -------
        object
            The vector that `vec` maps to.

        Examples
        --------
        
        >>> R3 = fn(Real, 3)
        >>> def mapping(vec): return [2*i for i in vec]
        >>> map1 = LinearMap(R3, R3, mapping)
        >>> map1([1, 2, 3])
        [2, 4, 6]
        """
        if vec not in self.domain:
            raise TypeError(f'{vec} is not an element of the domain.')
        return self.mapping(vec)
    
    def change_of_basis(self, domain_basis=None, codomain_basis=None):
        """
        pass
        """
        if domain_basis is None:
            domain_basechange = sp.eye(self.domain.dim)
        else:
            domain_basechange = self.domain.change_of_basis(domain_basis)
        if codomain_basis is None:
            codomain_basechange = sp.eye(self.codomain.dim)
        else:
            codomain_basechange = self.codomain.change_of_basis(codomain_basis) 

        map_matrix = codomain_basechange @ self.matrix @ domain_basechange.inv()
        return map_matrix, domain_basechange, codomain_basechange

    def composition(self, map2):
        """
        The composition of two linear maps.

        Parameters
        ----------
        map2 : LinearMap
            The linear map to compose with.

        Returns
        -------
        LinearMap
            The composition of `self` and `map2`.

        Raises
        ------
        LinearMapError
            If the domain of `self` is not equal to the codomain of `map2`.

        Examples
        --------
        
        >>> R3 = fn(Real, 3)
        >>> def mapping1(vec): return [2*i for i in vec]
        >>> def mapping2(vec): return [3*i for i in vec]
        >>> map1 = LinearMap(R3, R3, mapping1)
        >>> map2 = LinearMap(R3, R3, mapping2)
        >>> map3 = map1.composition(map2)
        >>> map3([1, 2, 3])
        [6, 12, 18]
        """
        if self.domain != map2.codomain:
            raise LinearMapError('The linear maps are not compatible.')
        
        def mapping(vec):
            return self.mapping(map2.mapping(vec))
        matrix = self.matrix @ map2.matrix
        if hasattr(self, '__name__') and hasattr(map2, '__name__'):
            name = f'{self.__name__} o {map2.__name__}'
        else:
            name = None
        return LinearMap(map2.domain, self.codomain, mapping, matrix, name)
    
    def image(self):
        """
        The image, or range, of the linear map.

        Returns
        -------
        VectorSpace
            The image of `self`.

        See Also
        --------
        LinearMap.range
        """
        basis = [vec.tolist() for vec in self.matrix.columnspace()]
        basis = [self.domain.from_coordinate(vec) for vec in basis]
        return self.domain.span(*basis)

    def kernel(self):
        """
        The kernel, or null space, of the linear map.

        Returns
        -------
        VectorSpace
            The kernel of `self`.

        See Also
        --------
        LinearMap.nullspace
        """
        basis = [vec.tolist() for vec in self.matrix.nullspace()]
        basis = [self.domain.from_coordinate(vec) for vec in basis]
        return self.domain.span(*basis)
    
    def adjoint(self):
        """
        The adjoint of the linear map.
        """
        raise NotImplementedError()
    
    def pseudoinverse(self):
        """
        The pseudoinverse of the linear map.
        """
        raise NotImplementedError()

    def is_injective(self):
        """
        Check whether the linear map is injective.

        Returns
        -------
        bool
            True if the linear map is injective, otherwise False.

        See Also
        --------
        LinearMap.is_surjective, LinearMap.is_bijective
        """
        return self.matrix.cols == self.rank

    def is_surjective(self):
        """
        Check whether the linear map is surjective.

        Returns
        -------
        bool
            True if the linear map is surjective, otherwise False.

        See Also
        --------
        LinearMap.is_injective, LinearMap.is_bijective
        """
        return self.matrix.rows == self.rank
    
    def is_bijective(self):
        """
        Check whether the linear map is bijective.

        Returns
        -------
        bool
            True if the linear map is bijective, otherwise False.

        See Also
        --------
        LinearMap.is_injective, LinearMap.is_surjective
        """
        return u.is_invertible(self.matrix)

    # Aliases
    range = image
    nullspace = kernel


class LinearOperator(LinearMap):
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        super().__init__(vectorspace, vectorspace, mapping, matrix, name)
    
    def change_of_basis(self, basis):
        """
        pass
        """
        basechange = self.domain.change_of_basis(basis)
        map_matrix = basechange @ self.matrix @ basechange.inv()
        return map_matrix, basechange
    
    def is_symmetric(self, innerproduct):
        """
        Check whether the linear operator is symmetric.

        Note that this method is only valid for operators defined on 
        real vector spaces. An exception is raised otherwise.

        Returns
        -------
        bool
            True if `self` is symmetric, otherwise False.

        Raises
        ------
        LinearMapError
            If `self` is not defined on a real vector space.

        See Also
        --------
        LinearOperator.is_hermitian
        """
        if self.domain.field is not Real:
            raise LinearMapError()
        matrix, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return matrix.is_symmetric()

    def is_hermitian(self, innerproduct):
        """
        Check whether the linear operator is hermitian.

        Returns
        -------
        bool
            True if `self` is hermitian, otherwise False.

        See Also
        --------
        LinearOperator.is_symmetric
        """
        matrix, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return matrix.is_hermitian

    def is_orthogonal(self, innerproduct):
        """
        Check whether the linear operator is orthogonal.

        Note that this method is only valid for operators defined on 
        real vector spaces. An exception is raised otherwise.

        Returns
        -------
        bool
            True if `self` is orthogonal, otherwise False.

        Raises
        ------
        LinearMapError
            If `self` is not defined on a real vector space.

        See Also
        --------
        LinearOperator.is_unitary
        """
        if self.domain.field is not Real:
            raise LinearMapError()
        matrix, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_orthogonal(matrix)

    def is_unitary(self, innerproduct):
        """
        Check whether the linear operator is unitary.

        Returns
        -------
        bool
            True if `self` is unitary, otherwise False.

        See Also
        --------
        LinearOperator.is_orthogonal
        """
        matrix, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_unitary(matrix)
    
    def is_normal(self, innerproduct):
        """
        Check whether the linear operator is normal.

        Returns
        -------
        bool
            True if `self` is normal, otherwise False.
        """
        matrix, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_normal(matrix)


class LinearFunctional(LinearMap):
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the linear functional is defined on.
        mapping : callable, optional
            A function that takes a vector in the vector space and 
            returns a scalar in the field.
        matrix : list of list or sympy.Matrix, optional
            The matrix representation of the linear functional with 
            respect to the basis of the vector space.
        name : str, optional
            pass

        Returns
        -------
        LinearFunctional
            pass

        Raises
        ------
        LinearMapError
            If neither the mapping nor the matrix is provided.
        """
        super().__init__(vectorspace, ..., mapping, matrix, name)


class Isomorphism(LinearMap):
    """
    pass
    """

    def __init__(self, domain, codomain, mapping=None, matrix=None, name=None):
        super().__init__(domain, codomain, mapping, matrix, name)

        if not self.is_bijective():
            raise LinearMapError('Linear map is not invertible.')

    def __repr__(self):
        return super().__repr__().replace('LinearMap', 'Isomorphism')
    
    def inverse(self):
        """
        The inverse of the linear map.

        Returns
        -------
        Isomorphism
            The inverse of `self`.
        """
        matrix = self.matrix.inv()
        return Isomorphism(self.codomain, self.domain, matrix=matrix)


class IdentityMap(Isomorphism):
    """
    pass
    """

    def __init__(self, vectorspace):
        """
        pass

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the identity map is defined on.

        Returns
        -------
        IdentityMap
            pass
        """
        super().__init__(vectorspace, vectorspace, lambda vec: vec)