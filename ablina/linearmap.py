import sympy as sp

from .utils import is_invertible, of_arity
from .vectorspace import VectorSpace


class LinearMapError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class IsomorphismError(Exception):
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
            raise TypeError('The domain must be a VectorSpace.')
        if not isinstance(codomain, VectorSpace):
            raise TypeError('The codomain must be a VectorSpace.')
        if mapping is None and matrix is None:
            raise LinearMapError('Either a matrix or mapping must be provided.')
        if domain.field is not codomain.field:
            raise LinearMapError(
                'The domain and codomain must be vector spaces over the same field.'
                )
        
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
        int: The rank of the linear map.
        """
        return self.matrix.rank()
    
    @property
    def nullity(self):
        """
        int: The nullity of the linear map.
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
            return NotImplemented
        return (
            self.domain == map2.domain 
            and self.codomain == map2.codomain 
            and self.matrix == map2.matrix
            )
    
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
        return is_invertible(self.matrix)

    # Aliases
    range = image
    nullspace = kernel


class LinearOperator(LinearMap):
    """
    pass
    """

    def __init__(self, vectorspace, mapping=None, matrix=None, name=None):
        super().__init__(vectorspace, vectorspace, mapping, matrix, name)


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
            raise IsomorphismError('Linear map is not invertible.')

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
            The domain and codomain of the identity map.

        Returns
        -------
        IdentityMap
            pass
        """
        super().__init__(vectorspace, vectorspace, lambda vec: vec)