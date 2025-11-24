"""
A module for working with linear maps between vector spaces.
"""

from __future__ import annotations

from typing import Any, Callable

from .field import Field, R
from .form import InnerProduct
from .matrix import Matrix, M
from . import utils as u
from .vectorspace import VectorSpace, fn


class LinearMapError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


class LinearMap:
    """
    A linear map between vector spaces.
    
    A linear map `T` is a function from one vector space to another, 
    satisfying the following properties:

    - `T(u + v) = T(u) + T(v)` for all vectors `u` and `v`
    - `T(av) = a T(v)` for all scalars `a` and vectors `v`
    """
    
    def __init__(
        self, 
        name: str, 
        domain: VectorSpace, 
        codomain: VectorSpace, 
        mapping: Callable[[Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        """
        Initialize a LinearMap instance.

        Parameters
        ----------
        name : str
            The name of the linear map.
        domain : VectorSpace
            The domain of the linear map.
        codomain : VectorSpace
            The codomain of the linear map.
        mapping : callable, optional
            A function that takes a vector in the domain and returns a 
            vector in the codomain.
        matrix : Matrix, optional
            The matrix representation of the linear map with respect to 
            the bases of the domain and codomain.

        Returns
        -------
        LinearMap
            A new LinearMap instance.

        Raises
        ------
        LinearMapError
            If neither the mapping nor the matrix is provided.
        LinearMapError
            If the fields of the domain and codomain are not the same.
        """
        if not isinstance(domain, VectorSpace):
            raise TypeError("Domain must be a VectorSpace.")
        if not isinstance(codomain, VectorSpace):
            raise TypeError("Codomain must be a VectorSpace.")
        if domain.field is not codomain.field:
            raise LinearMapError(
                "Domain and codomain must be vector spaces over the same field."
                )
        
        matrix = LinearMap._to_matrix(domain, codomain, mapping, matrix)
        mapping = LinearMap._to_mapping(domain, codomain, mapping, matrix)
        
        self.name = name
        self._domain = domain
        self._codomain = codomain
        self._mapping = mapping
        self._matrix = matrix
    
    @staticmethod
    def _to_matrix(
        domain: VectorSpace, 
        codomain: VectorSpace, 
        mapping: Callable[[Any], Any] | None, 
        matrix: Any | None
    ) -> Matrix:
        if matrix is not None:
            return LinearMap._validate_matrix(domain, codomain, matrix)
        if mapping is None:
            raise LinearMapError("Either a matrix or mapping must be provided.")
        if not u.of_arity(mapping, 1):
            raise TypeError("Mapping must be a callable of arity 1.")
        
        mat = []
        for vec in domain.basis:
            mapped_vec = mapping(vec)
            coord_vec = codomain.to_coordinate(mapped_vec)
            mat.append(coord_vec)
        return M.hstack(*mat)

    @staticmethod
    def _to_mapping(
        domain: VectorSpace, 
        codomain: VectorSpace, 
        mapping: Callable[[Any], Any] | None, 
        matrix: Matrix
    ) -> Callable[[Any], Any]:
        if mapping is not None:
            return mapping
        to_coord = domain.to_coordinate
        from_coord = codomain.from_coordinate
        return lambda vec: from_coord(matrix @ to_coord(vec))
    
    @staticmethod
    def _validate_matrix(
        domain: VectorSpace, 
        codomain: VectorSpace, 
        matrix: Any
    ) -> Matrix:
        mat = M(matrix)
        if mat.shape != (codomain.dim, domain.dim):
            raise ValueError("Matrix has invalid shape.")
        if not all(i in domain.field for i in mat):
            raise ValueError("Matrix entries must be elements of the field.")
        return mat

    @property
    def field(self) -> Field:
        """
        Field: The field of the domain and codomain.
        """
        return self.domain.field

    @property
    def domain(self) -> VectorSpace:
        """
        VectorSpace: The domain of the linear map.
        """
        return self._domain
    
    @property
    def codomain(self) -> VectorSpace:
        """
        VectorSpace: The codomain of the linear map.
        """
        return self._codomain
    
    @property
    def mapping(self) -> Callable[[Any], Any]:
        """
        callable: The function that maps vectors from the domain to the codomain.
        """
        return self._mapping
    
    @property
    def matrix(self) -> Matrix:
        """
        Matrix: The matrix representation of the linear map.
        """
        return self._matrix
    
    @property
    def rank(self) -> int:
        """
        int: The dimension of the image of the linear map.
        """
        return self.matrix.rank()
    
    @property
    def nullity(self) -> int:
        """
        int: The dimension of the kernel of the linear map.
        """
        return self.matrix.cols - self.rank
    
    def __repr__(self) -> str:
        return (
            f"LinearMap(name={self.name!r}, "
            f"domain={self.domain!r}, "
            f"codomain={self.codomain!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __str__(self) -> str:
        return self.name

    def __eq__(self, map2: Any) -> bool:
        """
        Check for equality of two linear maps.

        Parameters
        ----------
        map2 : LinearMap
            The linear map to compare with.

        Returns
        -------
        bool
            True if both linear maps are equal, otherwise False.
        """
        if not isinstance(map2, LinearMap):
            return False
        if not (self.domain == map2.domain and self.codomain == map2.codomain):
            return False
        basis1, basis2 = map2.domain.basis, map2.codomain.basis
        mat, _, _ = LinearMap.change_of_basis(self, basis1, basis2)
        return map2.matrix.equals(mat) is True
    
    def __add__(self, map2: LinearMap) -> LinearMap:
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

        Raises
        ------
        LinearMapError
            If the domains and codomains of `self` and `map2` are not equal.

        Examples
        --------
        
        >>> R3 = fn("R3", R, 3)
        >>> map1 = LinearMap("map1", R3, R3, lambda vec: 2 * vec)
        >>> map2 = LinearMap("map2", R3, R3, lambda vec: 3 * vec)
        >>> map3 = map1 + map2
        >>> map3([1, 2, 3])
        [5, 10, 15]
        """
        if not (self.domain == map2.domain and self.codomain == map2.codomain):
            raise LinearMapError("The linear maps are not compatible.")
        
        name = f"{self} + {map2}"
        def mapping(vec: Any) -> Any:
            vec1 = self.mapping(vec)
            vec2 = map2.mapping(vec)
            return self.codomain.add(vec1, vec2)
        mat = self.matrix + map2.matrix
        return LinearMap(name, self.domain, self.codomain, mapping, mat)
    
    def __mul__(self, scalar: Any) -> LinearMap:
        """
        The product of the linear map and a scalar.

        Parameters
        ----------
        scalar : object
            The scalar to multiply by.

        Returns
        -------
        LinearMap
            The product of `self` and `scalar`.

        Raises
        ------
        TypeError
            If `scalar` is not an element of the field.

        Examples
        --------
        
        >>> R3 = fn("R3", R, 3)
        >>> map1 = LinearMap("map1", R3, R3, lambda vec: 2 * vec)
        >>> map2 = 3 * map1
        >>> map2([1, 2, 3])
        [6, 12, 18]
        """
        if scalar not in self.field:
            raise TypeError("Scalar must be an element of the field.")
        
        name = f"{scalar} * {self}"
        def mapping(vec: Any) -> Any:
            return self.codomain.mul(scalar, self.mapping(vec))
        mat = scalar * self.matrix
        return LinearMap(name, self.domain, self.codomain, mapping, mat)
    
    def __rmul__(self, scalar: Any) -> LinearMap:
        return self.__mul__(scalar)
    
    def __call__(self, obj: Any) -> Any:
        """
        Apply the linear map to a vector or subspace.

        Parameters
        ----------
        obj : object
            The vector or subspace to map.

        Returns
        -------
        object
            The vector or subspace that `obj` maps to.

        Examples
        --------
        
        >>> R3 = fn("R3", R, 3)
        >>> map1 = LinearMap("map1", R3, R3, lambda vec: 2 * vec)
        >>> map1([1, 2, 3])
        [2, 4, 6]
        """
        if obj in self.domain:
            return self.mapping(obj)
        return self.restriction(obj).image()
    
    def info(self) -> str:
        """
        A description of the linear map.

        Returns
        -------
        str
            The formatted description.
        """
        signature = f"{self} : {self.domain} → {self.codomain}"

        lines = [
            signature,
            "-" * len(signature),
            f"Field        {self.field}",
            f"Rank         {self.rank}",
            f"Nullity      {self.nullity}",
            f"Injective?   {self.is_injective()}",  
            f"Surjective?  {self.is_surjective()}",
            f"Bijective?   {self.is_bijective()}",
            f"Matrix       {self.matrix}"
            ]
        return "\n".join(lines)
    
    def change_of_basis(
        self, 
        domain_basis: list[Any] | None = None, 
        codomain_basis: list[Any] | None = None
    ) -> tuple[Matrix, Matrix, Matrix]:
        """
        Change the basis representation of the linear map.

        Returns the matrix representation of the linear map with respect 
        to new bases for the domain and codomain, along with the 
        change-of-basis matrices.

        Parameters
        ----------
        domain_basis : list of object, optional
            A new basis for the domain. If None, the current basis is used.
        codomain_basis : list of object, optional
            A new basis for the codomain. If None, the current basis is used.

        Returns
        -------
        tuple of (Matrix, Matrix, Matrix)
            A tuple containing the matrix representation with respect to 
            the new bases, the domain change-of-basis matrix, and the 
            codomain change-of-basis matrix.
        """
        if domain_basis is None:
            domain_basechange = M.eye(self.domain.dim)
        else:
            domain_basechange = self.domain.change_of_basis(domain_basis)
        if codomain_basis is None:
            codomain_basechange = M.eye(self.codomain.dim)
        else:
            codomain_basechange = self.codomain.change_of_basis(codomain_basis) 

        map_matrix = codomain_basechange @ self.matrix @ domain_basechange.inv()
        return map_matrix, domain_basechange, codomain_basechange
    
    def restriction(self, subspace: VectorSpace) -> LinearMap:
        """
        Restrict the linear map to a subspace of the domain.

        Parameters
        ----------
        subspace : VectorSpace
            A subspace of the domain.

        Returns
        -------
        LinearMap
            The restriction of `self` to `subspace`.

        Raises
        ------
        TypeError
            If `subspace` is not a subspace of the domain.
        """
        if not self.domain.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the domain.")
        name = f"{self} | {subspace}"
        return LinearMap(name, subspace, self.codomain, self.mapping)

    def inverse(self) -> LinearMap:
        """
        The inverse of the linear map.

        Returns
        -------
        LinearMap
            The inverse of `self`.

        Raises
        ------
        LinearMapError
            If `self` is not invertible.
        """
        if not self.is_bijective():
            raise LinearMapError("Linear map is not invertible.")
        name = f"{self}^-1"
        mat = self.matrix.inv()
        return LinearMap(name, self.codomain, self.domain, matrix=mat)

    def composition(self, map2: LinearMap) -> LinearMap:
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
        
        >>> R3 = fn("R3", R, 3)
        >>> map1 = LinearMap("map1", R3, R3, lambda vec: 2 * vec)
        >>> map2 = LinearMap("map2", R3, R3, lambda vec: 3 * vec)
        >>> map3 = map1.composition(map2)
        >>> map3([1, 2, 3])
        [6, 12, 18]
        """
        if not isinstance(map2, LinearMap):
            raise TypeError("map2 must be of type LinearMap.")
        if self.domain != map2.codomain:
            raise LinearMapError("The linear maps are not compatible.")
        
        name = f"{self} ∘ {map2}"
        def mapping(vec: Any) -> Any:
            return self.mapping(map2.mapping(vec))
        mat = self.matrix @ map2.matrix
        return LinearMap(name, map2.domain, self.codomain, mapping, mat)
    
    def image(self) -> VectorSpace:
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
        name = f"im({self})"
        basis = self.matrix.columnspace()
        basis = [self.codomain.from_coordinate(vec) for vec in basis]
        return self.codomain.span(name, *basis)

    def kernel(self) -> VectorSpace:
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
        name = f"ker({self})"
        basis = self.matrix.nullspace()
        basis = [self.domain.from_coordinate(vec) for vec in basis]
        return self.domain.span(name, *basis)
    
    def adjoint(self) -> LinearMap:
        """
        The adjoint of the linear map.
        """
        raise NotImplementedError("This method is not yet implemented.")
    
    def pseudoinverse(self) -> LinearMap:
        """
        The pseudoinverse of the linear map.

        Returns
        -------
        LinearMap
            The pseudoinverse of `self`.
        """
        name = f"{self}^+"
        mat = self.matrix.pinv()
        return LinearMap(name, self.codomain, self.domain, matrix=mat)

    def is_injective(self) -> bool:
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

    def is_surjective(self) -> bool:
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
    
    def is_bijective(self) -> bool:
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
        return self.is_injective() and self.is_surjective()

    # Aliases
    range = image
    """An alias for the image method."""

    nullspace = kernel
    """An alias for the kernel method."""


class LinearOperator(LinearMap):
    """
    A linear operator on a vector space.
    
    A linear map from a vector space to itself. This is a special case of 
    a LinearMap where the domain and codomain are the same.
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        super().__init__(name, vectorspace, vectorspace, mapping, matrix)

    def __repr__(self) -> str:
        return (
            f"LinearOperator(name={self.name!r}, "
            f"vectorspace={self.domain!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __pow__(self, exp: int) -> LinearOperator:
        """
        Raise the linear operator to a power.

        Parameters
        ----------
        exp : int
            The exponent to raise the operator to.

        Returns
        -------
        LinearOperator
            The linear operator `self` raised to the power `exp`.
        """
        name = f"{self}^{exp}"
        mat = self.matrix ** exp
        return LinearOperator(name, self.domain, matrix=mat)
    
    def change_of_basis(self, basis: list[Any]) -> tuple[Matrix, Matrix]:
        """
        Change the basis representation of the linear operator.

        Returns the matrix representation of the linear operator with 
        respect to a new basis, along with the change-of-basis matrix.

        Parameters
        ----------
        basis : list of object
            A new basis for the vector space.

        Returns
        -------
        tuple of (Matrix, Matrix)
            A tuple containing the matrix representation with respect to 
            the new basis and the change-of-basis matrix.
        """
        basechange = self.domain.change_of_basis(basis)
        map_matrix = basechange @ self.matrix @ basechange.inv()
        return map_matrix, basechange
    
    def inverse(self) -> LinearOperator:
        """
        The inverse of the linear operator.

        Returns
        -------
        LinearOperator
            The inverse of `self`.

        Raises
        ------
        LinearMapError
            If `self` is not invertible.
        """
        if not self.is_bijective():
            raise LinearMapError("Linear map is not invertible.")
        name = f"{self}^-1"
        mat = self.matrix.inv()
        return LinearOperator(name, self.domain, matrix=mat)
    
    def is_invariant_subspace(self, subspace: VectorSpace) -> bool:
        """
        Check whether a subspace is invariant under the linear operator.

        A subspace is invariant if the image of the subspace under the 
        operator is contained in the subspace itself.

        Parameters
        ----------
        subspace : VectorSpace
            The subspace to check.

        Returns
        -------
        bool
            True if `subspace` is invariant under `self`, otherwise False.

        Raises
        ------
        TypeError
            If `subspace` is not a subspace of the domain.
        """
        if not self.domain.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the domain.")
        return subspace.is_subspace(self(subspace))
    
    def is_symmetric(self, innerproduct: InnerProduct) -> bool | None:
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
        if self.field is not R:
            raise LinearMapError("Operator must be defined on a real vector space.")
        mat, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return mat.is_symmetric()

    def is_hermitian(self, innerproduct: InnerProduct) -> bool | None:
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
        mat, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return mat.is_hermitian

    def is_orthogonal(self, innerproduct: InnerProduct) -> bool | None:
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
        if self.field is not R:
            raise LinearMapError("Operator must be defined on a real vector space.")
        mat, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_orthogonal(mat)

    def is_unitary(self, innerproduct: InnerProduct) -> bool | None:
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
        mat, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_unitary(mat)
    
    def is_normal(self, innerproduct: InnerProduct) -> bool | None:
        """
        Check whether the linear operator is normal.

        Returns
        -------
        bool
            True if `self` is normal, otherwise False.
        """
        mat, _ = self.change_of_basis(innerproduct.orthonormal_basis)
        return u.is_normal(mat)


class LinearFunctional(LinearMap):
    """
    A linear functional on a vector space.
    
    A linear map from a vector space to its field of scalars. This is a 
    special case of a LinearMap where the codomain is the underlying field.
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        """
        Initialize a LinearFunctional instance.

        Parameters
        ----------
        name : str
            The name of the linear functional.
        vectorspace : VectorSpace
            The vector space the linear functional is defined on.
        mapping : callable, optional
            A function that takes a vector in the vector space and 
            returns a scalar in the field.
        matrix : Matrix, optional
            The matrix representation of the linear functional with 
            respect to the basis of the vector space.

        Returns
        -------
        LinearFunctional
            A new LinearFunctional instance.

        Raises
        ------
        LinearMapError
            If neither the mapping nor the matrix is provided.
        """
        field = vectorspace.field
        codomain = fn(str(field), field, 1)
        super().__init__(name, vectorspace, codomain, mapping, matrix)

    def __repr__(self) -> str:
        return (
            f"LinearFunctional(name={self.name!r}, "
            f"vectorspace={self.domain!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def restriction(self, subspace: VectorSpace) -> LinearFunctional:
        """
        Restrict the linear functional to a subspace of the domain.

        Parameters
        ----------
        subspace : VectorSpace
            A subspace of the domain.

        Returns
        -------
        LinearFunctional
            The restriction of `self` to `subspace`.

        Raises
        ------
        TypeError
            If `subspace` is not a subspace of the domain.
        """
        if not self.domain.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the domain.")
        name = f"{self} | {subspace}"
        return LinearFunctional(name, subspace, self.mapping)


class Isomorphism(LinearMap):
    """
    An isomorphism between vector spaces.
    
    A bijective linear map between vector spaces. This is a special case 
    of a LinearMap that is both injective and surjective.
    """

    def __init__(
        self, 
        name: str, 
        domain: VectorSpace, 
        codomain: VectorSpace, 
        mapping: Callable[[Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        super().__init__(name, domain, codomain, mapping, matrix)

        if not self.is_bijective():
            raise LinearMapError("Linear map is not invertible.")

    def __repr__(self) -> str:
        return super().__repr__().replace("LinearMap", "Isomorphism")
    
    def info(self) -> str:
        """
        A description of the isomorphism.

        Returns
        -------
        str
            The formatted description.
        """
        signature = f"{self} : {self.domain} → {self.codomain}"

        lines = [
            signature,
            "-" * len(signature),
            f"Field   {self.field}",
            f"Matrix  {self.matrix}"
            ]
        return "\n".join(lines)
    
    def inverse(self) -> Isomorphism:
        """
        The inverse of the isomorphism.

        Returns
        -------
        Isomorphism
            The inverse of `self`.
        """
        name = f"{self}^-1"
        mat = self.matrix.inv()
        return Isomorphism(name, self.codomain, self.domain, matrix=mat)


class IdentityMap(LinearOperator):
    """
    The identity map on a vector space.
    
    A linear operator that maps every vector to itself. This is a special 
    case of a LinearOperator.
    """

    def __init__(self, vectorspace: VectorSpace) -> None:
        """
        Initialize an IdentityMap instance.

        Parameters
        ----------
        vectorspace : VectorSpace
            The vector space the identity map is defined on.

        Returns
        -------
        IdentityMap
            A new IdentityMap instance.
        """
        super().__init__("Id", vectorspace, lambda vec: vec)

    def __repr__(self) -> str:
        return f"IdentityMap(vectorspace={self.domain!r})"
    
    def info(self) -> str:
        """
        A description of the identity map.

        Returns
        -------
        str
            The formatted description.
        """
        signature = f"{self} : {self.domain} → {self.codomain}"

        lines = [
            signature,
            "-" * len(signature),
            f"Field   {self.field}",
            f"Matrix  {self.matrix}"
            ]
        return "\n".join(lines)
    
    def inverse(self) -> IdentityMap:
        """
        The inverse of the identity map.

        Returns
        -------
        IdentityMap
            The inverse of `self`.
        """
        return self