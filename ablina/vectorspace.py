"""
A module for working with finite-dimensional vector and affine spaces.
"""

from __future__ import annotations

from random import gauss
from typing import Any, Callable

import sympy as sp

from .field import Field, R
from .mathset import Set
from .matrix import Matrix, M
from .parser import split_constraint, sympify
from . import utils as u
from . import vs_utils as vsu


class NotAVectorSpaceError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


class Fn:
    """
    Subspace of the standard vector space F^n.

    Provides concrete implementations of the main vector space 
    operations (sum, intersection, span, etc.). This class should only be 
    instantiated when subclassing ``VectorSpace`` in order to define a 
    custom vector space. See the ``fn`` function for working with 
    subspaces of F^n.

    Parameters
    ----------
    field : Field
        The field of scalars for the vector space.
    n : int
        Length of the vectors in the vector space.
    constraints : list of str, optional
        Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").

    Raises
    ------
    NotAVectorSpaceError
        If the constraints do not define a valid subspace.
    """

    def __init__(
        self, 
        field: Field, 
        n: int, 
        constraints: list[str] | None = None, 
        *, 
        ns_matrix: Any | None = None, 
        rs_matrix: Any | None = None
    ) -> None:
        """
        Initialize an `Fn` instance.

        Validates the list of constraints and constructs the null space 
        and row space matrices to internally represent the subspace.

        Parameters
        ----------
        field : Field
            The field of scalars for the vector space.
        n : int
            Length of the vectors in the vector space.
        constraints : list of str, optional
            Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").

        Raises
        ------
        NotAVectorSpaceError
            If the constraints do not define a valid subspace.
        """
        if constraints is None:
            constraints = []
        if not isinstance(field, Field):
            raise TypeError("field must be of type Field.")

        if ns_matrix is None and rs_matrix is None:
            if not is_vectorspace(n, constraints):
                raise NotAVectorSpaceError(
                    "Constraints do not satisfy vector space axioms."
                    )

        add, mul, additive_inv = Fn._init_operations()
        ns, rs = Fn._init_matrices(n, constraints, ns_matrix, rs_matrix)

        self._field = field
        self._n = n

        self._add = add
        self._mul = mul
        self._additive_inv = additive_inv

        self._ns_matrix = ns
        self._rs_matrix = rs

    @staticmethod
    def _init_operations() -> tuple[
        Callable[[Matrix, Matrix], Matrix], 
        Callable[[Any, Matrix], Matrix], 
        Callable[[Matrix], Matrix]
        ]:
        def add(vec1: Matrix, vec2: Matrix) -> Matrix: return vec1 + vec2
        def mul(scalar: Any, vec: Matrix) -> Matrix: return scalar * vec
        def additive_inv(vec: Matrix) -> Matrix: return -vec
        return add, mul, additive_inv

    @staticmethod
    def _init_matrices(
        n: int, 
        constraints: list[str], 
        ns: Any | None, 
        rs: Any | None
    ) -> tuple[Matrix, Matrix]:
        if ns is not None:
            ns = M.zeros(0, n) if u.is_empty(ns) else M(ns)
        if rs is not None:
            rs = M.zeros(0, n) if u.is_empty(rs) else M(rs)

        # Initialize ns_matrix
        if ns is None:
            if rs is None:
                ns = vsu.to_ns_matrix(n, constraints)
            else:
                ns = vsu.to_complement(rs)
        
        # Initialize rs_matrix
        if rs is None:
            rs = vsu.to_complement(ns)
        return ns, rs

    @property
    def field(self) -> Field:
        return self._field
    
    @property
    def n(self) -> int:
        return self._n
    
    @property
    def add(self) -> Callable[[Matrix, Matrix], Matrix]:
        return self._add
    
    @property
    def mul(self) -> Callable[[Any, Matrix], Matrix]:
        return self._mul
    
    @property
    def additive_inv(self) -> Callable[[Matrix], Matrix]:
        return self._additive_inv
    
    @property
    def additive_id(self) -> Matrix:
        return M.zeros(self.n, 1)
    
    @property
    def basis(self) -> list[Matrix]:
        return [M(vec) for vec in self._rs_matrix.tolist()]
    
    @property
    def dim(self) -> int:
        return self._rs_matrix.rows

    def __repr__(self) -> str:
        return (
            f"Fn(field={self.field!r}, "
            f"n={self.n!r}, "
            f"ns_matrix={self._ns_matrix!r}, "
            f"rs_matrix={self._rs_matrix!r})"
            )
    
    def __contains__(self, vector: Any) -> bool:
        if not (isinstance(vector, M) and vector.shape == (self.n, 1)):
            return False
        if not all(i in self.field for i in vector):
            return False
        # Check if vector satisfies vector space constraints
        prod = self._ns_matrix @ vector
        return prod.is_zero_matrix is True

    def add_constraints(self, constraints: list[str]) -> Fn:
        constraints_fn = Fn(self.field, self.n, constraints)
        return self.intersection(constraints_fn)

    # Methods relating to vectors

    def vector(self, std: int | float = 1, arbitrary: bool = False) -> Matrix:
        size = self.dim
        if arbitrary:
            weights = list(u.symbols(f"c:{size}", field=self.field))
        else:
            weights = [round(gauss(0, std)) for _ in range(size)]
        vec = M([weights]) @ self._rs_matrix
        return vec.T

    def to_coordinate(self, vector: Matrix, basis: list[Matrix]) -> Matrix:
        if not basis:
            return M()
        mat = M.hstack(*basis)
        return mat.solve_least_squares(vector)

    def from_coordinate(self, coord_vec: Matrix, basis: list[Matrix]) -> Matrix:
        if not basis:
            return self.additive_id
        mat = M.hstack(*basis)
        return mat @ coord_vec
    
    def is_independent(self, *vectors: Matrix) -> bool:
        mat = M.hstack(*vectors)
        return mat.rank() == len(vectors)
    
    def is_basis(self, *vectors: Matrix) -> bool:
        return self.is_independent(*vectors) and len(vectors) == self.dim
    
    # Methods relating to vector spaces

    def sum(self, vs2: Fn) -> Fn:
        rs = M.vstack(self._rs_matrix, vs2._rs_matrix)
        rs = u.rref(rs, remove=True)
        return Fn(self.field, self.n, rs_matrix=rs)
    
    def intersection(self, vs2: Fn) -> Fn:
        ns = M.vstack(self._ns_matrix, vs2._ns_matrix)
        ns = u.rref(ns, remove=True)
        return Fn(self.field, self.n, ns_matrix=ns)
    
    def span(self, *vectors: Matrix, basis: list[Matrix] | None = None) -> Fn:
        if basis is None:
            rs = M.hstack(*vectors).T
            rs = u.rref(rs, remove=True)
        else:
            rs = M.hstack(*basis).T
        return Fn(self.field, self.n, rs_matrix=rs)

    def is_subspace(self, vs2: Fn) -> bool:
        return all(vec in self for vec in vs2.basis)
    
    # Methods involving the dot product

    def dot(self, vec1: Matrix, vec2: Matrix) -> Any:
        return M.dot(vec1, vec2)

    def norm(self, vector: Matrix) -> Any:
        return sp.sqrt(self.dot(vector, vector))
    
    def is_orthogonal(self, *vectors: Matrix) -> bool:
        for i, vec1 in enumerate(vectors, 1):
            for vec2 in vectors[i:]:
                if not self.dot(vec1, vec2).equals(0):
                    return False
        return True

    def is_orthonormal(self, *vectors: Matrix) -> bool:
        if not self.is_orthogonal(*vectors):
            return False
        return all(self.norm(vec).equals(1) for vec in vectors)
    
    def gram_schmidt(self, *vectors: Matrix) -> list[Matrix]:
        orthonormal_vecs = []
        for v in vectors:
            for q in orthonormal_vecs:
                factor = self.dot(v, q)
                proj = self.mul(factor, q)
                v = self.add(v, self.additive_inv(proj))
            unit_v = self.mul(1 / self.norm(v), v)
            orthonormal_vecs.append(unit_v)
        return orthonormal_vecs
    
    def ortho_projection(self, vector: Matrix, subspace: Fn) -> Matrix:
        mat = subspace._rs_matrix.T
        return mat @ (mat.T @ mat).inv() @ mat.T @ vector
    
    def ortho_complement(self, subspace: Fn) -> Fn:
        comp = Fn(self.field, self.n, rs_matrix=subspace._ns_matrix)
        return self.intersection(comp)


class VectorSpace:
    """
    Abstract base class for defining arbitrary vector spaces.

    Provides the core interface for finite-dimensional vector spaces 
    built on an underlying `Fn` space. Subclasses must define a `set` (of 
    type `Set`), an `fn` (of type `Fn`), and the methods `__push__` and 
    `__pull__` to establish the isomorphism between abstract vectors and 
    their concrete F^n representations.
    """

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._validate_subclass_contract()
        add, mul, additive_inv = cls._init_operations()

        cls.name = cls.__name__ if name is None else name
        cls._add = staticmethod(add)
        cls._mul = staticmethod(mul)
        cls._additive_inv = staticmethod(additive_inv)

    def __init__(
        self, 
        name: str, 
        constraints: list[str] | None = None, 
        basis: list[Any] | None = None, 
        *, 
        fn: Fn | None = None
    ) -> None:
        """
        Initialize a `VectorSpace` instance.

        Parameters
        ----------
        name : str
            The name of the vector space.
        constraints : list of str, optional
            Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").
        basis : list of object, optional
            A basis for the subspace.

        Raises
        ------
        ValueError
            If the provided basis vectors are not linearly independent.
        """
        self.name = name
        self.set = Set(name, self.set.cls, lambda vec: vec in self)

        if fn is not None:
            self.fn = fn
            return
        
        if constraints is not None:
            self.fn = self.fn.add_constraints(constraints)
        
        if basis is not None:
            if not self.is_independent(*basis):
                raise ValueError("Basis vectors must be linearly independent.")
            self.fn = self.fn.span(basis=[self.__push__(vec) for vec in basis])

    @classmethod
    def _validate_subclass_contract(cls) -> None:
        attributes = ["set", "fn"]
        methods = ["__push__", "__pull__"]
        
        for attr in attributes:
            if not hasattr(cls, attr):
                raise TypeError(f'{cls.__name__} must define "{attr}".')
        for method in methods:
            if not callable(getattr(cls, method, None)):
                raise TypeError(f'{cls.__name__} must define the method "{method}".')
        
        if not isinstance(cls.set, Set):
            raise TypeError(f"{cls.__name__}.set must be a Set.")
        if not isinstance(cls.fn, Fn):
            raise TypeError(f"{cls.__name__}.fn must be of type Fn.")
        
        cls.__push__: Callable[[Any], Matrix] = staticmethod(cls.__push__)
        cls.__pull__: Callable[[Matrix], Any] = staticmethod(cls.__pull__)

    @classmethod
    def _init_operations(cls) -> tuple[
        Callable[[Any, Any], Any], 
        Callable[[Any, Any], Any], 
        Callable[[Any], Any]
        ]:
        def add(vec1: Any, vec2: Any) -> Any:
            fn_vec1, fn_vec2 = cls.__push__(vec1), cls.__push__(vec2)
            sum = cls.fn.add(fn_vec1, fn_vec2)
            return cls.__pull__(sum)
        def mul(scalar: Any, vec: Any) -> Any:
            fn_vec = cls.__push__(vec)
            prod = cls.fn.mul(scalar, fn_vec)
            return cls.__pull__(prod)
        def additive_inv(vec: Any) -> Any:
            fn_vec = cls.__push__(vec)
            inv = cls.fn.additive_inv(fn_vec)
            return cls.__pull__(inv)
        return add, mul, additive_inv
    
    @property
    def field(self) -> Field:
        """
        Field: The field of scalars for the vector space.
        """
        return self.fn.field
    
    @property
    def add(self) -> Callable[[Any, Any], Any]:
        """
        callable: The addition operator on the vector space.
        """
        return self._add
    
    @property
    def mul(self) -> Callable[[Any, Any], Any]:
        """
        callable: The multiplication operator on the vector space.
        """
        return self._mul
    
    @property
    def additive_inv(self) -> Callable[[Any], Any]:
        """
        callable: Return the additive inverse of a vector.
        """
        return self._additive_inv
    
    @property
    def additive_id(self) -> Any:
        """
        object: The additive identity of the vector space.
        """
        return self.__pull__(self.fn.additive_id)
    
    @property
    def basis(self) -> list[Any]:
        """
        list of object: The basis of the vector space.
        """
        return [self.__pull__(vec) for vec in self.fn.basis]
    
    @property
    def dim(self) -> int:
        """
        int: The dimension of the vector space.
        """
        return self.fn.dim
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, basis={self.basis!r})"
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, vs2: Any) -> bool:
        """
        Check for equality of two vector spaces.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to compare with.

        Returns
        -------
        bool
            True if both vector spaces are equal, otherwise False.
        """
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)

    def __contains__(self, vector: Any) -> bool:
        """
        Check whether a vector is an element of the vector space.

        Parameters
        ----------
        vector : object
            The vector to check.

        Returns
        -------
        bool
            True if `vector` is an element of `self`, otherwise False.
        """
        if vector not in type(self).set:
            return False
        return self.__push__(vector) in self.fn
    
    def __pos__(self) -> VectorSpace:
        """
        Return `self`.
        """
        return self
    
    def __neg__(self) -> VectorSpace:
        """
        Return `self`.
        """
        return self
    
    def __add__(self, other: VectorSpace | Any) -> VectorSpace | AffineSpace:
        """
        Add a vector space or vector to `self`.

        Same as ``VectorSpace.sum`` if `other` is a vector space. 
        Otherwise, returns the affine coset of `self` through `other`.

        Parameters
        ----------
        other : VectorSpace or object
            The vector space or vector to add.

        Returns
        -------
        VectorSpace or AffineSpace
            The resulting subspace sum or coset.
        """
        if isinstance(other, VectorSpace):
            return self.sum(other)
        return self.coset(other)
    
    def __radd__(self, vector: Any) -> AffineSpace:
        return self.coset(vector)
    
    def __sub__(self, other: VectorSpace | Any) -> VectorSpace | AffineSpace:
        """
        Subtract a vector space or vector from `self`.

        Same as ``VectorSpace.sum`` if `other` is a vector space. 
        Otherwise, returns the affine coset of `self` through the 
        additive inverse of `other`.

        Parameters
        ----------
        other : VectorSpace or object
            The vector space or vector to subtract.

        Returns
        -------
        VectorSpace or AffineSpace
            The resulting subspace sum or coset.
        """
        if isinstance(other, VectorSpace):
            return self.sum(other)
        if other not in self.ambient_space():
            raise TypeError("Vector must be an element of the ambient space.")
        return self.coset(self.additive_inv(other))
    
    def __rsub__(self, vector: Any) -> AffineSpace:
        return self.coset(vector)
    
    def __truediv__(self, vs2: VectorSpace) -> VectorSpace:
        """
        Same as ``VectorSpace.quotient``.
        """
        return self.quotient(vs2)
    
    def __and__(self, vs2: VectorSpace) -> VectorSpace:
        """
        Same as ``VectorSpace.intersection``.
        """
        return self.intersection(vs2)

    def info(self) -> str:
        """
        A description of the vector space.

        Returns
        -------
        str
            The formatted description.
        """
        name = f"{self} (Subspace of {type(self).name})"
        lines = [
            name,
            "-" * len(name),
            f"Field      {self.field}",
            f"Identity   {self.additive_id}",
            f"Basis      [{', '.join(map(str, self.basis))}]",
            f"Dimension  {self.dim}",
            f"Vector     {self.vector(arbitrary=True)}"
            ]
        return "\n".join(lines)

    # Methods relating to vectors

    def vector(self, std: int | float = 1, arbitrary: bool = False) -> Any:
        """
        Return a vector from the vector space.

        If `arbitrary` is False, then the vector is randomly generated by 
        taking a linear combination of the basis vectors. The weights are 
        sampled from a normal distribution with standard deviation `std`. 
        If `arbitrary` is True, then the general form of the vectors in 
        the vector space is returned.

        Parameters
        ----------
        std : float
            The standard deviation used to generate weights.
        arbitrary : bool, default=False
            Determines whether a random or arbitrary vector is returned.
        
        Returns
        -------
        object
            A vector in the vector space.

        Examples
        --------

        >>> V = fn("V", R, 3, constraints=["2*v0 == v1"])
        >>> V.vector()
        [1, 2, 0]
        >>> V.vector()
        [-1, -2, 1]
        >>> V.vector(std=10)
        [11, 22, 13]
        >>> V.vector(arbitrary=True)
        [c0, 2*c0, c1]
        """
        fn_vec = self.fn.vector(std, arbitrary)
        return self.__pull__(fn_vec)
    
    def to_coordinate(self, vector: Any, basis: list[Any] | None = None) -> Matrix:
        """
        Convert a vector to its coordinate vector representation.

        Parameters
        ----------
        vector : object
            A vector in the vector space.
        basis : list of object, optional
            A basis for the vector space.

        Returns
        -------
        Matrix
            The coordinate vector representation of `vector`.

        Raises
        ------
        ValueError
            If the provided basis vectors do not form a basis.

        See Also
        --------
        VectorSpace.from_coordinate

        Examples
        --------

        >>> V = fn("V", R, 3, constraints=["v0 == 2*v1"])
        >>> V.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> V.to_coordinate([2, 1, 2])
        [2, 0]
        """
        if vector not in self:
            raise TypeError("Vector must be an element of the vector space.")
        if basis is None:
            fn_basis = self.fn.basis
        elif not self.is_basis(*basis):
            raise ValueError("Provided vectors do not form a basis.")
        else:
            fn_basis = [self.__push__(vec) for vec in basis]

        fn_vec = self.__push__(vector)
        return self.fn.to_coordinate(fn_vec, fn_basis)
    
    def from_coordinate(self, coord_vec: Any, basis: list[Any] | None = None) -> Any:
        """
        Convert a coordinate vector to the vector it represents.

        Returns a linear combination of the basis vectors whose weights 
        are given by the coordinates of `coord_vec`. If `basis` is None, 
        then ``self.basis`` is used. The length of `coord_vec` must be 
        equal to the number of vectors in the basis, or equivalently the 
        dimension of the vector space.

        Parameters
        ----------
        coord_vec : Matrix
            The coordinate vector to convert.
        basis : list of object, optional
            A basis for the vector space.

        Returns
        -------
        object
            The vector represented by `coord_vec`.

        Raises
        ------
        ValueError
            If `coord_vec` is not a valid coordinate vector.

        See Also
        --------
        VectorSpace.to_coordinate

        Examples
        --------

        >>> V = fn("V", R, 3, constraints=["v0 == 2*v1"])
        >>> V.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> V.from_coordinate([1, 1])
        [1, 1/2, 1]
        >>> new_basis = [[2, 1, 1], [0, 0, 1]]
        >>> V.from_coordinate([1, 1], basis=new_basis)
        [2, 1, 2]
        """
        vec = self._validate_coordinate(coord_vec)
        if basis is None:
            fn_basis = self.fn.basis
        elif not self.is_basis(*basis):
            raise ValueError("Provided vectors do not form a basis.")
        else:
            fn_basis = [self.__push__(vec) for vec in basis]
        
        fn_vec = self.fn.from_coordinate(vec, fn_basis)
        return self.__pull__(fn_vec)
    
    def is_independent(self, *vectors: Any) -> bool:
        """
        Check whether the given vectors are linearly independent.

        Returns True if no vectors are given since the empty list is 
        linearly independent by definition.

        Parameters
        ----------
        *vectors : object
            The vectors to check.

        Returns
        -------
        bool
            True if the vectors are linearly independent, otherwise False.

        Examples
        --------

        >>> V = fn("V", R, 3)
        >>> V.is_independent([1, 0, 0], [0, 1, 0])
        True
        >>> V.is_independent([1, 2, 3], [2, 4, 6])
        False
        >>> V.is_independent([0, 0, 0])
        False
        >>> V.is_independent()
        True
        """
        if not all(vec in self for vec in vectors):
            raise TypeError("Vectors must be elements of the vector space.")
        fn_vecs = [self.__push__(vec) for vec in vectors]
        return self.fn.is_independent(*fn_vecs)
    
    def is_basis(self, *vectors: Any) -> bool:
        """
        Check whether the given vectors form a basis.

        Parameters
        ----------
        *vectors : object
            The vectors to check.

        Returns
        -------
        bool
            True if the vectors form a basis, otherwise False.
        """
        if not all(vec in self for vec in vectors):
            raise TypeError("Vectors must be elements of the vector space.")
        fn_vecs = [self.__push__(vec) for vec in vectors]
        return self.fn.is_basis(*fn_vecs)
    
    def change_of_basis(self, basis: list[Any]) -> Matrix:
        """
        Compute the change-of-basis matrix to a new basis.

        Returns the matrix that transforms coordinate vectors from the 
        current basis to the new one.

        Parameters
        ----------
        basis : list of object
            A new basis for the vector space.

        Returns
        -------
        Matrix
            The change-of-basis matrix.

        Raises
        ------
        ValueError
            If the provided vectors do not form a basis.
        """
        if not self.is_basis(*basis):
            raise ValueError("Provided vectors do not form a basis.")
        basechange = [self.to_coordinate(vec) for vec in basis]
        basechange = M.hstack(*basechange)
        return basechange.inv()

    # Methods relating to vector spaces

    def ambient_space(self) -> VectorSpace:
        """
        The ambient space that `self` is a subspace of.

        Note that this method is equivalent to ``cls(name=cls.name)`` 
        where ``cls = type(self)``.

        Returns
        -------
        VectorSpace
            The ambient space of `self`.
        """
        cls = type(self)
        return cls(name=cls.name)

    def sum(self, vs2: VectorSpace) -> VectorSpace:
        """
        The sum of two vector spaces.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space being added.

        Returns
        -------
        VectorSpace
            The sum of `self` and `vs2`.

        Raises
        ------
        TypeError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.intersection

        Examples
        --------

        >>> U = fn("U", R, 3, constraints=["v0 == v1"])
        >>> V = fn("V", R, 3, constraints=["v1 == v2"])
        >>> W = U.sum(V)
        >>> W.basis
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> U + V == W
        True
        """
        self._validate_type(vs2)
        name = f"{self} + {vs2}"
        fn = self.fn.sum(vs2.fn)
        return type(self)(name, fn=fn)
    
    def intersection(self, vs2: VectorSpace) -> VectorSpace:
        """
        The intersection of two vector spaces.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to take the intersection with.

        Returns
        -------
        VectorSpace
            The intersection of `self` and `vs2`.

        Raises
        ------
        TypeError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.sum

        Examples
        --------

        >>> U = fn("U", R, 3, constraints=["v0 == v1"])
        >>> V = fn("V", R, 3, constraints=["v1 == v2"])
        >>> W = U.intersection(V)
        >>> W.basis
        [[1, 1, 1]]
        >>> U & V == W
        True
        """
        self._validate_type(vs2)
        name = f"{self} ∩ {vs2}"
        fn = self.fn.intersection(vs2.fn)
        return type(self)(name, fn=fn)
    
    def span(self, name: str, *vectors: Any, basis: list[Any] | None = None) -> VectorSpace:
        """
        The span of the given vectors.

        Returns the smallest subspace of `self` that contains the vectors 
        in `vectors`. In order to manually set the basis of the resulting 
        space, pass the vectors into `basis` instead. Note that the 
        vectors must be linearly independent if passed into `basis`.

        Parameters
        ----------
        name : str
            The name of the resulting subspace.
        *vectors : object
            The vectors to take the span of.
        basis : list of object, optional
            A linearly independent list of vectors in the vector space.

        Returns
        -------
        VectorSpace
            The span of the given vectors.

        Raises
        ------
        ValueError
            If the provided basis vectors are not linearly independent.

        Examples
        --------

        >>> V = fn("V", R, 3)
        >>> V.span("span1", [1, 2, 3], [4, 5, 6]).basis
        [[1, 0, -1], [0, 1, 2]]
        >>> V.span("span2", basis=[[1, 2, 3], [4, 5, 6]]).basis
        [[1, 2, 3], [4, 5, 6]]
        >>> V.span("span3").basis
        []
        """
        if basis is not None:
            return type(self)(name, basis=basis)
        if not all(vec in self for vec in vectors):
            raise TypeError("Vectors must be elements of the vector space.")
        
        fn_vecs = [self.__push__(vec) for vec in vectors]
        fn = self.fn.span(*fn_vecs)
        return type(self)(name, fn=fn)
    
    def is_subspace(self, vs2: VectorSpace) -> bool:
        """
        Check whether `vs2` is a linear subspace of `self`.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to check.

        Returns
        -------
        bool
            True if `vs2` is a subspace of `self`, otherwise False.

        Examples
        --------

        >>> V = fn("V", R, 3)
        >>> U = fn("U", R, 3, constraints=["v0 == v1"])
        >>> W = fn("W", R, 3, constraints=["v1 == v2"])
        >>> V.is_subspace(U)
        True
        >>> V.is_subspace(W)
        True
        >>> W.is_subspace(U)
        False
        >>> V.is_subspace(V)
        True
        """
        try:
            self._validate_type(vs2)
        except TypeError:
            return False
        return self.fn.is_subspace(vs2.fn)
    
    # Methods relating to affine spaces
    
    def coset(self, representative: Any) -> AffineSpace:
        """
        Return the affine coset through a point.

        Parameters
        ----------
        representative : object
            A vector in the ambient space.

        Returns
        -------
        AffineSpace
            The affine coset of `self` through `representative`.

        See Also
        --------
        VectorSpace.quotient
        """
        return AffineSpace(self, representative)
    
    def quotient(self, subspace: VectorSpace) -> VectorSpace:
        """
        The quotient of two vector spaces.

        Parameters
        ----------
        subspace : VectorSpace
            The vector space to divide by.

        Returns
        -------
        VectorSpace
            The quotient of `self` by `subspace`.

        Raises
        ------
        TypeError
            If `subspace` is not a subspace of `self`.

        See Also
        --------
        VectorSpace.coset
        """
        if not self.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the vector space.")
        
        vs = self.ambient_space()
        cls_name = f"{vs} / {subspace}"

        def in_quotient_space(coset: AffineSpace) -> bool:
            return coset.vectorspace == subspace

        class quotient_space(VectorSpace, name=cls_name):
            set = Set(cls_name, AffineSpace, in_quotient_space)
            fn = vs.fn.ortho_complement(subspace.fn)
            def __push__(coset: AffineSpace) -> Matrix:
                fn_vec = vs.__push__(coset.representative)
                return vs.fn.ortho_projection(fn_vec, fn)
            def __pull__(vec: Matrix) -> AffineSpace:
                return subspace.coset(vs.__pull__(vec))

        name = f"{self} / {subspace}"
        fn = self.fn.ortho_complement(subspace.fn)
        return quotient_space(name, fn=fn)

    def _validate_type(self, vs2: Any) -> None:
        if not isinstance(vs2, VectorSpace):
            raise TypeError(f"Expected a VectorSpace, got {type(vs2).__name__} instead.")
        if type(self).name != type(vs2).name:
            raise TypeError("Vector spaces must share the same ambient space.")
    
    def _validate_coordinate(self, coord_vec: Any) -> Matrix:
        vec = M(coord_vec)
        if vec.shape != (self.dim, 1):
            raise ValueError("Coordinate vector has invalid shape.")
        if not all(i in self.field for i in vec):
            raise ValueError("Coordinates must be elements of the field.")
        return vec


class AffineSpace:
    """
    Affine coset of a vector space.

    Represents a vector space translated by a fixed representative 
    vector. Implements various affine space operations.
    """

    def __init__(self, vectorspace: VectorSpace, representative: Any) -> None:
        """
        Initialize an `AffineSpace` instance.

        Parameters
        ----------
        vectorspace : VectorSpace
            The underlying vector space being translated.
        representative : object
            A vector in the ambient space to translate by.

        Raises
        ------
        TypeError
            If `representative` is not an element of the ambient space.
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError("vectorspace must be of type VectorSpace.")
        if representative not in vectorspace.ambient_space():
            raise TypeError("representative must be an element of the ambient space.")
        
        self.name = f"{vectorspace} + {representative}"
        self._vectorspace = vectorspace
        self._representative = representative

    @property
    def vectorspace(self) -> VectorSpace:
        """
        VectorSpace: The underlying vector space.
        """
        return self._vectorspace
    
    @property
    def representative(self) -> Any:
        """
        object: The representative point of the affine space.
        """
        return self._representative
    
    @property
    def set(self) -> Set:
        """
        Set: The set of points in the affine space.
        """
        vs = self.vectorspace
        return Set(self.name, vs.set.cls, lambda point: point in self)
    
    @property
    def dim(self) -> int:
        """
        int: The dimension of the affine space.
        """
        return self.vectorspace.dim
    
    def __repr__(self) -> str:
        return (
            f"AffineSpace(vectorspace={self.vectorspace!r}, "
            f"representative={self.representative!r})"
            )
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, as2: Any) -> bool:
        """
        Check for equality of two affine spaces.

        Parameters
        ----------
        as2 : AffineSpace
            The affine space to compare with.

        Returns
        -------
        bool
            True if both affine spaces are equal, otherwise False.
        """
        if not isinstance(as2, AffineSpace):
            return False
        return self.representative in as2

    def __contains__(self, point: Any) -> bool:
        """
        Check whether a point is an element of the affine space.

        Parameters
        ----------
        point : object
            The point to check.

        Returns
        -------
        bool
            True if `point` is an element of `self`, otherwise False.
        """
        vs = self.vectorspace
        if point not in vs.ambient_space():
            return False
        
        vec1 = self.representative
        vec2 = vs.additive_inv(point)
        return vs.add(vec1, vec2) in vs
    
    def __pos__(self) -> AffineSpace:
        """
        Return `self`.
        """
        return self
    
    def __neg__(self) -> AffineSpace:
        """
        Return the affine space with negated representative.

        Returns
        -------
        AffineSpace
            The negation of `self`.
        """
        vs = self.vectorspace
        repr = vs.additive_inv(self.representative)
        return AffineSpace(vs, repr)
    
    def __add__(self, other: AffineSpace | Any) -> AffineSpace:
        """
        Add an affine space or vector to `self`.

        Same as ``AffineSpace.sum`` if `other` is an affine space. 
        Otherwise, returns the translation of `self` by `other`.

        Parameters
        ----------
        other : AffineSpace or object
            The affine space or vector to add.

        Returns
        -------
        AffineSpace
            The sum of `self` and `other`.
        """
        vs = self.vectorspace
        if isinstance(other, AffineSpace):
            return self.sum(other)
        if other not in vs.ambient_space():
            raise TypeError("Vector must be an element of the ambient space.")
        
        repr = vs.add(self.representative, other)
        return AffineSpace(vs, repr)

    def __radd__(self, vector: Any) -> AffineSpace:
        return self.__add__(vector)
    
    def __sub__(self, other: AffineSpace | Any) -> AffineSpace:
        """
        Subtract an affine space or vector from `self`.

        If `other` is an affine space, returns the sum with its negation. 
        Otherwise, translates `self` by the additive inverse of `other`.

        Parameters
        ----------
        other : AffineSpace or object
            The affine space or vector to subtract.

        Returns
        -------
        AffineSpace
            The difference `self` - `other`.
        """
        vs = self.vectorspace
        if isinstance(other, AffineSpace):
            return self.sum(-other)
        if other not in vs.ambient_space():
            raise TypeError("Vector must be an element of the ambient space.")
        
        repr = vs.add(self.representative, vs.additive_inv(other))
        return AffineSpace(vs, repr)
    
    def __rsub__(self, vector: Any) -> AffineSpace:
        return (-self).__add__(vector)

    def __mul__(self, scalar: Any) -> AffineSpace:
        """
        Scale the affine space by a scalar.

        Parameters
        ----------
        scalar : number
            A scalar from the field of the underlying vector space.

        Returns
        -------
        AffineSpace
            The scaled affine space.

        Raises
        ------
        TypeError
            If `scalar` is not an element of the field.
        """
        vs = self.vectorspace
        if scalar not in vs.field:
            raise TypeError("Scalar must be an element of the field.")
        repr = vs.mul(scalar, self.representative)
        return AffineSpace(vs, repr)

    def __rmul__(self, scalar: Any) -> AffineSpace:
        return self.__mul__(scalar)
    
    def info(self) -> str:
        """
        A description of the affine space.

        Returns
        -------
        str
            The formatted description.
        """
        name = self.name
        lines = [
            name,
            "-" * len(name),
            f"Vector Space    {self.vectorspace}",
            f"Representative  {self.representative}",
            f"Dimension       {self.dim}",
            f"Point           {self.point(arbitrary=True)}"
            ]
        return "\n".join(lines)
    
    def point(self, std: int | float = 1, arbitrary: bool = False) -> Any:
        """
        Return a point from the affine space.

        Parameters
        ----------
        std : float
            The standard deviation used to generate weights.
        arbitrary : bool, default=False
            Determines whether a random or arbitrary point is returned.
        
        Returns
        -------
        object
            A point in the affine space.
        """
        vs = self.vectorspace
        vector = vs.vector(std, arbitrary)
        point = vs.add(vector, self.representative)
        return point
    
    def sum(self, as2: AffineSpace) -> AffineSpace:
        """
        The sum of two affine spaces.

        Parameters
        ----------
        as2 : AffineSpace
            The affine space being added.

        Returns
        -------
        AffineSpace
            The sum of `self` and `as2`.

        See Also
        --------
        AffineSpace.intersection
        """
        self._validate_type(as2)
        vs = self.vectorspace
        repr = vs.add(self.representative, as2.representative)
        return AffineSpace(vs, repr)

    def intersection(self, as2: AffineSpace) -> AffineSpace:
        """
        The intersection of two affine spaces.

        Parameters
        ----------
        as2 : AffineSpace
            The affine space to take the intersection with.

        Returns
        -------
        AffineSpace
            The intersection of `self` and `as2`.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.

        See Also
        --------
        AffineSpace.sum
        """
        raise NotImplementedError("This method is not yet implemented.")
    
    def _validate_type(self, as2: Any) -> None:
        if not isinstance(as2, AffineSpace):
            raise TypeError(f"Expected an AffineSpace, got {type(as2).__name__} instead.")
        if self.vectorspace != as2.vectorspace:
            raise TypeError("Affine spaces must be cosets of the same vector space.")


def fn(
    name: str, 
    field: Field, 
    n: int, 
    constraints: list[str] | None = None, 
    basis: list[Any] | None = None, 
    *, 
    ns_matrix: Any | None = None, 
    rs_matrix: Any | None = None
) -> VectorSpace:
    """
    Factory for subspaces of standard F^n.

    Parameters
    ----------
    name : str
        The name of the subspace.
    field : Field
        The field of scalars for the vector space.
    n : int
        Length of the vectors in the vector space.
    constraints : list of str, optional
        Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").
    basis : list of object, optional
        A basis for the subspace.

    Returns
    -------
    VectorSpace
        The resulting subspace of F^n.
    """
    if n == 1:
        cls_name = f"{field}"
        class fn(VectorSpace, name=cls_name):
            set = Set(cls_name, object)
            fn = Fn(field, 1)
            def __push__(vec: Any) -> Matrix: return M[vec]
            def __pull__(vec: Matrix) -> Any: return vec[0]
    else:
        def in_fn(vec: Any) -> bool:
            try: return M(vec).shape == (n, 1)
            except Exception: return False

        cls_name = f"{field}^{n}"
        class fn(VectorSpace, name=cls_name):
            set = Set(cls_name, object, in_fn)
            fn = Fn(field, n)
            def __push__(vec: Any) -> Matrix: return M(vec)
            def __pull__(vec: Matrix) -> Matrix: return vec

    if not (ns_matrix is None and rs_matrix is None):
        vs = Fn(field, n, constraints, ns_matrix=ns_matrix, rs_matrix=rs_matrix)
        return fn(name, fn=vs)
    return fn(name, constraints, basis)


def matrix_space(
    name: str, 
    field: Field, 
    shape: tuple[int, int], 
    constraints: list[str] | None = None, 
    basis: list[Any] | None = None
) -> VectorSpace:
    """
    Factory for subspaces of matrices of a given shape.

    Parameters
    ----------
    name : str
        The name of the subspace.
    field : Field
        The field of scalars for the vector space.
    shape : tuple of (int, int)
        Shape (rows, cols) of the matrices.
    constraints : list of str, optional
        Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").
    basis : list of object, optional
        A basis for the subspace.

    Returns
    -------
    VectorSpace
        The resulting subspace of matrices.
    """
    cls_name = f"{field}^({shape[0]} × {shape[1]})"
    n = sp.prod(shape)

    def in_matrix_space(mat: Any) -> bool:
        try: return M(mat).shape == shape
        except Exception: return False

    class matrix_space(VectorSpace, name=cls_name):
        set = Set(cls_name, object, in_matrix_space)
        fn = Fn(field, n)
        def __push__(mat: Any) -> Matrix: return M(mat).reshape(n, 1)
        def __pull__(vec: Matrix) -> Matrix: return vec.reshape(*shape)
    return matrix_space(name, constraints, basis)


def poly_space(
    name: str, 
    field: Field, 
    max_degree: int, 
    constraints: list[str] | None = None, 
    basis: list[Any] | None = None
) -> VectorSpace:
    """
    Factory for polynomial subspaces up to a given degree.

    Parameters
    ----------
    name : str
        The name of the subspace.
    field : Field
        The field of scalars for the vector space.
    max_degree : int
        Maximum degree of the polynomials.
    constraints : list of str, optional
        Constraints all vectors must satisfy (e.g. "v0 + 2*v1 == 0").
    basis : list of object, optional
        A basis for the subspace.

    Returns
    -------
    VectorSpace
        The resulting polynomial subspace.
    """
    cls_name = f"P_{max_degree}({field})"
    x = u.symbols("x")

    def in_poly_space(poly: Any) -> bool:
        try: return sp.degree(sp.Poly(poly, x)) <= max_degree
        except Exception: return False

    class poly_space(VectorSpace, name=cls_name):
        set = Set(cls_name, object, in_poly_space)
        fn = Fn(field, max_degree + 1)
        def __push__(poly: Any) -> Matrix:
            poly = sp.Poly(poly, x)
            coeffs = poly.all_coeffs()[::-1]  # Ascending order
            degree_diff = max_degree - len(coeffs) + 1
            return M(coeffs + [0] * degree_diff)
        def __pull__(vec: Matrix) -> Any:
            poly = sp.Poly(vec[::-1], x)
            return poly.as_expr()
    return poly_space(name, constraints, basis)


def hom(vs1: VectorSpace, vs2: VectorSpace) -> VectorSpace:
    """
    Factory for subspaces of linear maps between two vector spaces.

    Parameters
    ----------
    vs1 : VectorSpace
        Domain of the linear maps.
    vs2 : VectorSpace
        Codomain of the linear maps.

    Returns
    -------
    VectorSpace
        The matrix space representing hom(vs1, vs2).

    Raises
    ------
    TypeError
        If the fields of `vs1` and `vs2` are not the same.
    """
    if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
        raise TypeError("vs1 and vs2 must be of type VectorSpace.")
    if vs1.field is not vs2.field:
        raise TypeError("vs1 and vs2 must be vector spaces over the same field.")
    
    name = f"hom({vs1}, {vs2})"
    return matrix_space(name, vs1.field, (vs2.dim, vs1.dim))


def is_vectorspace(n: int, constraints: list[str]) -> bool:
    """
    Check whether the given constraints define a valid subspace of F^n.

    Parameters
    ----------
    n : int
        Length of the vectors in the vector space.
    constraints : list of str
        The constraints to check.

    Returns
    -------
    bool
        True if the constraints define a valid subspace, otherwise False.
    """
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))
    
    allowed_vars = u.symbols(f"v:{n}")
    for expr in exprs:
        expr = sympify(expr, allowed_vars)
        if not u.is_linear(expr):
            return False
        
        # Check for nonzero constant terms
        const, _ = expr.as_coeff_add(*allowed_vars)
        if not const.equals(0):
            return False
    return True


def rowspace(name: str, matrix: Any, field: Field = R) -> VectorSpace:
    """
    Return the row space of a matrix.

    Parameters
    ----------
    name : str
        The name of the row space.
    matrix : Matrix
        The matrix to take the row space of.
    field : Field
        The field of scalars.

    Returns
    -------
    VectorSpace
        The row space of `matrix`.

    See Also
    --------
    columnspace

    Examples
    --------

    >>> matrix = [[1, 2], [3, 4]]
    >>> V = rowspace("V", matrix)
    >>> V.basis
    [[1, 0], [0, 1]]
    """
    mat = M(matrix)
    n = mat.cols
    rs = u.rref(mat, remove=True)
    return fn(name, field, n, rs_matrix=rs)


def columnspace(name: str, matrix: Any, field: Field = R) -> VectorSpace:
    """
    Return the column space, or image, of a matrix.

    Parameters
    ----------
    name : str
        The name of the column space.
    matrix : Matrix
        The matrix to take the column space of.
    field : Field
        The field of scalars.

    Returns
    -------
    VectorSpace
        The column space of `matrix`.

    See Also
    --------
    image, rowspace

    Examples
    --------

    >>> matrix = [[1, 2], [3, 4]]
    >>> V = columnspace("V", matrix)
    >>> V.basis
    [[1, 0], [0, 1]]
    >>> U = image("U", matrix)
    >>> U.basis
    [[1, 0], [0, 1]]
    """
    mat = M(matrix).T
    return rowspace(name, mat, field)


def nullspace(name: str, matrix: Any, field: Field = R) -> VectorSpace:
    """
    Return the null space, or kernel, of a matrix.

    Parameters
    ----------
    name : str
        The name of the null space.
    matrix : Matrix
        The matrix to take the null space of.
    field : Field
        The field of scalars.

    Returns
    -------
    VectorSpace
        The null space of `matrix`.

    See Also
    --------
    kernel, left_nullspace

    Examples
    --------

    >>> matrix = [[1, 2], [3, 4]]
    >>> V = nullspace("V", matrix)
    >>> V.basis
    []
    >>> U = kernel("U", matrix)
    >>> U.basis
    []
    """
    mat = M(matrix)
    n = mat.cols
    ns = u.rref(mat, remove=True)
    return fn(name, field, n, ns_matrix=ns)


def left_nullspace(name: str, matrix: Any, field: Field = R) -> VectorSpace:
    """
    Return the left null space of a matrix.

    Parameters
    ----------
    name : str
        The name of the left null space.
    matrix : Matrix
        The matrix to take the left null space of.
    field : Field
        The field of scalars.

    Returns
    -------
    VectorSpace
        The left null space of `matrix`.

    See Also
    --------
    nullspace

    Examples
    --------

    >>> matrix = [[1, 2], [3, 4]]
    >>> V = left_nullspace("V", matrix)
    >>> V.basis
    []
    >>> U = nullspace("U", matrix.T)
    >>> U.basis
    []
    """
    mat = M(matrix).T
    return nullspace(name, mat, field)


image = columnspace
"""An alias for the columnspace function."""

kernel = nullspace
"""An alias for the nullspace function."""