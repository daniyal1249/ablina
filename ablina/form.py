"""
A module for working with forms and inner products.
"""

from __future__ import annotations

from typing import Any, Callable

import sympy as sp

from .field import R, C
from .matrix import Matrix, M
from .utils import is_invertible, of_arity
from .vectorspace import VectorSpace

# Note that methods/properties such as is_positive_definite 
# will return None if the matrix is symbolic


class FormError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


class InnerProductError(FormError):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


class SesquilinearForm:
    """
    A sesquilinear form on a vector space.

    A sesquilinear form `<,>` is a function that takes two vectors and 
    returns a scalar, satisfying the following properties:

    - `<cu, v> = involution(c) <u, v>` for all scalars `c` and vectors `u`, `v`
    - `<u + v, w> = <u, w> + <v, w>` for all vectors `u`, `v`, `w`
    - `<u, cv> = c <u, v>` for all scalars `c` and vectors `u`, `v`
    - `<u, v + w> = <u, v> + <u, w>` for all vectors `u`, `v`, `w`
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any, Any], Any] | None = None, 
        matrix: Any | None = None, 
        involution: Callable[[Any], Any] | None = None
    ) -> None:
        """
        Initialize a SesquilinearForm instance.

        Parameters
        ----------
        name : str
            The name of the form.
        vectorspace : VectorSpace
            The vector space the form is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : Matrix, optional
            The matrix representation of the form with respect to the 
            basis of the vector space.
        involution : callable, optional
            The involution to use for the form. Must be a callable of 
            arity 1. If None, defaults to conjugation.

        Returns
        -------
        SesquilinearForm
            A new SesquilinearForm instance.

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError("vectorspace must be of type VectorSpace.")
        
        involution = SesquilinearForm._to_involution(involution)
        matrix = SesquilinearForm._to_matrix(vectorspace, mapping, matrix)
        mapping = SesquilinearForm._to_mapping(vectorspace, mapping, matrix, involution)
        
        self.name = name
        self._vectorspace = vectorspace
        self._mapping = mapping
        self._matrix = matrix
        self._involution = involution

    @staticmethod
    def _to_involution(involution: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
        if involution is None:
            return sp.conjugate
        if not of_arity(involution, 1):
            raise TypeError("Involution must be a callable of arity 1.")
        return involution

    @staticmethod
    def _to_matrix(
        vectorspace: VectorSpace, 
        mapping: Callable[[Any, Any], Any] | None, 
        matrix: Any | None
    ) -> Matrix:
        if matrix is not None:
            return SesquilinearForm._validate_matrix(vectorspace, matrix)
        if mapping is None:
            raise FormError("Either a matrix or mapping must be provided.")
        if not of_arity(mapping, 2):
            raise TypeError("Mapping must be a callable of arity 2.")
        
        basis = vectorspace.basis
        n = len(basis)
        return M(n, n, lambda i, j: mapping(basis[i], basis[j]))

    @staticmethod
    def _to_mapping(
        vectorspace: VectorSpace, 
        mapping: Callable[[Any, Any], Any] | None, 
        matrix: Matrix, 
        involution: Callable[[Any], Any]
    ) -> Callable[[Any, Any], Any]:
        if mapping is not None:
            return mapping
        to_coord = vectorspace.to_coordinate

        def mapping(u: Any, v: Any) -> Any:
            u = to_coord(u).applyfunc(involution)
            return (u.T @ matrix @ to_coord(v))[0]
        return mapping
    
    @staticmethod
    def _validate_matrix(vectorspace: VectorSpace, matrix: Any) -> Matrix:
        mat = M(matrix)
        if not (mat.is_square and mat.rows == vectorspace.dim):
            raise ValueError("Matrix has invalid shape.")
        if not all(i in vectorspace.field for i in mat):
            raise ValueError("Matrix entries must be elements of the field.")
        return mat

    @property
    def vectorspace(self) -> VectorSpace:
        """
        VectorSpace: The vector space the form is defined on.
        """
        return self._vectorspace
    
    @property
    def mapping(self) -> Callable[[Any, Any], Any]:
        """
        callable: The function that maps vectors to scalars.
        """
        return self._mapping
    
    @property
    def matrix(self) -> Matrix:
        """
        Matrix: The matrix representation of the form.
        """
        return self._matrix
    
    @property
    def involution(self) -> Callable[[Any], Any]:
        """
        callable: The involution the form is defined with respect to.
        """
        return self._involution
    
    def __repr__(self) -> str:
        return (
            f"SesquilinearForm(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r}, "
            f"involution={self.involution!r})"
            )
    
    def __str__(self) -> str:
        return self.name

    def __eq__(self, form2: Any) -> bool:
        """
        Check for equality of two forms.

        Parameters
        ----------
        form2 : SesquilinearForm
            The form to compare with.

        Returns
        -------
        bool
            True if both forms are equal, otherwise False.
        """
        if not isinstance(form2, SesquilinearForm):
            return False
        return (
            self.vectorspace == form2.vectorspace 
            and self.matrix.equals(form2.matrix)
            )
    
    def __call__(self, vec1: Any, vec2: Any) -> Any:
        """
        Apply the form to two vectors.

        Parameters
        ----------
        vec1, vec2 : object
            The vectors to apply the form to.

        Returns
        -------
        object
            The scalar value `<vec1, vec2>`.

        Raises
        ------
        TypeError
            If the vectors are not elements of the vector space.
        """
        vs = self.vectorspace
        if not (vec1 in vs and vec2 in vs):
            raise TypeError("Vectors must be elements of the vector space.")
        return self.mapping(vec1, vec2)
    
    def info(self) -> str:
        """
        A description of the form.

        Returns
        -------
        str
            The formatted description.
        """
        vs = self.vectorspace
        signature = f"{self} : {vs} × {vs} → {vs.field}"

        try:
            pos_def = self.is_positive_definite()
        except FormError:
            pos_def = "N/A"

        lines = [
            signature,
            "-" * len(signature),
            f"Symmetric?          {self.is_symmetric()}",
            f"Positive Definite?  {pos_def}",
            f"Matrix              {self.matrix}"
            ]
        return "\n".join(lines)

    def inertia(self) -> tuple[int, int, int]:
        """
        Compute the inertia of the form.

        Returns a tuple (p, m, z) where:

        - p is the number of positive eigenvalues
        - m is the number of negative eigenvalues
        - z is the number of zero eigenvalues

        Returns
        -------
        tuple of (int, int, int)
            The inertia (p, m, z) of the form.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).
        """
        self._validate_form()
        tol = 1e-8
        eigenvals = self.matrix.evalf().eigenvals().items()

        p = sum(m for val, m in eigenvals if val >= tol)
        m = sum(m for val, m in eigenvals if val <= -tol)
        z = sum(m for val, m in eigenvals if abs(val) < tol)
        return p, m, z
    
    def signature(self) -> int:
        """
        Compute the signature of the form.

        The signature is the difference between the number of positive 
        and negative eigenvalues.

        Returns
        -------
        int
            The signature of the form.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).
        """
        p, m, _ = self.inertia()
        return p - m

    def is_degenerate(self) -> bool | None:
        """
        Check whether the form is degenerate.

        A form is degenerate if its matrix is not invertible.

        Returns
        -------
        bool
            True if `self` is degenerate, otherwise False.
        """
        is_inv = is_invertible(self.matrix)
        return None if is_inv is None else not is_inv
    
    def is_symmetric(self) -> bool | None:
        """
        Check whether the form is symmetric.

        This method checks whether `<u, v> = involution(<v, u>)` for all 
        `u` and `v`.

        Returns
        -------
        bool
            True if `self` is symmetric, otherwise False.
        """
        mat1 = self.matrix
        mat2 = mat1.applyfunc(self.involution).T
        return mat1.equals(mat2)
    
    def is_skew_symmetric(self) -> bool | None:
        """
        Check whether the form is skew-symmetric.

        This method checks whether `<u, v> = -involution(<v, u>)` for all 
        `u` and `v`.

        Returns
        -------
        bool
            True if `self` is skew-symmetric, otherwise False.

        See Also
        --------
        SesquilinearForm.is_alternating
        """
        mat1 = self.matrix
        mat2 = -1 * mat1.applyfunc(self.involution).T
        return mat1.equals(mat2)
    
    def is_alternating(self) -> bool | None:
        """
        Check whether the form is alternating.

        This method checks whether `<v, v> = 0` for all `v`.

        Returns
        -------
        bool
            True if `self` is alternating, otherwise False.

        See Also
        --------
        SesquilinearForm.is_skew_symmetric
        """
        is_skew = self.is_skew_symmetric()
        is_zero = self.matrix.diagonal().is_zero_matrix
        if is_skew is False or is_zero is False:
            return False
        return True if is_skew and is_zero else None

    def is_hermitian(self) -> bool | None:
        """
        Check whether the form is hermitian.

        This method checks whether `<u, v> = conjugate(<v, u>)` for all 
        `u` and `v`. Note that this method is equivalent to 
        ``self.is_symmetric`` when the involution is conjugation.

        Returns
        -------
        bool
            True if `self` is hermitian, otherwise False.

        See Also
        --------
        SesquilinearForm.is_symmetric
        """
        return self.matrix.is_hermitian

    def is_positive_definite(self) -> bool | None:
        """
        Check whether the form is positive definite.

        This method checks whether `<v, v> > 0` for all `v ≠ 0`. Note 
        that the form is required to be symmetric (for real spaces) or 
        hermitian (for complex spaces).

        Returns
        -------
        bool
            True if `self` is positive definite, otherwise False.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).

        See Also
        --------
        SesquilinearForm.is_positive_semidefinite
        """
        self._validate_form()
        return self.matrix.is_positive_definite

    def is_negative_definite(self) -> bool | None:
        """
        Check whether the form is negative definite.

        This method checks whether `<v, v> < 0` for all `v ≠ 0`. Note 
        that the form is required to be symmetric (for real spaces) or 
        hermitian (for complex spaces).

        Returns
        -------
        bool
            True if `self` is negative definite, otherwise False.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).

        See Also
        --------
        SesquilinearForm.is_negative_semidefinite
        """
        self._validate_form()
        return self.matrix.is_negative_definite

    def is_positive_semidefinite(self) -> bool | None:
        """
        Check whether the form is positive semidefinite.

        This method checks whether `<v, v> ≥ 0` for all `v`. Note that 
        the form is required to be symmetric (for real spaces) or 
        hermitian (for complex spaces).

        Returns
        -------
        bool
            True if `self` is positive semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).

        See Also
        --------
        SesquilinearForm.is_positive_definite
        """
        self._validate_form()
        return self.matrix.is_positive_semidefinite

    def is_negative_semidefinite(self) -> bool | None:
        """
        Check whether the form is negative semidefinite.

        This method checks whether `<v, v> ≤ 0` for all `v`. Note that 
        the form is required to be symmetric (for real spaces) or 
        hermitian (for complex spaces).

        Returns
        -------
        bool
            True if `self` is negative semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).

        See Also
        --------
        SesquilinearForm.is_negative_definite
        """
        self._validate_form()
        return self.matrix.is_negative_semidefinite

    def is_indefinite(self) -> bool | None:
        """
        Check whether the form is indefinite.

        This method checks whether `<u, u> > 0` and `<v, v> < 0` for some 
        `u` and `v`. Note that the form is required to be symmetric 
        (for real spaces) or hermitian (for complex spaces).

        Returns
        -------
        bool
            True if `self` is indefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not symmetric (real) or hermitian (complex).
        """
        self._validate_form()
        return self.matrix.is_indefinite
    
    def _validate_form(self) -> None:
        field = self.vectorspace.field
        if field is R:
            if not self.is_symmetric():
                raise FormError("Form must be symmetric for real vector spaces.")
        elif field is C:
            if self.involution is not sp.conjugate:
                raise FormError("Involution must be conjugation for complex vector spaces.")
            if not self.is_symmetric():
                raise FormError("Form must be hermitian for complex vector spaces.")
        else:
            raise FormError("Form must be defined on a real or complex vector space.")

    # Aliases
    is_anti_symmetric = is_skew_symmetric
    """An alias for the is_skew_symmetric method."""


class BilinearForm(SesquilinearForm):
    """
    A bilinear form on a vector space.

    A bilinear form `<,>` is a function that takes two vectors and 
    returns a scalar, satisfying the following properties:

    - `<cu, v> = c <u, v>` for all scalars `c` and vectors `u`, `v`
    - `<u + v, w> = <u, w> + <v, w>` for all vectors `u`, `v`, `w`
    - `<u, cv> = c <u, v>` for all scalars `c` and vectors `u`, `v`
    - `<u, v + w> = <u, v> + <u, w>` for all vectors `u`, `v`, `w`
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any, Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        """
        Initialize a BilinearForm instance.

        Parameters
        ----------
        name : str
            The name of the form.
        vectorspace : VectorSpace
            The vector space the form is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : Matrix, optional
            The matrix representation of the form with respect to the 
            basis of the vector space.

        Returns
        -------
        BilinearForm
            A new BilinearForm instance.

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        """
        super().__init__(name, vectorspace, mapping, matrix, lambda c: c)

    def __repr__(self) -> str:
        return (
            f"BilinearForm(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )


class InnerProduct(SesquilinearForm):
    """
    An inner product on a vector space.
    
    An inner product is a positive definite, symmetric (for real spaces) 
    or hermitian (for complex spaces) sesquilinear form.
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any, Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        """
        Initialize an InnerProduct instance.

        Parameters
        ----------
        name : str
            The name of the inner product.
        vectorspace : VectorSpace
            The vector space the inner product is defined on.
        mapping : callable, optional
            A function that takes two vectors in the vector space and 
            returns a scalar in the field.
        matrix : Matrix, optional
            The matrix representation of the inner product with respect 
            to the basis of the vector space.

        Returns
        -------
        InnerProduct
            A new InnerProduct instance.

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        InnerProductError
            If the form is not a valid inner product.
        """
        super().__init__(name, vectorspace, mapping, matrix)

        try:
            if not self.is_positive_definite():
                raise InnerProductError("Inner product must be positive definite.")
        except FormError as e:
            raise InnerProductError(*e.args)

        vs = self.vectorspace
        self._orthonormal_basis = self.gram_schmidt(*vs.basis)
        self._fn_orthonormal_basis = vs.fn.gram_schmidt(*vs.fn.basis)

    @property
    def orthonormal_basis(self) -> list[Any]:
        """
        list of object: An orthonormal basis for the vector space.
        """
        return self._orthonormal_basis
    
    def __repr__(self) -> str:
        return (
            f"InnerProduct(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __push__(self, vector: Any) -> Any:
        """
        Push a vector from the vector space to its F^n representation.

        Maps a vector in the abstract vector space to its coordinate 
        representation in F^n using the orthonormal basis.

        Parameters
        ----------
        vector : object
            A vector in the vector space.

        Returns
        -------
        object
            The coordinate representation of `vector` in F^n.
        """
        vs = self.vectorspace
        coord_vec = vs.to_coordinate(vector, basis=self.orthonormal_basis)
        vec = vs.fn.from_coordinate(coord_vec, basis=self._fn_orthonormal_basis)
        return vec
    
    def __pull__(self, vector: Any) -> Any:
        """
        Pull a vector from F^n to the vector space.

        Maps a coordinate vector in F^n back to the abstract vector space 
        using the orthonormal basis.

        Parameters
        ----------
        vector : object
            A coordinate vector in F^n.

        Returns
        -------
        object
            The corresponding vector in the vector space.
        """
        vs = self.vectorspace
        coord_vec = vs.fn.to_coordinate(vector, basis=self._fn_orthonormal_basis)
        vec = vs.from_coordinate(coord_vec, basis=self.orthonormal_basis)
        return vec
    
    def info(self) -> str:
        """
        A description of the inner product.

        Returns
        -------
        str
            The formatted description.
        """
        vs = self.vectorspace
        signature = f"{self} : {vs} × {vs} → {vs.field}"

        lines = [
            signature,
            "-" * len(signature),
            f"Orthonormal Basis  [{', '.join(map(str, self.orthonormal_basis))}]",
            f"Matrix             {self.matrix}"
            ]
        return "\n".join(lines)

    def norm(self, vector: Any) -> Any:
        """
        The norm, or magnitude, of a vector.

        Parameters
        ----------
        vector : object
            A vector in the vector space.

        Returns
        -------
        float
            The norm of `vector`.
        """
        return sp.sqrt(self(vector, vector))
    
    def is_orthogonal(self, *vectors: Any) -> bool:
        """
        Check whether the vectors are pairwise orthogonal.

        Parameters
        ----------
        *vectors : object
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors are orthogonal, otherwise False.
        """
        for i, vec1 in enumerate(vectors, 1):
            for vec2 in vectors[i:]:
                if not sp.Integer(0).equals(self(vec1, vec2)):
                    return False
        return True

    def is_orthonormal(self, *vectors: Any) -> bool:
        """
        Check whether the vectors are orthonormal.

        Parameters
        ----------
        *vectors : object
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors are orthonormal, otherwise False.
        """
        if not self.is_orthogonal(*vectors):
            return False
        return all(self.norm(vec).equals(1) for vec in vectors)
    
    def gram_schmidt(self, *vectors: Any) -> list[Any]:
        """
        Apply the Gram-Schmidt process to a set of vectors.

        Returns an orthonormal list of vectors that span the same 
        subspace as the input vectors.

        Parameters
        ----------
        *vectors : object
            The vectors in the vector space.

        Returns
        -------
        list
            An orthonormal list of vectors.
        
        Raises
        ------
        ValueError
            If the provided vectors are not linearly independent.
        """
        vs = self.vectorspace
        if not vs.is_independent(*vectors):
            raise ValueError("Vectors must be linearly independent.")
        
        orthonormal_vecs = []
        for v in vectors:
            for q in orthonormal_vecs:
                factor = self.mapping(v, q)
                proj = vs.mul(factor, q)
                v = vs.add(v, vs.additive_inv(proj))
            unit_v = vs.mul(1 / self.norm(v), v)
            orthonormal_vecs.append(unit_v)
        return orthonormal_vecs

    def ortho_projection(self, vector: Any, subspace: VectorSpace) -> Any:
        """
        The orthogonal projection of a vector.

        Parameters
        ----------
        vector : object
            The vector to project.
        subspace : VectorSpace
            The subspace to project onto.

        Returns
        -------
        object
            The orthogonal projection of `vector` onto `subspace`.
        """
        vs = self.vectorspace
        if vector not in vs:
            raise TypeError("Vector must be an element of the vector space.")
        if not vs.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the vector space.")
        
        fn_vec = self.__push__(vector)
        proj = vs.fn.ortho_projection(fn_vec, subspace.fn)
        return self.__pull__(proj)

    def ortho_complement(self, subspace: VectorSpace) -> VectorSpace:
        """
        The orthogonal complement of a vector space.

        Parameters
        ----------
        subspace : VectorSpace
            The subspace to take the orthogonal complement of.

        Returns
        -------
        VectorSpace
            The orthogonal complement of `subspace` in ``self.vectorspace``.
        """
        vs = self.vectorspace
        if not vs.is_subspace(subspace):
            raise TypeError("Subspace must be a subspace of the vector space.")

        name = f"perp({subspace})"
        fn_basis = [self.__push__(vec) for vec in subspace.basis]
        fn = vs.fn.span(*fn_basis)
        comp = vs.fn.ortho_complement(fn)
        basis = [self.__pull__(vec) for vec in comp.basis]
        return vs.span(name, *basis)


class QuadraticForm:
    """
    A quadratic form on a vector space.
    
    A quadratic form `q` is a function that takes a vector and returns a 
    scalar, satisfying the following properties:

    - `q(cv) = c^2 q(v)` for all scalars `c` and vectors `v`
    - `q(u + v) - q(u) - q(v)` is a bilinear form
    """

    def __init__(
        self, 
        name: str, 
        vectorspace: VectorSpace, 
        mapping: Callable[[Any], Any] | None = None, 
        matrix: Any | None = None
    ) -> None:
        """
        Initialize a QuadraticForm instance.

        Parameters
        ----------
        name : str
            The name of the quadratic form.
        vectorspace : VectorSpace
            The vector space the quadratic form is defined on.
        mapping : callable, optional
            A function that takes a vector in the vector space and 
            returns a scalar in the field.
        matrix : Matrix, optional
            The matrix representation of the quadratic form with respect 
            to the basis of the vector space.

        Returns
        -------
        QuadraticForm
            A new QuadraticForm instance.

        Raises
        ------
        FormError
            If neither the mapping nor the matrix is provided.
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError("vectorspace must be of type VectorSpace.")
        
        matrix = QuadraticForm._to_matrix(vectorspace, mapping, matrix)
        mapping = QuadraticForm._to_mapping(vectorspace, mapping, matrix)
        
        self.name = name
        self._vectorspace = vectorspace
        self._mapping = mapping
        self._matrix = matrix

    @staticmethod
    def _to_matrix(
        vectorspace: VectorSpace, 
        mapping: Callable[[Any], Any] | None, 
        matrix: Any | None
    ) -> Matrix:
        if matrix is not None:
            return QuadraticForm._validate_matrix(vectorspace, matrix)
        if mapping is None:
            raise FormError("Either a matrix or mapping must be provided.")
        if not of_arity(mapping, 1):
            raise TypeError("Mapping must be a callable of arity 1.")
        
        basis = vectorspace.basis
        n = len(basis)
        mat = M.zeros(n, n)
        
        for i in range(n):
            mat[i, i] = mapping(basis[i])
        
        for i in range(n):
            for j in range(i + 1, n):
                u, v = basis[i], basis[j]
                u_plus_v = vectorspace.add(u, v)
                value = (mapping(u_plus_v) - mapping(u) - mapping(v)) / 2
                mat[i, j], mat[j, i] = value, value
        
        return mat
    
    @staticmethod
    def _to_mapping(
        vectorspace: VectorSpace, 
        mapping: Callable[[Any], Any] | None, 
        matrix: Matrix
    ) -> Callable[[Any], Any]:
        if mapping is not None:
            return mapping
        to_coord = vectorspace.to_coordinate
        return lambda v: (to_coord(v).T @ matrix @ to_coord(v))[0]
    
    @staticmethod
    def _validate_matrix(vectorspace: VectorSpace, matrix: Any) -> Matrix:
        mat = M(matrix)
        if not (mat.is_square and mat.rows == vectorspace.dim):
            raise ValueError("Matrix has invalid shape.")
        if not all(i in vectorspace.field for i in mat):
            raise ValueError("Matrix entries must be elements of the field.")
        return (mat + mat.T) / 2

    @property
    def vectorspace(self) -> VectorSpace:
        """
        VectorSpace: The vector space the quadratic form is defined on.
        """
        return self._vectorspace
    
    @property
    def mapping(self) -> Callable[[Any], Any]:
        """
        callable: The function that maps vectors to scalars.
        """
        return self._mapping
    
    @property
    def matrix(self) -> Matrix:
        """
        Matrix: The matrix representation of the quadratic form.
        """
        return self._matrix
    
    @property
    def polarization(self) -> BilinearForm:
        """
        BilinearForm: The associated bilinear form.

        Returns the symmetric bilinear form `b` such that `q(v) = b(v, v)` 
        for all vectors `v`.
        """
        name = f"b_{self.name}"
        return BilinearForm(name, self.vectorspace, matrix=self.matrix)
    
    def __repr__(self) -> str:
        return (
            f"QuadraticForm(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __str__(self) -> str:
        return self.name

    def __eq__(self, form2: Any) -> bool:
        """
        Check for equality of two quadratic forms.

        Parameters
        ----------
        form2 : QuadraticForm
            The form to compare with.

        Returns
        -------
        bool
            True if both forms are equal, otherwise False.
        """
        if not isinstance(form2, QuadraticForm):
            return False
        return (
            self.vectorspace == form2.vectorspace 
            and self.matrix.equals(form2.matrix)
            )
    
    def __call__(self, vector: Any) -> Any:
        """
        Apply the quadratic form to a vector.

        Parameters
        ----------
        vector : object
            The vector to map.

        Returns
        -------
        object
            The scalar that `vector` maps to.

        Raises
        ------
        TypeError
            If `vector` is not an element of the vector space.
        """
        vs = self.vectorspace
        if vector not in vs:
            raise TypeError("Vector must be an element of the vector space.")
        return self.mapping(vector)
    
    def info(self) -> str:
        """
        A description of the quadratic form.

        Returns
        -------
        str
            The formatted description.
        """
        vs = self.vectorspace
        signature = f"{self} : {vs} → {vs.field}"

        lines = [
            signature,
            "-" * len(signature),
            f"Matrix  {self.matrix}"
            ]
        return "\n".join(lines)

    def inertia(self) -> tuple[int, int, int]:
        """
        Compute the inertia of the quadratic form.

        Returns a tuple (p, m, z) where:

        - p is the number of positive eigenvalues
        - m is the number of negative eigenvalues
        - z is the number of zero eigenvalues

        Returns
        -------
        tuple of (int, int, int)
            The inertia (p, m, z) of the quadratic form.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.
        """
        self._validate_form()
        tol = 1e-8
        eigenvals = self.matrix.evalf().eigenvals().items()

        p = sum(m for val, m in eigenvals if val >= tol)
        m = sum(m for val, m in eigenvals if val <= -tol)
        z = sum(m for val, m in eigenvals if abs(val) < tol)
        return p, m, z
    
    def signature(self) -> int:
        """
        Compute the signature of the quadratic form.

        The signature is the difference between the number of positive 
        and negative eigenvalues.

        Returns
        -------
        int
            The signature of the quadratic form.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.
        """
        p, m, _ = self.inertia()
        return p - m

    def is_degenerate(self) -> bool | None:
        """
        Check whether the quadratic form is degenerate.

        A quadratic form is degenerate if its associated bilinear form 
        is degenerate, i.e., if the matrix is not invertible.

        Returns
        -------
        bool
            True if `self` is degenerate, otherwise False.
        """
        is_inv = is_invertible(self.matrix)
        return None if is_inv is None else not is_inv
    
    def is_positive_definite(self) -> bool | None:
        """
        Check whether the quadratic form is positive definite.

        This method checks whether `q(v) > 0` for all `v ≠ 0`.

        Returns
        -------
        bool
            True if `self` is positive definite, otherwise False.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_positive_semidefinite
        """
        self._validate_form()
        return self.matrix.is_positive_definite

    def is_negative_definite(self) -> bool | None:
        """
        Check whether the quadratic form is negative definite.

        This method checks whether `q(v) < 0` for all `v ≠ 0`.

        Returns
        -------
        bool
            True if `self` is negative definite, otherwise False.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_negative_semidefinite
        """
        self._validate_form()
        return self.matrix.is_negative_definite

    def is_positive_semidefinite(self) -> bool | None:
        """
        Check whether the quadratic form is positive semidefinite.

        This method checks whether `q(v) ≥ 0` for all `v`.

        Returns
        -------
        bool
            True if `self` is positive semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_positive_definite
        """
        self._validate_form()
        return self.matrix.is_positive_semidefinite

    def is_negative_semidefinite(self) -> bool | None:
        """
        Check whether the quadratic form is negative semidefinite.

        This method checks whether `q(v) ≤ 0` for all `v`.

        Returns
        -------
        bool
            True if `self` is negative semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_negative_definite
        """
        self._validate_form()
        return self.matrix.is_negative_semidefinite

    def is_indefinite(self) -> bool | None:
        """
        Check whether the quadratic form is indefinite.

        This method checks whether `q(u) > 0` and `q(v) < 0` for some 
        `u` and `v`.

        Returns
        -------
        bool
            True if `self` is indefinite, otherwise False.

        Raises
        ------
        FormError
            If the form is not defined on a real vector space.
        """
        self._validate_form()
        return self.matrix.is_indefinite

    def _validate_form(self) -> None:
        if self.vectorspace.field is not R:
            raise FormError("Quadratic form must be defined on a real vector space.")