"""
A module for working with sesquilinear forms, inner products, and quadratic forms.
"""

import sympy as sp

from .field import R
from .matrix import M
from .utils import is_invertible, of_arity
from .vectorspace import VectorSpace

# Note that methods/properties such as is_positive_definite 
# will return None if the matrix is symbolic


class FormError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)


class InnerProductError(FormError):
    def __init__(self, msg=""):
        super().__init__(msg)


class BilinearForm:
    pass


class SesquilinearForm:
    """
    A sesquilinear form on a vector space.

    A sesquilinear form is a function that takes two vectors and returns 
    a scalar. It is conjugate-linear in the first argument and linear in 
    the second. For real vector spaces, this is equivalent to a 
    bilinear form.
    """

    def __init__(self, name, vectorspace, mapping=None, matrix=None):
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
        
        matrix = SesquilinearForm._to_matrix(vectorspace, mapping, matrix)
        mapping = SesquilinearForm._to_mapping(vectorspace, mapping, matrix)
        
        self.name = name
        self._vectorspace = vectorspace
        self._mapping = mapping
        self._matrix = matrix

    @staticmethod
    def _to_matrix(vectorspace, mapping, matrix):
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
    def _to_mapping(vectorspace, mapping, matrix):
        if mapping is not None:
            return mapping
        to_coord = vectorspace.to_coordinate
        return lambda u, v: (to_coord(u).H @ matrix @ to_coord(v))[0]
    
    @staticmethod
    def _validate_matrix(vectorspace, matrix):
        mat = M(matrix)
        if not (mat.is_square and mat.rows == vectorspace.dim):
            raise ValueError("Matrix has invalid shape.")
        if not all(i in vectorspace.field for i in mat):
            raise ValueError("Matrix entries must be elements of the field.")
        return mat

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
        Matrix: The matrix representation of the form.
        """
        return self._matrix
    
    def __repr__(self):
        return (
            f"SesquilinearForm(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __str__(self):
        return self.name

    def __eq__(self, form2):
        """
        Check for equality of two sesquilinear forms.

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
            and self.matrix == form2.matrix
            )
    
    def __call__(self, vec1, vec2):
        """
        Apply the sesquilinear form to two vectors.

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
            If either vector is not an element of the vector space.
        """
        vs = self.vectorspace
        if not (vec1 in vs and vec2 in vs):
            raise TypeError("Vectors must be elements of the vector space.")
        return self.mapping(vec1, vec2)
    
    def info(self):
        """
        A description of the sesquilinear form.

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
            f"Symmetric?          {self.is_symmetric()}",
            f"Hermitian?          {self.is_hermitian()}",
            f"Positive Definite?  {self.is_positive_definite()}",
            f"Matrix              {self.matrix}"
            ]
        return "\n".join(lines)

    def inertia(self):
        """
        Compute the inertia of the sesquilinear form.

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
        if self.vectorspace.field is R:
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
        """
        Compute the signature of the sesquilinear form.

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

    def is_degenerate(self):
        """
        Check whether the form is degenerate.

        A sesquilinear form is degenerate if its matrix is not invertible.

        Returns
        -------
        bool
            True if `self` is degenerate, otherwise False.
        """
        return not is_invertible(self.matrix)
    
    def is_symmetric(self):
        """
        Check whether the form is symmetric.

        This method checks whether `<x, y> = <y, x>` for all `x` and `y`.

        Returns
        -------
        bool
            True if `self` is symmetric, otherwise False.

        See Also
        --------
        SesquilinearForm.is_hermitian
        """
        return self.matrix.is_symmetric()

    def is_hermitian(self):
        """
        Check whether the form is hermitian.

        This method checks whether `<x, y> = conjugate(<y, x>)` for all 
        `x` and `y`. Note that this method is equivalent to 
        ``self.is_symmetric`` for forms defined on real vector spaces.

        Returns
        -------
        bool
            True if `self` is hermitian, otherwise False.

        See Also
        --------
        SesquilinearForm.is_symmetric
        """
        return self.matrix.is_hermitian

    def is_positive_definite(self):
        """
        Check whether the form is positive definite.

        This method checks whether `<x, x> > 0` for all `x ≠ 0`. Note 
        that the form is not required to be symmetric/hermitian.

        Returns
        -------
        bool
            True if `self` is positive definite, otherwise False.

        See Also
        --------
        SesquilinearForm.is_positive_semidefinite
        """
        return self.matrix.is_positive_definite

    def is_negative_definite(self):
        """
        Check whether the form is negative definite.

        This method checks whether `<x, x> < 0` for all `x ≠ 0`. Note 
        that the form is not required to be symmetric/hermitian.

        Returns
        -------
        bool
            True if `self` is negative definite, otherwise False.

        See Also
        --------
        SesquilinearForm.is_negative_semidefinite
        """
        return self.matrix.is_negative_definite

    def is_positive_semidefinite(self):
        """
        Check whether the form is positive semidefinite.

        This method checks whether `<x, x> ≥ 0` for all `x`. Note that 
        the form is not required to be symmetric/hermitian.

        Returns
        -------
        bool
            True if `self` is positive semidefinite, otherwise False.

        See Also
        --------
        SesquilinearForm.is_positive_definite
        """
        return self.matrix.is_positive_semidefinite

    def is_negative_semidefinite(self):
        """
        Check whether the form is negative semidefinite.

        This method checks whether `<x, x> ≤ 0` for all `x`. Note that 
        the form is not required to be symmetric/hermitian.

        Returns
        -------
        bool
            True if `self` is negative semidefinite, otherwise False.

        See Also
        --------
        SesquilinearForm.is_negative_definite
        """
        return self.matrix.is_negative_semidefinite

    def is_indefinite(self):
        """
        Check whether the form is indefinite.

        This method checks whether `<x, x> > 0` and `<y, y> < 0` for some 
        `x` and `y`. Note that the form is not required to be 
        symmetric/hermitian.

        Returns
        -------
        bool
            True if `self` is indefinite, otherwise False.
        """
        return self.matrix.is_indefinite


class InnerProduct(SesquilinearForm):
    """
    An inner product on a vector space.
    
    An inner product is a positive definite, symmetric (for real spaces) 
    or hermitian (for complex spaces) sesquilinear form.
    """

    def __init__(self, name, vectorspace, mapping=None, matrix=None):
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
        vs = self.vectorspace

        if vs.field is R:
            if not self.is_symmetric():
                raise InnerProductError("Real inner product must be symmetric.")
        elif not self.is_hermitian():
            raise InnerProductError("Complex inner product must be hermitian.")
        if not self.is_positive_definite():
            raise InnerProductError("Inner product must be positive definite.")

        self._orthonormal_basis = self.gram_schmidt(*vs.basis)
        self._fn_orthonormal_basis = vs.fn.gram_schmidt(*vs.fn.basis)

    @property
    def orthonormal_basis(self):
        """
        list of object: An orthonormal basis for the vector space.
        """
        return self._orthonormal_basis
    
    def __repr__(self):
        return super().__repr__().replace("SesquilinearForm", "InnerProduct")
    
    def __push__(self, vector):
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
    
    def __pull__(self, vector):
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
    
    def info(self):
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

    def norm(self, vector):
        """
        The norm, or magnitude, of a vector.

        Parameters
        ----------
        vector
            A vector in the vector space.

        Returns
        -------
        float
            The norm of `vector`.
        """
        return sp.sqrt(self(vector, vector))
    
    def is_orthogonal(self, *vectors):
        """
        Check whether the vectors are pairwise orthogonal.

        Parameters
        ----------
        *vectors
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

    def is_orthonormal(self, *vectors):
        """
        Check whether the vectors are orthonormal.

        Parameters
        ----------
        *vectors
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors are orthonormal, otherwise False.
        """
        if not self.is_orthogonal(*vectors):
            return False
        return all(self.norm(vec).equals(1) for vec in vectors)
    
    def gram_schmidt(self, *vectors):
        """
        Apply the Gram-Schmidt process to a set of vectors.

        Returns an orthonormal list of vectors that span the same 
        subspace as the input vectors.

        Parameters
        ----------
        *vectors
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

    def ortho_projection(self, vector, subspace):
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
            raise TypeError()
        if not vs.is_subspace(subspace):
            raise TypeError()
        
        fn_vec = self.__push__(vector)
        proj = vs.fn.ortho_projection(fn_vec, subspace.fn)
        return self.__pull__(proj)

    def ortho_complement(self, subspace):
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
            raise TypeError()

        name = f"perp({subspace})"
        fn_basis = [self.__push__(vec) for vec in subspace.basis]
        fn = vs.fn.span(*fn_basis)
        comp = vs.fn.ortho_complement(fn)
        basis = [self.__pull__(vec) for vec in comp.basis]
        return vs.span(name, *basis)


class QuadraticForm:
    """
    A quadratic form on a vector space.
    
    A quadratic form `Q` is a function that takes a vector and returns a 
    scalar, satisfying the following properties:

    - `Q(av) = a^2 Q(v)` for all scalars `a` and vectors `v`
    - `Q(u + v) - Q(u) - Q(v)` is a bilinear form
    """

    def __init__(self, name, vectorspace, mapping=None, matrix=None):
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
    def _to_matrix(vectorspace, mapping, matrix):
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
    def _to_mapping(vectorspace, mapping, matrix):
        if mapping is not None:
            return mapping
        to_coord = vectorspace.to_coordinate
        return lambda v: (to_coord(v).T @ matrix @ to_coord(v))[0]
    
    @staticmethod
    def _validate_matrix(vectorspace, matrix):
        mat = M(matrix)
        if not (mat.is_square and mat.rows == vectorspace.dim):
            raise ValueError("Matrix has invalid shape.")
        if not all(i in vectorspace.field for i in mat):
            raise ValueError("Matrix entries must be elements of the field.")
        return (mat + mat.T) / 2

    @property
    def vectorspace(self):
        """
        VectorSpace: The vector space the quadratic form is defined on.
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
        Matrix: The matrix representation of the quadratic form.
        """
        return self._matrix
    
    @property
    def polarization(self):
        """
        BilinearForm: The associated bilinear form.

        Returns the symmetric bilinear form `B` such that `Q(v) = B(v, v)` 
        for all vectors `v`.
        """
        name = f"B_{self.name}"
        return BilinearForm(name, self.vectorspace, matrix=self.matrix)
    
    def __repr__(self):
        return (
            f"QuadraticForm(name={self.name!r}, "
            f"vectorspace={self.vectorspace!r}, "
            f"mapping={self.mapping!r}, "
            f"matrix={self.matrix!r})"
            )
    
    def __str__(self):
        return self.name

    def __eq__(self, form2):
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
            and self.matrix == form2.matrix
            )
    
    def __call__(self, vector):
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
    
    def info(self):
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

    def inertia(self):
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
            If the quadratic form is not defined on a real vector space.
        """
        self._validate_form()
        tol = 1e-8
        eigenvals = self.matrix.evalf().eigenvals().items()
        p = sum(m for val, m in eigenvals if val >= tol)
        m = sum(m for val, m in eigenvals if val <= -tol)
        z = sum(m for val, m in eigenvals if abs(val) < tol)
        return p, m, z
    
    def signature(self):
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
            If the quadratic form is not defined on a real vector space.
        """
        p, m, _ = self.inertia()
        return p - m

    def is_degenerate(self):
        """
        Check whether the quadratic form is degenerate.

        A quadratic form is degenerate if its associated bilinear form 
        is degenerate, i.e., if the matrix is not invertible.

        Returns
        -------
        bool
            True if `self` is degenerate, otherwise False.
        """
        return not is_invertible(self.matrix)
    
    def is_positive_definite(self):
        """
        Check whether the quadratic form is positive definite.

        This method checks whether `Q(x) > 0` for all `x ≠ 0`.

        Returns
        -------
        bool
            True if `self` is positive definite, otherwise False.

        Raises
        ------
        FormError
            If the quadratic form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_positive_semidefinite
        """
        self._validate_form()
        return self.matrix.is_positive_definite

    def is_negative_definite(self):
        """
        Check whether the quadratic form is negative definite.

        This method checks whether `Q(x) < 0` for all `x ≠ 0`.

        Returns
        -------
        bool
            True if `self` is negative definite, otherwise False.

        Raises
        ------
        FormError
            If the quadratic form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_negative_semidefinite
        """
        self._validate_form()
        return self.matrix.is_negative_definite

    def is_positive_semidefinite(self):
        """
        Check whether the quadratic form is positive semidefinite.

        This method checks whether `Q(x) ≥ 0` for all `x`.

        Returns
        -------
        bool
            True if `self` is positive semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the quadratic form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_positive_definite
        """
        self._validate_form()
        return self.matrix.is_positive_semidefinite

    def is_negative_semidefinite(self):
        """
        Check whether the quadratic form is negative semidefinite.

        This method checks whether `Q(x) ≤ 0` for all `x`.

        Returns
        -------
        bool
            True if `self` is negative semidefinite, otherwise False.

        Raises
        ------
        FormError
            If the quadratic form is not defined on a real vector space.

        See Also
        --------
        QuadraticForm.is_negative_definite
        """
        self._validate_form()
        return self.matrix.is_negative_semidefinite

    def is_indefinite(self):
        """
        Check whether the quadratic form is indefinite.

        This method checks whether `Q(x) > 0` and `Q(y) < 0` for some 
        `x` and `y`.

        Returns
        -------
        bool
            True if `self` is indefinite, otherwise False.

        Raises
        ------
        FormError
            If the quadratic form is not defined on a real vector space.
        """
        self._validate_form()
        return self.matrix.is_indefinite

    def _validate_form(self):
        if self.vectorspace.field is not R:
            raise FormError("Form must be defined on a real vector space.")