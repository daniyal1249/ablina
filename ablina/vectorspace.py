from numbers import Complex, Real
from random import gauss

import sympy as sp

from .mathset import Set
from .operations import ScalarMul, VectorAdd
from .parser import split_constraint, sympify
from . import utils as u
from . import vs_utils as vsu


class VectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class NotAVectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class _StandardFn:
    def __init__(self, field, n, constraints=None, *, ns_matrix=None, rs_matrix=None):
        if field not in (Real, Complex):
            raise TypeError('Field must be either Real or Complex.')

        # Verify whether constraints satisfy vector space properties
        if constraints is None:
            constraints = []
        if ns_matrix is None and rs_matrix is None:
            if not is_vectorspace(n, constraints):
                raise NotAVectorSpaceError(
                    'Constraints do not satisfy vector space axioms.'
                    )
        ns, rs = _StandardFn._init_matrices(n, constraints, ns_matrix, rs_matrix)

        self._field = field
        self._n = n
        self._constraints = constraints
        self._ns_matrix = ns
        self._rs_matrix = rs

    @staticmethod
    def _init_matrices(n, constraints, ns_mat, rs_mat):
        if ns_mat is not None:
            ns_mat = sp.zeros(0, n) if u.is_empty(ns_mat) else sp.Matrix(ns_mat)
        if rs_mat is not None:
            rs_mat = sp.zeros(0, n) if u.is_empty(rs_mat) else sp.Matrix(rs_mat)
        
        # Initialize ns_matrix
        if ns_mat is None:
            if rs_mat is None:
                ns_mat = vsu.to_ns_matrix(n, constraints)
            else:
                ns_mat = vsu.to_complement(rs_mat)
        
        # Initialize rs_matrix
        if rs_mat is None:
            rs_mat = vsu.to_complement(ns_mat)
        return ns_mat, rs_mat

    @property
    def field(self):
        return self._field
    
    @property
    def n(self):
        return self._n
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def basis(self):
        return self._rs_matrix.tolist()
    
    @property
    def dim(self):
        return len(self.basis)
    
    def __contains__(self, vec):
        if not u.in_field(self.field, *vec):
            return False
        try:
            # Check if vec satisfies vector space constraints
            vec = sp.Matrix(vec)
            return bool((self._ns_matrix @ vec).is_zero_matrix)
        except Exception:
            return False
    
    def __eq__(self, vs2):
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)

    # Methods relating to vectors

    def vector(self, std=1, arbitrary=False):
        size = self._rs_matrix.rows
        if arbitrary:
            weights = list(u.symbols(f'c:{size}', field=self.field))
        else:
            weights = [round(gauss(0, std)) for _ in range(size)]
        vec = sp.Matrix([weights]) @ self._rs_matrix
        return vec.flat()  # Return list

    def to_coordinate(self, vector, basis=None):
        if basis is None:
            basis = self._rs_matrix.tolist()
        elif not self._is_basis(*basis):
            raise VectorSpaceError('The provided vectors do not form a basis.')
        if not basis:
            return []
        
        matrix, vec = sp.Matrix(basis).T, sp.Matrix(vector)
        coord_vec = matrix.solve_least_squares(vec)
        return coord_vec.flat()

    def from_coordinate(self, vector, basis=None):  # Check field
        if basis is None:
            basis = self._rs_matrix.tolist()
        elif not self._is_basis(*basis):
            raise VectorSpaceError('The provided vectors do not form a basis.')
        try:
            matrix, coord_vec = sp.Matrix(basis).T, sp.Matrix(vector)
            vec = matrix @ coord_vec
        except Exception as e:
            raise TypeError('Invalid coordinate vector.') from e
        return vec.flat() if vec else [0] * self.n
    
    def are_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows
    
    def _is_basis(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows and len(vectors) == self.dim

    # Methods relating to vector spaces

    def is_subspace(self, vs2):
        if not self.share_ambient_space(vs2):
            return False
        for i in range(self._rs_matrix.rows):
            vec = self._rs_matrix.row(i).T
            if not (vs2._ns_matrix @ vec).is_zero_matrix:
                return False
        return True

    def share_ambient_space(self, vs2):
        if type(self) is not type(vs2):
            return False
        return self.field is vs2.field and self.n == vs2.n

    # Methods involving the dot product

    def dot(self, vec1, vec2):
        return sum(i * j for i, j in zip(vec1, vec2))
    
    def norm(self, vector):
        return sp.sqrt(self.dot(vector, vector))
    
    def are_orthogonal(self, vec1, vec2):
        return self.dot(vec1, vec2) == 0
    
    def are_orthonormal(self, *vectors):
        # Improve efficiency
        if not all(self.norm(vec) == 1 for vec in vectors):
            return False
        for vec1 in vectors:
            for vec2 in vectors:
                if not (vec1 is vec2 or self.are_orthogonal(vec1, vec2)):
                    return False
        return True
    
    def gram_schmidt(self, *vectors):
        raise NotImplementedError()


class Fn(_StandardFn):
    """
    pass
    """

    def __init__(
            self, field, n, constraints=None, add=None, mul=None, 
            *, isomorphism=None, ns_matrix=None, rs_matrix=None
            ):
        """
        Parameters
        ----------
        field : {Real, Complex}
            The field of scalars.
        n : int
            The length of the vectors in V.
        constraints : list of str, optional
            The list of constraints that each vector in V must satisfy 
            (default: None). Refer to the notes for more information.
        add : callable, optional
            An addition function that takes two vectors in V and assigns 
            them to another vector in V. The function must obey the 
            vector space axioms: 

            - Commutativity
            - Associativity
            - Existence of an additive identity (zero)
            - Existence of an additive inverse
            
            The default is the standard addition on F^n.
        mul : callable, optional
            A multiplication function that takes a scalar and a vector 
            and assigns them to a vector in V. The function must obey the 
            vector space axioms: 

            - Associativity
            - Existence of a multiplicative identity
            - Distributivity with respect to vector addition
            - Distributivity with respect to scalar addition

            The default is the standard multiplication on F^n.

        Returns
        -------
        Fn
            The specified subspace of F^n.

        Raises
        ------
        NotAVectorSpaceError
            If the constraints, addition, and multiplication do not form 
            a vector space.
        """
        if isomorphism is not None:
            if not (isinstance(isomorphism, tuple) and len(isomorphism) == 2):
                raise TypeError('Isomorphism must be a 2-tuple of callables.')
        
        add, mul, iso = Fn._init_operations(field, n, add, mul, isomorphism)

        self._to_standard, self._from_standard = iso
        mapped_constraints = vsu.map_constraints(self._to_standard, constraints)
        super().__init__(
            field, n, mapped_constraints, ns_matrix=ns_matrix, rs_matrix=rs_matrix
            )
        
        self._add = add  # VectorAdd(field, n, add)
        self._mul = mul  # ScalarMul(field, n, mul)
        # Reassign constraints
        self._constraints = constraints

    @staticmethod
    def _init_operations(field, n, add, mul, iso):
        # For efficiency
        if add is None and mul is None:
            iso = (lambda vec: vec, lambda vec: vec)

        if add is None:
            def add(vec1, vec2): return [i + j for i, j in zip(vec1, vec2)]
        if mul is None:
            def mul(scalar, vec): return [scalar * i for i in vec]
        if iso is None:
            iso = vsu.standard_isomorphism(field, n, add, mul)

        return add, mul, iso

    @property
    def add(self):
        return self._add
    
    @property
    def mul(self):
        return self._mul
    
    @property
    def basis(self):
        return [self._from_standard(vec) for vec in super().basis]

    def __contains__(self, vec):
        try:
            standard_vec = self._to_standard(vec)
        except Exception:
            return False
        return super().__contains__(standard_vec)
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)

    # Methods relating to vectors

    def vector(self, std=1, arbitrary=False):
        standard_vec = super().vector(std, arbitrary)
        return self._from_standard(standard_vec)
    
    def to_coordinate(self, vector, basis=None):
        if basis is not None:
            basis = [self._to_standard(vec) for vec in basis]
        standard_vec = self._to_standard(vector)
        return super().to_coordinate(standard_vec, basis)
    
    def from_coordinate(self, vector, basis=None):
        if basis is not None:
            basis = [self._to_standard(vec) for vec in basis]
        standard_vec = super().from_coordinate(vector, basis)
        return self._from_standard(standard_vec)
    
    def are_independent(self, *vectors):
        standard_vecs = [self._to_standard(vec) for vec in vectors]
        return super().are_independent(*standard_vecs)

    # Methods relating to vector spaces

    def sum(self, vs2):
        rs_matrix = sp.Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix = u.rref(rs_matrix, remove=True)
        constraints = self.constraints  # Rework
        return Fn(
            self.field, self.n, constraints, self.add, self.mul, 
            isomorphism=(self._to_standard, self._from_standard), 
            rs_matrix=rs_matrix
            )
    
    def intersection(self, vs2):
        ns_matrix = sp.Matrix.vstack(self._ns_matrix, vs2._ns_matrix)
        ns_matrix = u.rref(ns_matrix, remove=True)
        constraints = self.constraints + vs2.constraints
        return Fn(
            self.field, self.n, constraints, self.add, self.mul, 
            isomorphism=(self._to_standard, self._from_standard), 
            ns_matrix=ns_matrix
            )
    
    def span(self, *vectors, basis=None):
        if basis is not None:
            vectors = basis
        standard_vecs = [self._to_standard(vec) for vec in vectors]
        if basis is None:
            standard_vecs = u.rref(standard_vecs, remove=True)
        constraints = [f'span({', '.join(map(str, vectors))})']
        return Fn(
            self.field, self.n, constraints, self.add, self.mul, 
            isomorphism=(self._to_standard, self._from_standard), 
            rs_matrix=standard_vecs
            )
    
    def share_ambient_space(self, vs2):
        # if not super().share_ambient_space(vs2):
        #     return False
        # return self.add == vs2.add and self.mul == vs2.mul
        return True

    # Methods involving the dot product
    
    def ortho_complement(self):
        constraints = [f'ortho_complement({', '.join(self.constraints)})']
        return Fn(
            self.field, self.n, constraints, self.add, self.mul, 
            isomorphism=(self._to_standard, self._from_standard), 
            rs_matrix=self._ns_matrix
            )
    
    def ortho_projection(self, vs2):
        raise NotImplementedError()


class VectorSpace:
    """
    pass
    """

    def __init__(self, vectors, fn, isomorphism):
        """
        pass

        Parameters
        ----------
        vectors : MathematicalSet
            pass
        fn : Fn
            pass
        isomorphism : callable or tuple of callable
            pass

        Returns
        -------
        pass

        Raises
        ------
        VectorSpaceError
            pass
        """
        if not isinstance(vectors, Set):
            raise TypeError('vectors must be a MathematicalSet.')
        if not isinstance(fn, Fn):
            raise TypeError('fn must be of type Fn.')
        iso = VectorSpace._check_isomorphism(isomorphism)
        
        self._vectors = vectors
        self._fn = fn
        self._to_fn, self._from_fn = iso

    @staticmethod
    def _check_isomorphism(iso):
        if isinstance(iso, tuple):
            if len(iso) == 2 and all(u.of_arity(i, 1) for i in iso):
                return iso
        elif u.of_arity(iso, 1):
            return iso, lambda vec: vec
        raise TypeError(
            'Isomorphism must be a callable or a 2-tuple of callables.'
            )
    
    @property
    def field(self):
        """
        {Real, Complex}: The field of scalars.
        """
        return self._fn.field
    
    @property
    def add(self):
        """
        callable: The addition operator on the vector space.
        """
        def add(vec1, vec2):
            if not (vec1 in self and vec2 in self):
                raise TypeError('Vectors must be elements of the vector space.')
            fn_vec1, fn_vec2 = self._to_fn(vec1), self._to_fn(vec2)
            sum = self._fn.add(fn_vec1, fn_vec2)
            return self._from_fn(sum)
        return add
    
    @property
    def mul(self):
        """
        callable: The multiplication operator on the vector space.
        """
        def mul(scalar, vec):
            if not u.in_field(self.field, scalar):
                raise TypeError('Scalar must be an element of the field.')
            if vec not in self:
                raise TypeError('Vector must be an element of the vector space.')
            fn_vec = self._to_fn(vec)
            prod = self._fn.mul(scalar, fn_vec)
            return self._from_fn(prod)
        return mul
    
    @property
    def add_id(self):
        """
        object: The additive identity of the vector space.
        """
        raise NotImplementedError()
    
    @property
    def add_inv(self):
        """
        callable: A function that returns the additive inverse of a given vector.
        """
        raise NotImplementedError()
    
    @property
    def mul_id(self):
        """
        object: The multiplicative identity of the vector space.
        """
        raise NotImplementedError()
    
    @property
    def basis(self):
        """
        list: The basis of the vector space.
        """
        return [self._from_fn(vec) for vec in self._fn.basis]
    
    @property
    def dim(self):
        """
        int: The dimension of the vector space.
        """
        return self._fn.dim
    
    @property
    def set(self):
        """
        MathematicalSet: The set containing the vectors in the vector space.
        """
        return Set(self._vectors.cls, lambda vec: vec in self)
    
    def __contains__(self, vec):
        if vec not in self._vectors:
            return False
        return self._to_fn(vec) in self._fn
    
    def __eq__(self, vs2):
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)

    # Methods relating to vectors

    def vector(self, std=1, arbitrary=False):
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
            Determines whether a random vector or arbitrary vector is returned.
        
        Returns
        -------
        object
            A vector in the vector space.

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3, constraints=['2*v0 == v1'])
        >>> vs.vector()
        [1, 2, 0]
        >>> vs.vector()
        [-1, -2, 1]
        >>> vs.vector(std=10)
        [11, 22, 13]
        >>> vs.vector(arbitrary=True)
        [c0, 2*c0, c1]
        """
        fn_vec = self._fn.vector(std, arbitrary)
        return self._from_fn(fn_vec)
    
    def to_coordinate(self, vector, basis=None):
        """
        Convert a vector to its coordinate vector representation.

        Parameters
        ----------
        vector : list
            A vector in the vector space.
        basis : list, optional
            pass

        Returns
        -------
        list
            The coordinate vector representation of `vector`.

        Raises
        ------
        VectorSpaceError
            If the provided basis vectors do not form a basis for the 
            vector space.

        See Also
        --------
        VectorSpace.from_coordinate

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3, constraints=['v0 == 2*v1'])
        >>> vs.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> vs.to_coordinate([2, 1, 2])
        [2, 0]
        """
        if vector not in self:
            raise TypeError('Vector must be an element of the vector space.')
        if basis is not None:
            if not all(vec in self for vec in basis):
                raise TypeError('Basis vectors must be elements of the vector space.')
            basis = [self._to_fn(vec) for vec in basis]

        fn_vec = self._to_fn(vector)
        return self._fn.to_coordinate(fn_vec, basis)
    
    def from_coordinate(self, vector, basis=None):
        """
        Convert a coordinate vector to the vector it represents.

        Returns a linear combination of the basis vectors whose weights 
        are given by the coordinates of `vector`. If `basis` is None, then 
        ``self.basis`` is used. The length of `vector` must be equal to 
        the number of vectors in the basis, or equivalently the dimension 
        of the vector space.

        Parameters
        ----------
        vector
            The coordinate vector to convert.
        basis : list, optional
            A basis of the vector space.

        Returns
        -------
        list
            The vector represented by `vector`.

        Raises
        ------
        TypeError
            If `vector` is of incorrect length.

        See Also
        --------
        VectorSpace.to_coordinate

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3, constraints=['v0 == 2*v1'])
        >>> vs.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> vs.from_coordinate([1, 1])
        [1, 1/2, 1]
        >>> new_basis = [[2, 1, 1], [0, 0, 1]]
        >>> vs.from_coordinate([1, 1], basis=new_basis)
        [2, 1, 2]
        """
        if basis is not None:
            if not all(vec in self for vec in basis):
                raise TypeError('Basis vectors must be elements of the vector space.')
            basis = [self._to_fn(vec) for vec in basis]
        
        fn_vec = self._fn.from_coordinate(vector, basis)
        return self._from_fn(fn_vec)
    
    def are_independent(self, *vectors):
        """
        Check whether the given vectors are linearly independent.

        Returns True if no vectors are given since the empty list is 
        linearly independent by definition.

        Parameters
        ----------
        *vectors
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors are linearly independent, otherwise False.

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.are_independent([1, 0, 0], [0, 1, 0])
        True
        >>> vs.are_independent([1, 2, 3], [2, 4, 6])
        False
        >>> vs.are_independent([0, 0, 0])
        False
        >>> vs.are_independent()
        True
        """
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vecs = [self._to_fn(vec) for vec in vectors]
        return self._fn.are_independent(*fn_vecs)

    # Methods relating to vector spaces

    def sum(self, vs2):
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
        VectorSpaceError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.intersection

        Examples
        --------

        >>> vs1 = VectorSpace.fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = VectorSpace.fn(Real, 3, constraints=['v1 == v2'])
        >>> vs = vs1.sum(vs2)
        >>> vs.basis
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> vs1 + vs2 == vs
        True
        """
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        fn = self._fn.sum(vs2._fn)
        return VectorSpace(self._vectors, fn, (self._to_fn, self._from_fn))
    
    def intersection(self, vs2):
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
        VectorSpaceError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.sum

        Examples
        --------

        >>> vs1 = VectorSpace.fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = VectorSpace.fn(Real, 3, constraints=['v1 == v2'])
        >>> vs = vs1.intersection(vs2)
        >>> vs.basis
        [[1, 1, 1]]
        >>> vs1 & vs2 == vs
        True
        """
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        fn = self._fn.intersection(vs2._fn)
        return VectorSpace(self._vectors, fn, (self._to_fn, self._from_fn))
    
    def span(self, *vectors, basis=None):
        """
        The span of the given vectors.

        Returns the smallest subspace of `self` that contains the vectors 
        in `vectors`. As with all instances of the ``VectorSpace`` class, 
        the basis of the resulting vector space is automatically 
        determined. To override this, pass the vectors into `basis` instead. 
        Note that the vectors must be linearly independent if passed into `basis`.

        Parameters
        ----------
        *vectors
            The vectors in the vector space.
        basis : list, optional
            A linearly independent list of vectors in the vector space.

        Returns
        -------
        VectorSpace
            The span of the given vectors.

        Raises
        ------
        VectorSpaceError
            If the provided basis vectors are not linearly independent.

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.span([1, 2, 3], [4, 5, 6]).basis
        [[1, 0, -1], [0, 1, 2]]
        >>> vs.span(basis=[[1, 2, 3], [4, 5, 6]]).basis
        [[1, 2, 3], [4, 5, 6]]
        >>> vs.span().basis
        []
        """
        if basis is not None:
            if not self.are_independent(*basis):
                raise VectorSpaceError('Basis vectors must be linearly independent.')
            vectors = basis
        elif not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        
        fn_vecs = [self._to_fn(vec) for vec in vectors]
        if basis is None:
            fn = self._fn.span(*fn_vecs)
        else:
            fn = self._fn.span(basis=fn_vecs)
        return VectorSpace(self._vectors, fn, (self._to_fn, self._from_fn))
    
    def is_subspace(self, vs2):
        """
        Check whether `self` is a linear subspace of `vs2`.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to check.

        Returns
        -------
        bool
            True if `self` is a subspace of `vs2`, otherwise False.

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs1 = VectorSpace.fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = VectorSpace.fn(Real, 3, constraints=['v1 == v2'])
        >>> vs1.is_subspace(vs)
        True
        >>> vs2.is_subspace(vs)
        True
        >>> vs1.is_subspace(vs2)
        False
        >>> vs.is_subspace(vs)
        True
        """
        if not self.share_ambient_space(vs2):
            return False
        return self._fn.is_subspace(vs2._fn)
    
    def share_ambient_space(self, vs2):
        """
        Check whether `self` and `vs2` are subspaces of the same vector space.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to check.

        Returns
        -------
        bool
            True if `self` and `vs2` share an ambient space, otherwise False.
        """
        # if self._vectors is not vs2._vectors:
        #     return False
        # return self._fn.share_ambient_space(vs2._fn)
        return True

    # Methods involving the dot product

    def dot(self, vec1, vec2):
        """
        The dot product between two vectors.

        Parameters
        ----------
        vec1, vec2
            The vectors in the vector space.

        Returns
        -------
        float
            The dot product between `vec1` and `vec2`.

        See Also
        --------
        VectorSpace.norm, VectorSpace.are_orthogonal

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.dot([1, 2, 3], [4, 5, 6])
        32
        >>> vs.dot([1, 0, 1], [0, 1, 0])
        0
        """
        if not (vec1 in self and vec2 in self):
            raise TypeError('Vectors must be elements of the vector space.')
        vec1, vec2 = self._to_fn(vec1), self._to_fn(vec2)
        return self._fn.dot(vec1, vec2)
    
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

        See Also
        --------
        VectorSpace.dot, VectorSpace.are_orthonormal

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.norm([1, 2, 3])
        sqrt(14)
        >>> vs.norm([0, 0, 0])
        0
        """
        return sp.sqrt(self.dot(vector, vector))
    
    def are_orthogonal(self, vec1, vec2):
        """
        Check whether two vectors are orthogonal.

        Parameters
        ----------
        vec1, vec2
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors are orthogonal, otherwise False.

        See Also
        --------
        VectorSpace.dot

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.are_orthogonal([1, 2, 3], [4, 5, 6])
        False
        >>> vs.are_orthogonal([1, 0, 1], [0, 1, 0])
        True
        """
        return self.dot(vec1, vec2) == 0
    
    def are_orthonormal(self, *vectors):
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

        See Also
        --------
        VectorSpace.dot, VectorSpace.norm

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3)
        >>> vs.are_orthonormal([1, 2, 3], [4, 5, 6])
        False
        >>> vs.are_orthonormal([1, 0, 0], [0, 1, 0])
        True
        >>> vs.are_orthonormal([1, 0, 0])
        True
        >>> vs.are_orthonormal()
        True
        """
        # Improve efficiency
        if not all(self.norm(vec) == 1 for vec in vectors):
            return False
        for vec1 in vectors:
            for vec2 in vectors:
                if not (vec1 is vec2 or self.are_orthogonal(vec1, vec2)):
                    return False
        return True
    
    def gram_schmidt(self, *vectors):
        """
        pass

        Parameters
        ----------
        *vectors
            pass

        Returns
        -------
        list
            pass

        See Also
        --------
        VectorSpace.are_orthonormal
        """
        raise NotImplementedError()
    
    def ortho_complement(self):
        """
        The orthogonal complement of a vector space.

        Returns
        -------
        VectorSpace
            The orthogonal complement of `self`.

        See Also
        --------
        VectorSpace.ortho_projection, VectorSpace.dot

        Examples
        --------

        >>> vs = VectorSpace.fn(Real, 3, constraints=['v0 == v1'])
        >>> vs.ortho_complement().basis
        [[1, -1, 0]]
        >>> vs.ortho_complement().ortho_complement() == vs
        True
        """
        fn = self._fn.ortho_complement()
        return VectorSpace(self._vectors, fn, (self._to_fn, self._from_fn))
    
    def ortho_projection(self, vs2):
        """
        The orthogonal projection of `self` onto `vs2`.

        Parameters
        ----------
        vs2 : VectorSpace
            pass

        Returns
        -------
        VectorSpace
            pass

        Raises
        ------
        VectorSpaceError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.ortho_complement, VectorSpace.dot
        """
        if not self.share_ambient_space(vs2):
            raise VectorSpaceError('Vector spaces must share the same ambient space.')
        fn = self._fn.ortho_projection(vs2._fn)
        return VectorSpace(self._vectors, fn, (self._to_fn, self._from_fn))
    
    # Vector space constructors
    
    @classmethod
    def fn(cls, field, n, constraints=None, basis=None, 
           add=None, mul=None, *, ns_matrix=None, rs_matrix=None):
        """
        pass
        """
        def in_fn(vec):
            try: return sp.Matrix(vec).shape == (n, 1)
            except Exception: return False
        
        vectors = Set(object, in_fn, name=f'F^{n}')
        fn = Fn(
            field, n, constraints, add, mul, 
            ns_matrix=ns_matrix, rs_matrix=rs_matrix
            )

        vectorspace = cls(vectors, fn, lambda vec: vec)
        if basis is not None:
            vectorspace = vectorspace.span(basis=basis)
        return vectorspace

    @classmethod
    def matrix(cls, field, shape, constraints=None, basis=None, 
               add=None, mul=None):
        """
        pass
        """
        def in_matrix(mat): return mat.shape == shape
        def to_fn(mat): return mat.flat()
        def from_fn(vec): return sp.Matrix(*shape, vec)
        
        name = f'{shape[0]} by {shape[1]} matrices'
        vectors = Set(sp.Matrix, in_matrix, name=name)
        fn = Fn(field, sp.prod(shape), constraints, add, mul)

        vectorspace = cls(vectors, fn, (to_fn, from_fn))
        if basis is not None:
            vectorspace = vectorspace.span(basis=basis)
        return vectorspace

    @classmethod
    def poly(cls, field, max_degree, constraints=None, basis=None, 
             add=None, mul=None):
        """
        pass
        """
        def in_poly(poly):
            return sp.degree(poly) <= max_degree
        def to_fn(poly):
            coeffs = poly.all_coeffs()[::-1]  # Ascending order
            degree_diff = max_degree - len(coeffs) + 1
            return coeffs + ([0] * degree_diff)
        def from_fn(vec):
            x = sp.symbols('x')
            return sp.Poly.from_list(vec[::-1], x)

        name = f'P_{max_degree}(F)'
        vectors = Set(sp.Poly, in_poly, name=name)
        fn = Fn(field, max_degree + 1, constraints, add, mul)

        vectorspace = cls(vectors, fn, (to_fn, from_fn))
        if basis is not None:
            vectorspace = vectorspace.span(basis=basis)
        return vectorspace

    @classmethod
    def hom(cls, vs1, vs2):
        """
        pass
        """
        if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
            raise TypeError()
        if vs1.field is not vs2.field:
            raise VectorSpaceError()
        return cls.matrix(vs1.field, (vs2.dim, vs1.dim))


def is_vectorspace(n, constraints):
    """
    Check whether F^n forms a vector space under the given constraints.

    Parameters
    ----------
    n : int
        The length of the vectors in the vector space.
    constraints : list of str
        The constraints ..

    Returns
    -------
    bool
        True if the constraints permit a vector space under standard 
        operations, otherwise False.
    """
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))
    
    allowed_vars = sp.symbols(f'v:{n}')
    for expr in exprs:
        expr = sympify(expr, allowed_vars)
        if not u.is_linear(expr):
            return False
        
        # Check for nonzero constant terms
        const, _ = expr.as_coeff_add()
        if const != 0:
            return False
    return True


def columnspace(matrix, field=Real):
    """
    Compute the column space, or image, of the matrix.

    Parameters
    ----------
    matrix : list of list or sympy.Matrix
        The matrix to take the column space of.
    field : {Real, Complex}
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
    >>> vs = columnspace(matrix)
    >>> print(vs.basis)
    [[1, 0], [0, 1]]
    >>> vs = image(matrix)
    >>> print(vs.basis)
    [[1, 0], [0, 1]]
    """
    constraints = [f'col({matrix})']
    matrix = sp.Matrix(matrix).T
    matrix = u.rref(matrix, remove=True)
    n = matrix.rows
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix)


def rowspace(matrix, field=Real):
    """
    Compute the row space of the matrix.

    Parameters
    ----------
    matrix : list of list or sympy.Matrix
        The matrix to take the row space of.
    field : {Real, Complex}
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
    >>> vs = rowspace(matrix)
    >>> print(vs.basis)
    [[1, 0], [0, 1]]
    """
    constraints = [f'row({matrix})']
    matrix = u.rref(matrix, remove=True)
    n = matrix.cols
    return VectorSpace.fn(field, n, constraints, rs_matrix=matrix)


def nullspace(matrix, field=Real):
    """
    Compute the null space, or kernel, of the matrix.

    Parameters
    ----------
    matrix : list of list or sympy.Matrix
        The matrix to take the null space of.
    field : {Real, Complex}
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
    >>> vs = nullspace(matrix)
    >>> print(vs.basis)
    []
    >>> vs = kernel(matrix)
    >>> print(vs.basis)
    []
    """
    constraints = [f'null({matrix})']
    matrix = u.rref(matrix, remove=True)
    n = matrix.cols
    return VectorSpace.fn(field, n, constraints, ns_matrix=matrix)


def left_nullspace(matrix, field=Real):
    """
    Compute the left null space of the matrix.

    Parameters
    ----------
    matrix : list of list or sympy.Matrix
        The matrix to take the left null space of.
    field : {Real, Complex}
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
    >>> vs = left_nullspace(matrix)
    >>> print(vs.basis)
    []
    >>> matrix = sympy.Matrix([[1, 2], [3, 4]])
    >>> vs1 = left_nullspace(matrix)
    >>> vs2 = nullspace(matrix.T)
    >>> print(vs1 == vs2)
    True
    """
    matrix = sp.Matrix(matrix).T
    return nullspace(matrix, field)


# Aliases
image = columnspace
kernel = nullspace