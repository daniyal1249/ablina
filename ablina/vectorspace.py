from numbers import Complex, Real
from random import gauss

import sympy as sp

from .mathset import Set
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
            raise VectorSpaceError('Invalid coordinate vector.') from e
        return vec.flat() if vec else [0] * self.n
    
    def are_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows
    
    def _is_basis(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows and len(vectors) == self.dim

    # Methods relating to vector spaces

    def is_subspace(self, vs2):
        for i in range(self._rs_matrix.rows):
            vec = self._rs_matrix.row(i).T
            if not (vs2._ns_matrix @ vec).is_zero_matrix:
                return False
        return True

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
        orthonormal_vecs = []
        for v in vectors:
            for q in orthonormal_vecs:
                factor = self.dot(v, q)
                proj = [factor * i for i in q]
                v = [i - j for i, j in zip(v, proj)]
            norm = self.norm(v)
            orthonormal_vecs.append([i / norm for i in v])
        return orthonormal_vecs


class Fn(_StandardFn):
    """
    pass
    """

    def __init__(
            self, field, n, constraints=None, add=None, mul=None, 
            *, isomorphism=None, ns_matrix=None, rs_matrix=None
            ):
        """
        pass
        """
        if constraints is None:
            constraints = []
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
    def additive_id(self):
        return [0] * self.n
    
    @property
    def additive_inv(self):
        def additive_inv(vec):
            return [-i for i in vec]
        return additive_inv
    
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

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        attributes = ['set', 'fn']
        methods = ['__to_fn__', '__from_fn__']

        for attr in attributes:
            if not hasattr(cls, attr):
                raise TypeError(f'{cls.__name__} must define "{attr}".')
        for method in methods:
            if not callable(getattr(cls, method, None)):
                raise TypeError(f'{cls.__name__} must define the method "{method}".')
        
        if not isinstance(cls.set, Set):
            raise TypeError(f'{cls.__name__}.set must be a MathematicalSet.')
        if not isinstance(cls.fn, Fn):
            raise TypeError(f'{cls.__name__}.fn must be of type Fn.')
        if name is not None:
            cls.__name__ = name

    def __init__(self, constraints=None, basis=None, *, fn=None):
        """
        pass
        """
        self.set = Set(self.set.cls, lambda vec: vec in self)
        if fn is not None:
            self.fn = fn
            return
        self.fn = Fn(
            self.fn.field, self.fn.n, constraints, self.fn.add, self.fn.mul, 
            isomorphism=(self.fn._to_standard, self.fn._from_standard)
            )
        if basis is not None:
            if not self.are_independent(*basis):
                raise VectorSpaceError('Basis vectors must be linearly independent.')
            self.fn = self.fn.span(basis=[self.__to_fn__(vec) for vec in basis])
    
    @property
    def field(self):
        """
        {Real, Complex}: The field of scalars.
        """
        return self.fn.field
    
    @property
    def add(self):
        """
        callable: The addition operator on the vector space.
        """
        def add(vec1, vec2):
            fn_vec1, fn_vec2 = self.__to_fn__(vec1), self.__to_fn__(vec2)
            sum = self.fn.add(fn_vec1, fn_vec2)
            return self.__from_fn__(sum)
        return add
    
    @property
    def mul(self):
        """
        callable: The multiplication operator on the vector space.
        """
        def mul(scalar, vec):
            fn_vec = self.__to_fn__(vec)
            prod = self.fn.mul(scalar, fn_vec)
            return self.__from_fn__(prod)
        return mul
    
    @property
    def additive_id(self):
        """
        object: The additive identity of the vector space.
        """
        return self.__from_fn__(self.fn.additive_id)
    
    @property
    def additive_inv(self):
        """
        callable: A function that returns the additive inverse of a given vector.
        """
        def additive_inv(vec):
            fn_vec = self.__to_fn__(vec)
            inv = self.fn.additive_inv(fn_vec)
            return self.__from_fn__(inv)
        return additive_inv
    
    @property
    def basis(self):
        """
        list: The basis of the vector space.
        """
        return [self.__from_fn__(vec) for vec in self.fn.basis]
    
    @property
    def dim(self):
        """
        int: The dimension of the vector space.
        """
        return self.fn.dim

    def __contains__(self, vec):
        """
        Check whether a vector is an element of the vector space.

        Parameters
        ----------
        vec : object
            The vector to check.

        Returns
        -------
        bool
            True if `vec` is an element of `self`, otherwise False.
        """
        if vec not in type(self).set:
            return False
        return self.__to_fn__(vec) in self.fn
    
    def __eq__(self, vs2):
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)
    
    def __add__(self, other):
        """
        pass
        """
        if isinstance(other, VectorSpace):
            return self.sum(other)
        return self.coset(other)
    
    def __radd__(self, other):
        return self.coset(other)
    
    def __truediv__(self, vs2):
        """
        Same as ``VectorSpace.quotient``.
        """
        return self.quotient(vs2)
    
    def __and__(self, vs2):
        """
        Same as ``VectorSpace.intersection``.
        """
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

        >>> vs = fn(Real, 3, constraints=['2*v0 == v1'])
        >>> vs.vector()
        [1, 2, 0]
        >>> vs.vector()
        [-1, -2, 1]
        >>> vs.vector(std=10)
        [11, 22, 13]
        >>> vs.vector(arbitrary=True)
        [c0, 2*c0, c1]
        """
        fn_vec = self.fn.vector(std, arbitrary)
        return self.__from_fn__(fn_vec)
    
    def to_coordinate(self, vector, basis=None):
        """
        Convert a vector to its coordinate vector representation.

        Parameters
        ----------
        vector : object
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

        >>> vs = fn(Real, 3, constraints=['v0 == 2*v1'])
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
            basis = [self.__to_fn__(vec) for vec in basis]

        fn_vec = self.__to_fn__(vector)
        return self.fn.to_coordinate(fn_vec, basis)
    
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
        vector : list
            The coordinate vector to convert.
        basis : list, optional
            A basis of the vector space.

        Returns
        -------
        object
            The vector represented by `vector`.

        Raises
        ------
        VectorSpaceError
            If `vector` is of incorrect length.

        See Also
        --------
        VectorSpace.to_coordinate

        Examples
        --------

        >>> vs = fn(Real, 3, constraints=['v0 == 2*v1'])
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
            basis = [self.__to_fn__(vec) for vec in basis]
        
        fn_vec = self.fn.from_coordinate(vector, basis)
        return self.__from_fn__(fn_vec)
    
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

        >>> vs = fn(Real, 3)
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
        fn_vecs = [self.__to_fn__(vec) for vec in vectors]
        return self.fn.are_independent(*fn_vecs)

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
        TypeError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.intersection

        Examples
        --------

        >>> vs1 = fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = fn(Real, 3, constraints=['v1 == v2'])
        >>> vs = vs1.sum(vs2)
        >>> vs.basis
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> vs1 + vs2 == vs
        True
        """
        self._validate_type(vs2)
        fn = self.fn.sum(vs2.fn)
        return type(self)(fn=fn)
    
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
        TypeError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.sum

        Examples
        --------

        >>> vs1 = fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = fn(Real, 3, constraints=['v1 == v2'])
        >>> vs = vs1.intersection(vs2)
        >>> vs.basis
        [[1, 1, 1]]
        >>> vs1 & vs2 == vs
        True
        """
        self._validate_type(vs2)
        fn = self.fn.intersection(vs2.fn)
        return type(self)(fn=fn)
    
    def span(self, *vectors, basis=None):
        """
        The span of the given vectors.

        Returns the smallest subspace of `self` that contains the vectors 
        in `vectors`. In order to manually set the basis of the resulting 
        space, pass the vectors into `basis` instead. Note that the 
        vectors must be linearly independent if passed into `basis`.

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

        >>> vs = fn(Real, 3)
        >>> vs.span([1, 2, 3], [4, 5, 6]).basis
        [[1, 0, -1], [0, 1, 2]]
        >>> vs.span(basis=[[1, 2, 3], [4, 5, 6]]).basis
        [[1, 2, 3], [4, 5, 6]]
        >>> vs.span().basis
        []
        """
        if basis is not None:
            return type(self)(basis=basis)
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        
        fn_vecs = [self.__to_fn__(vec) for vec in vectors]
        fn = self.fn.span(*fn_vecs)
        return type(self)(fn=fn)
    
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

        >>> vs = fn(Real, 3)
        >>> vs1 = fn(Real, 3, constraints=['v0 == v1'])
        >>> vs2 = fn(Real, 3, constraints=['v1 == v2'])
        >>> vs1.is_subspace(vs)
        True
        >>> vs2.is_subspace(vs)
        True
        >>> vs1.is_subspace(vs2)
        False
        >>> vs.is_subspace(vs)
        True
        """
        try:
            self._validate_type(vs2)
        except TypeError:
            return False
        return self.fn.is_subspace(vs2.fn)
    
    # Methods relating to affine spaces
    
    def coset(self, vector):
        """
        pass

        Parameters
        ----------
        vector : object
            A vector in the vector space.

        Returns
        -------
        AffineSpace
            pass

        See Also
        --------
        VectorSpace.quotient
        """
        return AffineSpace(self, vector)
    
    def quotient(self, vs2):
        """
        The quotient of two vector spaces.

        Parameters
        ----------
        vs2 : VectorSpace
            The vector space to divide by.

        Returns
        -------
        VectorSpace
            The quotient of `self` by `vs2`.

        Raises
        ------
        TypeError
            If `vs2` is not a subspace of `self`.

        See Also
        --------
        VectorSpace.coset
        """
        raise NotImplementedError()
        # if not isinstance(vs2, VectorSpace):
        #     raise TypeError()
        # if not vs2.is_subspace(self):
        #     raise TypeError()
        
        # name = f''

        # def in_quotient_space(coset):
        #     return

        # class quotient_space(VectorSpace, name=name):
        #     set = Set(AffineSpace, in_quotient_space, name=name)
        #     fn = Fn(self.field, None, add=self.fn.add, mul=self.fn.mul)
        #     def __to_fn__(self, coset): return
        #     def __from_fn__(self, vec): return
        # return quotient_space()

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

        >>> vs = fn(Real, 3)
        >>> vs.dot([1, 2, 3], [4, 5, 6])
        32
        >>> vs.dot([1, 0, 1], [0, 1, 0])
        0
        """
        if not (vec1 in self and vec2 in self):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vec1, fn_vec2 = self.__to_fn__(vec1), self.__to_fn__(vec2)
        return self.fn.dot(fn_vec1, fn_vec2)
    
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

        >>> vs = fn(Real, 3)
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

        >>> vs = fn(Real, 3)
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

        >>> vs = fn(Real, 3)
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
            The vectors in the vector space.

        Returns
        -------
        list
            An orthonormal list of vectors.
        
        Raises
        ------
        VectorSpaceError
            If the provided vectors are not linearly independent.

        See Also
        --------
        VectorSpace.are_orthonormal
        """
        if not self.are_independent(*vectors):
            raise VectorSpaceError('Vectors must be linearly independent.')
        fn_vecs = [self.__to_fn__(vec) for vec in vectors]
        orthonormal_vecs = self.fn.gram_schmidt(*fn_vecs)
        return [self.__from_fn__(vec) for vec in orthonormal_vecs]
    
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

        >>> vs = fn(Real, 3, constraints=['v0 == v1'])
        >>> vs.ortho_complement().basis
        [[1, -1, 0]]
        >>> vs.ortho_complement().ortho_complement() == vs
        True
        """
        fn = self.fn.ortho_complement()
        return type(self)(fn=fn)
    
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
        TypeError
            If `self` and `vs2` do not share the same ambient space.

        See Also
        --------
        VectorSpace.ortho_complement, VectorSpace.dot
        """
        self._validate_type(vs2)
        fn = self.fn.ortho_projection(vs2.fn)
        return type(self)(fn=fn)

    def _validate_type(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError()
        if type(self).__name__ != type(vs2).__name__:
            raise TypeError()


class AffineSpace:
    """
    pass
    """

    def __init__(self, vectorspace, vector):
        """
        pass
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError()
        if vector not in type(vectorspace)():
            raise TypeError()
        
        self._vectorspace = vectorspace
        self._representative = vector

    @property
    def vectorspace(self):
        """
        pass
        """
        return self._vectorspace
    
    @property
    def representative(self):
        """
        pass
        """
        return self._representative
    
    @property
    def set(self):
        """
        MathematicalSet: The set containing the points in the affine space.
        """
        return Set(self.vectorspace.set.cls, lambda point: point in self)
    
    @property
    def dim(self):
        """
        int: The dimension of the affine space.
        """
        return self.vectorspace.dim

    def __contains__(self, point):
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
        if point not in type(self.vectorspace)():
            return False
        vec1 = self.representative
        vec2 = self.vectorspace.additive_inv(point)
        return self.vectorspace.add(vec1, vec2) in self.vectorspace

    def __eq__(self, as2):
        if not isinstance(as2, AffineSpace):
            return False
        return self.representative in as2

    def __add__(self, as2):
        if not isinstance(as2, AffineSpace):
            raise TypeError()
        if self.vectorspace != as2.vectorspace:
            raise TypeError('Affine spaces must be cosets of the same vector space.')
        repr = self.vectorspace.add(self.representative, as2.representative)
        return AffineSpace(self.vectorspace, repr)

    def __mul__(self, scalar):
        if not isinstance(scalar, self.vectorspace.field):
            raise TypeError('Scalar must be an element of the field.')
        repr = self.vectorspace.mul(scalar, self.representative)
        return AffineSpace(self.vectorspace, repr)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


def fn(field, n, constraints=None, basis=None, 
       add=None, mul=None, *, ns_matrix=None, rs_matrix=None):
    """
    pass
    """
    name = f'{field.__name__}^{n}'

    def in_fn(vec):
        try: return sp.Matrix(vec).shape == (n, 1)
        except Exception: return False

    class fn(VectorSpace, name=name):
        set = Set(object, in_fn, name=name)
        fn = Fn(field, n, add=add, mul=mul)
        def __to_fn__(self, vec): return vec
        def __from_fn__(self, vec): return vec

    if not (ns_matrix is None and rs_matrix is None):
        vectorspace = Fn(
            field, n, constraints, add, mul, 
            ns_matrix=ns_matrix, rs_matrix=rs_matrix
            )
        return fn(fn=vectorspace)
    return fn(constraints, basis)


def matrix_space(field, shape, constraints=None, basis=None, 
                 add=None, mul=None):
    """
    pass
    """
    name = f'M({field.__name__}, {shape})'

    def in_matrix_space(mat):
        return mat.shape == shape

    class matrix_space(VectorSpace, name=name):
        set = Set(sp.Matrix, in_matrix_space, name=name)
        fn = Fn(field, sp.prod(shape), add=add, mul=mul)
        def __to_fn__(self, mat): return mat.flat()
        def __from_fn__(self, vec): return sp.Matrix(*shape, vec)
    return matrix_space(constraints, basis)


def poly_space(field, max_degree, constraints=None, basis=None, 
               add=None, mul=None):
    """
    pass
    """
    name = f'P({field.__name__}, {max_degree})'

    def in_poly_space(poly):
        return sp.degree(poly) <= max_degree

    class poly_space(VectorSpace, name=name):
        set = Set(sp.Poly, in_poly_space, name=name)
        fn = Fn(field, max_degree + 1, add=add, mul=mul)
        def __to_fn__(self, poly):
            coeffs = poly.all_coeffs()[::-1]  # Ascending order
            degree_diff = max_degree - len(coeffs) + 1
            return coeffs + ([0] * degree_diff)
        def __from_fn__(self, vec):
            x = sp.symbols('x')
            return sp.Poly.from_list(vec[::-1], x)
    return poly_space(constraints, basis)


def hom(vs1, vs2):
    """
    pass
    """
    if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
        raise TypeError()
    if vs1.field is not vs2.field:
        raise VectorSpaceError()
    return matrix_space(vs1.field, (vs2.dim, vs1.dim))


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
    Return the column space, or image, of a matrix.

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
    return fn(field, n, constraints, rs_matrix=matrix)


def rowspace(matrix, field=Real):
    """
    Return the row space of a matrix.

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
    return fn(field, n, constraints, rs_matrix=matrix)


def nullspace(matrix, field=Real):
    """
    Return the null space, or kernel, of a matrix.

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
    return fn(field, n, constraints, ns_matrix=matrix)


def left_nullspace(matrix, field=Real):
    """
    Return the left null space of a matrix.

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