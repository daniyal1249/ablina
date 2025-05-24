from random import gauss

import sympy as sp

from .field import R, C
from .mathset import MathSet
from .parser import split_constraint, sympify
from . import utils as u
from . import vs_utils as vsu


class NotAVectorSpaceError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class Fn:
    """
    pass
    """

    def __init__(self, field, n, constraints=None, *, ns_matrix=None, rs_matrix=None):
        """
        pass
        """
        if constraints is None:
            constraints = []
        if field not in (R, C):
            raise TypeError('Field must be either R or C.')

        # Verify whether constraints satisfy vector space properties
        if ns_matrix is None and rs_matrix is None:
            if not is_vectorspace(n, constraints):
                raise NotAVectorSpaceError(
                    'Constraints do not satisfy vector space axioms.'
                    )

        add, mul, additive_inv = Fn._init_operations()
        ns, rs = Fn._init_matrices(n, constraints, ns_matrix, rs_matrix)

        self._field = field
        self._n = n
        self._constraints = constraints

        self._add = add
        self._mul = mul
        self._additive_inv = additive_inv

        self._ns_matrix = ns
        self._rs_matrix = rs

    @staticmethod
    def _init_operations():
        def add(vec1, vec2):
            return [i + j for i, j in zip(vec1, vec2)]
        def mul(scalar, vec):
            return [scalar * i for i in vec]
        def additive_inv(vec):
            return [-i for i in vec]
        return add, mul, additive_inv

    @staticmethod
    def _init_matrices(n, constraints, ns, rs):
        if ns is not None:
            ns = sp.zeros(0, n) if u.is_empty(ns) else sp.Matrix(ns)
        if rs is not None:
            rs = sp.zeros(0, n) if u.is_empty(rs) else sp.Matrix(rs)
        
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
    def field(self):
        return self._field
    
    @property
    def n(self):
        return self._n
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def add(self):
        return self._add
    
    @property
    def mul(self):
        return self._mul
    
    @property
    def additive_inv(self):
        return self._additive_inv
    
    @property
    def additive_id(self):
        return [0] * self.n
    
    @property
    def basis(self):
        return self._rs_matrix.tolist()
    
    @property
    def dim(self):
        return self._rs_matrix.rows
    
    def __repr__(self):
        return (
            f'Fn(field={self.field!r}, '
            f'n={self.n!r}, '
            f'constraints={self.constraints!r}, '
            f'ns_matrix={self._ns_matrix!r}, '
            f'rs_matrix={self._rs_matrix!r})'
            )
    
    def __contains__(self, vector):
        if not all(i in self.field for i in vector):
            return False
        try:
            # Check if vector satisfies vector space constraints
            vector = sp.Matrix(vector)
            return bool((self._ns_matrix @ vector).is_zero_matrix)
        except Exception:
            return False

    # Methods relating to vectors

    def vector(self, std=1, arbitrary=False):
        size = self.dim
        if arbitrary:
            weights = list(u.symbols(f'c:{size}', field=self.field))
        else:
            weights = [round(gauss(0, std)) for _ in range(size)]
        vec = sp.Matrix([weights]) @ self._rs_matrix
        return vec.flat()

    def to_coordinate(self, vector, basis=None):
        if basis is None:
            basis = self.basis
        elif not self.is_basis(*basis):
            raise ValueError('Provided vectors do not form a basis.')
        if not basis:
            return []
        
        matrix, vec = sp.Matrix(basis).T, sp.Matrix(vector)
        coord_vec = matrix.solve_least_squares(vec)
        return coord_vec.flat()

    def from_coordinate(self, vector, basis=None):  # FIX: check field
        if basis is None:
            basis = self.basis
        elif not self.is_basis(*basis):
            raise ValueError('Provided vectors do not form a basis.')
        try:
            matrix, coord_vec = sp.Matrix(basis).T, sp.Matrix(vector)
            vec = matrix @ coord_vec
        except Exception as e:
            raise ValueError('Invalid coordinate vector.') from e
        return vec.flat() if vec else self.additive_id
    
    def is_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows
    
    def is_basis(self, *vectors):
        return self.is_independent(*vectors) and len(vectors) == self.dim
    
    # Methods relating to vector spaces

    def sum(self, vs2):
        rs_matrix = sp.Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix = u.rref(rs_matrix, remove=True)
        constraints = self.constraints  # FIX: rework
        return Fn(self.field, self.n, constraints, rs_matrix=rs_matrix)
    
    def intersection(self, vs2):
        ns_matrix = sp.Matrix.vstack(self._ns_matrix, vs2._ns_matrix)
        ns_matrix = u.rref(ns_matrix, remove=True)
        constraints = self.constraints + vs2.constraints
        return Fn(self.field, self.n, constraints, ns_matrix=ns_matrix)
    
    def span(self, *vectors, basis=None):
        vectors = u.rref(vectors, remove=True) if basis is None else basis
        constraints = [f'span({', '.join(map(str, vectors))})']
        return Fn(self.field, self.n, constraints, rs_matrix=vectors)

    def is_subspace(self, vs2):
        for i in range(vs2.dim):
            vec = vs2._rs_matrix.row(i).T
            if not (self._ns_matrix @ vec).is_zero_matrix:
                return False
        return True
    
    # Methods involving the dot product

    def dot(self, vec1, vec2):
        return sum(i * j for i, j in zip(vec1, vec2))
    
    def norm(self, vector):
        return sp.sqrt(self.dot(vector, vector))
    
    def is_orthogonal(self, *vectors):
        for i, vec1 in enumerate(vectors, 1):
            for vec2 in vectors[i:]:
                # FIX: consider tolerance
                if self.dot(vec1, vec2) != 0:
                    return False
        return True

    def is_orthonormal(self, *vectors):
        if not self.is_orthogonal(*vectors):
            return False
        return all(self.norm(vec).equals(1) for vec in vectors)
    
    def gram_schmidt(self, *vectors):
        orthonormal_vecs = []
        for v in vectors:
            for q in orthonormal_vecs:
                factor = self.dot(v, q)
                proj = self.mul(factor, q)
                v = self.add(v, self.additive_inv(proj))
            unit_v = self.mul(1 / self.norm(v), v)
            orthonormal_vecs.append(unit_v)
        return orthonormal_vecs
    
    def ortho_projection(self, vector, subspace):
        vector = sp.Matrix(vector)
        matrix = subspace._rs_matrix.T
        proj = matrix @ (matrix.T @ matrix).inv() @ matrix.T @ vector
        return proj.flat()
    
    def ortho_complement(self, subspace):
        constraints = [f'ortho_complement({', '.join(self.constraints)})']
        comp = Fn(self.field, self.n, constraints, rs_matrix=subspace._ns_matrix)
        return self.intersection(comp)


class VectorSpace:
    """
    pass
    """

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validate_subclass_contract()
        add, mul, additive_inv = cls._init_operations()

        cls.name = cls.__name__ if name is None else name
        cls._add = staticmethod(add)
        cls._mul = staticmethod(mul)
        cls._additive_inv = staticmethod(additive_inv)

    def __init__(self, name, constraints=None, basis=None, *, fn=None):
        """
        pass
        """
        self.name = name
        self.set = MathSet(name, self.set.cls, lambda vec: vec in self)

        if fn is not None:
            self.fn = fn
            return
        self.fn = Fn(self.fn.field, self.fn.n, constraints)

        if basis is not None:
            if not self.is_independent(*basis):
                raise ValueError('Basis vectors must be linearly independent.')
            self.fn = self.fn.span(basis=[self.__push__(vec) for vec in basis])

    @classmethod
    def _validate_subclass_contract(cls):
        attributes = ['set', 'fn']
        methods = ['__push__', '__pull__']
        
        for attr in attributes:
            if not hasattr(cls, attr):
                raise TypeError(f'{cls.__name__} must define "{attr}".')
        for method in methods:
            if not callable(getattr(cls, method, None)):
                raise TypeError(f'{cls.__name__} must define the method "{method}".')
        
        if not isinstance(cls.set, MathSet):
            raise TypeError(f'{cls.__name__}.set must be a MathSet.')
        if not isinstance(cls.fn, Fn):
            raise TypeError(f'{cls.__name__}.fn must be of type Fn.')
        
        cls.__push__ = staticmethod(cls.__push__)
        cls.__pull__ = staticmethod(cls.__pull__)

    @classmethod
    def _init_operations(cls):
        def add(vec1, vec2):
            fn_vec1, fn_vec2 = cls.__push__(vec1), cls.__push__(vec2)
            sum = cls.fn.add(fn_vec1, fn_vec2)
            return cls.__pull__(sum)
        def mul(scalar, vec):
            fn_vec = cls.__push__(vec)
            prod = cls.fn.mul(scalar, fn_vec)
            return cls.__pull__(prod)
        def additive_inv(vec):
            fn_vec = cls.__push__(vec)
            inv = cls.fn.additive_inv(fn_vec)
            return cls.__pull__(inv)
        return add, mul, additive_inv
    
    @property
    def field(self):
        """
        {R, C}: The field of scalars.
        """
        return self.fn.field
    
    @property
    def add(self):
        """
        callable: The addition operator on the vector space.
        """
        return self._add
    
    @property
    def mul(self):
        """
        callable: The multiplication operator on the vector space.
        """
        return self._mul
    
    @property
    def additive_inv(self):
        """
        callable: A function that returns the additive inverse of a given vector.
        """
        return self._additive_inv
    
    @property
    def additive_id(self):
        """
        object: The additive identity of the vector space.
        """
        return self.__pull__(self.fn.additive_id)
    
    @property
    def basis(self):
        """
        list: The basis of the vector space.
        """
        return [self.__pull__(vec) for vec in self.fn.basis]
    
    @property
    def dim(self):
        """
        int: The dimension of the vector space.
        """
        return self.fn.dim
    
    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r}, basis={self.basis!r})'
    
    def __str__(self):
        return self.name
    
    def __eq__(self, vs2):
        if self is vs2:
            return True
        return self.is_subspace(vs2) and vs2.is_subspace(self)

    def __contains__(self, vector):
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
    
    def __pos__(self):
        """
        Return `self`.
        """
        return self
    
    def __neg__(self):
        """
        Return `self`.
        """
        return self
    
    def __add__(self, other):
        """
        pass
        """
        if isinstance(other, VectorSpace):
            return self.sum(other)
        return self.coset(other)
    
    def __radd__(self, vector):
        return self.coset(vector)
    
    def __sub__(self, other):
        """
        pass
        """
        if isinstance(other, VectorSpace):
            return self.sum(other)
        if other not in self.ambient_space():
            raise TypeError()
        return self.coset(self.additive_inv(other))
    
    def __rsub__(self, vector):
        return self.coset(vector)
    
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

    def info(self):
        name = f'{self} (Subspace of {type(self).name})'
        lines = [
            name,
            '-' * len(name),
            f'Field      {self.field}',
            f'Identity   {self.additive_id}',
            f'Basis      [{', '.join(map(str, self.basis))}]',
            f'Dimension  {self.dim}',
            f'Vector     {self.vector(arbitrary=True)}'
            ]
        return '\n'.join(lines)

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

        >>> V = fn('V', R, 3, constraints=['2*v0 == v1'])
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
        ValueError
            If the provided basis vectors do not form a basis for the 
            vector space.

        See Also
        --------
        VectorSpace.from_coordinate

        Examples
        --------

        >>> V = fn('V', R, 3, constraints=['v0 == 2*v1'])
        >>> V.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> V.to_coordinate([2, 1, 2])
        [2, 0]
        """
        if vector not in self:
            raise TypeError('Vector must be an element of the vector space.')
        if basis is not None:
            if not all(vec in self for vec in basis):
                raise TypeError('Basis vectors must be elements of the vector space.')
            basis = [self.__push__(vec) for vec in basis]

        fn_vec = self.__push__(vector)
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
        ValueError
            If `vector` has invalid length.

        See Also
        --------
        VectorSpace.to_coordinate

        Examples
        --------

        >>> V = fn('V', R, 3, constraints=['v0 == 2*v1'])
        >>> V.basis
        [[1, 1/2, 0], [0, 0, 1]]
        >>> V.from_coordinate([1, 1])
        [1, 1/2, 1]
        >>> new_basis = [[2, 1, 1], [0, 0, 1]]
        >>> V.from_coordinate([1, 1], basis=new_basis)
        [2, 1, 2]
        """
        if basis is not None:
            if not all(vec in self for vec in basis):
                raise TypeError('Basis vectors must be elements of the vector space.')
            basis = [self.__push__(vec) for vec in basis]
        
        fn_vec = self.fn.from_coordinate(vector, basis)
        return self.__pull__(fn_vec)
    
    def is_independent(self, *vectors):
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

        >>> V = fn('V', R, 3)
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
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vecs = [self.__push__(vec) for vec in vectors]
        return self.fn.is_independent(*fn_vecs)
    
    def is_basis(self, *vectors):
        """
        Check whether the given vectors form a basis.

        Parameters
        ----------
        *vectors
            The vectors in the vector space.

        Returns
        -------
        bool
            True if the vectors form a basis, otherwise False.
        """
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vecs = [self.__push__(vec) for vec in vectors]
        return self.fn.is_basis(*fn_vecs)
    
    def change_of_basis(self, basis):
        """
        pass
        """
        if not self.is_basis(*basis):
            raise ValueError('Provided vectors do not form a basis.')
        basechange = [self.to_coordinate(vec) for vec in basis]
        return (sp.Matrix(basechange).T).inv()

    # Methods relating to vector spaces

    def ambient_space(self):
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

        >>> U = fn('U', R, 3, constraints=['v0 == v1'])
        >>> V = fn('V', R, 3, constraints=['v1 == v2'])
        >>> W = U.sum(V)
        >>> W.basis
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> U + V == W
        True
        """
        self._validate_type(vs2)
        name = f'{self} + {vs2}'
        fn = self.fn.sum(vs2.fn)
        return type(self)(name, fn=fn)
    
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

        >>> U = fn('U', R, 3, constraints=['v0 == v1'])
        >>> V = fn('V', R, 3, constraints=['v1 == v2'])
        >>> W = U.intersection(V)
        >>> W.basis
        [[1, 1, 1]]
        >>> U & V == W
        True
        """
        self._validate_type(vs2)
        name = f'{self} ∩ {vs2}'
        fn = self.fn.intersection(vs2.fn)
        return type(self)(name, fn=fn)
    
    def span(self, name, *vectors, basis=None):
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
        ValueError
            If the provided basis vectors are not linearly independent.

        Examples
        --------

        >>> V = fn('V', R, 3)
        >>> V.span('span1', [1, 2, 3], [4, 5, 6]).basis
        [[1, 0, -1], [0, 1, 2]]
        >>> V.span('span2', basis=[[1, 2, 3], [4, 5, 6]]).basis
        [[1, 2, 3], [4, 5, 6]]
        >>> V.span('span3').basis
        []
        """
        if basis is not None:
            return type(self)(name, basis=basis)
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        
        fn_vecs = [self.__push__(vec) for vec in vectors]
        fn = self.fn.span(*fn_vecs)
        return type(self)(name, fn=fn)
    
    def is_subspace(self, vs2):
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

        >>> V = fn('V', R, 3)
        >>> U = fn('U', R, 3, constraints=['v0 == v1'])
        >>> W = fn('W', R, 3, constraints=['v1 == v2'])
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
    
    def coset(self, representative):
        """
        pass

        Parameters
        ----------
        representative : object
            A vector in the vector space.

        Returns
        -------
        AffineSpace
            pass

        See Also
        --------
        VectorSpace.quotient
        """
        return AffineSpace(self, representative)
    
    def quotient(self, subspace):
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
            raise TypeError()
        
        vs = self.ambient_space()
        cls_name = f'{vs} / {subspace}'

        def in_quotient_space(coset):
            return coset.vectorspace == subspace  # FIX: check

        class quotient_space(VectorSpace, name=cls_name):
            set = MathSet(cls_name, AffineSpace, in_quotient_space)
            fn = vs.fn.ortho_complement(subspace.fn)
            def __push__(coset):
                fn_vec = vs.__push__(coset.representative)
                return vs.fn.ortho_projection(fn_vec, fn)
            def __pull__(vec):
                return subspace.coset(vs.__pull__(vec))

        name = f'{self} / {subspace}'
        fn = self.fn.ortho_complement(subspace.fn)
        return quotient_space(name, fn=fn)

    def _validate_type(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError(f'Expected a VectorSpace, got {type(vs2).__name__} instead.')
        if type(self).name != type(vs2).name:
            raise TypeError('Vector spaces must be subspaces of the same ambient space.')


class AffineSpace:
    """
    pass
    """

    def __init__(self, vectorspace, representative):
        """
        pass
        """
        if not isinstance(vectorspace, VectorSpace):
            raise TypeError()
        if representative not in vectorspace.ambient_space():
            raise TypeError()
        
        self.name = f'{vectorspace} + {representative}'
        self._vectorspace = vectorspace
        self._representative = representative

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
        MathSet: The set containing the points in the affine space.
        """
        vs = self.vectorspace
        return MathSet(self.name, vs.set.cls, lambda point: point in self)
    
    @property
    def dim(self):
        """
        int: The dimension of the affine space.
        """
        return self.vectorspace.dim
    
    def __repr__(self):
        return (
            f'AffineSpace(vectorspace={self.vectorspace!r}, '
            f'representative={self.representative!r})'
            )
    
    def __str__(self):
        return self.name
    
    def __eq__(self, as2):
        if not isinstance(as2, AffineSpace):
            return False
        return self.representative in as2

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
        vs = self.vectorspace
        if point not in vs.ambient_space():
            return False
        
        vec1 = self.representative
        vec2 = vs.additive_inv(point)
        return vs.add(vec1, vec2) in vs
    
    def __pos__(self):
        """
        Return `self`.
        """
        return self
    
    def __neg__(self):
        """
        pass
        """
        vs = self.vectorspace
        repr = vs.additive_inv(self.representative)
        return AffineSpace(vs, repr)
    
    def __add__(self, other):
        """
        pass
        """
        vs = self.vectorspace
        if isinstance(other, AffineSpace):
            return self.sum(other)
        if other not in vs.ambient_space():
            raise TypeError()
        
        repr = vs.add(self.representative, other)
        return AffineSpace(vs, repr)

    def __radd__(self, vector):
        return self.__add__(vector)
    
    def __sub__(self, other):
        """
        pass
        """
        vs = self.vectorspace
        if isinstance(other, AffineSpace):
            return self.sum(-other)
        if other not in vs.ambient_space():
            raise TypeError()
        
        repr = vs.add(self.representative, vs.additive_inv(other))
        return AffineSpace(vs, repr)
    
    def __rsub__(self, vector):
        return (-self).__add__(vector)

    def __mul__(self, scalar):
        """
        pass
        """
        vs = self.vectorspace
        if scalar not in vs.field:
            raise TypeError('Scalar must be an element of the field.')
        repr = vs.mul(scalar, self.representative)
        return AffineSpace(vs, repr)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def info(self):
        name = self.name
        lines = [
            name,
            '-' * len(name),
            f'Vector Space    {self.vectorspace}',
            f'Representative  {self.representative}',
            f'Dimension       {self.dim}',
            f'Point           {self.point(arbitrary=True)}'
            ]
        return '\n'.join(lines)
    
    def point(self, std=1, arbitrary=False):
        """
        Return a point from the affine space.

        Parameters
        ----------
        std : float
            The standard deviation used to generate weights.
        arbitrary : bool, default=False
            Determines whether a random point or arbitrary point is returned.
        
        Returns
        -------
        object
            A point in the affine space.
        """
        vs = self.vectorspace
        vector = vs.vector(std, arbitrary)
        point = vs.add(vector, self.representative)
        return point
    
    def sum(self, as2):
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
        vs = self.vectorspace
        if not isinstance(as2, AffineSpace):
            raise TypeError()
        if vs != as2.vectorspace:
            raise TypeError('Affine spaces must be cosets of the same vector space.')
        
        repr = vs.add(self.representative, as2.representative)
        return AffineSpace(vs, repr)

    def intersection(self, as2):
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

        See Also
        --------
        AffineSpace.sum
        """
        raise NotImplementedError()


def fn(name, field, n, constraints=None, basis=None, *, 
       ns_matrix=None, rs_matrix=None):
    """
    pass
    """
    if n == 1:
        cls_name = f'{field}'
        class fn(VectorSpace, name=cls_name):
            set = MathSet(cls_name, object)
            fn = Fn(field, 1)
            def __push__(vec): return [vec]
            def __pull__(vec): return vec[0]
    else:
        def in_fn(vec):
            try: return sp.Matrix(vec).shape == (n, 1)
            except Exception: return False

        cls_name = f'{field}^{n}'
        class fn(VectorSpace, name=cls_name):
            set = MathSet(cls_name, object, in_fn)
            fn = Fn(field, n)
            def __push__(vec): return vec
            def __pull__(vec): return vec

    if not (ns_matrix is None and rs_matrix is None):
        vs = Fn(field, n, constraints, ns_matrix=ns_matrix, rs_matrix=rs_matrix)
        return fn(name, fn=vs)
    return fn(name, constraints, basis)


def matrix_space(name, field, shape, constraints=None, basis=None):
    """
    pass
    """
    cls_name = f'{field}^({shape[0]} x {shape[1]})'

    def in_matrix_space(mat):
        try: return sp.Matrix(mat).shape == shape
        except Exception: return False

    class matrix_space(VectorSpace, name=cls_name):
        set = MathSet(cls_name, object, in_matrix_space)
        fn = Fn(field, sp.prod(shape))
        def __push__(mat):
            mat = sp.Matrix(mat)
            return mat.flat()
        def __pull__(vec):
            mat = sp.Matrix(*shape, vec)
            return mat.tolist()
    return matrix_space(name, constraints, basis)


def poly_space(name, field, max_degree, constraints=None, basis=None):
    """
    pass
    """
    cls_name = f'P_{max_degree}({field})'
    x = sp.symbols('x')

    def in_poly_space(poly):
        # FIX: check
        try: return sp.degree(sp.Poly(poly, x)) <= max_degree
        except Exception: return False

    class poly_space(VectorSpace, name=cls_name):
        set = MathSet(cls_name, object, in_poly_space)
        fn = Fn(field, max_degree + 1)
        def __push__(poly):
            poly = sp.Poly(poly, x)
            coeffs = poly.all_coeffs()[::-1]  # Ascending order
            degree_diff = max_degree - len(coeffs) + 1
            return coeffs + ([0] * degree_diff)
        def __pull__(vec):
            poly = sp.Poly(vec[::-1], x)
            return poly.as_expr()
    return poly_space(name, constraints, basis)


def hom(vs1, vs2):
    """
    pass
    """
    if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
        raise TypeError()
    if vs1.field is not vs2.field:
        raise TypeError()
    
    name = f'hom({vs1}, {vs2})'
    return matrix_space(name, vs1.field, (vs2.dim, vs1.dim))


def is_vectorspace(n, constraints):
    """
    Check whether F^n forms a vector space under the given constraints.

    Parameters
    ----------
    n : int
        The length of the vectors in the vector space.
    constraints : list of str
        The constraints to check.

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


def columnspace(name, matrix, field=R):
    """
    Return the column space, or image, of a matrix.

    Parameters
    ----------
    name : str
        The name of the column space.
    matrix : list of list or sympy.Matrix
        The matrix to take the column space of.
    field : {R, C}
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
    >>> V = columnspace('V', matrix)
    >>> V.basis
    [[1, 0], [0, 1]]
    >>> U = image('U', matrix)
    >>> U.basis
    [[1, 0], [0, 1]]
    """
    constraints = [f'col({matrix})']
    matrix = sp.Matrix(matrix).T
    matrix = u.rref(matrix, remove=True)
    n = matrix.rows
    return fn(name, field, n, constraints, rs_matrix=matrix)


def rowspace(name, matrix, field=R):
    """
    Return the row space of a matrix.

    Parameters
    ----------
    name : str
        The name of the row space.
    matrix : list of list or sympy.Matrix
        The matrix to take the row space of.
    field : {R, C}
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
    >>> V = rowspace('V', matrix)
    >>> V.basis
    [[1, 0], [0, 1]]
    """
    constraints = [f'row({matrix})']
    matrix = u.rref(matrix, remove=True)
    n = matrix.cols
    return fn(name, field, n, constraints, rs_matrix=matrix)


def nullspace(name, matrix, field=R):
    """
    Return the null space, or kernel, of a matrix.

    Parameters
    ----------
    name : str
        The name of the null space.
    matrix : list of list or sympy.Matrix
        The matrix to take the null space of.
    field : {R, C}
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
    >>> V = nullspace('V', matrix)
    >>> V.basis
    []
    >>> U = kernel('U', matrix)
    >>> U.basis
    []
    """
    constraints = [f'null({matrix})']
    matrix = u.rref(matrix, remove=True)
    n = matrix.cols
    return fn(name, field, n, constraints, ns_matrix=matrix)


def left_nullspace(name, matrix, field=R):
    """
    Return the left null space of a matrix.

    Parameters
    ----------
    name : str
        The name of the left null space.
    matrix : list of list or sympy.Matrix
        The matrix to take the left null space of.
    field : {R, C}
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
    >>> V = left_nullspace('V', matrix)
    >>> V.basis
    []
    >>> matrix = sympy.Matrix([[1, 2], [3, 4]])
    >>> U = left_nullspace('U', matrix)
    >>> W = nullspace('W', matrix.T)
    >>> U == W
    True
    """
    matrix = sp.Matrix(matrix).T
    return nullspace(name, matrix, field)


# Aliases
image = columnspace
kernel = nullspace