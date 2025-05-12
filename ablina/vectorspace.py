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
            f'Fn(field={self.field}, '
            f'n={self.n}, '
            f'constraints={self.constraints}, '
            f'ns_matrix={self._ns_matrix}, '
            f'rs_matrix={self._rs_matrix})'
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
    
    def __add__(self, vs2):
        return self.sum(vs2)
    
    def __and__(self, vs2):
        return self.intersection(vs2)

    # Methods relating to vectors

    def vector(self, std=1, arbitrary=False):
        size = self.dim
        if arbitrary:
            weights = list(u.symbols(f'c:{size}', field=self.field))
        else:
            weights = [round(gauss(0, std)) for _ in range(size)]
        vec = sp.Matrix([weights]) @ self._rs_matrix
        return vec.flat()  # Return list

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
    
    def are_independent(self, *vectors):
        matrix = sp.Matrix(vectors)
        return matrix.rank() == matrix.rows
    
    def is_basis(self, *vectors):
        return self.are_independent(*vectors) and len(vectors) == self.dim
    
    # Methods relating to vector spaces

    def sum(self, vs2):
        rs_matrix = sp.Matrix.vstack(self._rs_matrix, vs2._rs_matrix)
        rs_matrix = u.rref(rs_matrix, remove=True)
        constraints = self.constraints  # Rework
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
        for i in range(self.dim):
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
                proj = self.mul(factor, q)
                v = self.add(v, self.additive_inv(proj))
            unit_v = self.mul(1 / self.norm(v), v)
            orthonormal_vecs.append(unit_v)
        return orthonormal_vecs
    
    def ortho_complement(self, vs2):
        constraints = [f'ortho_complement({', '.join(self.constraints)})']
        comp = Fn(self.field, self.n, constraints, rs_matrix=vs2._ns_matrix)
        return self.intersection(comp)
    
    def ortho_projection(self, vs2):
        raise NotImplementedError()


class VectorSpace:
    """
    pass
    """

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validate_subclass_contract()
        add, mul, additive_inv = cls._init_operations()
        
        cls.name = cls.__name__
        if name is not None:
            if not isinstance(name, str):
                raise TypeError('Name must be a string.')
            cls.name = name

        cls._add = staticmethod(add)
        cls._mul = staticmethod(mul)
        cls._additive_inv = staticmethod(additive_inv)

    def __init__(self, name, constraints=None, basis=None, *, fn=None):
        """
        pass
        """
        if not isinstance(name, str):
            raise TypeError('Name must be a string.')
        self.name = name
        self.set = MathSet(name, self.set.cls, lambda vec: vec in self)

        if fn is not None:
            self.fn = fn
            return
        self.fn = Fn(self.fn.field, self.fn.n, constraints)

        if basis is not None:
            if not self.are_independent(*basis):
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
        return f'{type(self).__name__}(name="{self.name}", basis={self.basis})'
    
    def __str__(self):
        name = f'{self.name} (Subspace of {type(self).name})'
        lines = [
            name,
            '-' * len(name),
            f'Field      {self.field}',
            f'Identity   {self.additive_id}',
            f'Basis      {self.basis}',
            f'Dimension  {self.dim}',
            f'Vector     {self.vector(arbitrary=True)}'
            ]
        return '\n'.join(lines)

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

        >>> V = fn('V', R, 3)
        >>> V.are_independent([1, 0, 0], [0, 1, 0])
        True
        >>> V.are_independent([1, 2, 3], [2, 4, 6])
        False
        >>> V.are_independent([0, 0, 0])
        False
        >>> V.are_independent()
        True
        """
        if not all(vec in self for vec in vectors):
            raise TypeError('Vectors must be elements of the vector space.')
        fn_vecs = [self.__push__(vec) for vec in vectors]
        return self.fn.are_independent(*fn_vecs)
    
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
        name = f'sum({self.name}, {vs2.name})'
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
        name = f'inter({self.name}, {vs2.name})'
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

        >>> V = fn('V', R, 3)
        >>> U = fn('U', R, 3, constraints=['v0 == v1'])
        >>> W = fn('W', R, 3, constraints=['v1 == v2'])
        >>> U.is_subspace(V)
        True
        >>> W.is_subspace(V)
        True
        >>> U.is_subspace(W)
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
        # if not isinstance(vs2, VectorSpace):
        #     raise TypeError()
        # if not vs2.is_subspace(self):
        #     raise TypeError()
        
        # name = f''

        # def in_quotient_space(coset):
        #     return

        # class quotient_space(VectorSpace, name=name):
        #     set = MathSet(name, AffineSpace, in_quotient_space)
        #     fn = Fn(self.field, None, add=self.fn.add, mul=self.fn.mul)
        #     def __push__(self, coset): return
        #     def __pull__(self, vec): return
        # return quotient_space()
        raise NotImplementedError()

    def _validate_type(self, vs2):
        if not isinstance(vs2, VectorSpace):
            raise TypeError(f'Expected a VectorSpace, got {type(vs2).__name__} instead.')
        if type(self).name != type(vs2).name:
            raise TypeError(f'Vector spaces must be subspaces of the same ambient space.')


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
        if representative not in type(vectorspace)():
            raise TypeError()
        
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
        name = f'{self.representative} + {self.name})'  # FIX: rework
        return MathSet(name, vs.set.cls, lambda point: point in self)
    
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
        vs = self.vectorspace
        if point not in type(vs)():
            return False
        
        vec1 = self.representative
        vec2 = vs.additive_inv(point)
        return vs.add(vec1, vec2) in vs

    def __eq__(self, as2):
        if not isinstance(as2, AffineSpace):
            return False
        return self.representative in as2

    def __add__(self, as2):
        vs = self.vectorspace
        if not isinstance(as2, AffineSpace):
            raise TypeError()
        if vs != as2.vectorspace:
            raise TypeError('Affine spaces must be cosets of the same vector space.')
        
        repr = vs.add(self.representative, as2.representative)
        return AffineSpace(vs, repr)

    def __mul__(self, scalar):
        vs = self.vectorspace
        if scalar not in vs.field:
            raise TypeError('Scalar must be an element of the field.')
        repr = vs.mul(scalar, self.representative)
        return AffineSpace(vs, repr)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


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
    cls_name = f'M({field}, {shape})'

    def in_matrix_space(mat):
        return mat.shape == shape

    class matrix_space(VectorSpace, name=cls_name):
        set = MathSet(cls_name, sp.Matrix, in_matrix_space)
        fn = Fn(field, sp.prod(shape))
        def __push__(mat): return mat.flat()
        def __pull__(vec): return sp.Matrix(*shape, vec)
    return matrix_space(name, constraints, basis)


def poly_space(name, field, max_degree, constraints=None, basis=None):
    """
    pass
    """
    cls_name = f'P({field}, {max_degree})'

    def in_poly_space(poly):
        return sp.degree(poly) <= max_degree

    class poly_space(VectorSpace, name=cls_name):
        set = MathSet(cls_name, sp.Poly, in_poly_space)
        fn = Fn(field, max_degree + 1)
        def __push__(poly):
            coeffs = poly.all_coeffs()[::-1]  # Ascending order
            degree_diff = max_degree - len(coeffs) + 1
            return coeffs + ([0] * degree_diff)
        def __pull__(vec):
            x = sp.symbols('x')
            return sp.Poly.from_list(vec[::-1], x)
    return poly_space(name, constraints, basis)


def hom(vs1, vs2):
    """
    pass
    """
    if not (isinstance(vs1, VectorSpace) and isinstance(vs2, VectorSpace)):
        raise TypeError()
    if vs1.field is not vs2.field:
        raise TypeError()
    name = f'hom({vs1.name}, {vs2.name})'
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