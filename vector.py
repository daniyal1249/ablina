from numbers import Real, Complex
from sympy.matrices import Matrix, MatrixKind

class Vector:
    def __init__(self, field, vals):
        self._field = field
        self._vec = Matrix(vals)

    @property
    def field(self):
        return self._field

    def __repr__(self):
        return f'{type(self).__name__}{tuple(self._vec)}'
    
    def __iter__(self):
        return iter(self._vec)

    def __getitem__(self, idx):
        return self._vec[idx]
    
    # Unary Operators
    def __neg__(self):
        return type(self)(*(-1 * self._vec))
    
    def __pos__(self):
        return self

    def __len__(self):
        return len(self._vec)
    
    def __abs__(self):
        return self._vec.norm()
    
    def __hash__(self):
        return hash(tuple(self._vec))

    # Binary Operators
    def __eq__(self, vec2):
        if type(self) != type(vec2):
            return False
        return self._vec == vec2._vec
    
    def __contains__(self, val):
        return val in self._vec

    def __add__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot add type {type(vec2).__name__} to {self_type.__name__}.')
        return self_type(*(self._vec + vec2._vec))
    
    def __radd__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot add type {type(vec2).__name__} to {self_type.__name__}.')
        return self_type(*(vec2._vec + self._vec))
    
    def __sub__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot subtract type {type(vec2).__name__} from {self_type.__name__}.')
        return self_type(*(self._vec - vec2._vec))
    
    def __rsub__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot subtract {self_type.__name__} from type {type(vec2).__name__}.')
        return self_type(*(vec2._vec - self._vec))
    
    def __mul__(self, const):
        self_type = type(self)
        if self.in_field(const) is False:
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(self._vec * const))
    
    def __rmul__(self, const):
        self_type = type(self)
        if self.in_field(const) is False:
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(const * self._vec))
    
    def __truediv__(self, const):
        self_type = type(self)
        if self.in_field(const) is False:
            raise TypeError(f'Cannot divide {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(self._vec * (1 / const)))
    
    def __matmul__(self, matrix):
        self_type = type(self)
        if not hasattr(matrix, 'kind') or not isinstance(matrix.kind, MatrixKind):
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(matrix).__name__}.')
        return self_type(*(self._vec.T @ matrix))  # transpose vector
    
    def __rmatmul__(self, matrix):
        self_type = type(self)
        if not hasattr(matrix, 'kind') or not isinstance(matrix.kind, MatrixKind):
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(matrix).__name__}.')
        return self_type(*(matrix @ self._vec))
    
    def in_field(self, const):
        return is_real(const) if self.field == Real else is_complex(const)
    
    def normalize(self):
        return self / abs(self)

    def distance(self, vec2):
        pass
    
    def angle(self, vec2):
        pass


class R(Vector):
    def __new__(cls, *vals):
        if any(is_real(val) is False for val in vals):
            raise TypeError(f'One or more values are not valid. Ensure all inputs are either real or symbolic.')
        return super().__new__(cls)

    def __init__(self, *vals):
        super().__init__(Real, vals)

class C(Vector):
    def __new__(cls, *vals):
        if any(is_complex(val) is False for val in vals):
            raise TypeError(f'One or more values are not valid. Ensure all inputs are either complex or symbolic.')
        return super().__new__(cls)

    def __init__(self, *vals):
        super().__init__(Complex, vals)

def is_real(expr):
    '''
    Returns True if the expression is Real, None if its indeterminate, and 
    False otherwise.
    '''
    if hasattr(expr, 'is_real'):
        return expr.is_real  # sympy's is_real attribute
    return isinstance(expr, Real)

def is_complex(expr):
    '''
    Returns True if the expression is Complex, None if its indeterminate, and 
    False otherwise.
    '''
    if hasattr(expr, 'is_complex'):
        return expr.is_complex  # sympy's is_complex attribute
    return isinstance(expr, Complex)