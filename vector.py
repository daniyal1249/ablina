from numbers import Real, Complex
import numpy as np

class Vector:
    __array_priority__ = 1000

    def __new__(cls, field, vals):
        for val in vals:
            if not isinstance(val, field):
                raise TypeError(f'Expected {field.__name__} values, got {type(val).__name__} instead.')
        return super().__new__(cls)

    def __init__(self, field, vals):
        self.__field = field
        self.__vec = np.array(vals)

    @property
    def field(self):
        return self.__field

    def __repr__(self):
        return type(self).__name__ + str(tuple(self.__vec))
    
    def __getitem__(self, idx):
        return self.__vec[idx]
    
    # Unary Operators
    def __neg__(self):
        return type(self)(*(-1 * self.__vec))
    
    def __pos__(self):
        return self
    
    def __len__(self):
        return len(self.__vec)
    
    # Binary Operators
    def __add__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot add type {type(vec2).__name__} to {self_type.__name__}.')
        return self_type(*(self.__vec + vec2.__vec))
    
    def __radd__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot add type {type(vec2).__name__} to {self_type.__name__}.')
        return self_type(*(vec2.__vec + self.__vec))
    
    def __sub__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot subtract type {type(vec2).__name__} from {self_type.__name__}.')
        return self_type(*(self.__vec - vec2.__vec))
    
    def __rsub__(self, vec2):
        self_type = type(self)
        if not isinstance(vec2, self_type):
            raise TypeError(f'Cannot subtract {self_type.__name__} from type {type(vec2).__name__}.')
        return self_type(*(vec2.__vec - self.__vec))
    
    def __mul__(self, const):
        self_type = type(self)
        if not isinstance(const, self.field):
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(self.__vec * const))
    
    def __rmul__(self, const):
        self_type = type(self)
        if not isinstance(const, self.field):
            raise TypeError(f'Cannot multiply {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(const * self.__vec))
    
    def __truediv__(self, const):
        self_type = type(self)
        if not isinstance(const, self.field):
            raise TypeError(f'Cannot divide {self_type.__name__} with type {type(const).__name__}.')
        return self_type(*(self.__vec * (1 / const)))
    
    def __matmul__(self, matrix):
        self_type = type(self)
        result = self.__vec @ matrix
        if isinstance(result, (np.ndarray, self_type)):  # handles vector-vector multiplication
            return self_type(*result)
        else:
            return self_type(result)  # "result" is a numpy scalar
    
    def __rmatmul__(self, matrix):
        self_type = type(self)
        result = matrix @ self.__vec
        if isinstance(result, np.ndarray):
            return self_type(*result)
        else:
            return self_type(result)


class R(Vector):
    def __new__(cls, *vals):
        return super().__new__(cls, Real, vals)

    def __init__(self, *vals):
        super().__init__(Real, vals)

class C(Vector):
    def __new__(cls, *vals):
        return super().__new__(cls, Complex, vals)

    def __init__(self, *vals):
        super().__init__(Complex, vals)