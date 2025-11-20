"""
A module for working with vector space operations.
"""

from .parser import sympify
from .utils import of_arity, symbols


class OperationError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)


class Operation:
    """
    A wrapper for a function with a specified arity.
    
    Encapsulates a callable function along with its arity (number of 
    arguments), providing a consistent interface for operations.
    """
    def __init__(self, func, arity):
        if not of_arity(func, arity):
            raise OperationError()
        self._func = func
        self._arity = arity

    @property
    def func(self):
        return self._func
    
    @property
    def arity(self):
        return self._arity
    
    def __call__(self, *args):
        return self.func(*args)


class VectorAdd(Operation):
    """
    A vector addition operation on F^n.
    
    Represents a binary addition operation on vectors of length n over a 
    given field. Provides equality checking by comparing the operation 
    symbolically.
    """
    def __init__(self, field, n, func):
        super().__init__(func, 2)
        self._field = field
        self._n = n

    @property
    def field(self):
        return self._field

    @property
    def n(self):
        return self._n
    
    def __eq__(self, add2):
        if add2 is self:
            return True
        if not isinstance(add2, VectorAdd):
            return False
        if self.field is not add2.field or self.n != add2.n:
            return False
        # Initialize two arbitrary vectors (u and v)
        u, v = symbols((f"u:{self.n}", f"v:{self.n}"), field=self.field)
        u, v = list(u), list(v)
        try:
            for lhs, rhs in zip(self.func(u, v), add2.func(u, v)):
                if not sympify(lhs).equals(sympify(rhs)):
                    return False
            return True
        except Exception:
            return None


class ScalarMul(Operation):
    """
    A scalar multiplication operation on F^n.
    
    Represents a scalar multiplication operation that takes a scalar and 
    a vector of length n over a given field. Provides equality checking 
    by comparing the operation symbolically.
    """
    def __init__(self, field, n, func):
        super().__init__(func, 2)
        self._field = field
        self._n = n

    @property
    def field(self):
        return self._field

    @property
    def n(self):
        return self._n
    
    def __eq__(self, mul2):
        if mul2 is self:
            return True
        if not isinstance(mul2, ScalarMul):
            return False
        if self.field is not mul2.field or self.n != mul2.n:
            return False
        # Initialize an arbitrary vector (v) and scalar (c)
        v, c = symbols((f"v:{self.n}", "c"), field=self.field)
        v = list(v)
        try:
            for lhs, rhs in zip(self.func(c, v), mul2.func(c, v)):
                if not sympify(lhs).equals(sympify(rhs)):
                    return False
            return True
        except Exception:
            return None