import sympy as sp

from .utils import of_arity, symbols


class OperationError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class Operation:
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
        # Initialize two arbitrary vectors (u and v)
        u, v = symbols((f'u:{self.n}', f'v:{self.n}'), field=self.field)
        u, v = list(u), list(v)
        try:
            for lhs, rhs in zip(self.func(u, v), add2.func(u, v)):
                if not sp.sympify(lhs).equals(sp.sympify(rhs)):
                    return False
            return True
        except Exception:
            return None


class ScalarMul(Operation):
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
        # Initialize an arbitrary vector (v) and scalar (c)
        v, c = symbols((f'v:{self.n}', 'c'), field=self.field)
        v = list(v)
        try:
            for lhs, rhs in zip(self.func(c, v), mul2.func(c, v)):
                if not sp.sympify(lhs).equals(sp.sympify(rhs)):
                    return False
            return True
        except Exception:
            return None