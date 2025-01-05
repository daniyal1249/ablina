from numbers import Real

import sympy as sp
from utils import of_arity

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
        try:
            # Initialize two arbitrary vectors (xs and ys)
            if self.field is Real:
                xs, ys = sp.symbols((f'x:{self.n}', f'y:{self.n}'), real=True)
            else:
                xs, ys = sp.symbols((f'x:{self.n}', f'y:{self.n}'))

            xs, ys = list(xs), list(ys)
            for lhs, rhs in zip(self.func(xs, ys), add2.func(xs, ys)):
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
        try:
            # Initialize an arbitrary vector (xs) and scalar (c)
            if self.field is Real:
                xs, c = sp.symbols((f'x:{self.n}', 'c'), real=True)
            else:
                xs, c = sp.symbols((f'x:{self.n}', 'c'))

            xs = list(xs)
            for lhs, rhs in zip(self.func(c, xs), mul2.func(c, xs)):
                if not sp.sympify(lhs).equals(sp.sympify(rhs)):
                    return False
            return True
        except Exception:
            return None

