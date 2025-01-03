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



class VectorAdd(Operation):
    def __init__(self, func, n):
        super().__init__(func, 2)
        self._n = n

    @property
    def n(self):
        return self._n
    
    def __eq__(self, add2):
        pass


class ScalarMul(Operation):
    def __init__(self, func, n):
        super().__init__(func, 2)
        self._n = n

    @property
    def n(self):
        return self._n
    
    def __eq__(self, mul2):
        pass

