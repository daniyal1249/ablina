from sympy import Matrix as _M


class Matrix(_M):
    """
    pass
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        return super().__new__(cls, *args, **kwargs)

    def __class_getitem__(cls, mat):
        if isinstance(mat, tuple):
            return cls(mat)
        return cls([mat])

    def __repr__(self):
        if self.cols == 1:
            return str(self.flat())
        return str(self.tolist())

    def __str__(self):
        return self.__repr__()


M = Matrix