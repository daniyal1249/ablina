from sympy import Matrix as _M


class Matrix(_M):
    """
    pass
    """

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