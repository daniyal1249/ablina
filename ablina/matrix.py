"""
A module for working with matrices.
"""

from __future__ import annotations

from typing import Any

from sympy import Matrix as _M


class Matrix(_M):
    """
    A matrix class extending sympy's Matrix.
    
    Provides additional functionality and a custom representation for 
    matrices used in the ablina package.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Matrix:
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        return super().__new__(cls, *args, **kwargs)

    def __class_getitem__(cls, mat: Any) -> Matrix:
        if isinstance(mat, tuple):
            return cls(mat)
        return cls([mat])

    def __repr__(self) -> str:
        if self.cols == 1:
            return str(self.flat())
        return str(self.tolist())

    def __str__(self) -> str:
        return self.__repr__()


M: type[Matrix] = Matrix
"""An alias for the Matrix class."""