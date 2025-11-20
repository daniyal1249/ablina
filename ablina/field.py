"""
A module for working with fields.
"""

from sympy.sets.fancysets import Reals as _R, Complexes as _C


class Field:
    """
    Base class for fields.
    """
    pass


class Reals(Field, _R):
    """
    The field of real numbers.
    """

    def __repr__(self):
        return "R"
    
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, other):
        try:
            return super().__contains__(other)
        except Exception:
            return False


class Complexes(Field, _C):
    """
    The field of complex numbers.
    """

    def __repr__(self):
        return "C"
    
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, other):
        try:
            return super().__contains__(other)
        except Exception:
            return False


R = Reals()
C = Complexes()