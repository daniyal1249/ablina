"""
A module for working with fields.
"""

from __future__ import annotations

from typing import Any

from sympy.sets.fancysets import Reals as _R, Complexes as _C, Rationals as _Q


class Field:
    """
    Base class for fields.
    """
    pass


class Reals(Field, _R):
    """
    The field of real numbers.
    """

    def __repr__(self) -> str:
        return "R"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __contains__(self, other: Any) -> bool:
        try:
            return super().__contains__(other)
        except Exception:
            return False


class Complexes(Field, _C):
    """
    The field of complex numbers.
    """

    def __repr__(self) -> str:
        return "C"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __contains__(self, other: Any) -> bool:
        try:
            return super().__contains__(other)
        except Exception:
            return False


class Rationals(Field, _Q):
    """
    The field of rational numbers.
    """

    def __repr__(self) -> str:
        return "Q"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __contains__(self, other: Any) -> bool:
        try:
            return super().__contains__(other)
        except Exception:
            return False


R: Reals = Reals()
"""Singleton instance representing the field of real numbers."""

C: Complexes = Complexes()
"""Singleton instance representing the field of complex numbers."""

Q: Rationals = Rationals()
"""Singleton instance representing the field of rational numbers."""