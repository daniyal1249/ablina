from sympy.sets.fancysets import Reals, Complexes


class Field:
    """
    pass
    """
    pass


class Real(Field, Reals):
    """
    pass
    """

    def __repr__(self):
        return 'R'
    
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, other):
        try:
            return super().__contains__(other)
        except Exception:
            return False


class Complex(Field, Complexes):
    """
    pass
    """

    def __repr__(self):
        return 'C'
    
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, other):
        try:
            return super().__contains__(other)
        except Exception:
            return False


R = Real()
C = Complex()