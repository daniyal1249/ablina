from alapy.utils import of_arity


class MathematicalSet:
    """
    pass
    """

    def __init__(self, cls, *predicates, name=None):
        """
        pass

        Parameters
        ----------
        cls : type
            The class that all set elements will be an instance of.
        *predicates
            pass
        name : str, optional
            pass

        Returns
        -------
        MathematicalSet
            pass
        """
        if not isinstance(cls, type):
            raise TypeError()
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        if not all(of_arity(pred, 1) for pred in predicates):  # Make sure pred type is valid
            raise ValueError()
        
        self._cls = cls
        self._predicates = remove_duplicates(predicates)
        if name is not None:
            self.__name__ = name

    @property
    def cls(self):
        """
        type: The class that all set elements are instances of.
        """
        return self._cls
    
    @property
    def predicates(self):
        """
        list of callable: The list of predicates all set elements must satisfy.
        """
        return self._predicates
    
    def __repr__(self):
        return (f'Set({self.cls.__name__}, '
                f'{[pred.__name__ for pred in self.predicates]})')
    
    def __eq__(self, set2):
        if not isinstance(set2, Set):
            return False
        if hasattr(self, '__name__') and hasattr(set2, '__name__'):
            return self.__name__ == set2.__name__
        # Order of the predicates matters
        return self.cls is set2.cls and self.predicates == set2.predicates
    
    def __contains__(self, obj):
        if not isinstance(obj, self.cls):
            return False
        return all(pred(obj) for pred in self.predicates)
    
    def __neg__(self):
        return self.complement()
    
    def __pos__(self):
        return self
    
    def __and__(self, set2):
        return self.intersection(set2)
    
    def __or__(self, set2):
        return self.union(set2)
    
    def __sub__(self, set2):
        return self.difference(set2)

    def complement(self):
        """
        pass

        Returns
        -------
        MathematicalSet
            The complement of the set.
        """
        def complement_pred(obj):
            return all(pred(obj) for pred in self.predicates)
        return Set(self.cls, complement_pred)
    
    def intersection(self, set2):
        """
        pass

        Parameters
        ----------
        set2 : MathematicalSet
            The set to take the intersection with.

        Returns
        -------
        MathematicalSet
            The intersection of `self` and `set2`.
        """
        self._validate(set2)
        return Set(self.cls, self.predicates + set2.predicates)

    def union(self, set2):
        """
        pass

        Parameters
        ----------
        set2 : MathematicalSet
            The set to take the union with.

        Returns
        -------
        MathematicalSet
            The union of `self` and `set2`.
        """
        self._validate(set2)
        def union_pred(obj):
            return (all(pred(obj) for pred in self.predicates) 
                    or all(pred(obj) for pred in set2.predicates))
        return Set(self.cls, union_pred)

    def difference(self, set2):
        """
        pass

        Parameters
        ----------
        set2 : MathematicalSet
            The set that will be subtracted from `self`.

        Returns
        -------
        MathematicalSet
            The set difference `self` - `set2`.
        """
        return self.intersection(set2.complement())
    
    def is_subset(self, set2):
        """
        pass

        Parameters
        ----------
        set2 : MathematicalSet
            pass

        Returns
        -------
        bool
            True if all the predicates in `set2` are in `self`, otherwise 
            False.
        """
        self._validate(set2)
        return all(pred in self.predicates for pred in set2.predicates)
    
    def add_predicates(self, *predicates):
        """
        Add predicates to the set.

        Parameters
        ----------
        *predicates
            The predicates to be added.

        Returns
        -------
        MathematicalSet
            pass
        """
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        return Set(self.cls, *self.predicates, *predicates)

    def _validate(self, set2):
        if not isinstance(set2, Set):
            raise TypeError(f'Expected a MathematicalSet, got '
                            f'{type(set2).__name__} instead.')
        if self.cls is not set2.cls:
            raise ValueError('The cls attribute of both sets must be the same.')


def remove_duplicates(seq):
    """
    pass

    Parameters
    ----------
    seq : iterable
        pass

    Returns
    -------
    list
        The list containing the items in `seq` in order with duplicates 
        removed.
    """
    elems = set()
    return [x for x in seq if not (x in elems or elems.add(x))]


def negate(pred):
    """
    pass

    Parameters
    ----------
    pred : callable
        The predicate to negate.

    Returns
    -------
    callable:
        The negation of `pred`.
    """
    def negation(obj): not pred(obj)
    negation.__name__ = f'not_{pred.__name__}'
    return negation


# Set alias
Set = MathematicalSet