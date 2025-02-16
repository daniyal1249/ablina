from .utils import of_arity


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
            The class all set elements must be instances of.
        *predicates
            The predicates all set elements must satisfy.
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
        return (
            f'Set({self.cls.__name__}, '
            f'{[pred.__name__ for pred in self.predicates]})'
            )
    
    def __eq__(self, set2):
        if not isinstance(set2, Set):
            return False
        if hasattr(self, '__name__') and hasattr(set2, '__name__'):
            return self.__name__ == set2.__name__
        # Order of the predicates matters
        return self.cls is set2.cls and self.predicates == set2.predicates
    
    def __contains__(self, obj):
        """
        Check whether an object is an element of the set.

        Parameters
        ----------
        obj : object
            The object to check.

        Returns
        -------
        bool
            True if `obj` is an element of `self`, otherwise False.
        """
        if not isinstance(obj, self.cls):
            return False
        return all(pred(obj) for pred in self.predicates)
    
    def __pos__(self):
        """
        Return `self`.
        """
        return self
    
    def __neg__(self):
        """
        Same as ``Set.complement``.
        """
        return self.complement()
    
    def __and__(self, set2):
        """
        Same as ``Set.intersection``.
        """
        return self.intersection(set2)
    
    def __or__(self, set2):
        """
        Same as ``Set.union``.
        """
        return self.union(set2)
    
    def __sub__(self, set2):
        """
        Same as ``Set.difference``.
        """
        return self.difference(set2)

    def complement(self):
        """
        The complement of a set.

        Returns the set of all objects in the universal set that are not 
        in `self`. The universal set is always ``Set(self.cls)``. In other 
        words, the resulting set contains all instances of ``self.cls`` 
        that are not in `self`.

        Returns
        -------
        MathematicalSet
            The complement of `self`.

        Examples
        --------

        >>> set1 = Set(list, lambda x: len(x) == 3)
        >>> set2 = set1.complement()
        >>> [1, 2, 3] in set1
        True
        >>> [1, 2, 3] in set2
        False
        >>> [1, 2] in set2
        True
        >>> (1, 2) in set2
        False
        >>> None in set2
        False
        """
        def complement_pred(obj):
            return not all(pred(obj) for pred in self.predicates)
        return Set(self.cls, complement_pred)
    
    def intersection(self, set2):
        """
        The intersection of two sets.

        Returns the set of all objects contained in both `self` and 
        `set2`. Note that the ``cls`` attribute of both sets must be the 
        same.

        Parameters
        ----------
        set2 : MathematicalSet
            The set to take the intersection with.

        Returns
        -------
        MathematicalSet
            The intersection of `self` and `set2`.

        Raises
        ------
        TypeError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> set1 = Set(list, lambda x: len(x) == 3)
        >>> set2 = Set(list, lambda x: 1 in x)
        >>> set3 = set1.intersection(set2)
        >>> [2, 3, 4] in set3
        False
        >>> [1, 2] in set3
        False
        >>> [1, 2, 3] in set3
        True
        """
        self._validate(set2)
        return Set(self.cls, self.predicates + set2.predicates)

    def union(self, set2):
        """
        The union of two sets.

        Parameters
        ----------
        set2 : MathematicalSet
            The set to take the union with.

        Returns
        -------
        MathematicalSet
            The union of `self` and `set2`.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> set1 = Set(list, lambda x: len(x) == 3)
        >>> set2 = Set(list, lambda x: 1 in x)
        >>> set3 = set1.union(set2)
        >>> [2, 3, 4] in set3
        True
        >>> [1, 2] in set3
        True
        >>> [1, 2, 3] in set3
        True
        """
        self._validate(set2)
        def union_pred(obj):
            return (
                all(pred(obj) for pred in self.predicates) 
                or all(pred(obj) for pred in set2.predicates)
                )
        return Set(self.cls, union_pred)

    def difference(self, set2):
        """
        The difference of two sets.

        Returns the set of all objects in `self` that are not in `set2`.

        Parameters
        ----------
        set2 : MathematicalSet
            The set that will be subtracted from `self`.

        Returns
        -------
        MathematicalSet
            The set difference `self` - `set2`.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> set1 = Set(list, lambda x: len(x) == 3)
        >>> set2 = Set(list, lambda x: 1 in x)
        >>> set3 = set1.difference(set2)
        >>> [2, 3, 4] in set3
        True
        >>> [1, 2] in set3
        False
        >>> [1, 2, 3] in set3
        False
        """
        return self.intersection(set2.complement())
    
    def is_subset(self, set2):
        """
        Check whether `self` is a subset of `set2`.

        Note that this function

        Parameters
        ----------
        set2 : MathematicalSet
            pass

        Returns
        -------
        bool
            True if every predicate in `set2` is also in `self`, 
            otherwise False.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------
        >>> def pred1(x): return len(x) == 3
        >>> def pred2(x): return 1 in x
        >>> set1 = Set(list, pred1)
        >>> set2 = Set(list, pred1, pred2)
        >>> set3 = Set(list, pred1, lambda x: 1 in x)
        >>> set2.is_subset(set1)
        True
        >>> set1.is_subset(set2)
        False
        >>> set3.is_subset(set1)
        True
        >>> set1.is_subset(set3)
        False
        >>> set3.is_subset(set2)
        False
        >>> set2.is_subset(set3)
        False
        """
        self._validate(set2)
        return all(pred in self.predicates for pred in set2.predicates)
    
    def add_predicates(self, *predicates):
        """
        Add predicates to a set.

        Parameters
        ----------
        *predicates
            The predicates to be added.

        Returns
        -------
        MathematicalSet
            `self` with the additional predicates.

        Examples
        --------

        >>> set1 = Set(int, lambda x: x > 0)
        >>> set2 = set1.add_predicates(lambda x: x > 2)
        >>> 1 in set1
        True
        >>> 1 in set2
        False
        >>> 3 in set2
        True
        """
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        return Set(self.cls, *self.predicates, *predicates)

    def _validate(self, set2):
        if not isinstance(set2, Set):
            raise TypeError(
                f'Expected a MathematicalSet, got {type(set2).__name__} instead.'
                )
        if self.cls is not set2.cls:
            raise ValueError('The cls attribute of both sets must be the same.')


def remove_duplicates(seq):
    """
    Remove duplicate elements in an iterable while preserving the order.

    Parameters
    ----------
    seq : iterable
        The iterable to remove duplicates from.

    Returns
    -------
    list
        `seq` with duplicates removed.

    Examples
    --------

    >>> lst = [1, 2, 2, 3, 4, 3]
    >>> remove_duplicates(lst)
    [1, 2, 3, 4]
    """
    elems = set()
    return [x for x in seq if not (x in elems or elems.add(x))]


def negate(pred):
    """
    The negation of a predicate.

    Parameters
    ----------
    pred : callable
        The predicate to negate.

    Returns
    -------
    callable:
        The negation of `pred`.

    Examples
    --------

    >>> def pred1(x): return len(x) == 3
    >>> pred2 = negate(pred)
    >>> pred1([1, 2, 3])
    True
    >>> pred2([1, 2, 3])
    False
    """
    def negation(obj): return not pred(obj)
    negation.__name__ = f'not_{pred.__name__}'
    return negation


# Set alias
Set = MathematicalSet