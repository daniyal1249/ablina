"""
A module for working with sets defined by predicates.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .utils import of_arity


class Set:
    """
    A set defined by a class and predicates.
    
    A set is defined by a class that all elements must be instances of, 
    and a list of predicates that all elements must satisfy.
    """

    def __init__(
        self, 
        name: str, 
        cls: type, 
        *predicates: Callable[[Any], bool]
    ) -> None:
        """
        Initialize a Set instance.

        Parameters
        ----------
        name : str
            The name of the set.
        cls : type
            The class all set elements must be instances of.
        *predicates
            The predicates all set elements must satisfy.

        Returns
        -------
        Set
            A new Set instance.
        """
        if not isinstance(cls, type):
            raise TypeError("cls must be a class.")
        
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        if not all(callable(pred) for pred in predicates):
            raise TypeError("All predicates must be callable.")
        if not all(of_arity(pred, 1) for pred in predicates):
            raise ValueError("All predicates must have arity 1.")
        
        self.name = name
        self._cls = cls
        self._predicates = remove_duplicates(predicates)

    @property
    def cls(self) -> type:
        """
        type: The class that all set elements are instances of.
        """
        return self._cls
    
    @property
    def predicates(self) -> list[Callable[[Any], bool]]:
        """
        list of callable: The list of predicates all set elements must satisfy.
        """
        return self._predicates
    
    def __repr__(self) -> str:
        return (
            f"Set({self.name!r}, "
            f"{self.cls!r}, "
            f"{self.predicates!r})"
            )
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, set2: Any) -> bool:
        if not isinstance(set2, Set):
            return False
        return self.name == set2.name
    
    def __contains__(self, obj: Any) -> bool:
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
    
    def __pos__(self) -> Set:
        """
        Return `self`.
        """
        return self
    
    def __neg__(self) -> Set:
        """
        Same as ``Set.complement``.
        """
        return self.complement()
    
    def __and__(self, set2: Set) -> Set:
        """
        Same as ``Set.intersection``.
        """
        return self.intersection(set2)
    
    def __or__(self, set2: Set) -> Set:
        """
        Same as ``Set.union``.
        """
        return self.union(set2)
    
    def __sub__(self, set2: Set) -> Set:
        """
        Same as ``Set.difference``.
        """
        return self.difference(set2)

    def complement(self) -> Set:
        """
        The complement of a set.

        Returns the set of all objects in the universal set that are not 
        in `self`. The universal set is always `self` without any 
        predicates. In other words, the resulting set contains all 
        instances of ``self.cls`` that are not in `self`.

        Returns
        -------
        Set
            The complement of `self`.

        Examples
        --------

        >>> A = Set("A", list, lambda x: len(x) == 3)
        >>> B = A.complement()
        >>> [1, 2, 3] in A
        True
        >>> [1, 2, 3] in B
        False
        >>> [1, 2] in B
        True
        >>> (1, 2) in B
        False
        >>> None in B
        False
        """
        name = f"{self}^C"
        def complement_pred(obj: Any) -> bool:
            return not all(pred(obj) for pred in self.predicates)
        return Set(name, self.cls, complement_pred)
    
    def intersection(self, set2: Set) -> Set:
        """
        The intersection of two sets.

        Returns the set of all objects contained in both `self` and 
        `set2`. Note that the ``cls`` attribute of both sets must be the 
        same.

        Parameters
        ----------
        set2 : Set
            The set to take the intersection with.

        Returns
        -------
        Set
            The intersection of `self` and `set2`.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> A = Set("A", list, lambda x: len(x) == 3)
        >>> B = Set("B", list, lambda x: 1 in x)
        >>> C = A.intersection(B)
        >>> [2, 3, 4] in C
        False
        >>> [1, 2] in C
        False
        >>> [1, 2, 3] in C
        True
        """
        self._validate_type(set2)
        name = f"{self} ∩ {set2}"
        return Set(name, self.cls, self.predicates + set2.predicates)
    
    def union(self, set2: Set) -> Set:
        """
        The union of two sets.

        Returns the set of all objects contained in either `self` or 
        `set2`. Note that the ``cls`` attribute of both sets must be the 
        same.

        Parameters
        ----------
        set2 : Set
            The set to take the union with.

        Returns
        -------
        Set
            The union of `self` and `set2`.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> A = Set("A", list, lambda x: len(x) == 3)
        >>> B = Set("B", list, lambda x: 1 in x)
        >>> C = A.union(B)
        >>> [2, 3, 4] in C
        True
        >>> [1, 2] in C
        True
        >>> [1, 2, 3] in C
        True
        """
        self._validate_type(set2)
        name = f"{self} ∪ {set2}"
        def union_pred(obj: Any) -> bool:
            return (
                all(pred(obj) for pred in self.predicates) 
                or all(pred(obj) for pred in set2.predicates)
                )
        return Set(name, self.cls, union_pred)
    
    def difference(self, set2: Set) -> Set:
        """
        The difference of two sets.

        Returns the set of all objects in `self` that are not in `set2`.

        Parameters
        ----------
        set2 : Set
            The set to be subtracted from `self`.

        Returns
        -------
        Set
            The set difference `self` - `set2`.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> A = Set("A", list, lambda x: len(x) == 3)
        >>> B = Set("B", list, lambda x: 1 in x)
        >>> C = A.difference(B)
        >>> [2, 3, 4] in C
        True
        >>> [1, 2] in C
        False
        >>> [1, 2, 3] in C
        False
        """
        self._validate_type(set2)
        return self.intersection(set2.complement())
    
    def is_subset(self, set2: Set) -> bool:
        """
        Check whether `set2` is a subset of `self`.

        Note that this method is NOT equivalent to the mathematical notion 
        of subset. Due to programmatic limitations, this method instead 
        checks whether every predicate object in `self` is also in `set2`. 
        If so, it returns True, and `set2` is a subset of `self` in the 
        mathematical sense. Otherwise, it returns False, and nothing can 
        be said about the relationship between `self` and `set2`. The 
        examples below illustrate this.

        Parameters
        ----------
        set2 : Set
            The set to check.

        Returns
        -------
        bool
            True if every predicate in `self` is also in `set2`, 
            otherwise False.

        Raises
        ------
        ValueError
            If ``self.cls`` and ``set2.cls`` are not the same.

        Examples
        --------

        >>> def pred1(x): return len(x) == 3
        >>> def pred2(x): return 1 in x
        >>> A = Set("A", list, pred1)
        >>> B = Set("B", list, pred1, pred2)
        >>> C = Set("C", list, pred1, lambda x: 1 in x)
        >>> A.is_subset(B)
        True
        >>> B.is_subset(A)
        False
        >>> A.is_subset(C)
        True
        >>> C.is_subset(A)
        False
        >>> B.is_subset(C)
        False
        >>> C.is_subset(B)
        False
        """
        self._validate_type(set2)
        return all(pred in set2.predicates for pred in self.predicates)

    def _validate_type(self, set2: Any) -> None:
        if not isinstance(set2, Set):
            raise TypeError(f"Expected a Set, got {type(set2).__name__} instead.")
        if self.cls is not set2.cls:
            raise ValueError("The cls attribute of both sets must be the same.")


def remove_duplicates(seq: Iterable[Any]) -> list[Any]:
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

    >>> x = [1, 2, 2, 3, 4, 3]
    >>> remove_duplicates(x)
    [1, 2, 3, 4]
    """
    elems = set()
    return [x for x in seq if not (x in elems or elems.add(x))]


def negate(pred: Callable[[Any], bool]) -> Callable[[Any], bool]:
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
    >>> pred2 = negate(pred1)
    >>> pred1([1, 2, 3])
    True
    >>> pred2([1, 2, 3])
    False
    """
    def negation(obj: Any) -> bool:
        return not pred(obj)
    negation.__name__ = f"not_{pred.__name__}"
    return negation
