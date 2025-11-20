"""
A module providing utility functions for linear algebra operations.
"""

import inspect

import sympy as sp

from .field import R
from .matrix import M


def symbols(names, field=None, **kwargs):
    """
    Returns sympy symbols with the specified names and field (``R`` or ``C``).

    Additional constraints can be specified as keyword args.
    """
    if field is None:
        return sp.symbols(names, **kwargs)
    if field is R:
        return sp.symbols(names, real=True, **kwargs)
    return sp.symbols(names, complex=True, **kwargs)


def is_linear(expr, vars=None):
    """
    Determines if an equation or expression is linear with respect to the 
    variables in `vars`. 
    
    If `vars` is None, all variables in `expr` are checked.
    """
    if vars is None:
        vars = expr.free_symbols
    if not vars:
        return True
    try:
        terms = sp.Poly(expr, *vars).monoms()
    except sp.PolynomialError:
        return False  # Return False if not a polynomial
    
    for exponents in terms:
        if sum(exponents) not in (0, 1):
            return False
        if not all(i in (0, 1) for i in exponents):
            return False
    return True


def is_empty(matrix):
    """
    Returns True if the matrix contains no elements, otherwise False.
    """
    mat = M(matrix)
    return mat.rows == 0 or mat.cols == 0


def is_invertible(matrix):
    """
    Returns True if the matrix is invertible, otherwise False.
    """
    mat = M(matrix)
    return mat.is_square and not mat.det().equals(0)


def is_orthogonal(matrix):
    """
    Returns True if the matrix is orthogonal, otherwise False.
    """
    # Make sure the matrix is real
    return is_unitary(matrix)


def is_unitary(matrix):
    """
    Returns True if the matrix is unitary, otherwise False.
    """
    mat = M(matrix)
    if not mat.is_square:
        return False
    identity = M.eye(mat.rows)
    return (mat @ mat.H).equals(identity)


def is_normal(matrix):
    """
    Returns True if the matrix is normal, otherwise False.
    """
    mat = M(matrix)
    if not mat.is_square:
        return False
    adjoint = mat.H
    return (mat @ adjoint).equals(adjoint @ mat)


def rref(matrix, remove=False):
    """
    Returns the rref of the matrix.

    If `remove` is True, all zero rows are removed.
    """
    mat = M(matrix)
    rref, _ = mat.rref()
    if not remove:
        return M(rref)
    
    for i in range(rref.rows - 1, -1, -1):
        if any(rref.row(i)):
            break
        rref.row_del(i)
    return M(rref)


def of_arity(func, arity):
    """
    Returns True if the function can accept `arity` positional arguments, 
    otherwise False.

    Raises
    ------
    TypeError
        If `func` is not callable.
    """
    sig = inspect.signature(func)
    
    # Check for required keyword-only parameters
    # If any exist, function cannot be called with only positional args
    for param in sig.parameters.values():
        if (param.kind == inspect.Parameter.KEYWORD_ONLY 
            and param.default == inspect.Parameter.empty):
            return False
    
    has_var_positional = False
    min_positional = 0
    max_positional = 0
    
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif param.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
            max_positional += 1
            if param.default == inspect.Parameter.empty:
                min_positional += 1
    
    if has_var_positional:
        # Function can accept any number >= min_positional
        return arity >= min_positional
    else:
        # Function can accept exactly min_positional to max_positional
        return min_positional <= arity <= max_positional


def add_attributes(cls, *attributes):
    """
    Dynamically create a subclass with additional attributes.

    Creates a new class that inherits from the given class and adds the 
    specified attributes as class attributes.

    Parameters
    ----------
    cls : type
        The base class to subclass.
    *attributes
        The attributes to add to the new class.

    Returns
    -------
    type
        A new subclass with the added attributes.
    """
    attributes = {attr.__name__: attr for attr in attributes}
    return type(f"{cls.__name__}_subclass", (cls,), attributes)