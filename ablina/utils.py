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
    return matrix.rows == 0 or matrix.cols == 0


def is_invertible(matrix):
    """
    Returns True if the matrix is invertible, otherwise False.
    """
    return matrix.is_square and not matrix.det().equals(0)


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
    if not matrix.is_square:
        return False
    identity = M.eye(matrix.rows)
    return (matrix.H @ matrix).equals(identity)


def is_normal(matrix):
    """
    Returns True if the matrix is normal, otherwise False.
    """
    if not matrix.is_square:
        return False
    adjoint = matrix.H
    return (matrix @ adjoint).equals(adjoint @ matrix)


def rref(matrix, remove=False):
    """
    Returns the rref of the matrix.

    If `remove` is True, all zero rows are removed.
    """
    rref, _ = matrix.rref()
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
    if len(sig.parameters) < arity:
        return False
    
    count_req_pos = 0  # Number of required positional args
    for param in sig.parameters.values():
        # Return False if there are required keyword-only args
        if (param.kind == inspect.Parameter.KEYWORD_ONLY 
            and param.default == inspect.Parameter.empty):
            return False
        if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, 
            inspect.Parameter.POSITIONAL_OR_KEYWORD) 
            and param.default == inspect.Parameter.empty):
            count_req_pos += 1

        if count_req_pos > arity:
            return False
    return True


def add_attributes(cls, *attributes):
    attributes = {attr.__name__: attr for attr in attributes}
    return type(f'{cls.__name__}_subclass', (cls,), attributes)