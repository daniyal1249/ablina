import sympy as sp
from sympy.solvers.solveset import NonlinearError

from .parser import ConstraintError, split_constraint, sympify
from .utils import rref


# To test associativity of multiplication (2 scalars one vector), define
# operation to be normal mul if both are scalars, and scalar mul otherwise


def to_ns_matrix(n, lin_constraints):
    """
    Return the matrix representation of the given linear constraints.

    Parameters
    ----------
    n : int
        pass
    lin_constraints : list of str
        The list of constraints.

    Returns
    -------
    ns_matrix : sympy.Matrix
        A sympy matrix with the linear constraints as rows.
    """
    exprs = set()
    for constraint in lin_constraints:
        exprs.update(split_constraint(constraint))

    matrix = []
    allowed_vars = sp.symbols(f'v:{n}')
    for expr in exprs:
        row = [0] * n
        try:
            expr = sympify(expr, allowed_vars)
        except Exception as e:
            raise ConstraintError('Invalid constraint format.') from e

        for var in expr.free_symbols:
            var_idx = int(var.name.lstrip('v'))
            var_coeff = expr.coeff(var, 1)
            row[var_idx] = var_coeff
        matrix.append(row)
    
    ns_matrix = rref(matrix, remove=True) if matrix else sp.zeros(0, n)
    return ns_matrix


def to_complement(matrix):
    """
    pass

    Parameters
    ----------
    matrix : sympy.Matrix
        pass

    Returns
    -------
    sympy.Matrix
        pass
    """
    if matrix.rows == 0:
        return sp.eye(matrix.cols)
    
    ns_basis = matrix.nullspace()
    if not ns_basis:
        return sp.zeros(0, matrix.cols)
    return rref([vec.T for vec in ns_basis], remove=True)


# For testing
def standard_isomorphism(field, n, add, mul):
    return lambda x: x, lambda x: x


def map_constraints(mapping, constraints):
    """
    pass

    Parameters
    ----------
    mapping : callable
        pass
    constraints : list of str
        pass

    Returns
    -------
    list of str
        pass
    """
    return constraints