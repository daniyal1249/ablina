from .matrix import M
from .parser import ConstraintError, split_constraint, sympify
from .utils import rref, symbols


def to_ns_matrix(n, constraints):
    """
    Return the matrix representation of the given linear constraints.

    Parameters
    ----------
    n : int
        pass
    constraints : list of str
        The list of constraints.

    Returns
    -------
    Matrix
        A matrix with the linear constraints as rows.
    """
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))

    mat = []
    allowed_vars = symbols(f'v:{n}')
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
        mat.append(row)
    
    return rref(mat, remove=True) if mat else M.zeros(0, n)


def to_complement(matrix):
    """
    pass

    Parameters
    ----------
    matrix : Matrix
        pass

    Returns
    -------
    Matrix
        pass
    """
    mat = M(matrix)
    if mat.rows == 0:
        return M.eye(mat.cols)
    
    basis = mat.nullspace()
    if not basis:
        return M.zeros(0, mat.cols)
    comp = M.hstack(*basis).T
    return rref(comp, remove=True)