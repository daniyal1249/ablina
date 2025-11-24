"""
A module providing utility functions for vector space operations.
"""

from __future__ import annotations

from typing import Any

from .matrix import Matrix, M
from .parser import ConstraintError, split_constraint, sympify
from .utils import rref, symbols


def to_ns_matrix(n: int, constraints: list[str]) -> Matrix:
    """
    Return the matrix representation of the given linear constraints.

    Parameters
    ----------
    n : int
        The dimension of the vector space (length of vectors).
    constraints : list of str
        The list of linear constraints.

    Returns
    -------
    Matrix
        A matrix with the linear constraints as rows.

    Raises
    ------
    ConstraintError
        If any constraint has an invalid format.
    """
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))

    mat = []
    allowed_vars = symbols(f"v:{n}")
    for expr in exprs:
        row = [0] * n
        try:
            expr = sympify(expr, allowed_vars)
        except Exception as e:
            raise ConstraintError("Invalid constraint format.") from e

        for var in expr.free_symbols:
            var_idx = int(var.name.lstrip("v"))
            var_coeff = expr.coeff(var, 1)
            row[var_idx] = var_coeff
        mat.append(row)
    
    return rref(mat, remove=True) if mat else M.zeros(0, n)


def to_complement(matrix: Any) -> Matrix:
    """
    Return the complement of a matrix.

    This function works bidirectionally: if given a null space matrix, it 
    returns a row space matrix, and if given a row space matrix, it 
    returns a null space matrix.

    Parameters
    ----------
    matrix : Matrix
        The matrix to take the complement of.

    Returns
    -------
    Matrix
        The complement of `matrix`.
    """
    mat = M(matrix)
    if mat.rows == 0:
        return M.eye(mat.cols)
    
    basis = mat.nullspace()
    if not basis:
        return M.zeros(0, mat.cols)
    comp = M.hstack(*basis).T
    return rref(comp, remove=True)