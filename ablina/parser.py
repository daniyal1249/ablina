"""
A module for parsing and processing mathematical expressions and constraints.
"""

from __future__ import annotations

from typing import Any, Iterable

from sympy import sympify as _sympify


class ParsingError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


class ConstraintError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


def sympify(expr: str, allowed_vars: Iterable[Any] | None = None) -> Any:
    """
    Return the sympy representation of the given expression.

    Parameters
    ----------
    expr : str
        The expression to convert to a sympy representation.
    allowed_vars : iterable, optional
        The set of allowed variables in the expression.

    Returns
    -------
    Any
        The sympy representation of `expr`.

    Raises
    ------
    ParsingError
        If `expr` contains variables not in `allowed_vars`.
    """
    # Filter unrecognized characters for safety (consider regex)
    expr = _sympify(expr, rational=True)  # evaluate flag
    if allowed_vars is not None:
        if not all(var in allowed_vars for var in expr.free_symbols):
            invalid_vars = expr.free_symbols - set(allowed_vars)
            raise ParsingError(f"Unrecognized variables found: {invalid_vars}")
    return expr


def split_constraint(constraint: str) -> set[str]:
    """
    Split a constraint with multiple relational operators into separate 
    relations.

    Parameters
    ----------
    constraint : str
        The constraint string to split, containing one or more "==" operators.

    Returns
    -------
    relations : set of str
        A set of relation strings, each representing one equality from 
        the constraint.
    """
    exprs = constraint.split("==")
    expr_count = len(exprs)
    if expr_count == 1:
        raise ConstraintError('Constraints must include at least one "==".')
    
    eqs = set()
    for i in range(expr_count - 1):
        eq = f"{exprs[i]} - ({exprs[i + 1]})"
        eqs.add(eq)
    return eqs