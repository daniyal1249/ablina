import re

import sympy as sp


class ParsingError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class ConstraintError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


def sympify(expr, allowed_vars=None):
    """
    Return the sympy representation of the given expression.

    Parameters
    ----------
    expr : str
        x
    allowed_vars : iterable, optional
        x

    Returns
    -------
    sympy.Basic
        x

    Raises
    ------
    sympy.SympifyError
        x
    ParsingError
        If `expr` contains variables not in `allowed_vars`.
    """
    # Filter unrecognized characters for safety (consider regex)
    expr = sp.sympify(expr, rational=True, evaluate=False)
    if allowed_vars is not None:
        if not all(var in allowed_vars for var in expr.free_symbols):
            invalid_vars = expr.free_symbols - set(allowed_vars)
            raise ParsingError(f'Unrecognized variables found: {invalid_vars}')
    return expr


def split_constraint(constraint):
    """
    Split a constraint with multiple relational operators into separate 
    relations.

    Parameters
    ----------
    constraint : str
        x

    Returns
    -------
    relations : set
        x
    """
    operators = r'(==|!=|>=|<=|>|<)'
    exprs = re.split(operators, constraint)
    expr_count = len(exprs)
    if expr_count == 1:
        raise ConstraintError('Constraints must include at least one '
                              'relational operator from: ==, !=, >=, <=, >, <')
    
    relations = set()
    for i in range(1, expr_count - 1, 2):
        left_operand = exprs[i - 1].strip()
        operator = exprs[i].strip()
        right_operand = exprs[i + 1].strip()

        if not (left_operand and right_operand):
            raise ConstraintError(f'Missing operand for {operator}.')
        relations.add(f'{left_operand} {operator} {right_operand}')
    return relations