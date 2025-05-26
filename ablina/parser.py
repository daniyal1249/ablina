from sympy import sympify as _sympify


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
        pass
    allowed_vars : iterable, optional
        pass

    Returns
    -------
    Any
        pass

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
            raise ParsingError(f'Unrecognized variables found: {invalid_vars}')
    return expr


def split_constraint(constraint: str):
    """
    Split a constraint with multiple relational operators into separate 
    relations.

    Parameters
    ----------
    constraint : str
        pass

    Returns
    -------
    relations : set of str
        pass
    """
    exprs = constraint.split('==')
    expr_count = len(exprs)
    if expr_count == 1:
        raise ConstraintError('Constraints must include at least one "==".')
    
    eqs = set()
    for i in range(expr_count - 1):
        eq = f'{exprs[i]} - ({exprs[i + 1]})'
        eqs.add(eq)
    return eqs