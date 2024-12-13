from sympy import *

class ConstraintError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

def split_constraint(constraint):
    '''
    Splits a constraint with multiple equal signs into separate equations. 
    Returns a set of expressions (str) of the form LHS - RHS.
    '''
    exprs = constraint.split('=')
    expr_count = len(exprs)
    if expr_count == 1:
        raise ConstraintError('Constraints must include an equal sign.')
    
    eqs = set()
    for i in range(expr_count - 1):
        eq = f'{exprs[i]} - ({exprs[i + 1]})'
        eqs.add(eq)  # maybe .strip()
    return eqs

def is_linear(expr, vars=None):
    '''
    Determines if an equation or expression is linear.
    '''
    if vars is None:
        vars = expr.free_symbols

    # Convert equation into an expression
    if isinstance(expr, Eq):
        lhs, rhs = expr.lhs, expr.rhs
        expr = lhs - rhs
    try:
        return all(degree(expr, var) == 1 for var in vars)
    except PolynomialError:
        # Return false if not a polynomial
        return False

def parse_expression(n, expr):
    '''
    Checks the syntax/variables of an expression and returns a sympy expression
    '''
    # # Filter unrecognized characters for safety (consider regex)
    # for char in expr:
    #     if char.isalpha() and char != 'x':
    #         raise ConstraintError(f'"{char}" is not recognized.')

    try:
        expr = sympify(expr)
    except Exception as e:
        raise ConstraintError(f'Invalid equation format: {e}')

    allowed_vars = set(symbols(f'x0:{n}'))
    vars = expr.free_symbols
    if vars - allowed_vars:
        raise ConstraintError(f'Unrecognized variables found: {vars - allowed_vars}')
    return expr

def to_ns_matrix(n, constraints):
    '''
    Returns a sympy matrix representing the given set of constraints
    '''
    exprs = set()
    for constraint in constraints:
        exprs.update(split_constraint(constraint))

    rows = []
    for expr in exprs:
        row = [0] * n
        expr = parse_expression(n, expr)
        for var in expr.free_symbols:
            var_idx = int(var.name.lstrip('x'))
            var_coeff = expr.coeff(var, 1)
            row[var_idx] = var_coeff
        rows.append(row)
    
    matrix = Matrix(rows)
    rref_matrix, _ = matrix.rref()
    ns_matrix = [row for row in rref_matrix.tolist() if any(val != 0 for val in row)]
    return Matrix(ns_matrix)
