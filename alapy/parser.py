import re
import sympy as sp

class ParsingError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class ConstraintError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

def split_constraint(constraint):
    '''
    Splits a constraint with multiple relational operators into separate 
    relations. Returns a set of relations as strings.
    '''
    operators = r'(==|!=|>=|<=|>|<)'
    exprs = re.split(operators, constraint)
    expr_count = len(exprs)
    if expr_count == 1:
        raise ConstraintError('Constraints must include at least one ' \
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

def is_linear(expr, vars=None):
    '''
    Determines if an equation or expression is linear with respect to the 
    variables in vars. If vars is None, all variables in expr are checked.
    '''
    if vars is None:
        vars = expr.free_symbols

    # Convert an equation into an expression
    if isinstance(expr, sp.Eq):
        expr = expr.lhs - expr.rhs
    try:
        return all(sp.degree(expr, var) == 1 for var in vars)
    except sp.PolynomialError:
        return False  # Return false if not a polynomial

def sympify(expr, allowed_vars=None):
    '''
    Returns the sympy representation of expr. Raises a ParsingError if 
    allowed_vars is given and expr contains variables not in it.
    '''
    # Filter unrecognized characters for safety (consider regex)
    expr = sp.sympify(expr, rational=True, evaluate=False)
    if allowed_vars is not None:
        if not all(var in allowed_vars for var in expr.free_symbols):
            invalid_vars = expr.free_symbols - set(allowed_vars)
            raise ParsingError(f'Unrecognized variables found: {invalid_vars}')
    return expr

def to_ns_matrix(n, lin_constraints):
    '''
    Returns a sympy matrix with the linear constraints as rows.
    '''
    # Return zero matrix if there are no constraints
    if not lin_constraints:
        return sp.zeros(1, n)

    exprs = set()
    for constraint in lin_constraints:
        exprs.update(split_constraint(constraint))

    rows, allowed_vars = [], sp.symbols(f'v:{n}')
    for expr in exprs:
        row = [0] * n
        try:
            expr = sympify(expr, allowed_vars)
            expr = expr.lhs - expr.rhs  # convert equation to an expression
        except Exception as e:
            raise ConstraintError(f'Invalid constraint format: {e}')

        for var in expr.free_symbols:
            var_idx = int(var.name.lstrip('v'))
            var_coeff = expr.coeff(var, 1)
            row[var_idx] = var_coeff
        rows.append(row)
    
    matrix = sp.Matrix(rows)
    ns_matrix, _ = matrix.rref()
    return ns_matrix
