import re
import sympy as sp

class ParsingError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

class ConstraintError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)

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