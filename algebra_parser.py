import numpy as np

# <1> = <2>
def split_constraint(constraint):
    expressions = constraint.split('=')
    expr_length = len(expressions)
    if expr_length == 1:
        raise SyntaxError('Invalid constraint')  # check error msg
    
    eqs = set()
    for i in range(expr_length - 1):
        eq = expressions[i] + ' = ' + expressions[i + 1]
        eqs.add(eq.strip())
    return eqs


def parse_equation(n, eq):
    symbols = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '>', 
               '+', '-', '*', '='}
    for char in eq:
        if char != ' ' and char not in symbols:
            raise SyntaxError(f'"{char}" not recognized.')


def to_matrix(n, constraints):
    pass