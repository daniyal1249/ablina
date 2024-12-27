from numbers import Real
from sympy import *

def additive_id(field, n, add):
    # Initialize an arbitrary vector (xs) and the identity (ys)
    if field is Real:
        xs, ys = symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = symbols((f'x:{n}', f'y:{n}'))

    # Equations that must be satisfied
    exprs = [expand(lhs - rhs) for lhs, rhs in zip(add(list(xs), list(ys)), xs)]

    try:
        ids = linsolve(exprs, *ys)
    except Exception as e:  # replace with NonlinearError
        print(e)
        ids = nonlinsolve(exprs, ys)  # check output type

    if isinstance(ids, ConditionSet):
        print('ConditionSet received.')
        return []
    
    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not any(coord.has(*xs) for coord in id):
            valid_ids.append(id)
    return valid_ids

def additive_inv(field, n, add, additive_id):
    # Initialize an arbitrary vector (xs) and the inverse (ys)
    if field is Real:
        xs, ys = symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = symbols((f'x:{n}', f'y:{n}'))

    # Equations that must be satisfied
    exprs = [expand(lhs - rhs) for lhs, rhs in zip(add(list(xs), list(ys)), additive_id)]

    try:
        inverses = linsolve(exprs, *ys)
    except Exception as e:  # replace with NonlinearError
        print(e)
        inverses = nonlinsolve(exprs, ys)

    if isinstance(inverses, ConditionSet):
        print('ConditionSet received.')
        return []

    # Substitute zero for all parameters if a parametric solution is given
    valid_inverses = []
    sub_zero = {y: 0 for y in ys}
    for inv in inverses:
        valid_inv = []
        for coord in inv:
            valid_inv.append(coord.subs(sub_zero))
        valid_inverses.append(lambdify([list(xs)], valid_inv))

    return valid_inverses

def multiplicative_id(field, n, mul):
    # Initialize an arbitrary vector (xs) and scalar (c)
    if field is Real:
        xs, c = symbols((f'x:{n}', 'c'), real=True)
    else:
        xs, c = symbols((f'x:{n}', 'c'))

    # Equations that must be satisfied
    exprs = [lhs - rhs for lhs, rhs in zip(mul(c, list(xs)), xs)]

    ids = nonlinsolve(exprs, [c])  # check output type

    if isinstance(ids, ConditionSet):
        print('ConditionSet received.')
        return []
    
    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not id[0].has(*xs):
            valid_ids.append(id[0])  # append a scalar instead of a tuple
    return valid_ids

def is_commutative(field, n, operation):
    # Initialize two arbitrary vectors (xs and ys)
    if field is Real:
        xs, ys = symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = symbols((f'x:{n}', f'y:{n}'))

    for lhs, rhs in zip(operation(xs, ys), operation(ys, xs)):
        if not sympify(lhs).equals(sympify(rhs)):
            return False
    return True

def is_associative(field, n, operation):
    # Initialize three arbitrary vectors (xs, ys, and zs)
    if field is Real:
        xs, ys, zs = symbols((f'x:{n}', f'y:{n}', f'z:{n}'), real=True)
    else:
        xs, ys, zs = symbols((f'x:{n}', f'y:{n}', f'z:{n}'))

    lhs_vec = operation(xs, operation(ys, zs))
    rhs_vec = operation(operation(xs, ys), zs)
    
    for lhs, rhs in zip(lhs_vec, rhs_vec):
        if not sympify(lhs).equals(sympify(rhs)):
            return False
    return True

# to test associativity of multiplication (2 scalars one vector), define
# operation to be normal mul if both are scalars, and scalar mul otherwise


def solve_func_eq(equation, func):
    """
    Attempts to solve a univariate functional equation by guessing common forms of solutions.
    """
    a, b, x = symbols('a b x')
    c, d = symbols('c d', nonzero=True)
    w = Wild('w')
    
    # Solution forms
    solution_forms = [
        lambda x: a * x + b,           # Linear
        lambda x: c * log(x) + a,      # Logarithmic
        lambda x: c * exp(d * x) + a,  # Exponential
        lambda x: c * (x ** d)         # Power
    ]

    solutions = set()
    for form in solution_forms:
        # Substitute the forms into the equation
        subbed_eq = equation.replace(func(w), form(w))
        free_vars = subbed_eq.free_symbols
        print('free vars:', free_vars)
        # for expr in mapping:
        #     # Extract the free variables in the arguments of the functions
        #     free_vars.update(expr.args[0].free_symbols)
        try:
            # Try to solve for the parameters
            params = dict(*solve(subbed_eq, [a, b, c, d], dict=True))  # extract dict in list
            print(params)

            for param, expr in params.copy().items():
                if free_vars.intersection(expr.free_symbols):
                    params.pop(param)

            if params or is_tautology(subbed_eq):
                solution = simplify(form(x).subs(params))
                solutions.add(solution)
        except Exception:
            print('skipped')
            continue
    return solutions

def is_tautology(equation):
    if simplify(equation) == 0:
        return True
    return False

x, y, a, b = symbols('x y a b')
f = Function('f')
eq = Eq(f(x) , x + y)
print(solve_func_eq(eq, f))
# eq = b*log(x) + b*log(y) - b*log(x*y)
# eq = logcombine(eq, force=True)
# print(simplify(eq, force=True))
# print(solve(eq, b))
