from numbers import Real
import sympy as sp
from sympy.solvers.solveset import NonlinearError

def additive_id(field, n, add):
    # Initialize an arbitrary vector (xs) and the identity (ys)
    if field is Real:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'))

    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(list(xs), list(ys)), xs)]

    try:
        ids = sp.linsolve(exprs, *ys)
    except NonlinearError:
        ids = sp.nonlinsolve(exprs, ys)  # check output type

    if isinstance(ids, sp.ConditionSet):
        return []

    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not any(coord.has(*xs) for coord in id):
            valid_ids.append(list(id))
    return valid_ids

def additive_inv(field, n, add, additive_id, lambdify=True):
    # Initialize an arbitrary vector (xs) and the inverse (ys)
    if field is Real:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'))

    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(list(xs), list(ys)), additive_id)]

    try:
        inverses = sp.linsolve(exprs, *ys)
    except NonlinearError:
        inverses = sp.nonlinsolve(exprs, ys)

    if isinstance(inverses, sp.ConditionSet):
        return []
    if not lambdify:
        return [list(inv) for inv in inverses]

    # Substitute zero for all parameters if a parametric solution is given
    valid_inverses = []
    sub_zero = {y: 0 for y in ys}
    for inv in inverses:
        valid_inv = []
        for coord in inv:
            valid_inv.append(coord.subs(sub_zero))
        valid_inverses.append(sp.lambdify([list(xs)], valid_inv))

    return valid_inverses

def multiplicative_id(field, n, mul):
    # Initialize an arbitrary vector (xs) and scalar (c)
    if field is Real:
        xs, c = sp.symbols((f'x:{n}', 'c'), real=True)
    else:
        xs, c = sp.symbols((f'x:{n}', 'c'))

    # Equations that must be satisfied
    exprs = [lhs - rhs for lhs, rhs in zip(mul(c, list(xs)), xs)]
    ids = sp.nonlinsolve(exprs, [c])  # check output type

    if isinstance(ids, sp.ConditionSet):
        return []
    
    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not id[0].has(*xs):
            valid_ids.append(id[0])  # append scalar instead of tuple
    return valid_ids

def is_commutative(field, n, operation):
    # Initialize two arbitrary vectors (xs and ys)
    if field is Real:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'), real=True)
    else:
        xs, ys = sp.symbols((f'x:{n}', f'y:{n}'))

    for lhs, rhs in zip(operation(xs, ys), operation(ys, xs)):
        if not sp.sympify(lhs).equals(sp.sympify(rhs)):
            return False
    return True

def is_associative(field, n, operation):
    # Initialize three arbitrary vectors (xs, ys, and zs)
    if field is Real:
        xs, ys, zs = sp.symbols((f'x:{n}', f'y:{n}', f'z:{n}'), real=True)
    else:
        xs, ys, zs = sp.symbols((f'x:{n}', f'y:{n}', f'z:{n}'))

    lhs_vec = operation(xs, operation(ys, zs))
    rhs_vec = operation(operation(xs, ys), zs)
    
    for lhs, rhs in zip(lhs_vec, rhs_vec):
        if not sp.sympify(lhs).equals(sp.sympify(rhs)):
            return False
    return True

# to test associativity of multiplication (2 scalars one vector), define
# operation to be normal mul if both are scalars, and scalar mul otherwise


def solve_func_eq(equation, func):
    """
    Attempts to solve a univariate functional equation by guessing common forms of solutions.
    """
    _a, _b, x = sp.symbols('_a _b x')
    w = sp.Wild('w')
    
    solution_forms = [
        lambda x: _a * x + _b,          # Linear
        lambda x: _a * sp.log(x) + _b,  # Logarithmic
        lambda x: _a * sp.exp(x),       # Exponential
    ]
    
    valid_funcs = set()
    for form in solution_forms:
        # Substitute the forms into the equation
        subbed_eq = equation.replace(func(w), form(w))
        try:
            sols = sp.solve(subbed_eq, [_a, _b], rational=True, dict=True)
            sols = [dict()] if not sols else sols
        except Exception:
            continue

        for sol in sols:
            invalid_expr = False
            for expr in sol.copy().values():
                invalid_vars = subbed_eq.free_symbols.difference({_a, _b})
                if expr.free_symbols.intersection(invalid_vars):
                    invalid_expr = True
                    break
            if invalid_expr:
                continue
            
            if sol or is_tautology(subbed_eq):  # include tautologies
                valid_func = sp.simplify(form(x).subs(sol))
                valid_funcs.add(valid_func)
    return valid_funcs

def is_tautology(equation):
    '''
    Returns True if the equation always holds, otherwise False
    '''
    eq = sp.simplify(equation)
    if isinstance(eq, sp.Eq):
        return False
    return bool(eq)  # must be a bool if not Eq

# Need to account for nested functions using while loop

# x, y, a, b, c = symbols('x y a b c', positive=True)
# f = Function('f')
# g = Function('g')
# eq = Eq(f(x) + f(y), f(a) + 1)
# print(solve_func_eq(eq, f))
