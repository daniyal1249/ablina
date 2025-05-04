import sympy as sp
from sympy.solvers.solveset import NonlinearError

from .parser import ConstraintError, split_constraint, sympify
from .utils import rref, symbols


def additive_id(field, n, add):
    """
    The identity element of an addition function on F^n.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the addition function accepts.
    add : callable
        The addition function on F^n.

    Returns
    -------
    pass
    """
    # Initialize an arbitrary vector (v) and the identity (e)
    v, e = symbols((f'v:{n}', f'e:{n}'), field=field)
    v, e = list(v), list(e)
    
    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(v, e), v)]

    try:
        ids = sp.linsolve(exprs, *e)
    except NonlinearError:
        ids = sp.nonlinsolve(exprs, e)  # Check output type
    if isinstance(ids, sp.ConditionSet):
        return []

    valid_ids = []
    for id in ids:
        # Ensure the ids do not depend on v
        if not any(coord.has(*v) for coord in id):
            valid_ids.append(list(id))
    return valid_ids


def additive_inv(field, n, add, add_id, lambdify=False):
    """
    The additive inverse of an addition function on F^n.
    """
    # Initialize an arbitrary vector (v) and the inverse (u)
    v, u = symbols((f'v:{n}', f'u:{n}'), field=field)
    v, u = list(v), list(u)

    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(v, u), add_id)]

    try:
        invs = sp.linsolve(exprs, *u)
    except NonlinearError:
        invs = sp.nonlinsolve(exprs, u)
    if isinstance(invs, sp.ConditionSet):
        return []
    
    if not lambdify:
        return [list(inv) for inv in invs]

    # Substitute zero for all params if a parametric solution is given
    valid_invs = []
    sub_zero = {coord: 0 for coord in u}
    for inv in invs:
        valid_inv = []
        for coord in inv:
            valid_inv.append(coord.subs(sub_zero))
        valid_invs.append(sp.lambdify([v], valid_inv))
    return valid_invs


def is_commutative(field, n, operation):
    """
    Check whether a binary operation on F^n is commutative.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the operation accepts.
    operation : callable
        The operation to check.

    Examples
    --------
    pass
    """
    # Initialize two arbitrary vectors (u and v)
    u, v = symbols((f'u:{n}', f'v:{n}'), field=field)
    u, v = list(u), list(v)

    for lhs, rhs in zip(operation(u, v), operation(v, u)):
        value = sp.sympify(lhs).equals(sp.sympify(rhs))
        if value is False or value is None:
            return value
    return True


def is_associative(field, n, operation):
    """
    Check whether a binary operation on F^n is associative.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the operation accepts.
    operation : callable
        The operation to check.

    Examples
    --------
    pass
    """
    # Initialize three arbitrary vectors (u, v, and w)
    u, v, w = symbols((f'u:{n}', f'v:{n}', f'w:{n}'), field=field)
    u, v, w = list(u), list(v), list(w)

    lhs_vec = operation(u, operation(v, w))
    rhs_vec = operation(operation(u, v), w)
    for lhs, rhs in zip(lhs_vec, rhs_vec):
        value = sp.sympify(lhs).equals(sp.sympify(rhs))
        if value is False or value is None:
            return value
    return True

# To test associativity of multiplication (2 scalars one vector), define
# operation to be normal mul if both are scalars, and scalar mul otherwise

def solve_func_eq(equation, f):
    """
    Attempt to solve a univariate functional equation by guessing common 
    forms of solutions.

    Parameters
    ----------
    equation : sympy.Expr or sympy.Eq
        The functional equation to solve.
    func : sympy.Function
        The function to solve for.

    Returns
    -------
    valid_funcs : set of sympy.Expr
        pass
    """
    _a, _b, x = sp.symbols('_a _b x')
    w = sp.Wild('w')
    
    solution_forms = [
        lambda x: _a * x + _b,          # Linear
        lambda x: _a * sp.log(x) + _b,  # Logarithmic
        lambda x: _a * sp.exp(x)        # Exponential
        ]
    
    valid_funcs = set()
    for form in solution_forms:
        # Substitute the forms into the equation
        subbed_eq = equation.replace(f(w), form(w))
        invalid_vars = subbed_eq.free_symbols - {_a, _b}
        try:
            sols = sp.solve(subbed_eq, [_a, _b], dict=True)
            sols = sols if sols else [dict()]
        except Exception:
            continue

        for sol in sols:
            invalid_expr = False
            for expr in sol.values():
                if expr.free_symbols.intersection(invalid_vars):
                    invalid_expr = True
                    break
            if invalid_expr:
                continue
            
            if sol or is_tautology(subbed_eq):  # Include tautologies
                valid_func = sp.simplify(form(x).subs(sol))
                valid_funcs.add(valid_func)
    return valid_funcs


def is_tautology(equation):
    """
    Check whether an equation is a tautology.

    Parameters
    ----------
    equation : sympy.Eq
        pass

    Returns
    -------
    bool
        True if `equation` always holds, otherwise False.
    """
    eq = sp.simplify(equation)
    if isinstance(eq, sp.Eq):
        return False
    return bool(eq)  # Must be a sympy bool if not Eq


def standard_isomorphism(field, n, add, mul):
    """
    pass

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors in the vector space.
    add : callable
        pass
    mul : callable
        pass

    Returns
    -------
    pass
    """
    # Need to support custom domains
    # Need to implement an intersection function
    # Return separate functions for each coordinate

    f = sp.Function('f')
    u, v = symbols((f'u:{n}', f'v:{n}'), field=field)

    init_set = False
    for i in range(len(add)):
        func_eq = sp.Eq(f(u[i]) + f(v[i]), f(add[i]))
        if not init_set:
            valid_funcs = solve_func_eq(func_eq, f)
            init_set = True
        else:
            valid_funcs.intersection_update(solve_func_eq(func_eq, f))
        if not valid_funcs:
            return valid_funcs
    
    for i in range(len(mul)):
        func_eq = sp.Eq(f(u[i]) * f(v[i]), f(mul[i]))
        valid_funcs.intersection_update(solve_func_eq(func_eq, f))
        if not valid_funcs:
            return valid_funcs
    return valid_funcs


# Need to account for nested functions using while loop

# x, y, a, b, c = sp.symbols('x y a b c', real=True)
# xs, ys = sp.symbols((f'x:3', f'y:3'), real=True)
# f = sp.Function('f')
# g = sp.Function('g')
# eq = sp.Eq(f(x) * f(y), f(x + y))
# # print(solve_func_eq(eq, f))

# add = [i + j for i, j in zip(xs, ys)]
# mul = [i * j for i, j in zip(xs, ys)]

# print(isomorphism(Real, 3, add, mul))