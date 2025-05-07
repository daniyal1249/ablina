import sympy as sp
from sympy.solvers.solveset import NonlinearError

from .utils import symbols


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


def truth_eval(equation):
    """
    Check whether an equation is a tautology or contradiction.

    Parameters
    ----------
    equation : sympy.Eq
        pass

    Returns
    -------
    bool
        pass
    """
    eq = sp.simplify(equation)
    if isinstance(eq, sp.Eq):
        return None
    return bool(eq)  # eq must be a sympy bool if not Eq


def substitute_form(equation, f, form):
    w = sp.Wild('w')
    return equation.replace(f(w), form(w))


def find_valid_params(equation, f, form, params):
    x = sp.symbols('x')
    subbed_eq = substitute_form(equation, f, form)
    if truth_eval(subbed_eq):
        return form(x)
    
    try:
        sols = sp.solve(subbed_eq, params, dict=True)
    except Exception:
        return None
    
    # valid_sols = []
    # for sol in sols:
    #     if all(expr.free_symbols <= set(params) for expr in sol.values()):
    #         valid_sols.append(sol)
    # return [form(x).subs(sol) for sol in valid_sols]

    for sol in sols:
        if all(expr.free_symbols <= set(params) for expr in sol.values()):
            return form(x).subs(sol)


def solve_func_eq(equation, f):
    """
    Attempt to solve a univariate functional equation by guessing 
    common forms of solutions.

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
    a0, a1 = sp.symbols('_a:2')
    b0, b1 = sp.symbols('_b:2', nonzero=True)

    forms = [
        # (lambda x: a, [a]),                       # Constant
        (lambda x: b0 * x + a0, [a0, b0]),          # Linear
        (lambda x: b1 * sp.log(x) + a1, [a1, b1]),  # Logarithmic
        # (lambda x: b * 2**x, [b]),                # Exponential (base 2)
        # (lambda x: b * sp.exp(x), [b]),           # Exponential (base e)
        ]
    
    # valid_sols = set()
    # for form, params in forms:
    #     sols = find_valid_params(equation, f, form, params)
    #     valid_sols.update(sols)
    # return valid_sols

    for form, params in forms:
        sol = find_valid_params(equation, f, form, params)
        if sol:
            return sol


def find_add_isomorphism(field, n, add):
    f = sp.Function('f')
    u, v = symbols((f'u:{n}', f'v:{n}'), field=field)


def find_mul_isomorphism(field, n, mul):
    f = sp.Function('f')
    u, v = symbols((f'u:{n}', f'v:{n}'), field=field)


def internal_isomorphism(field, n, add, mul):
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