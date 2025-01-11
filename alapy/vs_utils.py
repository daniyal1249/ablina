import sympy as sp
from sympy.solvers.solveset import NonlinearError

from alapy.parser import ConstraintError, split_constraint, sympify
from alapy.utils import rref, symbols


def additive_id(field, n, add):
    """
    The additive identity of an addition function on F^n.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the addition function takes.
    add : callable
        The addition function ..

    Returns
    -------
    x
    """
    # Initialize an arbitrary vector (xs) and the identity (ys)
    xs, ys = symbols((f'x:{n}', f'y:{n}'), field=field)
    xs, ys = list(xs), list(ys)
    
    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(xs, ys), xs)]

    try:
        ids = sp.linsolve(exprs, *ys)
    except NonlinearError:
        ids = sp.nonlinsolve(exprs, ys)  # Check output type
    if isinstance(ids, sp.ConditionSet):
        return []

    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not any(coord.has(*xs) for coord in id):
            valid_ids.append(list(id))
    return valid_ids


def additive_inv(field, n, add, add_id, lambdify=False):
    """
    The additive inverse of an addition function on F^n.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the addition function takes.
    add : callable
        The addition function ..

    Returns
    -------
    x
    """
    # Initialize an arbitrary vector (xs) and the inverse (ys)
    xs, ys = symbols((f'x:{n}', f'y:{n}'), field=field)
    xs, ys = list(xs), list(ys)

    # Equations that must be satisfied
    exprs = [sp.expand(lhs - rhs) for lhs, rhs in zip(add(xs, ys), add_id)]

    try:
        inverses = sp.linsolve(exprs, *ys)
    except NonlinearError:
        inverses = sp.nonlinsolve(exprs, ys)
    if isinstance(inverses, sp.ConditionSet):
        return []
    if not lambdify:
        return [list(inv) for inv in inverses]

    # Substitute zero for all params if a parametric solution is given
    valid_inverses = []
    sub_zero = {y: 0 for y in ys}
    for inv in inverses:
        valid_inv = []
        for coord in inv:
            valid_inv.append(coord.subs(sub_zero))
        valid_inverses.append(sp.lambdify([xs], valid_inv))
    return valid_inverses


def multiplicative_id(field, n, mul):
    """
    The multiplicative identity of a multiplication function on F^n.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the multiplication function takes.
    mul : callable
        The multiplication function ..

    Returns
    -------
    x
    """
    # Initialize an arbitrary vector (xs) and scalar (c)
    xs, c = symbols((f'x:{n}', 'c'), field=field)
    xs = list(xs)

    # Equations that must be satisfied
    exprs = [lhs - rhs for lhs, rhs in zip(mul(c, xs), xs)]
    ids = sp.nonlinsolve(exprs, [c])  # Check output type
    if isinstance(ids, sp.ConditionSet):
        return []
    
    valid_ids = []
    for id in ids:
        # Ensure the ids dont depend on xs
        if not id[0].has(*xs):
            valid_ids.append(id[0])  # Append scalar instead of tuple
    return valid_ids


def is_commutative(field, n, operation):
    """
    Check whether a binary operation on F^n is commutative.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the operation takes.
    operation : callable
        The function ..

    Examples
    --------
    x
    """
    # Initialize two arbitrary vectors (xs and ys)
    xs, ys = symbols((f'x:{n}', f'y:{n}'), field=field)
    xs, ys = list(xs), list(ys)

    for lhs, rhs in zip(operation(xs, ys), operation(ys, xs)):
        if not sp.sympify(lhs).equals(sp.sympify(rhs)):
            return False
    return True


def is_associative(field, n, operation):
    """
    Check whether a binary operation on F^n is associative.

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors the operation takes.
    operation : callable
        The function ..

    Examples
    --------
    x
    """
    # Initialize three arbitrary vectors (xs, ys, and zs)
    xs, ys, zs = symbols((f'x:{n}', f'y:{n}', f'z:{n}'), field=field)
    xs, ys, zs = list(xs), list(ys), list(zs)

    lhs_vec = operation(xs, operation(ys, zs))
    rhs_vec = operation(operation(xs, ys), zs)
    for lhs, rhs in zip(lhs_vec, rhs_vec):
        if not sp.sympify(lhs).equals(sp.sympify(rhs)):
            return False
    return True

# To test associativity of multiplication (2 scalars one vector), define
# operation to be normal mul if both are scalars, and scalar mul otherwise

def solve_func_eq(equation, func):
    """
    Attempt to solve a univariate functional equation by guessing common 
    forms of solutions.

    Parameters
    ----------
    equation : sympy.Expr or sympy.Eq
        The functional equation to be solved.
    func : sympy.Function
        The unknown function to solve for.

    Returns
    -------
    valid_funcs : set
        x
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
        invalid_vars = subbed_eq.free_symbols - {_a, _b}
        try:
            sols = sp.solve(subbed_eq, [_a, _b], rational=True, dict=True)
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
        x

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

    Parameters
    ----------
    field : {Real, Complex}
        The field of scalars.
    n : int
        The length of the vectors in the vector space.
    add : callable
        x
    mul : callable
        x

    Returns
    -------
    x
    """
    # Need to support custom domains
    # Need to implement an intersection function
    # Return separate functions for each coordinate

    f = sp.Function('f')
    xs, ys = symbols((f'x:{n}', f'y:{n}'), field=field)

    init_set = False
    for i in range(len(add)):
        func_eq = sp.Eq(f(xs[i]) + f(ys[i]), f(add[i]))
        if not init_set:
            valid_funcs = solve_func_eq(func_eq, f)
            init_set = True
        else:
            valid_funcs.intersection_update(solve_func_eq(func_eq, f))
        if not valid_funcs:
            return valid_funcs
    
    for i in range(len(mul)):
        func_eq = sp.Eq(f(xs[i]) * f(ys[i]), f(mul[i]))
        valid_funcs.intersection_update(solve_func_eq(func_eq, f))
        if not valid_funcs:
            return valid_funcs
    return valid_funcs


# For testing
def standard_isomorphism(field, n, add, mul):
    return lambda x: x, lambda x: x


def map_constraints(mapping, constraints):
    """

    Parameters
    ----------
    mapping : callable
        x
    constraints : list of str
        x

    Returns
    -------
    list of str
    """
    return constraints


def to_ns_matrix(n, lin_constraints):
    """
    Return the matrix representation of the given linear constraints.

    Parameters
    ----------
    n : int
        x
    lin_constraints : list of str
        The list of constraints.

    Returns
    -------
    ns_matrix : sympy.Matrix
        A sympy matrix with the linear constraints as rows.
    """
    exprs = set()
    for constraint in lin_constraints:
        exprs.update(split_constraint(constraint))

    matrix = []
    allowed_vars = sp.symbols(f'v:{n}')
    for expr in exprs:
        row = [0] * n
        try:
            expr = sympify(expr, allowed_vars)
            expr = expr.lhs - expr.rhs  # Convert equation to an expression
        except Exception as e:
            raise ConstraintError('Invalid constraint format.') from e

        for var in expr.free_symbols:
            var_idx = int(var.name.lstrip('v'))
            var_coeff = expr.coeff(var, 1)
            row[var_idx] = var_coeff
        matrix.append(row)
    
    ns_matrix = rref(matrix, remove=True) if matrix else sp.zeros(0, n)
    return ns_matrix


def to_complement(matrix):
    """

    Parameters
    ----------
    matrix : sympy.Matrix
        x

    Returns
    -------
    sympy.Matrix
        x
    """
    if matrix.rows == 0:
        return sp.eye(matrix.cols)
    
    ns_basis = matrix.nullspace()
    if not ns_basis:
        return sp.zeros(0, matrix.cols)
    return rref([vec.T for vec in ns_basis], remove=True)


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
