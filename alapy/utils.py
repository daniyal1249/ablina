from numbers import Real, Complex
import inspect
import sympy as sp

def symbols(names, field=Complex, **kwargs):
    '''
    Returns sympy symbols with the specified names and field (Real or Complex).
    Additional constraints can be specified as keyword args.
    '''
    if field is Real:
        return sp.symbols(names, real=True, **kwargs)
    else:
        return sp.symbols(names, **kwargs)

def is_real(expr):
    '''
    Returns True if the expression is Real, None if its indeterminate, 
    and False otherwise.
    '''
    if hasattr(expr, 'is_real'):
        return expr.is_real  # sympy's is_real attribute
    return isinstance(expr, Real)

def is_complex(expr):
    '''
    Returns True if the expression is Complex, None if its indeterminate, 
    and False otherwise.
    '''
    if hasattr(expr, 'is_complex'):
        return expr.is_complex  # sympy's is_complex attribute
    return isinstance(expr, Complex)

def is_linear(expr, vars=None):
    '''
    Determines if an equation or expression is linear with respect to the 
    variables in vars. If vars is None, all variables in expr are checked.
    '''
    if vars is None:
        vars = expr.free_symbols
    try:
        terms = sp.Poly(expr, *vars).monoms()
    except sp.PolynomialError:
        return False  # return False if not a polynomial
    
    for exponents in terms:
        if sum(exponents) not in (0, 1):
            return False
        if not all(i in (0, 1) for i in exponents):
            return False
    return True

def is_empty(matrix):
    '''
    Returns True if the matrix contains no elements, otherwise False.
    '''
    matrix = sp.Matrix(matrix)
    return matrix.cols == 0 or matrix.rows == 0

def is_invertible(matrix):
    '''
    Returns True if the matrix is invertible, otherwise False.
    '''
    matrix = sp.Matrix(matrix)
    return matrix.is_square and matrix.det() != 0

def rref(matrix, remove=False):
    '''
    Returns the rref of the matrix. Removes all zero rows if remove is True.
    '''
    matrix = sp.Matrix(matrix)
    rref, _ = matrix.rref()
    
    if not remove:
        return rref
    for i in range(rref.rows - 1, -1, -1):
        if any(rref.row(i)):
            break
        rref.row_del(i)
    return rref

def of_arity(func, n):
    '''
    Returns True if the function can accept n positional arguments, 
    otherwise False. Raises a TypeError if func is not callable.
    '''
    sig = inspect.signature(func)
    if len(sig.parameters) < n:
        return False
    
    count_req_pos = 0  # Number of required positional args
    for param in sig.parameters.values():
        # Return False if there are required keyword-only args
        if (param.kind == inspect.Parameter.KEYWORD_ONLY and 
            param.default == inspect.Parameter.empty):
            return False
        if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, 
            inspect.Parameter.POSITIONAL_OR_KEYWORD) and 
            param.default == inspect.Parameter.empty):
            count_req_pos += 1

        if count_req_pos > n:
            return False
    return True

def add_attributes(cls, *attributes):
    attributes = {attr.__name__: attr for attr in attributes}
    return type(f'{cls.__name__}_subclass', (cls,), attributes)