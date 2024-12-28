import inspect
import sympy as sp

def ns_to_rs(matrix):
    matrix = sp.Matrix(matrix)
    if is_empty(matrix):
        return matrix
    
    ns_basis = matrix.nullspace()
    if not ns_basis:
        return sp.zeros(1, matrix.cols)
    
    rs_matrix, _ = sp.Matrix([vec.T for vec in ns_basis]).rref()
    return rs_matrix

def rs_to_ns(matrix):
    matrix = sp.Matrix(matrix)
    if is_empty(matrix):
        return matrix
    
    rs_basis = matrix.rowspace()
    if not rs_basis:
        return sp.eye(matrix.cols)
    
    ns_basis = sp.Matrix(rs_basis).nullspace()
    if not ns_basis:
        return sp.zeros(1, matrix.cols)

    ns_matrix, _ = sp.Matrix([vec.T for vec in ns_basis]).rref()
    return ns_matrix

def is_empty(matrix):
    '''
    Returns True if the matrix contains no elements otherwise False
    '''
    matrix = sp.Matrix(matrix)
    return matrix.cols == 0 or matrix.rows == 0

def of_arity(func, n):
    '''
    Returns True if the function can accept n positional arguments, otherwise False.
    '''
    sig = inspect.signature(func)
    if len(sig.parameters) < n:
        return False
    
    count_req_pos = 0  # Number of required positional args
    for param in sig.parameters.values():
        # Return False if there are required keyword-only args
        if param.kind == inspect.Parameter.KEYWORD_ONLY and param.default == inspect.Parameter.empty:
            return False
        if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) 
            and param.default == inspect.Parameter.empty):
            count_req_pos += 1

        if count_req_pos > n:
            return False
    return True

def add_attributes(cls, *attributes):
    attributes = {attr.__name__: attr for attr in attributes}
    return type(f'{cls.__name__}_subclass', (cls,), attributes)