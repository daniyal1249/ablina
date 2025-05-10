__version__ = '0.3.0'
__author__ = 'Daniyal Akif'
__email__ = 'daniyalakif@gmail.com'
__license__ = 'MIT'
__description__ = 'A Python package for abstract linear algebra'
__url__ = 'https://github.com/daniyal1249/ablina'


from .field import Field, R, C
from .form import SesquilinearForm, InnerProduct, QuadraticForm
from .linearmap import (
    LinearMap, LinearOperator, LinearFunctional, Isomorphism, IdentityMap
    )
from .mathset import MathSet, negate
from .vectorspace import (
    Fn, VectorSpace, AffineSpace, fn, matrix_space, poly_space, hom, 
    is_vectorspace, columnspace, rowspace, nullspace, left_nullspace, 
    image, kernel
    )

__all__ = [
    'Field', 'R', 'C',
    'SesquilinearForm', 'InnerProduct', 'QuadraticForm',
    'LinearMap', 'LinearOperator', 'LinearFunctional', 'Isomorphism',
    'IdentityMap',
    'MathSet', 'negate',
    'Fn', 'VectorSpace', 'AffineSpace', 'fn', 'matrix_space', 'poly_space',
    'hom', 'is_vectorspace', 'columnspace', 'rowspace', 'nullspace',
    'left_nullspace', 'image', 'kernel'
    ]