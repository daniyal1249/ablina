__version__ = '0.3.0'
__author__ = 'Daniyal Akif'
__email__ = 'daniyalakif@gmail.com'
__license__ = 'MIT'
__description__ = 'A Python package for abstract linear algebra'
__url__ = 'https://github.com/daniyal1249/ablina'


from numbers import Complex, Real

from .form import InnerProduct, QuadraticForm, SesquilinearForm
from .linearmap import LinearMap, Isomorphism, IdentityMap
from .mathset import MathSet
from .vectorspace import (
    Fn, VectorSpace, AffineSpace, fn, matrix_space, poly_space, hom, 
    is_vectorspace, columnspace, rowspace, nullspace, left_nullspace, 
    image, kernel
    )

__all__ = [
    'Complex', 'Real',
    'InnerProduct', 'QuadraticForm', 'SesquilinearForm',
    'LinearMap', 'Isomorphism', 'IdentityMap',
    'MathSet',
    'Fn', 'VectorSpace', 'AffineSpace', 'fn', 'matrix_space', 'poly_space',
    'hom', 'is_vectorspace', 'columnspace', 'rowspace', 'nullspace',
    'left_nullspace', 'image', 'kernel'
    ]