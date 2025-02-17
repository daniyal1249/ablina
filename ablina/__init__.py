__version__ = '0.2.1'
__author__ = 'Daniyal Akif'
__email__ = 'daniyalakif@gmail.com'
__license__ = 'MIT'
__description__ = 'A Python package for abstract linear algebra'
__url__ = 'https://github.com/daniyal1249/ablina'


from numbers import Complex, Real

from .innerproduct import InnerProductSpace
from .linearmap import LinearMap, Isomorphism, IdentityMap
from .mathset import MathematicalSet, Set
from .vectorspace import (
    VectorSpace, Fn, is_vectorspace, columnspace, rowspace, 
    nullspace, left_nullspace, image, kernel
    )


__all__ = [
    'Complex', 'Real',
    'InnerProductSpace',
    'LinearMap', 'Isomorphism', 'IdentityMap',
    'MathematicalSet', 'Set',
    'VectorSpace', 'Fn', 'is_vectorspace', 'columnspace', 'rowspace',
    'nullspace', 'left_nullspace', 'image', 'kernel'
    ]