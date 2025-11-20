__version__ = "1.0.0"
__author__ = "Daniyal Akif"
__email__ = "daniyalakif@gmail.com"
__license__ = "MIT"
__description__ = "A Python package for abstract linear algebra"
__url__ = "https://github.com/daniyal1249/ablina"


from .field import Field, Reals, Complexes, R, C
from .form import (
    FormError, InnerProductError, SesquilinearForm, InnerProduct, QuadraticForm
    )
from .linearmap import (
    LinearMapError, LinearMap, LinearOperator, LinearFunctional, Isomorphism, 
    IdentityMap
    )
from .mathset import Set, negate, remove_duplicates
from .matrix import Matrix, M
from .parser import ParsingError, ConstraintError, sympify, split_constraint
from .utils import (
    symbols, is_linear, is_empty, is_invertible, is_orthogonal, is_unitary, 
    is_normal, rref, of_arity, add_attributes
    )
from .vectorspace import (
    NotAVectorSpaceError, Fn, VectorSpace, AffineSpace, fn, matrix_space, 
    poly_space, hom, is_vectorspace, columnspace, rowspace, nullspace, 
    left_nullspace, image, kernel
    )
from .vs_utils import to_ns_matrix, to_complement


__all__ = [
    "Field", "Reals", "Complexes", "R", "C",
    "FormError", "InnerProductError", "SesquilinearForm", "InnerProduct", "QuadraticForm",
    "LinearMapError", "LinearMap", "LinearOperator", "LinearFunctional", "Isomorphism",
    "IdentityMap",
    "Set", "negate", "remove_duplicates",
    "Matrix", "M",
    "ParsingError", "ConstraintError", "sympify", "split_constraint",
    "symbols", "is_linear", "is_empty", "is_invertible", "is_orthogonal", "is_unitary",
    "is_normal", "rref", "of_arity", "add_attributes",
    "NotAVectorSpaceError", "Fn", "VectorSpace", "AffineSpace", "fn", "matrix_space",
    "poly_space", "hom", "is_vectorspace", "columnspace", "rowspace", "nullspace",
    "left_nullspace", "image", "kernel",
    "to_ns_matrix", "to_complement"
    ]