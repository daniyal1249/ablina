try:
    import manim
except ImportError:
    raise ImportError(
        'Manim is required for the visualize module. Install it using: \n\n'
        '    pip install ablina[visualize]')

from .linearmap import LinearMap
from .vectorspace import VectorSpace
