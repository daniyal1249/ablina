"""
A module for visualizing linear algebra objects.
"""

try:
    import manim
except ImportError:
    raise ImportError(
        "Manim is required for the visualize module. Install it using: \n\n"
        "    pip install ablina[visualize]"
        )
