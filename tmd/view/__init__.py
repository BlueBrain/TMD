"""
Visualize neurons and plots persistence barcode, diagrams, images.
Matplotlib required.
"""


try:
    import matplotlib
except ImportError as exc:
    raise ImportError('tmd[viewer] is not installed. ' +
                      'Please install it by doing: pip install tmd[viewer]') from exc

from tmd.view import plot, view
