"""Visualize neurons and plots persistence barcode, diagrams, images.

Matplotlib required.
"""


try:
    import matplotlib  # noqa
except ImportError as exc:
    raise ImportError(
        "tmd[viewer] is not installed. " + "Please install it by doing: pip install tmd[viewer]"
    ) from exc

from tmd.view import plot  # noqa
from tmd.view import view  # noqa
