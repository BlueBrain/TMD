"""TMD transformation algorithms."""
import numpy as np


def tmd_scale(barcode, thickness):
    """Scale the first two components according to the thickness parameter.

    Only these components are scaled because they correspond to spatial coordinates.
    """
    scaling_factor = np.ones(len(barcode[0]), dtype=float)
    scaling_factor[:2] = thickness
    return np.multiply(barcode, scaling_factor).tolist()
