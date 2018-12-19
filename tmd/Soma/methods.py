'''
tmd Soma's methods
'''
import numpy as np


def get_center(self):
    """
    Soma method to get the center of the soma.
    """
    x_center = np.mean(self.x)
    y_center = np.mean(self.y)
    z_center = np.mean(self.z)

    return np.array([x_center, y_center, z_center])


def get_diameter(self):
    """
    Soma method to get the diameter of the soma.
    """
    if len(self.x) == 1:
        diameter = self.d[0]
    else:
        center = self.get_center()
        diameter = np.mean(np.sqrt(np.power(self.x - center[0], 2) +
                                   np.power(self.y - center[1], 2) +
                                   np.power(self.z - center[2], 2)))
    return diameter
