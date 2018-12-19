'''
tmd Neuron's methods
'''
import numpy as np


def size(self, neurite_type='all'):
    """
    Neuron method to get size.
    """
    if neurite_type == 'all':
        neurite_list = ['basal', 'axon', 'apical']

    s = np.sum([len(getattr(self, neu)) for neu in neurite_list])

    return int(s)


def get_bounding_box(self):
    """
    Input
    ------
    neuron: tmd neuron

    Returns
    ---------
    bounding_box: np.array
        ([xmin,ymin,zmin], [xmax,ymax,zmax])
    """
    x = []
    y = []
    z = []

    for tree in self.neurites:
        x = x + tree.x.tolist()
        y = y + tree.y.tolist()
        z = z + tree.z.tolist()

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
