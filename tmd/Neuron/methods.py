'''
tmd Neuron's methods
'''


def size(self, neurite_type='all'):
    """
    Neuron method to get size.
    """
    if neurite_type == 'all':
        neurite_list = ['basal', 'axon', 'apical']
    else:
        neurite_list = list([neurite_type])

    s = 0

    for neu in neurite_list:

        s = s + len(getattr(self, neu))

    return int(s)


def get_section_lengths(self, neurite_type='all'):
    """
    Neuron method to get section lengths.
    """
    import numpy as np

    if neurite_type == 'all':
        neurite_list = ['basal', 'axon', 'apical']
    else:
        neurite_list = list([neurite_type])

    lengths = []

    for neu in neurite_list:

        for tree in getattr(self, neu):

            lengths = lengths + list(tree.get_section_lengths())

    return np.array(lengths)


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
    import numpy as np
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


def simplify(self):
    '''Creates a copy of itself and simplifies all trees
       to create a skeleton of the neuron
    '''
    from tmd.utils import tree_type as td
    from tmd import Neuron

    neu = Neuron.Neuron()
    neu.soma = self.soma.copy_soma()

    for tr in self.neurites:
        t = tr.extract_simplified()
        neu.append_tree(t, td)

    return neu
