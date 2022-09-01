"""
tmd class : Neuron
"""
import copy
import warnings

import numpy as np

from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT


class Neuron(object):
    """
    A Neuron object is a container for Trees
    (basal_dendrite, apical_dendrite and axon) and a Soma.
    """

    # pylint: disable=import-outside-toplevel
    from tmd.Neuron.methods import get_bounding_box
    from tmd.Neuron.methods import size

    def __init__(self, name="Neuron"):
        """Creates an empty Neuron object."""
        self.soma = Soma.Soma()
        self.axon = []
        self.apical_dendrite = []
        self.basal_dendrite = []
        self.undefined = []
        self.name = name

    @property
    def neurites(self):
        """Get neurites."""
        return self.apical_dendrite + self.axon + self.basal_dendrite + self.undefined

    @property
    def dendrites(self):
        """Get dendrites."""
        return self.apical_dendrite + self.basal_dendrite

    @property
    def apical(self):
        """Get apical dendrites."""
        warnings.warn(
            "The 'apical' property is deprecated, please use 'apical_dendrite' instead",
            DeprecationWarning,
        )
        return self.apical_dendrite

    @property
    def basal(self):
        """Get basal dendrites."""
        warnings.warn(
            "The 'basal' property is deprecated, please use 'basal_dendrite' instead",
            DeprecationWarning,
        )
        return self.basal_dendrite

    def rename(self, new_name):
        """Modifies the name of the Neuron to new_name."""
        self.name = new_name

    def set_soma(self, new_soma):
        """Set the given Soma object as the soma of the current Neuron."""
        if isinstance(new_soma, Soma.Soma):
            self.soma = new_soma

    def append_tree(self, new_tree, tree_types):
        """Append a Tree object to the Neuron.

        If type of object is tree this function finds the type of tree and adds the new_tree to the
        correct list of trees in neuron.
        """
        if isinstance(new_tree, Tree.Tree):

            if int(np.median(new_tree.t)) in tree_types.keys():
                neurite_type = tree_types[int(np.median(new_tree.t))]
            else:
                neurite_type = "undefined"
            getattr(self, neurite_type).append(new_tree)

    def copy_neuron(self):
        """Returns a deep copy of the Neuron."""
        return copy.deepcopy(self)

    def is_equal(self, neu):
        """Tests if all neuron structures are the same."""
        eq = np.alltrue(
            [
                self.soma.is_equal(neu.soma),
                np.alltrue([t1.is_equal(t2) for t1, t2 in zip(self.neurites, neu.neurites)]),
            ]
        )
        return eq

    def is_same(self, neu):
        """Tests if all neuron data are the same."""
        eq = np.alltrue(
            [
                self.name == neu.name,
                self.soma.is_equal(neu.soma),
                np.alltrue([t1.is_equal(t2) for t1, t2 in zip(self.neurites, neu.neurites)]),
            ]
        )
        return eq

    def simplify(self):
        """Creates a copy of itself and simplifies all trees to create a skeleton of the neuron."""
        neu = Neuron()
        neu.soma = self.soma.copy_soma()

        for tr in self.neurites:
            t = tr.extract_simplified()
            neu.append_tree(t, TREE_TYPE_DICT)

        return neu
