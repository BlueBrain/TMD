'''
tmd class : Neuron
'''


class Neuron(object):
    """
    A Neuron object is a container for Trees
    (basal, apical and axon) and a Soma.
    """
    from tmd.Neuron.methods import size
    from tmd.Neuron.methods import get_bounding_box

    def __init__(self, name='Neuron'):
        '''Creates an empty Neuron object.
        '''
        from tmd.Soma import Soma

        self.soma = Soma.Soma()
        self.axon = list()
        self.apical = list()
        self.basal = list()
        self.undefined = list()
        self.name = name

    @property
    def neurites(self):
        '''Get neurites'''
        return self.apical + self.axon + self.basal + self.undefined

    @property
    def dendrites(self):
        '''Get dendrites'''
        return self.apical + self.basal

    def rename(self, new_name):
        """
        Modifies the name of the Neuron to new_name.
        """
        self.name = new_name

    def set_soma(self, new_soma):
        """
        If type of object is soma
        sets the neuron soma to new_soma
        """
        from tmd.Soma import Soma

        if isinstance(new_soma, Soma.Soma):
            self.soma = new_soma

    def append_tree(self, new_tree, td):
        """
        If type of object is tree
        this function finds the type of tree
        and adds the new_tree to the correct
        list of trees in neuron.
        """
        from tmd.Tree import Tree
        import numpy as np
        # from tmd.utils import tree_type as td

        if isinstance(new_tree, Tree.Tree):

            if int(np.median(new_tree.t)) in td.keys():
                tree_type = td[int(np.median(new_tree.t))]
            else:
                tree_type = 'undefined'
            getattr(self, tree_type).append(new_tree)

    def copy_neuron(self):
        """
        Returns a deep copy of the Neuron.
        """
        import copy

        return copy.deepcopy(self)

    def is_equal(self, neu):
        '''Tests if all neuron structures are the same'''
        import numpy as np

        eq = np.alltrue([self.soma.is_equal(neu.soma),
                         np.alltrue([t1.is_equal(t2) for t1, t2 in
                                     zip(self.neurites, neu.neurites)])])

        return eq

    def is_same(self, neu):
        '''Tests if all neuron data are the same'''
        import numpy as np

        eq = np.alltrue([self.name == neu.name,
                         self.soma.is_equal(neu.soma),
                         np.alltrue([t1.is_equal(t2) for t1, t2 in
                                     zip(self.neurites, neu.neurites)])])

        return eq

    def simplify(self):
        '''Creates a copy of itself and simplifies all trees
           to create a skeleton of the neuron
        '''
        from tmd.utils import tree_type as td

        neu = Neuron()
        neu.soma = self.soma.copy_soma()

        for tr in self.neurites:
            t = tr.extract_simplified()
            neu.append_tree(t, td)

        return neu
