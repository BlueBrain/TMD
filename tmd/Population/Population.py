'''
tmd class : Population
'''


class Population(object):
    """
    A Population object is a container for Neurons.
    """

    def __init__(self, name='Pop'):
        '''Creates an empty Population object.
        '''
        self.neurons = []
        self.name = name

    @property
    def axon(self):
        return [a for n in self.neurons for a in n.axon]

    @property
    def apical(self):
        return [a for n in self.neurons for a in n.apical]

    @property
    def basal(self):
        return [a for n in self.neurons for a in n.basal]

    @property
    def undefined(self):
        return [a for n in self.neurons for a in n.undefined]

    @property
    def neurites(self):
        return self.apical + self.axon + self.basal + self.undefined

    @property
    def dendrites(self):
        return self.apical + self.basal

    def append_neuron(self, new_neuron):
        """
        If type of object is neuron it adds
        the new_neuron to the list of neurons
        of the population.
        """
        from tmd.Neuron import Neuron

        if isinstance(new_neuron, Neuron.Neuron):
            self.neurons.append(new_neuron)

    def extract_ph(self, neurite_type='all', output_folder='./',
                   feature='radial_distances'):
        """Extract the persistent homology of all
           neurites in the population and saves
           them in files according to the tree type.
        """
        from tmd.Topology.methods import extract_ph as eph
        import os

        def try_except(tree, ntree, feature, output_folder, ttype='basal'):
            '''Try except extract ph from tree.
            '''
            try:
                eph(tree, feature=feature,
                    output_file=os.path.join(output_folder, ttype +
                                             '_' + str(ntree) + '.txt'))
            except ValueError:
                print(tree)

        if neurite_type == 'all':
            _ = [try_except(ap, enap, feature, output_folder, ttype='apical')
                 for enap, ap in enumerate(self.apicals)]

            _ = [try_except(ax, enax, feature, output_folder, ttype='axon')
                 for enax, ax in enumerate(self.axons)]

            _ = [try_except(bas, enbas, feature, output_folder, ttype='basal')
                 for enbas, bas in enumerate(self.basals)]

        else:
            _ = [try_except(ax, enax, feature, output_folder, ttype=neurite_type)
                 for enax, ax in enumerate(getattr(self, neurite_type + 's'))]

    def extract_ph_names(self, neurite_type='all', output_folder='./',
                         feature='radial_distances'):
        """Extract the persistent homology of all
           neurites in the population and saves
           them in files according to the tree type.
        """
        from tmd.Topology.methods import extract_ph as eph
        import os

        def try_except(tree, ntree, feature, output_folder, ttype='basal'):
            '''Try except extract ph from tree.
            '''
            try:
                eph(tree, feature=feature,
                    output_file=os.path.join(output_folder, ttype +
                                             '_' + str(ntree) + '.txt'))
            except ValueError:
                print(tree)

        if neurite_type == 'all':
            _ = [[try_except(ap, enap, feature, output_folder,
                             ttype='apical_' + n.name.split('/')[-1])
                  for enap, ap in enumerate(n.apical)] for n in self.neurons]

            _ = [[try_except(ax, enax, feature, output_folder,
                             ttype='axon_' + n.name.split('/')[-1])
                  for enax, ax in enumerate(n.axon)] for n in self.neurons]

            _ = [[try_except(bas, enbas, feature, output_folder,
                             ttype='basal_' + n.name.split('/')[-1])
                  for enbas, bas in enumerate(n.basal)] for n in self.neurons]

        else:
            _ = [[try_except(ax, enax, feature, output_folder,
                             ttype=neurite_type + '_' + n.name.split('/')[-1])
                  for enax, ax in enumerate(getattr(n, neurite_type + 's'))] for n in self.neurons]

    def extract_ph_neurons(self, neurite_type='all', output_folder='./',
                           feature='radial_distances'):
        """Extract the persistent homology of all
           neurites in the population and saves
           them in files according to the tree type.
        """
        from tmd.Topology.methods import extract_ph as eph
        import os

        def try_except(neuron, feature, output_folder, ttype=neurite_type):
            '''Try except extract ph from tree.
            '''
            try:
                eph(neuron, feature=feature, function='get_ph_neuron',
                    neurite_type=ttype,
                    output_file=os.path.join(output_folder, neuron.name + '.txt'))

            except ValueError:
                print(neuron.name)

        _ = [try_except(n, feature, output_folder, ttype=neurite_type)
             for n in self.neurons]
