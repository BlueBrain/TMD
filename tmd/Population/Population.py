'''
tmd class : Population
'''
from tmd.Neuron import Neuron


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
        '''Get axon'''
        return [a for n in self.neurons for a in n.axon]

    @property
    def apical(self):
        '''Get apical'''
        return [a for n in self.neurons for a in n.apical]

    @property
    def basal(self):
        '''Get basal'''
        return [a for n in self.neurons for a in n.basal]

    @property
    def undefined(self):
        '''I dont know'''
        return [a for n in self.neurons for a in n.undefined]

    @property
    def neurites(self):
        '''Get neurites'''
        return self.apical + self.axon + self.basal + self.undefined

    @property
    def dendrites(self):
        '''Get dendrites'''
        return self.apical + self.basal

    def append_neuron(self, new_neuron):
        """
        If type of object is neuron it adds
        the new_neuron to the list of neurons
        of the population.
        """
        if isinstance(new_neuron, Neuron.Neuron):
            self.neurons.append(new_neuron)
