'''
tmd class : Population
'''
import warnings
from tmd.Neuron import Neuron


class Population(object):
    """
    A Population object is a container for Neurons.
    """

    def __init__(self, name='Pop', neurons=None):
        '''Creates an empty Population object.
        '''
        self.neurons = []
        self.name = name

        if neurons:
            for neuron in neurons:
                self.append_neuron(neuron)

    @property
    def axon(self):
        '''Get axon'''
        return [a for n in self.neurons for a in n.axon]

    @property
    def apical(self):
        '''Get apical'''
        warnings.warn(
            "The 'apical' property is deprecated, please use 'apical_dendrite' instead",
            DeprecationWarning
        )
        return self.apical_dendrite

    @property
    def apical_dendrite(self):
        '''Get apical'''
        return [a for n in self.neurons for a in n.apical_dendrite]

    @property
    def basal(self):
        '''Get basal'''
        warnings.warn(
            "The 'basal' property is deprecated, please use 'basal_dendrite' instead",
            DeprecationWarning
        )
        return self.basal_dendrite

    @property
    def basal_dendrite(self):
        '''Get basal'''
        return [a for n in self.neurons for a in n.basal_dendrite]

    @property
    def undefined(self):
        '''I dont know'''
        return [a for n in self.neurons for a in n.undefined]

    @property
    def neurites(self):
        '''Get neurites'''
        return self.apical_dendrite + self.axon + self.basal_dendrite + self.undefined

    @property
    def dendrites(self):
        '''Get dendrites'''
        return self.apical_dendrite + self.basal_dendrite

    def append_neuron(self, new_neuron):
        """
        If type of object is neuron it adds
        the new_neuron to the list of neurons
        of the population.
        """
        if isinstance(new_neuron, Neuron.Neuron):
            self.neurons.append(new_neuron)
