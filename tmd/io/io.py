'''
Python module that contains the functions
about reading and writing files.
'''
from __future__ import print_function

import os

import morphio
from morphio import Morphology  # pylint: disable=no-name-in-module

from tmd.Neuron import Neuron
from tmd.Tree import Tree
from tmd.Soma import Soma
from tmd.Population import Population


# Definition of tree types
TYPE_DCT = {'soma': 1,
            'basal': 3,
            'apical': 4,
            'axon': 2}


class LoadNeuronError(Exception):
    '''Captures the exception of failing to load a single neuron
    '''


def load_neuron(input_file, tree_types=None):
    ''' neuron loader '''
    morphology = Morphology(input_file, options=morphio.no_duplicates)

    neuron = Neuron.Neuron(name=input_file.replace('.swc', ''))

    neuron.set_soma(Soma.Soma(x=morphology.soma.points[:, 0],
                              y=morphology.soma.points[:, 1],
                              z=morphology.soma.points[:, 2],
                              d=morphology.soma.diameters))

    for root in morphology.root_sections:
        neuron.append_tree(Tree.Tree.from_morphio(root))

    return neuron


def load_population(neurons, tree_types=None, name=None):
    '''Loads all data of recognised format (swc, h5)
       into a Population object.
       Takes as input a directory or a list of files to load.
    '''
    if isinstance(neurons, (list, tuple)):
        files = neurons
        name = name if name is not None else 'Population'
    elif os.path.isdir(neurons):  # Assumes given input is a directory
        files = [os.path.join(neurons, l) for l in os.listdir(neurons)]
        name = name if name is not None else os.path.basename(neurons)

    pop = Population.Population(name=name)

    files2load = [i for i in files if (i.endswith(".h5") or i.endswith(".swc"))]

    for i in files2load:
        try:
            pop.append_neuron(load_neuron(i, tree_types=tree_types))
        except LoadNeuronError:
            print('File failed to load: {}'.format(i))

    return pop
