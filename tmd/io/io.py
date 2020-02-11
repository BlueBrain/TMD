'''
Python module that contains the functions
about reading and writing files.
'''
from __future__ import print_function

import os
import numpy as _np
from scipy import sparse as sp
from scipy.sparse import csgraph as cs
from tmd.io.swc import SWC_DCT
from tmd.io.swc import read_swc
from tmd.io.swc import swc_to_data
from tmd.io.h5 import read_h5
from tmd.Neuron import Neuron
from tmd.Tree import Tree
from tmd.Soma import Soma
from tmd.Population import Population
from tmd.utils import tree_type as td

# Definition of tree types
TYPE_DCT = {'soma': 1,
            'basal': 3,
            'apical': 4,
            'axon': 2}


class LoadNeuronError(Exception):
    '''Captures the exception of failing to load a single neuron
    '''


def make_tree(data):
    '''Make tree structure from loaded data.
       Returns a tree of tmd.Tree
       type.
    '''
    tr_data = _np.transpose(data)

    parents = [_np.where(tr_data[0] == i)[0][0]
               if len(_np.where(tr_data[0] == i)[0]) > 0
               else - 1 for i in tr_data[6]]

    return Tree.Tree(x=tr_data[SWC_DCT['x']], y=tr_data[SWC_DCT['y']], z=tr_data[SWC_DCT['z']],
                     d=tr_data[SWC_DCT['radius']], t=tr_data[SWC_DCT['type']], p=parents)


def load_neuron(input_file, line_delimiter='\n', soma_type=None,
                tree_types=None, remove_duplicates=True):
    '''
    Io method to load an swc or h5 file into a Neuron object.
    TODO: Check if tree is connected to soma, otherwise do
    not include it in neuron structure and warn the user
    that there are disconnected components
    '''

    tree_types_final = td.copy()
    if tree_types is not None:
        tree_types_final.update(tree_types)

    # Definition of swc types from type_dict function
    if soma_type is None:
        soma_index = TYPE_DCT['soma']
    else:
        soma_index = soma_type

    # Make neuron with correct filename and load data
    if os.path.splitext(input_file)[-1] == '.swc':
        data = swc_to_data(read_swc(input_file=input_file,
                                    line_delimiter=line_delimiter))
        neuron = Neuron.Neuron(name=input_file.replace('.swc', ''))

    elif os.path.splitext(input_file)[-1] == '.h5':
        data = read_h5(input_file=input_file, remove_duplicates=remove_duplicates)
        neuron = Neuron.Neuron(name=input_file.replace('.h5', ''))

    try:
        soma_ids = _np.where(_np.transpose(data)[1] == soma_index)[0]
    except IndexError:
        raise LoadNeuronError('Soma points not in the expected format')

    # Extract soma information from swc
    soma = Soma.Soma(x=_np.transpose(data)[SWC_DCT['x']][soma_ids],
                     y=_np.transpose(data)[SWC_DCT['y']][soma_ids],
                     z=_np.transpose(data)[SWC_DCT['z']][soma_ids],
                     d=_np.transpose(data)[SWC_DCT['radius']][soma_ids])

    # Save soma in Neuron
    neuron.set_soma(soma)
    p = _np.array(_np.transpose(data)[6], dtype=int) - _np.transpose(data)[0][0]
    # return p, soma_ids
    try:
        dA = sp.csr_matrix((_np.ones(len(p) - len(soma_ids)),
                           (range(len(soma_ids), len(p)),
                            p[len(soma_ids):])), shape=(len(p), len(p)))
    except Exception:
        raise LoadNeuronError('Cannot create connectivity, nodes not connected correctly.')

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids):, len(soma_ids):])

    # Extract trees
    for i in range(comp[0]):
        tree_ids = _np.where(comp[1] == i)[0] + len(soma_ids)
        tree = make_tree(data[tree_ids])
        neuron.append_tree(tree, td=tree_types_final)

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

    for filename in files:
        try:
            assert filename.endswith(('.h5', '.swc'))
            pop.append_neuron(load_neuron(filename, tree_types=tree_types))
        except AssertionError:
            raise Warning("{} is not a valid h5 or swc file".format(filename))
        except LoadNeuronError:
            print('File failed to load: {}'.format(filename))

    return pop
