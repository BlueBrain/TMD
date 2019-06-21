'''
Python module that contains the functions
about reading and writing files.
'''
from __future__ import print_function

import os
import sys
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
    """
    Io method to load an swc or h5 file into a Neuron object.
    TODO: Check if tree is connected to soma, otherwise do
    not include it in neuron structure and warn the user
    that there are disconnected components
    """

    if tree_types is not None:
        td.update(tree_types)

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
    else:
        return None

    # Get soma ids
    soma_ids = [node[SWC_DCT['index']] for node in data if node[SWC_DCT['type']] == soma_index]
    soma_ids = _np.array(soma_ids, dtype=_np.int)
    if len(soma_ids) == 0:
        print("WARNING: file contains no soma.", file=sys.stderr)

    # Extract soma information from swc;
    # This should normally work even if len(soma_ids) == 0
    soma = Soma.Soma(x=data[soma_ids, SWC_DCT['x']],
                     y=data[soma_ids, SWC_DCT['y']],
                     z=data[soma_ids, SWC_DCT['z']],
                     d=data[soma_ids, SWC_DCT['radius']])

    # Save soma in Neuron
    neuron.set_soma(soma)
    soma_ids = set(soma_ids)

    # Extract neurites by computing connected components of the adjacency matrix
    def construct_adj():
        """ Construct adjacency matrix for all nodes except the soma

        The adjacency matrix will be such that the nodes belonging to
        a given neurite end up in the same connected component.

        Nodes that either connect to soma, or have no parent get
        self-connections so that they end up in the same right connected
        component.
        """

        # Construct (child, parent) pairs
        pairs = data[:, [SWC_DCT['index'], SWC_DCT['parent']]].astype(_np.int)
        max_index = pairs.max()
        # Remove points belonging to soma
        pairs = [pair for pair in pairs if pair[0] not in soma_ids]
        # Add self-connections to end-points of trees
        pairs = [(p0, p1) if p1 not in soma_ids and p1 != -1 else (p0, p0) for p0, p1 in pairs]
        pairs = _np.array(pairs)

        # Prepare data for sparse matrix construction
        values = _np.ones(shape=len(pairs))
        dim = max_index + 1

        return sp.coo_matrix((values, pairs.T), shape=(dim, dim))

    def get_connected_components(adj):
        """ Extracts non-trivial connected components from adj disregarding the soma """

        n_conn, conn = cs.connected_components(adj)
        indices = [_np.where(conn == c)[0] for c in range(n_conn)]

        # Remove trivial components originating from the soma
        indices = [idx for idx in indices if len(idx) > 1]

        return indices

    # Extract trees
    for tree_ids in get_connected_components(construct_adj()):
        selector = _np.vectorize(lambda x: x in tree_ids)
        selection = selector(data[:, SWC_DCT['index']].astype(_np.int))
        tree = make_tree(data[selection])
        neuron.append_tree(tree, td=td)

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
