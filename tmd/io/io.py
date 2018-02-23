'''
Python module that contains the functions
about reading and writing files.
'''

import numpy as _np
from tmd.io.swc import swc_dct
from tmd.io.swc import read_swc
from tmd.io.swc import swc_to_data
from tmd.io.h5 import read_h5
from tmd.Neuron import Neuron
from tmd.Tree import Tree
from tmd.Soma import Soma

# Definition of tree types
type_dct = {'soma': 1,
            'basal': 3,
            'apical': 4,
            'axon': 2}


def make_tree(data):
    '''Make tree structure from loaded data.
       Returns a tree of tmd.Tree
       type.
    '''
    tr_data = _np.transpose(data)

    parents = [_np.where(tr_data[0] == i)[0][0]
               if len(_np.where(tr_data[0] == i)[0]) > 0
               else - 1 for i in tr_data[6]]

    return Tree.Tree(x=tr_data[swc_dct['x']], y=tr_data[swc_dct['y']], z=tr_data[swc_dct['z']],
                     d=tr_data[swc_dct['radius']], t=tr_data[swc_dct['type']], p=parents)


def load_neuron(input_file, line_delimiter='\n', tree_types=None, remove_duplicates=True):
    '''
    Io method to load an swc or h5 file into a Neuron object.
    TODO: Check if tree is connected to soma, otherwise do
    not include it in neuron structure and warn the user
    that there are disconnected components
    '''
    import os
    from scipy import sparse as sp
    from scipy.sparse import csgraph as cs
    from tmd.utils import tree_type as td

    if tree_types is not None:
        td.update(tree_types)

    # Definition of swc types from type_dict function
    soma_index = type_dct['soma']

    # Make neuron with correct filename and load data
    if os.path.splitext(input_file)[-1] == '.swc':
        # print 'Loading swc file ', input_file
        data = swc_to_data(read_swc(input_file=input_file,
                                    line_delimiter=line_delimiter))
        neuron = Neuron.Neuron(name=input_file.replace('.swc', ''))

    elif os.path.splitext(input_file)[-1] == '.h5':
        # print 'Loading h5 file ', input_file
        data = read_h5(input_file=input_file, remove_duplicates=remove_duplicates)
        neuron = Neuron.Neuron(name=input_file.replace('.h5', ''))

    soma_ids = _np.where(_np.transpose(data)[1] == soma_index)[0]

    # Extract soma information from swc
    soma = Soma.Soma(x=_np.transpose(data)[swc_dct['x']][soma_ids],
                     y=_np.transpose(data)[swc_dct['y']][soma_ids],
                     z=_np.transpose(data)[swc_dct['z']][soma_ids],
                     d=_np.transpose(data)[swc_dct['radius']][soma_ids])

    # Save soma in Neuron
    neuron.set_soma(soma)
    p = _np.array(_np.transpose(data)[6], dtype=int) - _np.transpose(data)[0][0]
    dA = sp.csr_matrix((_np.ones(len(p) - len(soma_ids)),
                        (range(len(soma_ids), len(p)),
                         p[len(soma_ids):])), shape=(len(p), len(p)))

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids):, len(soma_ids):])

    # Extract trees
    for i in xrange(comp[0]):
        tree_ids = _np.where(comp[1] == i)[0] + len(soma_ids)
        tree = make_tree(data[tree_ids])
        neuron.append_tree(tree, td=td)

    return neuron


def load_population(input_directory, tree_types=None):
    '''Loads all data of recognised format (swc, h5)
       into a Population object.
    '''
    import os
    from tmd.Population import Population

    files_h5 = [i for i in os.listdir(input_directory) if i.endswith(".h5")]
    files_swc = [i for i in os.listdir(input_directory) if i.endswith(".swc")]

    pop = Population.Population(name=os.path.relpath(input_directory))

    for i in files_h5 + files_swc:
        print 'Loading ' + i + ' ...'
        pop.append_neuron(load_neuron(os.path.join(input_directory, i),
                                      tree_types=tree_types))

    pop.apicals = [ap for n in pop.neurons for ap in n.apical]
    pop.axons = [ax for n in pop.neurons for ax in n.axon]
    pop.basals = [bas for n in pop.neurons for bas in n.basal]

    return pop
