'''Test tmd.io'''
import os
from nose import tools as nt
from numpy.testing import assert_array_equal
from tmd.io import io
import numpy as np
from tmd.Soma import Soma
from tmd.Tree import Tree
import glob

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')
POP_PATH = os.path.join(DATA_PATH, 'valid')

# Filenames for testing
basic_file = os.path.join(DATA_PATH, 'basic.swc')
nosecids_file = os.path.join(DATA_PATH, 'basic_no_sec_ids.swc')
sample_file = os.path.join(DATA_PATH, 'sample.swc')

sample_h5_v1_file = os.path.join(DATA_PATH, 'sample_v1.h5')
sample_h5_v2_file = os.path.join(DATA_PATH, 'sample_v2.h5')
sample_h5_v0_file = os.path.join(DATA_PATH, 'sample_v0.h5')

neuron_v1 = io.load_neuron(sample_h5_v1_file)
neuron_v2 = io.load_neuron(sample_h5_v2_file)
neuron1 = io.load_neuron(sample_file)

soma_test = Soma.Soma([0.], [0.], [0.], [12.])
soma_test1 = Soma.Soma([0.], [0.], [0.], [6.])
apical_test = Tree.Tree(x=np.array([5., 5.]), y=np.array([6., 6.]), z=np.array([7., 7.]),
                        d=np.array([16., 16.]), t=np.array([4, 4]), p=np.array([-1,  0]))
basal_test = Tree.Tree(x=np.array([4.]), y=np.array([5.]), z=np.array([6.]),
                       d=np.array([14.]), t=np.array([3]), p=np.array([-1]))
axon_test = Tree.Tree(x=np.array([3.]), y=np.array([4.]), z=np.array([5.]),
                      d=np.array([12.]), t=np.array([2]), p=np.array([-1]))


def test_type_dict():
    nt.ok_(io.TYPE_DCT == {'soma': 1,
                           'basal': 3,
                           'apical': 4,
                           'axon': 2})


def test_load_swc_neuron():
    neuron = io.load_neuron(basic_file)
    nt.ok_(neuron.soma.is_equal(soma_test))
    nt.ok_(len(neuron.apical) == 1)
    nt.ok_(len(neuron.basal) == 1)
    nt.ok_(len(neuron.axon) == 1)
    nt.ok_(neuron.apical[0].is_equal(apical_test))
    nt.ok_(neuron.basal[0].is_equal(basal_test))
    nt.ok_(neuron.axon[0].is_equal(axon_test))
    neuron1 = io.load_neuron(sample_file)
    nt.ok_(neuron1.soma.is_equal(soma_test1))
    nt.ok_(len(neuron1.apical) == 0)
    nt.ok_(len(neuron1.basal) == 1)
    nt.ok_(len(neuron1.axon) == 1)
    tree_1 = neuron1.axon[0]
    nt.ok_(np.allclose(tree_1.get_bifurcations(), np.array([10, 20])))
    nt.ok_(np.allclose(tree_1.get_terminations(), np.array([15, 25, 30])))
    try:
        neuron = io.load_neuron(nosecids_file)
        nt.ok_(False)
    except:
        nt.ok_(True)


def test_load_h5_neuron():
    nt.ok_(neuron_v1.soma.is_equal(neuron_v2.soma))
    nt.ok_(neuron_v1.basal[0].is_equal(neuron_v2.basal[0]))
    nt.ok_(neuron_v1.axon[0].is_equal(neuron_v2.axon[0]))
    try:
        neuron = io.load_neuron(sample_h5_v0_file)
        nt.ok_(False)
    except:
        nt.ok_(True)


def test_io_load():
    #neuron_v1 = io.load_neuron(sample_h5_v1_file)
    #neuron_v2 = io.load_neuron(sample_h5_v2_file)
    nt.ok_(neuron1.is_equal(neuron_v1))
    nt.ok_(neuron1.is_equal(neuron_v2))


def test_load_population():
    population = io.load_population(POP_PATH)
    nt.ok_(len(population.neurons) == 5)
    names = np.array([os.path.basename(n.name) for n in population.neurons])

    L = glob.glob(POP_PATH + '/*')
    population1 = io.load_population(L)
    nt.ok_(len(population.neurons) == 5)
    names1 = np.array([os.path.basename(n.name) for n in population1.neurons])
    assert_array_equal(names, names1)
    nt.ok_(population.neurons[0].is_equal(population1.neurons[0]))


def test_tree_type():
    tree_types = {5: 'soma',
                  6: 'axon',
                  7: 'basal',
                  8: 'apical'}

    neuron = io.load_neuron(os.path.join(DATA_PATH, 'basic_exotic_section_types.swc'),
                            soma_type=5,
                            tree_types=tree_types)

    def point(section):
        return np.column_stack([section.x, section.y, section.z])
    assert_array_equal(point(neuron.apical[0]), [[3, 4, 5]])
    assert_array_equal(point(neuron.basal[0]), [[4, 5, 6]])
    assert_array_equal(point(neuron.axon[0]), [[5, 6, 7], [5, 6, 7]])
