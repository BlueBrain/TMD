'''Test tmd.io'''
import os
from numpy import testing as npt
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


def test_load_swc_neuron():
    neuron = io.load_neuron(basic_file)
    assert neuron.soma.is_equal(soma_test)
    assert len(neuron.apical) == 1
    assert len(neuron.basal) == 1
    assert len(neuron.axon) == 1
    assert neuron.apical[0].is_equal(apical_test)
    assert neuron.basal[0].is_equal(basal_test)
    assert neuron.axon[0].is_equal(axon_test)

    neuron1 = io.load_neuron(sample_file)
    assert neuron1.soma.is_equal(soma_test1)
    assert len(neuron1.apical) == 0
    assert len(neuron1.basal) == 1
    assert len(neuron1.axon) == 1

    tree_1 = neuron1.axon[0]
    npt.assert_allclose(tree_1.get_bifurcations(), np.array([10, 20]))
    npt.assert_allclose(tree_1.get_terminations(), np.array([15, 25, 30]))
    try:
        neuron = io.load_neuron(nosecids_file)
        assert False
    except:
        assert True


def test_load_h5_neuron():
    assert neuron_v1.soma.is_equal(neuron_v2.soma)
    assert neuron_v1.basal[0].is_equal(neuron_v2.basal[0])
    assert neuron_v1.axon[0].is_equal(neuron_v2.axon[0])
    try:
        neuron = io.load_neuron(sample_h5_v0_file)
        assert False
    except:
        assert True


def test_io_load():
    #neuron_v1 = io.load_neuron(sample_h5_v1_file)
    #neuron_v2 = io.load_neuron(sample_h5_v2_file)
    assert neuron1.is_equal(neuron_v1)
    assert neuron1.is_equal(neuron_v2)


def test_load_population():
    population = io.load_population(POP_PATH)
    assert len(population.neurons) == 5
    names = np.array([os.path.basename(n.name) for n in population.neurons])

    L = glob.glob(POP_PATH + '/*')
    population1 = io.load_population(L)
    assert len(population.neurons) == 5

    names1 = np.array([os.path.basename(n.name) for n in population1.neurons])
    npt.assert_array_equal(names, names1)
    assert population.neurons[0].is_equal(population1.neurons[0])


def test_tree_type():
    tree_types = {5: 'soma',
                  6: 'axon',
                  7: 'basal',
                  8: 'apical'}

    neuron = io.load_neuron(os.path.join(DATA_PATH, 'basic_exotic_section_types.swc'),
                            soma_type=5,
                            user_tree_types=tree_types)

    def point(section):
        return np.column_stack([section.x, section.y, section.z])

    npt.assert_array_equal(point(neuron.apical[0]), [[3, 4, 5]])
    npt.assert_array_equal(point(neuron.basal[0]), [[4, 5, 6]])
    npt.assert_array_equal(point(neuron.axon[0]), [[5, 6, 7], [5, 6, 7]])
