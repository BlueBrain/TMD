'''Test tmd.Neuron'''
import numpy as np
from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT as td

soma_test = Soma.Soma([0.],[0.],[0.],[12.])
soma_test1 = Soma.Soma([0.],[0.],[0.],[6.])
apical_test = Tree.Tree(x=np.array([5., 5.]), y=np.array([6., 6.]), z=np.array([ 7., 7.]),
                       d=np.array([16., 16.]), t=np.array([4, 4]), p=np.array([-1,  0]))
basal_test = Tree.Tree(x=np.array([4.]), y=np.array([5.]), z=np.array([6.]),
                       d=np.array([14.]), t=np.array([3]), p=np.array([-1]))
axon_test = Tree.Tree(x=np.array([ 3.]), y=np.array([4.]), z=np.array([5.]),
                      d=np.array([12.]), t=np.array([2 ]), p=np.array([-1]))

neu_test = Neuron.Neuron()
neu_test.set_soma(soma_test)
neu_test.append_tree(apical_test, td)

def test_neuron_init_():
    neu1 = Neuron.Neuron()

    assert neu1.name == 'Neuron'
    assert isinstance(neu1.soma, Soma.Soma)
    assert neu1.axon == []
    assert neu1.basal == []
    assert neu1.apical == []
    assert neu1.neurites == []
    neu1 = Neuron.Neuron(name='test')
    assert neu1.name == 'test'

def test_neuron_rename():
    neu1 = Neuron.Neuron()
    neu1.rename('test')
    assert neu1.name == 'test'

def test_copy_neuron():
    neu1 = Neuron.Neuron()
    neu2 = neu1.copy_neuron()
    assert neu1.is_equal(neu2)
    assert neu1 != neu2

def test_neuron_is_equal():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(apical_test, td)
    assert neu1.is_equal(neu_test)

    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test1)
    neu1.append_tree(apical_test, td)
    assert not neu1.is_equal(neu_test)

    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(basal_test, td)
    assert not neu1.is_equal(neu_test)

def test_neuron_is_same():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(apical_test, td)
    assert neu1.is_same(neu_test)

    neu1.name = 'test_not_same'
    assert not neu1.is_same(neu_test)

def test_neuron_set_soma():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    assert neu1.soma.is_equal(soma_test)

def test_append_tree():
    neu1 = Neuron.Neuron()
    neu1.append_tree(apical_test, td)
    assert len(neu1.neurites) == 1

    neu1.append_tree(basal_test, td)
    neu1.append_tree(axon_test, td)
    assert len(neu1.neurites) == 3
    assert len(neu1.basal) == 1
    assert len(neu1.axon) == 1
    assert len(neu1.apical) == 1
