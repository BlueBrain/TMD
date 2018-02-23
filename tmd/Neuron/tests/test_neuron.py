'''Test tmd.Neuron'''
from nose import tools as nt
import numpy as np
from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import tree_type as td

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
    nt.ok_(neu1.name == 'Neuron')
    nt.ok_(isinstance(neu1.soma, Soma.Soma))
    nt.ok_(neu1.axon == [])
    nt.ok_(neu1.basal == [])
    nt.ok_(neu1.apical == [])
    nt.ok_(neu1.neurites == [])
    neu1 = Neuron.Neuron(name='test')
    nt.ok_(neu1.name == 'test')

def test_neuron_rename():
    neu1 = Neuron.Neuron()
    neu1.rename('test')
    nt.ok_(neu1.name == 'test')

def test_copy_neuron():
    neu1 = Neuron.Neuron()
    neu2 = neu1.copy_neuron()
    nt.ok_(neu1.is_equal(neu2))
    nt.ok_(neu1 != neu2)

def test_neuron_is_equal():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(apical_test, td)
    nt.ok_(neu1.is_equal(neu_test))
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test1)
    neu1.append_tree(apical_test, td)
    nt.ok_(not neu1.is_equal(neu_test))
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(basal_test, td)
    nt.ok_(not neu1.is_equal(neu_test))

def test_neuron_is_same():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    neu1.append_tree(apical_test, td)
    nt.ok_(neu1.is_same(neu_test))
    neu1.name = 'test_not_same'
    nt.ok_(not neu1.is_same(neu_test))

def test_neuron_set_soma():
    neu1 = Neuron.Neuron()
    neu1.set_soma(soma_test)
    nt.ok_(neu1.soma.is_equal(soma_test))

def test_append_tree():
    neu1 = Neuron.Neuron()
    neu1.append_tree(apical_test, td)
    nt.ok_(len(neu1.neurites) == 1)
    neu1.append_tree(basal_test, td)
    neu1.append_tree(axon_test, td)
    nt.ok_(len(neu1.neurites) == 3)
    nt.ok_(len(neu1.basal) == 1)
    nt.ok_(len(neu1.axon) == 1)
    nt.ok_(len(neu1.apical) == 1)
