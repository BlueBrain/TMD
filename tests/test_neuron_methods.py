"""Test tmd.Neuron"""
import numpy as np
import pytest
from numpy import testing as npt

from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT as td


@pytest.fixture
def soma():
    return Soma.Soma([0.0], [0.0], [0.0], [12.0])


@pytest.fixture
def apical_tree():
    return Tree.Tree(
        x=np.array([5.0, 5.0]),
        y=np.array([6.0, 6.0]),
        z=np.array([7.0, 7.0]),
        d=np.array([16.0, 16.0]),
        t=np.array([4, 4]),
        p=np.array([-1, 0]),
    )


@pytest.fixture
def basal_tree():
    return Tree.Tree(
        x=np.array([4.0]),
        y=np.array([5.0]),
        z=np.array([6.0]),
        d=np.array([14.0]),
        t=np.array([3]),
        p=np.array([-1]),
    )


@pytest.fixture
def axon_tree():
    return Tree.Tree(
        x=np.array([3.0]),
        y=np.array([4.0]),
        z=np.array([5.0]),
        d=np.array([12.0]),
        t=np.array([2]),
        p=np.array([-1]),
    )


@pytest.fixture
def neuron(soma, apical_tree):
    neu_test = Neuron.Neuron()
    neu_test.set_soma(soma)
    neu_test.append_tree(apical_tree, td)
    return neu_test


def test_size(neuron):
    assert neuron.size() == 1


def test_get_bounding_box(neuron):
    npt.assert_array_equal(neuron.get_bounding_box(), np.array([[5, 6, 7], [5, 6, 7]]))
