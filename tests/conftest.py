"""Configuration for the pytest test suite."""
import os

import numpy as np
import pytest

from tmd.io import io
from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT as td

_path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def DATA_PATH():
    return os.path.join(_path, "data")


@pytest.fixture
def POP_PATH(DATA_PATH):
    return os.path.join(DATA_PATH, "valid")


@pytest.fixture
def basic_file(DATA_PATH):
    return os.path.join(DATA_PATH, "basic.swc")


@pytest.fixture
def basic_nosecids_file(DATA_PATH):
    return os.path.join(DATA_PATH, "basic_no_sec_ids.swc")


@pytest.fixture
def sample_file(DATA_PATH):
    return os.path.join(DATA_PATH, "sample.swc")


@pytest.fixture
def sample_h5_v1_file(DATA_PATH):
    return os.path.join(DATA_PATH, "sample_v1.h5")


@pytest.fixture
def sample_h5_v2_file(DATA_PATH):
    return os.path.join(DATA_PATH, "sample_v2.h5")


@pytest.fixture
def sample_h5_v0_file(DATA_PATH):
    return os.path.join(DATA_PATH, "sample_v0.h5")


@pytest.fixture
def neuron_v1(sample_h5_v1_file):
    return io.load_neuron(sample_h5_v1_file)


@pytest.fixture
def neuron_v2(sample_h5_v2_file):
    return io.load_neuron(sample_h5_v2_file)


@pytest.fixture
def neuron1(sample_file):
    return io.load_neuron(sample_file)


@pytest.fixture
def soma_test():
    return Soma.Soma([0.0], [0.0], [0.0], [12.0])


@pytest.fixture
def soma_test1():
    return Soma.Soma([0.0], [0.0], [0.0], [6.0])


@pytest.fixture
def apical_test():
    return Tree.Tree(
        x=np.array([5.0, 5.0]),
        y=np.array([6.0, 6.0]),
        z=np.array([7.0, 7.0]),
        d=np.array([16.0, 16.0]),
        t=np.array([4, 4]),
        p=np.array([-1, 0]),
    )


@pytest.fixture
def basal_test():
    return Tree.Tree(
        x=np.array([4.0]),
        y=np.array([5.0]),
        z=np.array([6.0]),
        d=np.array([14.0]),
        t=np.array([3]),
        p=np.array([-1]),
    )


@pytest.fixture
def axon_test():
    return Tree.Tree(
        x=np.array([3.0]),
        y=np.array([4.0]),
        z=np.array([5.0]),
        d=np.array([12.0]),
        t=np.array([2]),
        p=np.array([-1]),
    )


@pytest.fixture
def neuron(soma_test, apical_test):
    neu_test = Neuron.Neuron()
    neu_test.set_soma(soma_test)
    neu_test.append_tree(apical_test, td)
    return neu_test


@pytest.fixture
def population(POP_PATH):
    return io.load_population(POP_PATH)
