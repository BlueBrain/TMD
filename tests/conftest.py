"""Configuration for the pytest test suite."""
# pylint: disable=redefined-outer-name
import os

import numpy as np
import pytest

from tmd.io import io
from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Topology import analysis
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT as td

_path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def DATA_PATH():
    """Path to the directory containing the data used in tests."""
    return os.path.join(_path, "data")


@pytest.fixture
def POP_PATH(DATA_PATH):
    """Path to the directory containing the morphologies used to create the population in tests."""
    return os.path.join(DATA_PATH, "valid")


@pytest.fixture
def basic_file(DATA_PATH):
    """Path to a basic morphology."""
    return os.path.join(DATA_PATH, "basic.swc")


@pytest.fixture
def basic_nosecids_file(DATA_PATH):
    """Path to a basic morphology with duplicated IDs."""
    return os.path.join(DATA_PATH, "basic_no_sec_ids.swc")


@pytest.fixture
def sample_file(DATA_PATH):
    """Path to a sample morphology."""
    return os.path.join(DATA_PATH, "sample.swc")


@pytest.fixture
def sample_h5_v0_file(DATA_PATH):
    """Path to a sample morphology in HDF5 V0 format."""
    return os.path.join(DATA_PATH, "sample_v0.h5")


@pytest.fixture
def sample_h5_v1_file(DATA_PATH):
    """Path to a sample morphology in HDF5 V1 format."""
    return os.path.join(DATA_PATH, "sample_v1.h5")


@pytest.fixture
def sample_h5_v2_file(DATA_PATH):
    """Path to a sample morphology in HDF5 V2 format."""
    return os.path.join(DATA_PATH, "sample_v2.h5")


@pytest.fixture
def neuron_v1(sample_h5_v1_file):
    """A sample morphology loaded from HDF5 V1 format."""
    return io.load_neuron(sample_h5_v1_file)


@pytest.fixture
def neuron_v2(sample_h5_v2_file):
    """A sample morphology loaded from HDF5 V2 format."""
    return io.load_neuron(sample_h5_v2_file)


@pytest.fixture
def neuron1(sample_file):
    """A sample morphology loaded from SWC format."""
    return io.load_neuron(sample_file)


@pytest.fixture
def soma_test():
    """A simple Soma."""
    return Soma.Soma([0.0], [0.0], [0.0], [12.0])


@pytest.fixture
def soma_test1():
    """A simple Soma."""
    return Soma.Soma([0.0], [0.0], [0.0], [6.0])


@pytest.fixture
def apical_test():
    """A simple apical Tree."""
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
    """A simple basal Tree."""
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
    """A simple axon Tree."""
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
    """A neuron build from the simple soma and axon tree."""
    neu_test = Neuron.Neuron()
    neu_test.set_soma(soma_test)
    neu_test.append_tree(apical_test, td)
    return neu_test


@pytest.fixture
def population(POP_PATH):
    """A small population."""
    return io.load_population(POP_PATH)


@pytest.fixture
def neuron_ph_1(DATA_PATH):
    """A raw persistence image data."""
    return analysis.load_file(os.path.join(DATA_PATH, "neuron_ph_1.txt"))


@pytest.fixture
def neuron_ph_2(DATA_PATH):
    """Another raw persistence image data."""
    return analysis.load_file(os.path.join(DATA_PATH, "neuron_ph_2.txt"))
