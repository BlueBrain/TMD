"""Test Neuron conversions."""
# pylint: disable=protected-access
import os

import mock
import morphio
import numpy as np
from numpy import testing as npt

from tmd.io import conversion as tested
from tmd.io.io import load_neuron
from tmd.io.io import load_neuron_from_morphio

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")


class MockSection:
    """A Mock for the Section class."""

    def __init__(self, id, points, diameters, type, parent=None):
        # pylint: disable=redefined-builtin
        self.id = id
        self.points = points
        self.diameters = diameters
        self.type = type
        self.parent = parent
        self.traversal = []

    def iter(self):
        # noqa: D102 ; pylint: disable=missing-function-docstring
        return self.traversal

    @property
    def is_root(self):
        # noqa: D102 ; pylint: disable=missing-function-docstring
        return self.parent is None


class MockNeuron:
    """A Mock for the Neuron class."""

    def __init__(self):
        root = MockSection(
            id=0,
            points=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            diameters=np.array([0.1, 0.2, 0.3]),
            type=2,
            parent=None,
        )

        child1 = MockSection(
            id=1,
            points=np.array([[0.7, 0.8, 0.9], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            diameters=np.array([0.3, 0.4, 0.5]),
            type=2,
            parent=root,
        )

        child2 = MockSection(
            id=2,
            points=np.array([[0.7, 0.8, 0.9], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            diameters=np.array([0.3, 0.6, 0.7]),
            type=2,
            parent=root,
        )

        root.traversal = [root, child1, child2]

        self.root_sections = [root]

    @property
    def diameters(self):
        # noqa: D102 ; pylint: disable=missing-function-docstring
        return np.hstack([s.diameters for root in self.root_sections for s in root.iter()])

    @property
    def points(self):
        # noqa: D102 ; pylint: disable=missing-function-docstring
        return np.vstack([s.points for root in self.root_sections for s in root.iter()])

    @property
    def n_points(self):
        # noqa: D102 ; pylint: disable=missing-function-docstring
        return len(self.points)


def test_convert_morphio_soma():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    morphio_soma = mock.Mock(
        points=np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        diameters=np.array([2.1, 3.4]),
    )

    soma = tested.convert_morphio_soma(morphio_soma)

    npt.assert_allclose(soma.x, [0.0, 2.0])
    npt.assert_allclose(soma.y, [1.0, 3.0])
    npt.assert_allclose(soma.z, [2.0, 4.0])
    npt.assert_allclose(soma.d, [2.1, 3.4])


def test_section_to_data():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    section = MockSection(
        id=0,
        points=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        diameters=np.array([1.2, 1.3, 1.4]),
        type=3,
        parent=None,
    )

    n, data = tested._section_to_data(section, tree_length=11, start=0, parent=-1)

    npt.assert_equal(n, 3)
    npt.assert_allclose(data.points, section.points)
    npt.assert_allclose(section.diameters, data.diameters)
    npt.assert_equal(data.section_type, 3)
    npt.assert_array_equal(data.parents, [-1, 11 + 0, 11 + 1])

    n, data = tested._section_to_data(section, tree_length=2, start=1, parent=5)

    npt.assert_equal(n, 2)
    npt.assert_allclose(data.points, section.points[1:])
    npt.assert_allclose(data.diameters, section.diameters[1:])
    npt.assert_equal(data.section_type, 3)
    npt.assert_array_equal(data.parents, [5, 2 + 0])


def test_convert_morphio_trees():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    morphio_neuron = MockNeuron()

    trees = list(tested.convert_morphio_trees(morphio_neuron))

    assert len(trees) == 1

    tree = trees[0]

    npt.assert_allclose(tree.x, [0.1, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7])
    npt.assert_allclose(tree.y, [0.2, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8])
    npt.assert_allclose(tree.z, [0.3, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9])
    npt.assert_allclose(tree.d, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    npt.assert_array_equal(tree.t, [2, 2, 2, 2, 2, 2, 2])
    npt.assert_array_equal(tree.p, [-1, 0, 1, 2, 3, 2, 5])


def _assert_neurons_equal(neuron1, neuron2):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(neuron1.soma.x, neuron2.soma.x)
    npt.assert_allclose(neuron1.soma.y, neuron2.soma.y)
    npt.assert_allclose(neuron1.soma.z, neuron2.soma.z)
    npt.assert_allclose(neuron1.soma.d, neuron2.soma.d)

    for neurite1, neurite2 in zip(neuron1.neurites, neuron2.neurites):
        npt.assert_allclose(neurite1.x, neurite2.x)
        npt.assert_allclose(neurite1.y, neurite2.y)
        npt.assert_allclose(neurite1.z, neurite2.z)
        npt.assert_allclose(neurite1.d, neurite2.d)
        npt.assert_array_equal(neurite1.t, neurite2.t)
        npt.assert_array_equal(neurite1.p, neurite2.p)


def test_neuron_building_consistency__h5():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    path = f"{DATA_PATH}/valid/C010398B-P2.h5"

    neuron1 = load_neuron(path)
    neuron2 = load_neuron_from_morphio(path)

    _assert_neurons_equal(neuron1, neuron2)

    neuron2 = load_neuron_from_morphio(morphio.Morphology(path))

    _assert_neurons_equal(neuron1, neuron2)


def test_neuron_building_consistency__swc():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    path = f"{DATA_PATH}/valid/C010398B-P2.CNG.swc"

    neuron1 = load_neuron(path)
    neuron2 = load_neuron_from_morphio(path)

    _assert_neurons_equal(neuron1, neuron2)

    neuron2 = load_neuron_from_morphio(morphio.Morphology(path))

    _assert_neurons_equal(neuron1, neuron2)
