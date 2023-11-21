"""Test tmd.io."""
import glob
import os
from pathlib import Path

import numpy as np
import pytest
from numpy import testing as npt

from tmd.io import io


def test_load_swc_neuron(
    soma_test,
    soma_test1,
    apical_test,
    basal_test,
    axon_test,
    basic_file,
    sample_file,
    basic_nosecids_file,
):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    neuron = io.load_neuron(basic_file)
    assert neuron.soma.is_equal(soma_test)
    assert len(neuron.apical_dendrite) == 1
    assert len(neuron.basal_dendrite) == 1
    assert len(neuron.axon) == 1
    assert neuron.apical_dendrite[0].is_equal(apical_test)
    assert neuron.basal_dendrite[0].is_equal(basal_test)
    assert neuron.axon[0].is_equal(axon_test)

    neuron1 = io.load_neuron(sample_file)
    assert neuron1.soma.is_equal(soma_test1)
    assert len(neuron1.apical_dendrite) == 0
    assert len(neuron1.basal_dendrite) == 1
    assert len(neuron1.axon) == 1

    tree_1 = neuron1.axon[0]
    npt.assert_allclose(tree_1.get_bifurcations(), np.array([10, 20]))
    npt.assert_allclose(tree_1.get_terminations(), np.array([15, 25, 30]))

    with pytest.warns(UserWarning, match=r"The following IDs are duplicated: \[3\.\]"):
        io.load_neuron(basic_nosecids_file)


def test_load_h5_neuron(neuron_v1, neuron_v2, sample_h5_v0_file):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert neuron_v1.soma.is_equal(neuron_v2.soma)
    assert neuron_v1.basal_dendrite[0].is_equal(neuron_v2.basal_dendrite[0])
    assert neuron_v1.axon[0].is_equal(neuron_v2.axon[0])

    with pytest.raises(Exception, match="Not recognized h5 version"):
        io.load_neuron(sample_h5_v0_file)


def test_io_load(neuron1, neuron_v1, neuron_v2):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert neuron1.is_equal(neuron_v1)
    assert neuron1.is_equal(neuron_v2)


def test_load_population(POP_PATH):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    # Testing with a directory
    population = io.load_population(POP_PATH)
    assert len(population.neurons) == 5
    names = np.array([os.path.basename(n.name) for n in population.neurons])

    sample_neuron = population.get_by_name(POP_PATH + "/sample")[0]
    assert sample_neuron.name == POP_PATH + "/sample"

    # Testing with a glob
    L = glob.glob(POP_PATH + "/*")
    population1 = io.load_population(L)
    assert len(population.neurons) == 5

    names1 = np.array([os.path.basename(n.name) for n in population1.neurons])
    npt.assert_array_equal(names, names1)
    assert sample_neuron.is_equal(population1.get_by_name(POP_PATH + "/sample")[0])

    # Testing with a posix path
    population2 = io.load_population(os.path.join(POP_PATH, "sample.swc"))
    assert len(population2.neurons) == 1

    names2 = np.array([os.path.basename(n.name) for n in population2.neurons])
    npt.assert_array_equal(["sample"], names2)
    assert sample_neuron.is_equal(population2.neurons[0])

    # Testing with a list of posix paths
    population3 = io.load_population([os.path.join(POP_PATH, "sample.swc")])
    assert len(population3.neurons) == 1

    names3 = np.array([os.path.basename(n.name) for n in population3.neurons])
    npt.assert_array_equal(["sample"], names3)
    assert sample_neuron.is_equal(population3.neurons[0])

    # Testing with a posix pathlib.Path object
    population4 = io.load_population(Path(os.path.join(POP_PATH, "sample.swc")))
    assert len(population4.neurons) == 1

    names4 = np.array([os.path.basename(n.name) for n in population4.neurons])
    npt.assert_array_equal(["sample"], names4)
    assert sample_neuron.is_equal(population4.neurons[0])

    # Testing with a list of posix pathlib.Path objects
    population5 = io.load_population([Path(os.path.join(POP_PATH, "sample.swc"))])
    assert len(population5.neurons) == 1

    names5 = np.array([os.path.basename(n.name) for n in population5.neurons])
    npt.assert_array_equal(["sample"], names5)
    assert sample_neuron.is_equal(population5.neurons[0])

    # A posix path pointing to a file that does not exist should raise an exception
    with pytest.raises(
        TypeError,
        match=(
            "The format of the given neurons is not supported. "
            "Expected an iterable of files, or a directory, or a single morphology file. "
            f'Got: {os.path.join(POP_PATH, "UNKNOWN")}'
        ),
    ):
        io.load_population(os.path.join(POP_PATH, "UNKNOWN"))

    # A pathlib.Path object pointing to a file that does not exist should raise an exception
    with pytest.raises(
        TypeError,
        match=(
            "The format of the given neurons is not supported. "
            "Expected an iterable of files, or a directory, or a single morphology file. "
            f'Got: {Path(os.path.join(POP_PATH, "UNKNOWN"))}'
        ),
    ):
        io.load_population(Path(os.path.join(POP_PATH, "UNKNOWN")))

    # A wrong type should raise an exception
    with pytest.raises(
        TypeError,
        match=(
            "The format of the given neurons is not supported. "
            "Expected an iterable of files, or a directory, or a single morphology file. "
            f"Got: {0}"
        ),
    ):
        io.load_population(0)


def test_tree_type(DATA_PATH):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    tree_types = {5: "soma", 6: "axon", 7: "basal_dendrite", 8: "apical_dendrite"}

    neuron = io.load_neuron(
        os.path.join(DATA_PATH, "basic_exotic_section_types.swc"),
        soma_type=5,
        user_tree_types=tree_types,
    )

    def point(section):
        return np.column_stack([section.x, section.y, section.z])

    npt.assert_array_equal(point(neuron.apical_dendrite[0]), [[3, 4, 5]])
    npt.assert_array_equal(point(neuron.basal_dendrite[0]), [[4, 5, 6]])
    npt.assert_array_equal(point(neuron.axon[0]), [[5, 6, 7], [5, 6, 7]])


@pytest.mark.parametrize("use_morphio", [True, False])
def test_upper_case(DATA_PATH, use_morphio):
    """Read files with upper-case extensions."""
    path = Path(DATA_PATH) / "upper_case_names"
    if use_morphio:
        files = path
    else:
        files = [i for i in path.iterdir() if i.suffix.lower() != ".asc"]
    io.load_population(files, use_morphio=use_morphio)
