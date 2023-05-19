"""Test tmd.topology.analysis."""
import os

from numpy import testing as npt

from tmd.Topology import analysis
from tmd.Topology import distances

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")

# Filenames for testing
neuron_ph_1_file = os.path.join(DATA_PATH, "neuron_ph_1.txt")
neuron_ph_2_file = os.path.join(DATA_PATH, "neuron_ph_2.txt")


def test_persistence_image_diff():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    p2 = analysis.load_file(neuron_ph_2_file)
    dist1 = distances.max_persistence_image_diff(p1, p2)
    npt.assert_almost_equal(dist1, 0.74316633, decimal=5)
    dist2 = distances.total_persistence_image_diff(p1, p2)
    npt.assert_almost_equal(dist2, 2836.788888, decimal=5)


def test_betti_diff():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    p2 = analysis.load_file(neuron_ph_2_file)
    dist1 = distances.max_betti_diff(p1, p2)
    npt.assert_almost_equal(dist1, 3.0, decimal=1)
    dist2 = distances.total_betti_diff(p1, p2, num_bins=10)
    npt.assert_almost_equal(dist2, 7.0, decimal=1)


def test_entropy_diff():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    p2 = analysis.load_file(neuron_ph_2_file)
    dist1 = distances.max_entropy_diff(p1, p2)
    npt.assert_almost_equal(dist1, 0.534525, decimal=5)
    dist2 = distances.total_entropy_diff(p1, p2, num_bins=4)
    npt.assert_almost_equal(dist2, 0.294013, decimal=5)
