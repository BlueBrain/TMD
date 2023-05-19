"""Test tmd.topology.analysis."""
from numpy import testing as npt

from tmd.Topology import distances


def test_persistence_image_diff(neuron_ph_1, neuron_ph_2):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dist1 = distances.max_persistence_image_diff(neuron_ph_1, neuron_ph_2)
    npt.assert_almost_equal(dist1, 0.74316633, decimal=5)
    dist2 = distances.total_persistence_image_diff(neuron_ph_1, neuron_ph_2)
    npt.assert_almost_equal(dist2, 2836.788888, decimal=5)


def test_betti_diff(neuron_ph_1, neuron_ph_2):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dist1 = distances.max_betti_diff(neuron_ph_1, neuron_ph_2)
    npt.assert_almost_equal(dist1, 3.0, decimal=1)
    dist2 = distances.total_betti_diff(neuron_ph_1, neuron_ph_2, num_bins=10)
    npt.assert_almost_equal(dist2, 7.0, decimal=1)


def test_entropy_diff(neuron_ph_1, neuron_ph_2):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dist1 = distances.max_entropy_diff(neuron_ph_1, neuron_ph_2)
    npt.assert_almost_equal(dist1, 0.534525, decimal=5)
    dist2 = distances.total_entropy_diff(neuron_ph_1, neuron_ph_2, num_bins=4)
    npt.assert_almost_equal(dist2, 0.294013, decimal=5)
