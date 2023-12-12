"""Test tmd.topology.analysis."""
import numpy as np
from numpy import testing as npt

from tmd.Topology import vectorizations


def test_get_limits(neuron_ph_1, neuron_ph_2):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = neuron_ph_1.tolist()
    p2 = neuron_ph_2.tolist()
    lims1 = vectorizations.get_limits(p1)
    lims1_2 = vectorizations.get_limits([p1] + [p2])
    npt.assert_almost_equal(lims1, [[155.88089, 633.59656], [0.0, 561.04985]], decimal=5)
    npt.assert_almost_equal(lims1_2, [[40.62391, 633.59656], [0.0, 561.04985]], decimal=5)


def test_persistence_image_data(neuron_ph_1):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dt1 = vectorizations.persistence_image_data(neuron_ph_1)
    npt.assert_equal(np.shape(dt1), (100, 100))
    npt.assert_equal(np.max(dt1), 1.0)
    dt2 = vectorizations.persistence_image_data(neuron_ph_1, resolution=10, norm_factor=1.0)
    npt.assert_equal(np.shape(dt2), (10, 10))
    npt.assert_almost_equal(np.max(dt2), 3.367e-06, decimal=5)


def test_betti_curve(neuron_ph_1):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dt1 = vectorizations.betti_curve(neuron_ph_1)[0]
    npt.assert_equal(len(dt1), 1000)
    dt1 = vectorizations.betti_curve(neuron_ph_1, num_bins=4)[0]
    npt.assert_equal(len(dt1), 4)
    npt.assert_almost_equal(dt1, [1, 2, 1, 1], decimal=1)


def test_life_entropy_curve(neuron_ph_1):
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    dt1 = vectorizations.life_entropy_curve(neuron_ph_1)[0]
    npt.assert_equal(len(dt1), 1000)
    dt1 = vectorizations.life_entropy_curve(neuron_ph_1, num_bins=4)[0]
    npt.assert_equal(len(dt1), 4)
    npt.assert_almost_equal(dt1, [0.35397, 0.61112, 0.35397, 0.35397], decimal=5)
