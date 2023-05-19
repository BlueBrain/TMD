"""Test tmd.topology.analysis."""
import os

import numpy as np
from numpy import testing as npt

from tmd.Topology import analysis
from tmd.Topology import vectorizations

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")

# Filenames for testing
neuron_ph_1_file = os.path.join(DATA_PATH, "neuron_ph_1.txt")
neuron_ph_2_file = os.path.join(DATA_PATH, "neuron_ph_2.txt")


def test_persistence_image_data():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    dt1 = vectorizations.persistence_image_data(p1)
    npt.assert_equal(np.shape(dt1), (100, 100))
    npt.assert_equal(np.max(dt1), 1.0)
    dt2 = vectorizations.persistence_image_data(p1, resolution=10, norm_factor=1.0)
    npt.assert_equal(np.shape(dt2), (10, 10))
    npt.assert_almost_equal(np.max(dt2), 3.367e-06, decimal=5)


def test_betti_curve():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    dt1 = vectorizations.betti_curve(p1)[0]
    npt.assert_equal(len(dt1), 1000)
    dt1 = vectorizations.betti_curve(p1, num_bins=4)[0]
    npt.assert_equal(len(dt1), 4)
    npt.assert_almost_equal(dt1, [1, 2, 1, 1], decimal=1)


def test_life_entropy_curve():
    """Tests the validity of persistence image data
    with respect to resolution, and norm factor
    """
    p1 = analysis.load_file(neuron_ph_1_file)
    dt1 = vectorizations.life_entropy_curve(p1)[0]
    npt.assert_equal(len(dt1), 1000)
    dt1 = vectorizations.life_entropy_curve(p1, num_bins=4)[0]
    npt.assert_equal(len(dt1), 4)
    npt.assert_almost_equal(dt1, [0.35397, 0.61112, 0.35397, 0.35397], decimal=5)
