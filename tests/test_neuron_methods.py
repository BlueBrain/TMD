"""Test tmd.Neuron."""
import numpy as np
from numpy import testing as npt


def test_size(neuron):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert neuron.size() == 1


def test_get_bounding_box(neuron):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_array_equal(neuron.get_bounding_box(), np.array([[5, 6, 7], [5, 6, 7]]))
