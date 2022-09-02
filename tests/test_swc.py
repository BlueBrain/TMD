"""Test tmd.io.swc."""
# pylint: disable=redefined-outer-name
import os

import numpy as np
import pytest
from numpy import testing as npt

from tmd.io import swc


@pytest.fixture
def nocom_file(DATA_PATH):
    """File with no comment."""
    return os.path.join(DATA_PATH, "basic_no_comments.swc")


@pytest.fixture
def options_file(DATA_PATH):
    """File with basic options."""
    return os.path.join(DATA_PATH, "basic_options.swc")


@pytest.fixture
def basic_data():
    """Basic data."""
    return np.array(
        ["1 1 0 0 0 6 -1", "2 2 3 4 5 6 1", "3 3 4 5 6 7 1", "4 4 5 6 7 8 1", "5 4 5 6 7 8 4"]
    )


@pytest.fixture
def options_data():
    """Basic option data."""
    return np.array(["2", "3", "4", "5", "6", "7", "8", "9\n"])


bx1 = np.array([0.0, 3.0, 4.0, 5.0, 5.0])
by1 = np.array([0.0, 4.0, 5.0, 6.0, 6.0])
bz1 = np.array([0.0, 5.0, 6.0, 7.0, 7.0])
bd1 = np.array([12.0, 12.0, 14.0, 16.0, 16.0])
bt1 = np.array([1, 2, 3, 4, 4])
bp1 = np.array([-1, 0, 0, 0, 3])
bch1 = {0: [1, 2, 3], 1: [], 2: [], 3: [4], 4: []}


def test_swc_dict():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert swc.SWC_DCT == {"index": 0, "parent": 6, "radius": 5, "type": 1, "x": 2, "y": 3, "z": 4}


def test_read_swc(basic_file, basic_data, options_file, options_data):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    data1 = swc.read_swc(basic_file)
    npt.assert_array_equal(data1, basic_data)
    data2 = swc.read_swc(options_file, line_delimiter=" ")
    npt.assert_array_equal(data2, options_data)


def test_swc_data_to_lists(basic_data):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    x1, y1, z1, d1, t1, p1, ch1 = swc.swc_data_to_lists(basic_data)
    npt.assert_allclose(x1, bx1)
    npt.assert_allclose(y1, by1)
    npt.assert_allclose(z1, bz1)
    npt.assert_allclose(d1, bd1)
    npt.assert_allclose(t1, bt1)
    npt.assert_allclose(p1, bp1)
    assert ch1 == bch1
