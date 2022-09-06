"""Test tmd.io.h5."""
# pylint: disable=protected-access
import os

import h5py
import numpy as np
from numpy import testing as npt

from tmd.io import h5

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")

# Filenames for testing
basic_file = os.path.join(DATA_PATH, "basic.swc")
nosecids_file = os.path.join(DATA_PATH, "basic_no_sec_ids.swc")
sample_file = os.path.join(DATA_PATH, "sample.swc")

sample_h5_v1_file = os.path.join(DATA_PATH, "sample_v1.h5")
sample_h5_v2_file = os.path.join(DATA_PATH, "sample_v2.h5")
sample_h5_v0_file = os.path.join(DATA_PATH, "sample_v0.h5")

h5file_v0 = h5py.File(sample_h5_v0_file, mode="r")
h5file_v1 = h5py.File(sample_h5_v1_file, mode="r")
h5file_v2 = h5py.File(sample_h5_v2_file, mode="r")

points_1 = np.array(
    [
        [0.0, 0.0, 0.0, 6.0],
        [0.0, 0.0, 0.0, 0.40000001],
        [0.0, 1.0, 0.0, 0.40000001],
        [0.0, 2.0, 0.0, 0.40000001],
        [0.0, 3.0, 0.0, 0.40000001],
        [0.0, 4.0, 0.0, 0.40000001],
        [0.0, 5.0, 0.0, 0.40000001],
        [0.0, 6.0, 0.0, 0.40000001],
        [0.0, 7.0, 0.0, 0.40000001],
        [0.0, 8.0, 0.0, 0.40000001],
    ]
)
groups_1 = np.array(
    [
        [0, 1, -1],
        [1, 2, 0],
        [12, 2, 1],
        [18, 2, 1],
        [24, 2, 3],
        [30, 2, 3],
        [36, 3, 0],
        [47, 3, 6],
        [53, 3, 6],
    ]
)

data_1 = np.array(
    [
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, -1.0],
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.40000001, 0.0],
        [2.0, 2.0, 0.0, 1.0, 0.0, 0.40000001, 1.0],
        [3.0, 2.0, 0.0, 2.0, 0.0, 0.40000001, 2.0],
        [4.0, 2.0, 0.0, 3.0, 0.0, 0.40000001, 3.0],
        [5.0, 2.0, 0.0, 4.0, 0.0, 0.40000001, 4.0],
        [6.0, 2.0, 0.0, 5.0, 0.0, 0.40000001, 5.0],
        [7.0, 2.0, 0.0, 6.0, 0.0, 0.40000001, 6.0],
        [8.0, 2.0, 0.0, 7.0, 0.0, 0.40000001, 7.0],
        [9.0, 2.0, 0.0, 8.0, 0.0, 0.40000001, 8.0],
    ]
)

bx1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
by1 = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
bz1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
bd1 = np.array(
    [
        6.0,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
        0.40000001,
    ]
)
bt1 = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
bp1 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
bch1 = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: []}


def test_h5_dict():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert h5.h5_dct == {"PX": 0, "PY": 1, "PZ": 2, "PD": 3, "GPFIRST": 0, "GTYPE": 1, "GPID": 2}


def test_find_group():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    g0 = h5._find_group(0, groups_1)
    npt.assert_allclose(g0, [0, 1, -1])

    g12 = h5._find_group(12, groups_1)
    g13 = h5._find_group(13, groups_1)
    npt.assert_allclose(g12, [12, 2, 1])
    npt.assert_allclose(g13, [12, 2, 1])


def test_find_parent_id():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    pid0 = h5._find_parent_id(0, groups_1)
    assert pid0 == -1

    pid0 = h5._find_parent_id(53, groups_1)
    assert pid0 == 46

    pid0 = h5._find_parent_id(59, groups_1)
    assert pid0 == 58


def test_find_last_point():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    lp0 = h5._find_last_point(0, groups_1, points_1)
    assert lp0 == 0

    lp1 = h5._find_last_point(1, groups_1, points_1)
    assert lp1 == 11

    lp2 = h5._find_last_point(2, groups_1, points_1)
    assert lp2 == 17

    lp8 = h5._find_last_point(8, groups_1, points_1)
    assert lp8 == 9


def test_remove_duplicate_points():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    points, groups = h5._unpack_v1(h5file_v1)
    p1, g1 = h5.remove_duplicate_points(points, groups)
    assert len(p1) == 53
    npt.assert_allclose(
        np.transpose(g1)[0], np.array([0.0, 1.0, 12.0, 17.0, 22.0, 27.0, 32.0, 43.0, 48.0])
    )
    npt.assert_allclose(np.transpose(g1)[1], np.transpose(groups)[1])
    npt.assert_allclose(np.transpose(g1)[2], np.transpose(groups)[2])


def test_get_h5_version():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    version_0 = h5._get_h5_version(h5file_v0)
    version_1 = h5._get_h5_version(h5file_v1)
    version_2 = h5._get_h5_version(h5file_v2)
    assert version_0 is None
    assert version_1 == 1
    assert version_2 == 2


def test_unpack_v1():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    _, groups = h5._unpack_v1(h5file_v1)
    npt.assert_allclose(groups[:10], groups_1)


def test_unpack_v2():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    _, groups = h5._unpack_v2(h5file_v2, "2")
    npt.assert_allclose(groups[:10], groups_1)


def test_unpack_data():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    data = h5._unpack_data(points_1, groups_1)
    npt.assert_allclose(data, data_1)


def test_read_h5():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    data_v1 = h5.read_h5(sample_h5_v1_file)
    data_v2 = h5.read_h5(sample_h5_v2_file)
    npt.assert_allclose(data_v1[:10], data_1)
    npt.assert_allclose(data_v2[:10], data_1)


def test_h5_data_to_lists():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    x1, y1, z1, d1, t1, p1, ch1 = h5.h5_data_to_lists(data_1)
    npt.assert_allclose(x1, bx1)
    npt.assert_allclose(y1, by1)
    npt.assert_allclose(z1, bz1)
    npt.assert_allclose(d1, bd1)
    npt.assert_allclose(t1, bt1)
    npt.assert_allclose(p1, bp1)
    assert ch1 == bch1
