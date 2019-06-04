'''Test tmd.io.h5'''
import os
from nose import tools as nt
from tmd.io import h5
import numpy as np
import h5py

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')

# Filenames for testing
basic_file = os.path.join(DATA_PATH, 'basic.swc')
nosecids_file = os.path.join(DATA_PATH, 'basic_no_sec_ids.swc')
sample_file = os.path.join(DATA_PATH, 'sample.swc')

sample_h5_v1_file = os.path.join(DATA_PATH, 'sample_v1.h5')
sample_h5_v2_file = os.path.join(DATA_PATH, 'sample_v2.h5')
sample_h5_v0_file = os.path.join(DATA_PATH, 'sample_v0.h5')

h5file_v0 = h5py.File(sample_h5_v0_file, mode='r')
h5file_v1 = h5py.File(sample_h5_v1_file, mode='r')
h5file_v2 = h5py.File(sample_h5_v2_file, mode='r')

points_1 = np.array([[0.,   0.,   0.,   6.],
                     [0.,   0.,   0.,   0.40000001],
                     [0.,   1.,   0.,   0.40000001],
                     [0.,   2.,   0.,   0.40000001],
                     [0.,   3.,   0.,   0.40000001],
                     [0.,   4.,   0.,   0.40000001],
                     [0.,   5.,   0.,   0.40000001],
                     [0.,   6.,   0.,   0.40000001],
                     [0.,   7.,   0.,   0.40000001],
                     [0.,   8.,   0.,   0.40000001]])
groups_1 = np.array([[0,  1, -1],
                     [1,  2,  0],
                     [12,  2,  1],
                     [18,  2,  1],
                     [24,  2,  3],
                     [30,  2,  3],
                     [36,  3,  0],
                     [47,  3,  6],
                     [53,  3,  6]])

data_1 = np.array([[0.,  1.,  0.,  0.,  0.,
                    6., -1.],
                   [1.,  2.,  0.,  0.,  0.,
                    0.40000001,  0.],
                   [2.,  2.,  0.,  1.,  0.,
                    0.40000001,  1.],
                   [3.,  2.,  0.,  2.,  0.,
                    0.40000001,  2.],
                   [4.,  2.,  0.,  3.,  0.,
                    0.40000001,  3.],
                   [5.,  2.,  0.,  4.,  0.,
                    0.40000001,  4.],
                   [6.,  2.,  0.,  5.,  0.,
                    0.40000001,  5.],
                   [7.,  2.,  0.,  6.,  0.,
                    0.40000001,  6.],
                   [8.,  2.,  0.,  7.,  0.,
                    0.40000001,  7.],
                   [9.,  2.,  0.,  8.,  0.,
                    0.40000001,  8.]])

bx1 = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
by1 = np.array([0.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
bz1 = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
bd1 = np.array([6.,  0.40000001,  0.40000001,  0.40000001,  0.40000001,
                0.40000001,  0.40000001,  0.40000001,  0.40000001,  0.40000001])
bt1 = np.array([1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
bp1 = np.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
bch1 = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: []}


def test_h5_dict():
    nt.ok_(h5.h5_dct == {'PX': 0,
                         'PY': 1,
                         'PZ': 2,
                         'PD': 3,
                         'GPFIRST': 0,
                         'GTYPE': 1,
                         'GPID': 2})


def test_find_group():
    g0 = h5._find_group(0, groups_1)
    nt.ok_(np.allclose(g0, [0,  1, -1]))
    g12 = h5._find_group(12, groups_1)
    g13 = h5._find_group(13, groups_1)
    nt.ok_(np.allclose(g12, [12,  2,  1]))
    nt.ok_(np.allclose(g13, [12,  2,  1]))


def test_find_parent_id():
    pid0 = h5._find_parent_id(0, groups_1)
    nt.ok_(pid0 == -1)
    pid0 = h5._find_parent_id(53, groups_1)
    nt.ok_(pid0 == 46)
    pid0 = h5._find_parent_id(59, groups_1)
    nt.ok_(pid0 == 58)


def test_find_last_point():
    lp0 = h5._find_last_point(0, groups_1, points_1)
    nt.ok_(lp0 == 0)
    lp1 = h5._find_last_point(1, groups_1, points_1)
    nt.ok_(lp1 == 11)
    lp2 = h5._find_last_point(2, groups_1, points_1)
    nt.ok_(lp2 == 17)
    lp8 = h5._find_last_point(8, groups_1, points_1)
    nt.ok_(lp8 == 9)


def test_remove_duplicate_points():
    points, groups = h5._unpack_v1(h5file_v1)
    p1, g1 = h5.remove_duplicate_points(points, groups)
    nt.ok_(len(p1) == 53)
    nt.ok_(np.allclose(np.transpose(g1)[0],
                       np.array([0.,   1.,  12.,  17.,
                                 22.,  27.,  32.,  43.,  48.])))
    nt.ok_(np.allclose(np.transpose(g1)[1], np.transpose(groups)[1]))
    nt.ok_(np.allclose(np.transpose(g1)[2], np.transpose(groups)[2]))


def test_get_h5_version():
    version_0 = h5._get_h5_version(h5file_v0)
    version_1 = h5._get_h5_version(h5file_v1)
    version_2 = h5._get_h5_version(h5file_v2)
    nt.ok_(version_0 is None)
    nt.ok_(version_1 == 1)
    nt.ok_(version_2 == 2)


def test_unpack_v1():
    points, groups = h5._unpack_v1(h5file_v1)
    nt.ok_(np.allclose(groups[:10], groups_1))


def test_unpack_v2():
    points, groups = h5._unpack_v2(h5file_v2, '2')
    nt.ok_(np.allclose(groups[:10], groups_1))


def test_unpack_data():
    data = h5._unpack_data(points_1, groups_1)
    nt.ok_(np.allclose(data, data_1))


def test_read_h5():
    data_v1 = h5.read_h5(sample_h5_v1_file)
    data_v2 = h5.read_h5(sample_h5_v2_file)
    nt.ok_(np.allclose(data_v1[:10], data_1))
    nt.ok_(np.allclose(data_v2[:10], data_1))


def test_h5_data_to_lists():
    x1, y1, z1, d1, t1, p1, ch1 = h5.h5_data_to_lists(data_1)
    nt.ok_(np.allclose(x1, bx1))
    nt.ok_(np.allclose(y1, by1))
    nt.ok_(np.allclose(z1, bz1))
    nt.ok_(np.allclose(d1, bd1))
    nt.ok_(np.allclose(t1, bt1))
    nt.ok_(np.allclose(p1, bp1))
    nt.ok_(ch1 == bch1)
