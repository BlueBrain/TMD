'''Test tmd.topology.analysis'''
from nose import tools as nt
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tmd.Topology import analysis
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')

# Filenames for testing
sample_ph_0_file = os.path.join(DATA_PATH, 'sample_ph_0.txt')
sample_ph_1_file = os.path.join(DATA_PATH, 'sample_ph_1.txt')

sample_data_0 = np.array([[10, 8], [9, 8], [9, 8]])
sample_data_1 = np.array([[10, 7], [9, 8], [9, 8]])

def test_load_file():
    f0 = analysis.load_file(sample_ph_0_file)
    f1 = analysis.load_file(sample_ph_1_file)
    nt.ok_(np.allclose(f0, np.array([[ 12.24744871,  11.18033989],
                                     [ 11.18033989,  10.        ],
                                     [ 12.24744871,  0.        ]])))
    nt.ok_(np.allclose(f1, np.array([[ 11.18033989,  10.        ],
                                     [ 11.18033989,  0.        ]])))

def test_histogram_horizontal():
    bins, data = analysis.histogram_horizontal(sample_data_0, num_bins=11, min_bin=8.0, max_bin=10.0)
    nt.ok_(np.allclose(bins, np.array([ 8., 8.2, 8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10. ])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  3.,  3.,  1.,  3.,  1.,  1.,  1.,  1.])))
    bins, data = analysis.histogram_horizontal(sample_data_0, num_bins=10)
    nt.ok_(np.allclose(bins, np.array([ 8.        , 8.22222222,   8.44444444,   8.66666667,
                                        8.88888889, 9.11111111,   9.33333333,   9.55555556,
                                        9.77777778, 10.])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  3.,  3.,  3.,  1.,  1.,  1.,  1.])))
    bins, data = analysis.histogram_horizontal(sample_data_0, num_bins=4, min_bin=7., max_bin=10.)
    nt.ok_(np.allclose(bins, np.array([ 7.,   8.,   9.,  10.])))
    nt.ok_(np.allclose(data, np.array([ 0.,  3.,  1.])))
    bins, data = analysis.histogram_horizontal(sample_data_1, num_bins=4, min_bin=7., max_bin=10.)
    nt.ok_(np.allclose(bins, np.array([ 7.,   8.,   9.,  10.])))
    nt.ok_(np.allclose(data, np.array([ 1.,  3.,  1.])))

def test_histogram_stepped():
    bins, data = analysis.histogram_stepped(sample_data_0)
    nt.ok_(np.allclose(bins, np.array([ 8,  9, 10])))
    nt.ok_(np.allclose(data, np.array([ 3.,  1.])))
    bins, data = analysis.histogram_stepped(sample_data_1)
    nt.ok_(np.allclose(bins, np.array([ 7,  8,  9, 10])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  1.])))

def test_distance_stepped():
    f0 = analysis.load_file(sample_ph_0_file)
    f1 = analysis.load_file(sample_ph_1_file)
    nt.ok_(analysis.distance_stepped(f0, f0) == 0.0)
    nt.ok_(analysis.distance_stepped(f0, f0, order=1) == 0.0)
    nt.ok_(analysis.distance_stepped(f0, f0, order=2) == 0.0)
    nt.ok_(np.allclose(analysis.distance_stepped(f0, f1), 34.01795854515))
    nt.ok_(np.allclose(analysis.distance_stepped(f0, f1, order=1), 34.01795854515))
    nt.ok_(np.allclose(analysis.distance_stepped(f0, f1, order=2), 25.71017265966))

def test_distance_horizontal():
    nt.ok_(analysis.distance_horizontal(sample_data_0, sample_data_0)==0.0)
    nt.ok_(analysis.distance_horizontal(sample_data_0, sample_data_1, bins=4) == 2.0)

def test_distance_horizontal_unnormed():
    nt.ok_(analysis.distance_horizontal_unnormed(sample_data_0, sample_data_0)==0.0)
    nt.ok_(analysis.distance_horizontal_unnormed(sample_data_0, sample_data_1, bins=4) == 1.0)


def test_closest_ph():

    random_values = np.random.random(4).tolist()

    ph_list = [
                [
                    [16.90, 6.68],
                    [10.52, 5.98],
                    [74.11, 0.00]
                ],
                [
                    [3.01, 1.58],
                    [15.22, 0.18],
                    [60.48, 0.00]
                ],
                [
                    [9.78, 5.30],
                    [7.66, 1.60],
                    [24.00, 0.00]
                ],
                [
                    [6.05, 2.01],
                    [3.91, 1.41],
                    [8.05, 0.00]
                ],
                [
                    [2.78, 0.87],
                    [6.12, 0.21],
                    [21.24, 0.00]
                ],
                [
                    [4.99, 4.06],
                    [4.38, 2.92],
                    [6.79, 0.00]
                ],
                [
                    [4.99, 4.06],
                    [4.38, 2.92],
                    [4.79, 0.00]
                ]
    ]

    for ph in ph_list:
        for bar in ph:
            bar += random_values

    target_extent = 6.0

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_above', return_index=True)

    assert closest_index == 5, (closest_index, 5)
    assert_array_equal(ph_list[5], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_below', return_index=True)

    assert closest_index == 6, (closest_index, 6)
    assert_array_equal(ph_list[6], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='nearest', return_index=True)

    assert closest_index == 5, (closest_index, 5)
    assert_array_equal(ph_list[5], closest_ph)

    # extreme case, target_extent bigger than any other extent
    target_extent = 100.

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_above', return_index=True)

    assert closest_index == 0, (closest_index, 0)
    assert_array_equal(ph_list[0], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_below', return_index=True)

    assert closest_index == 0, (closest_index, 0)
    assert_array_equal(ph_list[0], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='nearest', return_index=True)

    assert closest_index == 0, (closest_index, 0)
    assert_array_equal(ph_list[0], closest_ph)

    # extreme case, target_extent smaller than any other extent
    target_extent = 2.0

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_above', return_index=True)

    assert closest_index == 6, (closest_index, 6)
    assert_array_equal(ph_list[6], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_below', return_index=True)

    assert closest_index == 6, (closest_index, 6)
    assert_array_equal(ph_list[6], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='nearest', return_index=True)

    assert closest_index == 6, (closest_index, 6)
    assert_array_equal(ph_list[6], closest_ph)

    # particular case of exact match of extents
    target_extent = 24.0

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_above', return_index=True)

    assert closest_index == 2, (closest_index, 2)
    assert_array_equal(ph_list[2], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='from_below', return_index=True)

    assert closest_index == 2, (closest_index, 2)
    assert_array_equal(ph_list[2], closest_ph)

    closest_ph, closest_index = \
    analysis.closest_ph(ph_list, target_extent, method='nearest', return_index=True)

    assert closest_index == 2, (closest_index, 2)
    assert_array_equal(ph_list[2], closest_ph)
