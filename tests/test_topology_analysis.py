'''Test tmd.topology.analysis'''
from nose import tools as nt
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_almost_equal
from tmd.Topology import analysis
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')

# Filenames for testing
sample_ph_0_file = os.path.join(DATA_PATH, 'sample_ph_0.txt')
sample_ph_1_file = os.path.join(DATA_PATH, 'sample_ph_1.txt')
neuron_ph_1_file = os.path.join(DATA_PATH, 'neuron_ph_1.txt')
neuron_ph_2_file = os.path.join(DATA_PATH, 'neuron_ph_2.txt')

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


def _ph_list():

    return [
                [
                    [16.90, 6.68, 0.1, 0.1, 0.1, 0.1],
                    [10.52, 5.98, 0.1, 0.1, 0.1, 0.1],
                    [74.11, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [3.010, 1.58, 0.1, 0.1, 0.1, 0.1],
                    [15.22, 0.18, 0.1, 0.1, 0.1, 0.1],
                    [60.48, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [9.78, 5.30, 0.1, 0.1, 0.1, 0.1],
                    [7.66, 1.60, 0.1, 0.1, 0.1, 0.1],
                    [24.0, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [6.05, 2.01, 0.1, 0.1, 0.1, 0.1],
                    [3.91, 1.41, 0.1, 0.1, 0.1, 0.1],
                    [8.05, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [2.78, 0.87, 0.1, 0.1, 0.1, 0.1],
                    [6.12, 0.21, 0.1, 0.1, 0.1, 0.1],
                    [21.2, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [4.99, 4.06, 0.1, 0.1, 0.1, 0.1],
                    [4.38, 2.92, 0.1, 0.1, 0.1, 0.1],
                    [6.79, 0.00, 0.1, 0.1, 0.1, 0.1]
                ],
                [
                    [4.99, 4.06, 0.1, 0.1, 0.1, 0.1],
                    [4.38, 2.92, 0.1, 0.1, 0.1, 0.1],
                    [4.79, 0.00, 0.1, 0.1, 0.1, 0.1]
                ]
    ]

def test_closest_ph__reasonable_target_extent():

    ph_list = _ph_list()

    target_extent = 6.0

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_above')
    assert_equal(closest_index, 5)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_below')
    assert_equal(closest_index, 6)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='nearest')
    assert_equal(closest_index, 5)


def test_closest_ph__very_big_target_extent():

    ph_list = _ph_list()

    target_extent = 100.

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_above')
    assert_equal(closest_index, 0)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_below')
    assert_equal(closest_index, 0)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='nearest')
    assert_equal(closest_index, 0)


def test_closest_ph__very_small_target_extent():

    ph_list = _ph_list()

    target_extent = 2.0

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_above')
    assert_equal(closest_index, 6)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_below')
    assert_equal(closest_index, 6)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='nearest')
    assert_equal(closest_index, 6)


def test_closest_ph__exact_match_target_extent():

    ph_list = _ph_list()

    target_extent = 24.0

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_above')
    assert_equal(closest_index, 2)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='from_below')
    assert_equal(closest_index, 2)

    closest_index = analysis.closest_ph(ph_list, target_extent, method='nearest')
    assert_equal(closest_index, 2)


def test_apical_point():
    p1 = analysis.load_file(neuron_ph_1_file)
    p2 = analysis.load_file(neuron_ph_2_file)
    ap1 = analysis.find_apical_point_distance(p1)
    ap2 = analysis.find_apical_point_distance(p2)
    assert_almost_equal(ap1, 413.2151457659, decimal=5)
    assert_almost_equal(ap2, 335.8844214625, decimal=5)
