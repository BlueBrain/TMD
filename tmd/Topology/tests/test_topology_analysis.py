'''Test tmd.topology.analysis'''
from nose import tools as nt
import numpy as np
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

def test_horizontal_hist():
    bins, data = analysis.horizontal_hist(sample_data_0, num_bins=11, min_bin=8.0, max_bin=10.0)
    nt.ok_(np.allclose(bins, np.array([ 8., 8.2, 8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10. ])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  3.,  3.,  1.,  3.,  1.,  1.,  1.,  1.])))
    bins, data = analysis.horizontal_hist(sample_data_0, num_bins=10)
    nt.ok_(np.allclose(bins, np.array([ 8.        , 8.22222222,   8.44444444,   8.66666667,
                                        8.88888889, 9.11111111,   9.33333333,   9.55555556,
                                        9.77777778, 10.])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  3.,  3.,  3.,  1.,  1.,  1.,  1.])))
    bins, data = analysis.horizontal_hist(sample_data_0, num_bins=4, min_bin=7., max_bin=10.)
    nt.ok_(np.allclose(bins, np.array([ 7.,   8.,   9.,  10.])))
    nt.ok_(np.allclose(data, np.array([ 0.,  3.,  1.])))
    bins, data = analysis.horizontal_hist(sample_data_1, num_bins=4, min_bin=7., max_bin=10.)
    nt.ok_(np.allclose(bins, np.array([ 7.,   8.,   9.,  10.])))
    nt.ok_(np.allclose(data, np.array([ 1.,  3.,  1.])))

def test_step_hist():
    bins, data = analysis.step_hist(sample_data_0)
    nt.ok_(np.allclose(bins, np.array([ 8,  9, 10])))
    nt.ok_(np.allclose(data, np.array([ 3.,  1.])))
    bins, data = analysis.step_hist(sample_data_1)
    nt.ok_(np.allclose(bins, np.array([ 7,  8,  9, 10])))
    nt.ok_(np.allclose(data, np.array([ 3.,  3.,  1.])))

def test_step_dist():
    f0 = analysis.load_file(sample_ph_0_file)
    f1 = analysis.load_file(sample_ph_1_file)
    nt.ok_(analysis.step_dist(f0, f0) == 0.0)
    nt.ok_(analysis.step_dist(f0, f0, order=1) == 0.0)
    nt.ok_(analysis.step_dist(f0, f0, order=2) == 0.0)
    nt.ok_(np.allclose(analysis.step_dist(f0, f1), 34.01795854515))
    nt.ok_(np.allclose(analysis.step_dist(f0, f1, order=1), 34.01795854515))
    nt.ok_(np.allclose(analysis.step_dist(f0, f1, order=2), 25.71017265966))

def test_hist_distance():
    nt.ok_(analysis.hist_distance(sample_data_0, sample_data_0)==0.0)
    nt.ok_(analysis.hist_distance(sample_data_0, sample_data_1, bins=4) == 2.0)

def test_get_total_length():
    nt.ok_(analysis.get_total_length(sample_data_0)==4.0)
    nt.ok_(analysis.get_total_length(sample_data_1)==5.0)

def test_hist_distance_un():
    nt.ok_(analysis.hist_distance_un(sample_data_0, sample_data_0)==0.0)
    nt.ok_(analysis.hist_distance_un(sample_data_0, sample_data_1, bins=4) == 1.0)
