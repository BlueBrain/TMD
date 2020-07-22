'''Test tmd.io.swc'''
import os
from nose import tools as nt
from tmd.io import swc
import numpy as np

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')

# Filenames for testing
basic_file = os.path.join(DATA_PATH, 'basic.swc')
nocom_file = os.path.join(DATA_PATH, 'basic_no_comments.swc')
nosecids_file = os.path.join(DATA_PATH, 'basic_no_sec_ids.swc')
options_file = os.path.join(DATA_PATH, 'basic_options.swc')
sample_file = os.path.join(DATA_PATH, 'sample.swc')

basic_data = np.array(['1 1 0 0 0 6 -1', '2 2 3 4 5 6 1',
                       '3 3 4 5 6 7 1', '4 4 5 6 7 8 1', '5 4 5 6 7 8 4'])
options_data = np.array(['2', '3', '4', '5', '6', '7', '8', '9\n'])

bx1 = np.array([0.,  3.,  4.,  5.,  5.])
by1 = np.array([0.,  4.,  5.,  6.,  6.])
bz1 = np.array([0.,  5.,  6.,  7.,  7.])
bd1 = np.array([12.,  12.,  14.,  16.,  16.])
bt1 = np.array([1, 2, 3, 4, 4])
bp1 = np.array([-1,  0,  0,  0,  3])
bch1 = {0: [1, 2, 3], 1: [], 2: [], 3: [4], 4: []}


def test_swc_dict():
    nt.ok_(swc.SWC_DCT == {'index': 0,
                           'parent': 6,
                           'radius': 5,
                           'type': 1,
                           'x': 2,
                           'y': 3,
                           'z': 4})


def test_read_swc():
    data1 = swc.read_swc(basic_file)
    nt.ok_(np.alltrue(data1 == basic_data))
    data2 = swc.read_swc(options_file, line_delimiter=' ')
    nt.ok_(np.alltrue(data2 == options_data))


def test_swc_data_to_lists():
    x1, y1, z1, d1, t1, p1, ch1 = swc.swc_data_to_lists(basic_data)
    nt.ok_(np.allclose(x1, bx1))
    nt.ok_(np.allclose(y1, by1))
    nt.ok_(np.allclose(z1, bz1))
    nt.ok_(np.allclose(d1, bd1))
    nt.ok_(np.allclose(t1, bt1))
    nt.ok_(np.allclose(p1, bp1))
    nt.ok_(ch1 == bch1)
