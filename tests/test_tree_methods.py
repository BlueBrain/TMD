'''Test tmd.Tree'''
from nose import tools as nt
import numpy as np
from tmd.Tree import Tree
from tmd.io import io
from tmd.Tree import methods
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')
POP_PATH = os.path.join(DATA_PATH, 'valid')

# Filenames for testing
sample_file = os.path.join(DATA_PATH, 'sample.swc')
sample_file2 = os.path.join(POP_PATH, 'C010398B-P2.h5')

neu1 = io.load_neuron(sample_file)
neu3 = io.load_neuron(sample_file2)
tree_h5 = neu3.axon[0]
tree_h5_ap = neu3.apical[0]
tree0 = neu1.neurites[0]
tree1 = neu1.neurites[1]

secs_h5_beg = np.array([0,  16,  17,  21,  30,  52,  78,  78,  52,  30, 196, 219, 219,
                        196,  21,  17, 301, 301, 334, 334, 406, 409, 409, 406,  16, 508,
                        519, 522, 612, 640, 645, 645, 640, 612, 710, 730, 738, 738, 730,
                        710, 522, 519, 508])

secs_h5_end = np.array([16,  17,  21,  30,  52,  78,  86,  91, 190, 196, 219, 222, 230,
                        249, 256, 301, 330, 334, 385, 406, 409, 454, 482, 494, 508, 519,
                        522, 612, 640, 645, 678, 682, 684, 710, 730, 738, 772, 795, 804,
                        828, 829, 832, 838])

secs_h5_beg_ap = np.array([0,   4,   5,   8,   9,  20,  33, 109, 109,  33,  20,   9,   8,
                           213, 213,   5,   4])

secs_h5_end_ap = np.array([4,   5,   8,   9,  20,  33, 109, 121, 130, 143, 176, 205, 213,
                           225, 251, 267, 292])

secs_h5_beg_points_ap = np.array([0,   5,   6,   9,  10,  21,  34, 110, 122, 131, 144, 177, 206,
                                  214, 226, 252, 268])

secs_h5_end_points_ap = np.array([4,   5,   8,   9,  20,  33, 109, 121, 130, 143, 176, 205, 213,
                                  225, 251, 267, 292])

secs_h5_beg_points = np.array([0,  17,  18,  22,  31,  53,  79,  87,  92, 191, 197, 220, 223,
                               231, 250, 257, 302, 331, 335, 386, 407, 410, 455, 483, 495, 509,
                               520, 523, 613, 641, 646, 679, 683, 685, 711, 731, 739, 773, 796,
                               805, 829, 830, 833])

secs_h5_end_points = np.array([16,  17,  21,  30,  52,  78,  86,  91, 190, 196, 219, 222, 230,
                               249, 256, 301, 330, 334, 385, 406, 409, 454, 482, 494, 508, 519,
                               522, 612, 640, 645, 678, 682, 684, 710, 730, 738, 772, 795, 804,
                               828, 829, 832, 838])

x1 = np.array([0.,  1.,  2.,  3., 4.])
y1 = np.array([0.,  2.,  3.,  4., 5.])
z1 = np.array([0.,  3.,  4.,  5., 6.])
d1 = np.array([2.,  4.,  6.,  8., 10.])
t1 = np.array([2,   2,   2,   2,  2])
p1 = np.array([-1,  0,   1,   2,  1])

tree = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)

def test_rd():
    nt.ok_(methods._rd([0, 0], [0, 1]) == 1.)
    nt.ok_(methods._rd([0, 0, 0], [0, 0, 1]) == 1.)
    nt.ok_(methods._rd([1, 2, 0], [0, 2, 1]) == np.sqrt(2.))

# def test_rd_w():
#    nt.ok_(methods._rd_w([0,0], [0,1], w=[0., 1.]) == 1.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 2.], normed=False) == 2.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 0.], normed=False) == 0.)
#    nt.ok_(methods._rd_w([1,2,0], [0,2,1], normed=False) == methods._rd([1,2,0], [0,2,1]))


def test_size():
    nt.ok_(methods.size(tree0) == 31.)
    nt.ok_(methods.size(tree1) == 21.)


def test_get_type():
    nt.ok_(tree0.get_type() == 2)
    nt.ok_(tree1.get_type() == 3)


def test_get_bounding_box():
    nt.ok_(np.allclose(tree0.get_bounding_box(),
                       np.array([[-5.,  0., -5.],
                                 [5., 10.,  5.]])))
    nt.ok_(np.allclose(tree1.get_bounding_box(),
                       np.array([[-5.,  0., 0.],
                                 [5., 10., 0.]])))

def test_get_segments():
    seg0 = tree0.get_segments()
    seg1 = tree1.get_segments()
    seg = tree.get_segments()
    nt.ok_(len(seg0) == 30)
    nt.ok_(len(seg1) == 20)
    nt.ok_(np.allclose(seg, [np.array([[0.,  0.,  0.],
                                       [1.,  2.,  3.]]),
                             np.array([[1.,  2.,  3.],
                                       [2.,  3.,  4.]]),
                             np.array([[2.,  3.,  4.],
                                       [3.,  4.,  5.]]),
                             np.array([[1.,  2.,  3.],
                                       [4.,  5.,  6.]])]))

def test_get_point_radial_dist():
    prds = tree.get_point_radial_distances()
    nt.ok_(np.allclose(prds, np.array([0., 3.74165739, 5.38516481, 7.07106781, 8.77496439])))

def test_get_point_path_dist():
    pds = tree.get_point_path_distances()
    nt.ok_(np.allclose(pds, np.array([0., 3.74165739, 5.47370819, 7.205759  , 8.93780981])))

def test_get_sections_2():
    secs = tree.get_sections_2()
    nt.ok_(np.allclose(secs[0], np.array([0, 1, 1])))
    nt.ok_(np.allclose(secs[1], np.array([1, 3, 4])))
    secs = tree_h5.get_sections_2()
    nt.ok_(np.allclose(secs[0], secs_h5_beg))
    nt.ok_(np.allclose(secs[1], secs_h5_end))
    secs = tree_h5_ap.get_sections_2()
    nt.ok_(np.allclose(secs[0], secs_h5_beg_ap))
    nt.ok_(np.allclose(secs[1], secs_h5_end_ap))

def test_get_sections_only_points():
    secs = tree.get_sections_only_points()
    nt.ok_(np.allclose(secs[0], np.array([0, 2, 4])))
    nt.ok_(np.allclose(secs[1], np.array([1, 3, 4])))
    secs = tree_h5.get_sections_only_points()
    nt.ok_(np.allclose(secs[0], secs_h5_beg_points))
    nt.ok_(np.allclose(secs[1], secs_h5_end_points))
    secs = tree_h5_ap.get_sections_only_points()
    nt.ok_(np.allclose(secs[0], secs_h5_beg_points_ap))
    nt.ok_(np.allclose(secs[1], secs_h5_end_points_ap))

def test_get_bif_term():
    nt.ok_(np.allclose(tree.get_bif_term(), np.array([1.,  2.,  1.,  0.,  0.])))

def test_get_bifurcations():
    nt.ok_(np.allclose(tree.get_bifurcations(), np.array([1])))

def test_get_terminations():
    nt.ok_(np.allclose(tree.get_terminations(), np.array([3, 4])))

def test_get_way_to_root():
    nt.ok_(np.allclose(methods.get_way_to_root(tree), np.array([-1])))
