'''Test tmd.topology.methods'''
from nose import tools as nt
import numpy as np
from tmd.Topology import methods
from tmd.Topology import analysis
from tmd.Tree import Tree
from tmd.io import io
import os
from collections import OrderedDict

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')

sample_ph_0 = os.path.join(DATA_PATH, 'sample_ph_0.txt')
sample_ph_1 = os.path.join(DATA_PATH, 'sample_ph_1.txt')

x1 = np.array([0.,  0.,   0.,  1.,   1.])
y1 = np.array([0.,  1.,  -1.,  1.,   0.])
z1 = np.array([0.,  0.,   0,   0.,  -1.])
d1 = np.array([2.,  2.,  2.,  2.,  2.])
t1 = np.array([ 1,  1,  1,  1,  1])
p1 = np.array([-1,  0,  1,  2,  3])

x2 = np.array([0.,  3.,  4.,  5.,  4.])
p2 = np.array([-1,  0,  1,  1,  1])

tree = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
tree_trifork = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p2)

# ===================================== tree0 =======================================

x2 = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -2.,
       -3., -4., -5.,  1.,  2.,  3.,  4.,  5.,  5.,  5.,  5.,  5.,  5.,
        5.,  5.,  5.,  5.,  5.])
y2 = np.array([0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 10., 10.,
       10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
       10., 10., 10., 10., 10.])
z2 = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  4.,  5.,
       -1., -2., -3., -4., -5.])
d2 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
       0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
       0.4, 0.4, 0.4, 0.4, 0.4])
t2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2])
p2 = np.array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 10,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 20, 26, 27, 28, 29])

tree0 = Tree.Tree(x=x2, y=y2, z=z2, d=d2, t=t2, p=p2)

# ===================================== tree1 =======================================
x2 = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -2.,
       -3., -4., -5.,  1.,  2.,  3.,  4.,  5.])
y2 = np.array([0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 10., 10.,
       10., 10., 10., 10., 10., 10., 10., 10.])
z2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0.])
d2 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
       0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
t2 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
p2 = np.array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 10,
       16, 17, 18, 19])

tree1 = Tree.Tree(x=x2, y=y2, z=z2, d=d2, t=t2, p=p2)


def test_get_persistence_diagram():
    ph0 = methods.get_persistence_diagram(tree0)
    ph1 = methods.get_persistence_diagram(tree1)
    nt.ok_(np.allclose(ph0, [[12.24744871391589, 11.180339887498949],
                             [11.180339887498949, 10.0],
                             [12.24744871391589, 0]]))
    nt.ok_(np.allclose(ph1, [[11.180339887498949, 10.0],
                             [11.180339887498949, 0]]))
    ph0 = methods.get_persistence_diagram(tree0, feature='path_distances')
    ph1 = methods.get_persistence_diagram(tree1, feature='path_distances')
    nt.ok_(np.allclose(ph0, [[20.0, 15.0], [15.0, 10.0], [20.0, 0]]))
    nt.ok_(np.allclose(ph1, [[15.0, 10.0], [15.0, 0]]))

def test_extract_persistence_diagram():
    import filecmp
    if os.path.isfile('test_ph.txt'):
        os.remove('test_ph.txt')
    methods.extract_ph(tree0, output_file='./test_ph.txt')
    ph_file = analysis.load_file('./test_ph.txt')
    ph_0 = analysis.load_file(sample_ph_0)
    nt.ok_(np.allclose(ph_file, ph_0))
    os.remove('test_ph.txt')

def test_get_lifetime():
    lf0 = methods.get_lifetime(tree0)
    lf1 = methods.get_lifetime(tree1)
    nt.ok_(np.allclose(lf0, np.array([[  0.        ,  10.        ],
                                      [ 10.        ,  11.18033989],
                                      [ 10.        ,  11.18033989],
                                      [ 11.18033989,  12.24744871],
                                      [ 11.18033989,  12.24744871]])))
    nt.ok_(np.allclose(lf1, np.array([[  0.        ,  10.        ],
                                      [ 10.        ,  11.18033989],
                                      [ 10.        ,  11.18033989]])))

def test_extract_connectivity_from_points():
    dist = methods.extract_connectivity_from_points(tree, threshold=0.1)
    nt.ok_(np.allclose(dist, np.array([[ True, False, False, False, False],
                                       [False,  True, False, False, False],
                                       [False, False,  True, False, False],
                                       [False, False, False,  True, False],
                                       [False, False, False, False,  True]], dtype=bool)))
    dist = methods.extract_connectivity_from_points(tree, threshold=1.1)
    nt.ok_(np.allclose(dist, np.array([[ True,  True,  True, False, False],
                                       [ True,  True, False,  True, False],
                                       [ True, False,  True, False, False],
                                       [False,  True, False,  True, False],
                                       [False, False, False, False,  True]], dtype=bool)))
    dist = methods.extract_connectivity_from_points(tree, threshold=2.1)
    nt.ok_(np.allclose(dist, np.array([[ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True, False,  True],
                                       [ True,  True, False,  True,  True],
                                       [ True,  True,  True,  True,  True]], dtype=bool)))
    dist = methods.extract_connectivity_from_points(tree, threshold=3.1)
    nt.ok_(np.allclose(dist, np.array([[ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True]], dtype=bool)))
