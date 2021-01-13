'''Test tmd.topology.methods'''
import mock
import numpy as np
from numpy import testing as npt
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

    npt.assert_allclose(ph0, [[12.24744871391589, 11.180339887498949],
                             [11.180339887498949, 10.0],
                             [12.24744871391589, 0]])
    npt.assert_allclose(ph1, [[11.180339887498949, 10.0],
                             [11.180339887498949, 0]])
    ph0 = methods.get_persistence_diagram(tree0, feature='path_distances')
    ph1 = methods.get_persistence_diagram(tree1, feature='path_distances')
    npt.assert_allclose(ph0, [[20.0, 15.0], [15.0, 10.0], [20.0, 0]])
    npt.assert_allclose(ph1, [[15.0, 10.0], [15.0, 0]])

def test_extract_persistence_diagram():
    import filecmp
    if os.path.isfile('test_ph.txt'):
        os.remove('test_ph.txt')
    methods.extract_ph(tree0, output_file='./test_ph.txt')
    ph_file = analysis.load_file('./test_ph.txt')
    ph_0 = analysis.load_file(sample_ph_0)
    npt.assert_allclose(ph_file, ph_0)
    os.remove('test_ph.txt')

def test_get_lifetime():
    lf0 = methods.get_lifetime(tree0)
    lf1 = methods.get_lifetime(tree1)
    npt.assert_allclose(lf0, np.array([[  0.        ,  10.        ],
                                      [ 10.        ,  11.18033989],
                                      [ 10.        ,  11.18033989],
                                      [ 11.18033989,  12.24744871],
                                      [ 11.18033989,  12.24744871]]))
    npt.assert_allclose(lf1, np.array([[  0.        ,  10.        ],
                                      [ 10.        ,  11.18033989],
                                      [ 10.        ,  11.18033989]]))

def test_extract_connectivity_from_points():
    dist = methods.extract_connectivity_from_points(tree, threshold=0.1)
    npt.assert_allclose(dist, np.array([[ True, False, False, False, False],
                                       [False,  True, False, False, False],
                                       [False, False,  True, False, False],
                                       [False, False, False,  True, False],
                                       [False, False, False, False,  True]], dtype=bool))
    dist = methods.extract_connectivity_from_points(tree, threshold=1.1)
    npt.assert_allclose(dist, np.array([[ True,  True,  True, False, False],
                                       [ True,  True, False,  True, False],
                                       [ True, False,  True, False, False],
                                       [False,  True, False,  True, False],
                                       [False, False, False, False,  True]], dtype=bool))
    dist = methods.extract_connectivity_from_points(tree, threshold=2.1)
    npt.assert_allclose(dist, np.array([[ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True, False,  True],
                                       [ True,  True, False,  True,  True],
                                       [ True,  True,  True,  True,  True]], dtype=bool))
    dist = methods.extract_connectivity_from_points(tree, threshold=3.1)
    npt.assert_allclose(dist, np.array([[ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True],
                                       [ True,  True,  True,  True,  True]], dtype=bool))


def test_filtration_function():

    feature = 'radial_distances'
    func = methods._filtration_function(feature)
    npt.assert_allclose(func(tree), [0. , 1. , 1., 1.41421356, 1.41421356])

    feature = 'path_distances'
    func = methods._filtration_function(feature)
    npt.assert_allclose(func(tree), [0., 1., 3., 5.236068, 6.650282])


def test_tree_to_property_barcode():

    filtration_function = lambda tree: np.array([0., 1., 3., 5., 7.])

    prop = lambda *args: mock.Mock(
        get=lambda v: [],
        infinite_component=lambda v: []
    )

    ph = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop)

    npt.assert_array_equal(ph, [
        [3., 1.],
        [5., 1.],
        [7., 0.],
    ])

    prop = lambda *args: mock.Mock(
        get=lambda v: [v],
        infinite_component=lambda v: [0]
    )

    ph = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop)

    npt.assert_array_equal(ph, [
        [3., 1., 1.],
        [5., 1., 1.],
        [7., 0., 0.],
    ])

    prop = lambda *args: mock.Mock(
        get=lambda v: list(range(v, v + 5)),
        infinite_component=lambda v: [np.nan] * 5
    )

    ph = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop)

    npt.assert_array_equal(ph, [
        [3., 1., 1., 2., 3., 4., 5.],
        [5., 1., 1., 2., 3., 4., 5.],
        [7., 0., np.nan, np.nan, np.nan, np.nan, np.nan],
    ])


def _integration_tree():

    xyzdtp = np.array([
       [  5.,  -2.,   5.,   0.2,   3.,  -1.],
       [ 11.,  -5.,  12.,   0.2,   3.,   0.],
       [ 18.,  -8.,  19.,   0.4,   3.,   1.],
       [ 24., -11.,  26.,   0.2,   3.,   2.],
       [ 31., -14.,  33.,   0.4,   3.,   3.],
       [ 38., -16.,  40.,   0.6,   3.,   4.],
       [ 45., -18.,  47.,   0.8,   3.,   5.],
       [ 52., -19.,  53.,   0.6,   3.,   6.],
       [ 37., -17.,  39.,   0.4,   3.,   4.],
       [ 44., -20.,  46.,   0.2,   3.,   8.],
       [ 50., -23.,  53.,   0.2,   3.,   9.],
       [ 57., -26.,  60.,   0.4,   3.,  10.],
       [ 63., -29.,  67.,   0.6,   3.,  11.],
       [ 70., -32.,  74.,   0.8,   3.,  12.],
       [ 24., -10.,  26.,   1.0,   3.,   2.],
       [ 31., -12.,  33.,   0.8,   3.,  14.],
       [ 39., -13.,  40.,   0.6,   3.,  15.]])

    return Tree.Tree(
        x=xyzdtp[:, 0], y=xyzdtp[:, 1], z=xyzdtp[:, 2],
        d=xyzdtp[:, 3], t=xyzdtp[:, 4].astype(int), p=xyzdtp[:, 5].astype(int)
    )


def test_get_persistence_diagram__integration():

    npt.assert_allclose(
        methods.get_persistence_diagram(_integration_tree()),
        [
            [69.29646455628166, 40.049968789001575],
            [50.0199960015992, 20.024984394500787],
            [99.42836617384397, 0]
        ],
        atol=1e-6
    )


def test_get_ph_angles__integration():

    npt.assert_allclose(
        methods.get_ph_angles(_integration_tree()),
        [
            [69.296464556281, 40.0499687890015, 0.0, 0.012041319646904, 0.19866459470163, 0.0148790141866820],
            [50.01999600159, 20.0249843945007, 0.0, 0.0, 0.198664594701636, 0.0025605561025675],
            [99.428366173843, 0, np.nan, np.nan, np.nan, np.nan]
        ],
        atol=1e-6
    )


def test_get_ph_radii__integration():

    npt.assert_allclose(
        methods.get_ph_radii(_integration_tree()),
        [
            [69.29646455628166, 40.049968789001575, 0.3],
            [50.0199960015992, 20.024984394500787, 0.15],
            [99.42836617384397, 0, 0.1]
        ],
        atol=1e-6
    )

