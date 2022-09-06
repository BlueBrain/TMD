"""Test tmd.topology.methods."""
# pylint: disable=protected-access
import os

import mock
import numpy as np
from numpy import testing as npt

from tmd.Topology import analysis
from tmd.Topology import methods
from tmd.Tree import Tree

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")

sample_ph_0 = os.path.join(DATA_PATH, "sample_ph_0.txt")
sample_ph_1 = os.path.join(DATA_PATH, "sample_ph_1.txt")

x1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
y1 = np.array([0.0, 1.0, -1.0, 1.0, 0.0])
z1 = np.array([0.0, 0.0, 0, 0.0, -1.0])
d1 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
t1 = np.array([1, 1, 1, 1, 1])
p1 = np.array([-1, 0, 1, 2, 3])

x2 = np.array([0.0, 3.0, 4.0, 5.0, 4.0])
p2 = np.array([-1, 0, 1, 1, 1])

tree = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
tree_trifork = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p2)

# fmt: off
# ===================================== tree0 =======================================
x2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -2.,
               -3., -4., -5., 1., 2., 3., 4., 5., 5., 5., 5., 5., 5.,
               5., 5., 5., 5., 5.])
y2 = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 10., 10.,
               10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
               10., 10., 10., 10., 10.])
z2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5.,
               -1., -2., -3., -4., -5.])
d2 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
               0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
               0.4, 0.4, 0.4, 0.4, 0.4])
t2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2])
p2 = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10,
               16, 17, 18, 19, 20, 21, 22, 23, 24, 20, 26, 27, 28, 29])

tree0 = Tree.Tree(x=x2, y=y2, z=z2, d=d2, t=t2, p=p2)

# ===================================== tree1 =======================================
x2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -2.,
               -3., -4., -5., 1., 2., 3., 4., 5.])
y2 = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 10., 10.,
               10., 10., 10., 10., 10., 10., 10., 10.])
z2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.])
d2 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
               0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
t2 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
p2 = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10,
               16, 17, 18, 19])
# fmt: on

tree1 = Tree.Tree(x=x2, y=y2, z=z2, d=d2, t=t2, p=p2)


def test_get_persistence_diagram():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    ph0 = methods.get_persistence_diagram(tree0)
    ph1 = methods.get_persistence_diagram(tree1)

    npt.assert_allclose(
        ph0,
        [
            [12.24744871391589, 11.180339887498949],
            [11.180339887498949, 10.0],
            [12.24744871391589, 0],
        ],
    )
    npt.assert_allclose(ph1, [[11.180339887498949, 10.0], [11.180339887498949, 0]])
    ph0 = methods.get_persistence_diagram(tree0, feature="path_distances")
    ph1 = methods.get_persistence_diagram(tree1, feature="path_distances")
    npt.assert_allclose(ph0, [[20.0, 15.0], [15.0, 10.0], [20.0, 0]])
    npt.assert_allclose(ph1, [[15.0, 10.0], [15.0, 0]])


def test_extract_persistence_diagram():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    if os.path.isfile("test_ph.txt"):
        os.remove("test_ph.txt")
    methods.extract_ph(tree0, output_file="./test_ph.txt")
    ph_file = analysis.load_file("./test_ph.txt")
    ph_0 = analysis.load_file(sample_ph_0)
    npt.assert_allclose(ph_file, ph_0)
    os.remove("test_ph.txt")


def test_get_lifetime():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    lf0 = methods.get_lifetime(tree0)
    lf1 = methods.get_lifetime(tree1)
    npt.assert_allclose(
        lf0,
        np.array(
            [
                [0.0, 10.0],
                [10.0, 11.18033989],
                [10.0, 11.18033989],
                [11.18033989, 12.24744871],
                [11.18033989, 12.24744871],
            ]
        ),
    )
    npt.assert_allclose(lf1, np.array([[0.0, 10.0], [10.0, 11.18033989], [10.0, 11.18033989]]))


def test_extract_connectivity_from_points():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    dist = methods.extract_connectivity_from_points(tree, threshold=0.1)
    npt.assert_allclose(
        dist,
        np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            dtype=bool,
        ),
    )
    dist = methods.extract_connectivity_from_points(tree, threshold=1.1)
    npt.assert_allclose(
        dist,
        np.array(
            [
                [True, True, True, False, False],
                [True, True, False, True, False],
                [True, False, True, False, False],
                [False, True, False, True, False],
                [False, False, False, False, True],
            ],
            dtype=bool,
        ),
    )
    dist = methods.extract_connectivity_from_points(tree, threshold=2.1)
    npt.assert_allclose(
        dist,
        np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    dist = methods.extract_connectivity_from_points(tree, threshold=3.1)
    npt.assert_allclose(
        dist,
        np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
    )


def test_filtration_function():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    feature = "radial_distances"
    func = methods._filtration_function(feature)
    npt.assert_allclose(func(tree), [0.0, 1.0, 1.0, 1.41421356, 1.41421356])

    feature = "path_distances"
    func = methods._filtration_function(feature)
    npt.assert_allclose(func(tree), [0.0, 1.0, 3.0, 5.236068, 6.650282])


def test_tree_to_property_barcode():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    def filtration_function(_):
        return np.array([0.0, 1.0, 3.0, 5.0, 7.0])

    def prop1(*args):  # pylint: disable=unused-argument
        return mock.Mock(get=lambda v: [], infinite_component=lambda v: [])

    ph, bars_to_points = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop1)

    npt.assert_array_equal(
        ph,
        [
            [3.0, 1.0],
            [5.0, 1.0],
            [7.0, 0.0],
        ],
    )

    assert bars_to_points == [[2], [3], [4, 1]]

    def prop2(*args):  # pylint: disable=unused-argument
        return mock.Mock(get=lambda v: [v], infinite_component=lambda v: [0])

    ph, bars_to_points = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop2)

    npt.assert_array_equal(
        ph,
        [
            [3.0, 1.0, 1.0],
            [5.0, 1.0, 1.0],
            [7.0, 0.0, 0.0],
        ],
    )

    assert bars_to_points == [[2], [3], [4, 1]]

    def prop3(*args):  # pylint: disable=unused-argument
        return mock.Mock(
            get=lambda v: list(range(v, v + 5)), infinite_component=lambda v: [np.nan] * 5
        )

    ph, bars_to_points = methods.tree_to_property_barcode(tree_trifork, filtration_function, prop3)

    npt.assert_array_equal(
        ph,
        [
            [3.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [7.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
    )

    assert bars_to_points == [[2], [3], [4, 1]]


def _integration_tree():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    xyzdtp = np.array(
        [
            [5.0, -2.0, 5.0, 0.2, 3.0, -1.0],
            [11.0, -5.0, 12.0, 0.2, 3.0, 0.0],
            [18.0, -8.0, 19.0, 0.4, 3.0, 1.0],
            [24.0, -11.0, 26.0, 0.2, 3.0, 2.0],
            [31.0, -14.0, 33.0, 0.4, 3.0, 3.0],
            [38.0, -16.0, 40.0, 0.6, 3.0, 4.0],
            [45.0, -18.0, 47.0, 0.8, 3.0, 5.0],
            [52.0, -19.0, 53.0, 0.6, 3.0, 6.0],
            [37.0, -17.0, 39.0, 0.4, 3.0, 4.0],
            [44.0, -20.0, 46.0, 0.2, 3.0, 8.0],
            [50.0, -23.0, 53.0, 0.2, 3.0, 9.0],
            [57.0, -26.0, 60.0, 0.4, 3.0, 10.0],
            [63.0, -29.0, 67.0, 0.6, 3.0, 11.0],
            [70.0, -32.0, 74.0, 0.8, 3.0, 12.0],
            [24.0, -10.0, 26.0, 1.0, 3.0, 2.0],
            [31.0, -12.0, 33.0, 0.8, 3.0, 14.0],
            [39.0, -13.0, 40.0, 0.6, 3.0, 15.0],
        ]
    )

    return Tree.Tree(
        x=xyzdtp[:, 0],
        y=xyzdtp[:, 1],
        z=xyzdtp[:, 2],
        d=xyzdtp[:, 3],
        t=xyzdtp[:, 4].astype(int),
        p=xyzdtp[:, 5].astype(int),
    )


def test_get_persistence_diagram__integration():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(
        methods.get_persistence_diagram(_integration_tree()),
        [
            [69.29646455628166, 40.049968789001575],
            [50.0199960015992, 20.024984394500787],
            [99.42836617384397, 0],
        ],
        atol=1e-6,
    )


def test_get_ph_angles__integration():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(
        methods.get_ph_angles(_integration_tree()),
        [
            [
                69.296464556281,
                40.0499687890015,
                0.0,
                0.012041319646904,
                0.19866459470163,
                0.0148790141866820,
            ],
            [50.01999600159, 20.0249843945007, 0.0, 0.0, 0.198664594701636, 0.0025605561025675],
            [99.428366173843, 0, np.nan, np.nan, np.nan, np.nan],
        ],
        atol=1e-6,
    )


def test_get_ph_radii__integration():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(
        methods.get_ph_radii(_integration_tree()),
        [
            [69.29646455628166, 40.049968789001575, 0.3],
            [50.0199960015992, 20.024984394500787, 0.15],
            [99.42836617384397, 0, 0.1],
        ],
        atol=1e-6,
    )
