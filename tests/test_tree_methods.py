"""Test tmd.Tree methods."""
# pylint: disable=protected-access
import os

import numpy as np
from numpy import testing as npt

from tmd.io import io
from tmd.Tree import Tree
from tmd.Tree import methods

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")
POP_PATH = os.path.join(DATA_PATH, "valid")

# Filenames for testing
sample_file = os.path.join(DATA_PATH, "sample.swc")
sample_file2 = os.path.join(POP_PATH, "C010398B-P2.h5")

neu1 = io.load_neuron(sample_file)
neu3 = io.load_neuron(sample_file2)
tree_h5 = neu3.axon[0]
tree_h5_ap = neu3.apical[0]
tree0 = neu1.neurites[0]
tree1 = neu1.neurites[1]

# fmt: off
secs_h5_beg = np.array([0, 16, 17, 21, 30, 52, 78, 78, 52, 30, 196, 219, 219,
                        196, 21, 17, 301, 301, 334, 334, 406, 409, 409, 406, 16, 508,
                        519, 522, 612, 640, 645, 645, 640, 612, 710, 730, 738, 738, 730,
                        710, 522, 519, 508])

secs_h5_end = np.array([16, 17, 21, 30, 52, 78, 86, 91, 190, 196, 219, 222, 230,
                        249, 256, 301, 330, 334, 385, 406, 409, 454, 482, 494, 508, 519,
                        522, 612, 640, 645, 678, 682, 684, 710, 730, 738, 772, 795, 804,
                        828, 829, 832, 838])

secs_h5_beg_ap = np.array([0, 4, 5, 8, 9, 20, 33, 109, 109, 33, 20, 9, 8,
                           213, 213, 5, 4])

secs_h5_end_ap = np.array([4, 5, 8, 9, 20, 33, 109, 121, 130, 143, 176, 205, 213,
                           225, 251, 267, 292])

secs_h5_beg_points_ap = np.array([0, 5, 6, 9, 10, 21, 34, 110, 122, 131, 144, 177, 206,
                                  214, 226, 252, 268])

secs_h5_end_points_ap = np.array([4, 5, 8, 9, 20, 33, 109, 121, 130, 143, 176, 205, 213,
                                  225, 251, 267, 292])

secs_h5_beg_points = np.array([0, 17, 18, 22, 31, 53, 79, 87, 92, 191, 197, 220, 223,
                               231, 250, 257, 302, 331, 335, 386, 407, 410, 455, 483, 495, 509,
                               520, 523, 613, 641, 646, 679, 683, 685, 711, 731, 739, 773, 796,
                               805, 829, 830, 833])

secs_h5_end_points = np.array([16, 17, 21, 30, 52, 78, 86, 91, 190, 196, 219, 222, 230,
                               249, 256, 301, 330, 334, 385, 406, 409, 454, 482, 494, 508, 519,
                               522, 612, 640, 645, 678, 682, 684, 710, 730, 738, 772, 795, 804,
                               828, 829, 832, 838])
# fmt: on

x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y1 = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
z1 = np.array([0.0, 3.0, 4.0, 5.0, 6.0])
d1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
t1 = np.array([2, 2, 2, 2, 2])
p1 = np.array([-1, 0, 1, 2, 1])

tree = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
long_tree = Tree.Tree(
    x=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    y=np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    z=np.array([0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    d=np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
    t=np.array([2, 2, 2, 2, 2, 2, 2]),
    p=np.array([-1, 0, 1, 2, 2, 4, 5]),
)


def test_rd():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert methods._rd([0, 0], [0, 1]) == 1.0
    assert methods._rd([0, 0, 0], [0, 0, 1]) == 1.0
    assert methods._rd([1, 2, 0], [0, 2, 1]) == np.sqrt(2.0)


# def test_rd_w():
#    nt.ok_(methods._rd_w([0,0], [0,1], w=[0., 1.]) == 1.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 2.], normed=False) == 2.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 0.], normed=False) == 0.)
#    nt.ok_(methods._rd_w([1,2,0], [0,2,1], normed=False) == methods._rd([1,2,0], [0,2,1]))


def test_size():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert methods.size(tree0) == 31.0
    assert methods.size(tree1) == 21.0


def test_get_type():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert tree0.get_type() == 2
    assert tree1.get_type() == 3


def test_get_bounding_box():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    np.allclose(tree0.get_bounding_box(), np.array([[-5.0, 0.0, -5.0], [5.0, 10.0, 5.0]]))
    np.allclose(tree1.get_bounding_box(), np.array([[-5.0, 0.0, 0.0], [5.0, 10.0, 0.0]]))


def test_get_segments():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    seg0 = tree0.get_segments()
    seg1 = tree1.get_segments()
    seg = tree.get_segments()
    assert len(seg0) == 30
    assert len(seg1) == 20
    npt.assert_allclose(
        seg,
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ],
    )


def test_get_point_radial_dist():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    prds = tree.get_point_radial_distances()
    npt.assert_allclose(prds, np.array([0.0, 3.74165739, 5.38516481, 7.07106781, 8.77496439]))


def test_get_point_path_dist():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    pds = tree.get_point_path_distances()
    npt.assert_allclose(pds, np.array([0.0, 3.74165739, 5.47370819, 7.205759, 8.93780981]))


def test_get_point_section_lengths():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    pds = tree.get_point_section_lengths()
    npt.assert_array_almost_equal(pds, np.array([0.0, 3.7416575, 0.0, 3.46410161, 5.19615221]))

    pds = long_tree.get_point_section_lengths()
    npt.assert_array_almost_equal(
        pds, np.array([0.0, 0.0, 5.47370827, 1.73205078, 0.0, 0.0, 6.92820311])
    )


def test_get_trunk_length():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    pds = tree.get_trunk_length()
    npt.assert_almost_equal(pds, 3.7416575)

    pds = long_tree.get_trunk_length()
    npt.assert_almost_equal(pds, 5.4737083)


def test_get_sections_2():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    secs = tree.get_sections_2()
    npt.assert_allclose(secs[0], np.array([0, 1, 1]))
    npt.assert_allclose(secs[1], np.array([1, 3, 4]))
    secs = tree_h5.get_sections_2()
    npt.assert_allclose(secs[0], secs_h5_beg)
    npt.assert_allclose(secs[1], secs_h5_end)
    secs = tree_h5_ap.get_sections_2()
    npt.assert_allclose(secs[0], secs_h5_beg_ap)
    npt.assert_allclose(secs[1], secs_h5_end_ap)


def test_get_sections_only_points():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    secs = tree.get_sections_only_points()
    npt.assert_allclose(secs[0], np.array([0, 2, 4]))
    npt.assert_allclose(secs[1], np.array([1, 3, 4]))
    secs = tree_h5.get_sections_only_points()
    npt.assert_allclose(secs[0], secs_h5_beg_points)
    npt.assert_allclose(secs[1], secs_h5_end_points)
    secs = tree_h5_ap.get_sections_only_points()
    npt.assert_allclose(secs[0], secs_h5_beg_points_ap)
    npt.assert_allclose(secs[1], secs_h5_end_points_ap)


def test_get_bif_term():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(tree.get_bif_term(), np.array([1.0, 2.0, 1.0, 0.0, 0.0]))


def test_get_bifurcations():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(tree.get_bifurcations(), np.array([1]))


def test_get_terminations():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(tree.get_terminations(), np.array([3, 4]))


def test_get_way_to_root():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(methods.get_way_to_root(tree), np.array([-1]))


def test_parents_children():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    tested_tree = Tree.Tree(
        x=np.zeros(5),
        y=np.zeros(5),
        z=np.zeros(5),
        d=np.zeros(5),
        t=np.zeros(5),
        p=np.array([-1, 0, 1, 2, 2]),
    )

    parents, children = tested_tree.parents_children

    assert parents == {0: -1, 2: 0, 3: 2, 4: 2}

    expected_children = {2: [3, 4]}

    for key, values in expected_children.items():
        npt.assert_array_equal(children[key], values)
