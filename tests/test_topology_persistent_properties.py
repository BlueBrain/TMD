from mock import patch, Mock
import numpy as np
from numpy import testing as npt
from tmd.Topology import persistent_properties as tested


class MockTree:

    def __init__(self, points, parents=None, children=None, begs=None, ends=None):

        self._points = points
        self._parents = parents
        self._children = children
        self._begs = begs
        self._ends = ends

    def get_direction_between(self, start_id, end_id):
        vec = self._points[end_id] - self._points[start_id]
        vec /= np.linalg.norm(vec)
        return vec

    @property
    def sections(self):
        return self._begs, self._ends

    @property
    def parents_children(self):
        return self._parents, self._children


def test_no_property():

    prop = tested.NoProperty(None)

    res_get = prop.get(None)
    res_inf = prop.infinite_component(None)

    assert isinstance(res_get, list)
    assert isinstance(res_inf, list)

    npt.assert_equal(len(res_get), 0)
    npt.assert_equal(len(res_inf), 0)


def test_persistent_angles():

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ]),
        parents={1: 0},
        children={1: [2, 3]},
        begs=[-1, 1]
    )

    prop = tested.PersistentAngles(tree)

    res_get = prop.get(1)
    res_inf = prop.infinite_component(0)

    assert isinstance(res_get, list)
    assert isinstance(res_inf, list)

    npt.assert_allclose(res_get, [0., 0., 0.5 * np.pi, -0.5 * np.pi])
    npt.assert_array_equal(res_inf, [np.nan, np.nan, np.nan, np.nan])


def test_section_mean_radii():

    radii = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    section_begs = np.array([0, 2, 5])
    section_ends = np.array([2, 5, 6])

    expected_mean_radii = [0.15, 0.4, 0.6]

    npt.assert_allclose(
        tested.PersistentMeanRadius._section_mean_radii(radii, section_begs, section_ends),
        expected_mean_radii
    )


def test_persistent_mean_radius():

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ]),
        begs=np.array([0, 1, 2, 3]),
        ends=np.array([1, 2, 3, 4])
    )
    tree.d = 2.0 * np.array([0.1, 0.2, 0.3, 0.4])

    prop = tested.PersistentMeanRadius(tree)

    res_get = prop.get(2)
    res_inf = prop.infinite_component(0)

    assert isinstance(res_get, list)
    assert isinstance(res_inf, list)

    npt.assert_array_equal(res_get, [0.3])
    npt.assert_array_equal(res_inf, [0.1])


def test_phi_theta():

    func = tested.PersistentAngles._phi_theta

    u = np.array([1., 0., 0.])
    v = np.array([1., 0., 0.])

    npt.assert_allclose(func(u, v), [0., 0.])
    npt.assert_allclose(func(v, u), [0., 0.])
    npt.assert_allclose(func(3. * u, 4. * v), [0., 0.])

    u = np.array([1., 0., 0.])
    v = np.array([0., 0., 1.])

    npt.assert_allclose(func(u, v), [0., -np.pi * 0.5])
    npt.assert_allclose(func(v, u), [0., +np.pi * 0.5])
    npt.assert_allclose(func(2. * v, 3. * u), [0., np.pi * 0.5])

    u = np.array([1., 0., 0.])
    v = np.array([-1., 0., 0.])

    npt.assert_allclose(func(u, v), [np.pi, 0.])
    npt.assert_allclose(func(v, u), [-np.pi, 0.])
    npt.assert_allclose(func(2. * v, 3. * u), [-np.pi, 0.])

    u = np.array([1., 1., 0.])
    v = np.array([1., 0., 1.])

    npt.assert_allclose(func(u, v), [-0.25 * np.pi, -0.25 * np.pi])
    npt.assert_allclose(func(v, u), [0.25 * np.pi, 0.25 * np.pi])
    npt.assert_allclose(func(2. * v, 3. * u), [0.25 * np.pi, 0.25 * np.pi])


def test_angles_tree():

    func = tested.PersistentAngles._angles_tree

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.]
        ])
    )

    angles = func(tree, parID=0, parEND=1, ch1ID=2, ch2ID=3)
    npt.assert_allclose(angles, [0., -0.5 * np.pi, 0.5 * np.pi, 0.])

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ])
    )

    angles = func(tree, parID=0, parEND=1, ch1ID=2, ch2ID=3)
    npt.assert_allclose(angles, [0., 0., 0.5 * np.pi, -0.5 * np.pi])

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.]
        ])
    )

    angles = func(tree, parID=0, parEND=1, ch1ID=2, ch2ID=3)
    npt.assert_allclose(angles, [0., 0., 0.5 * np.pi, -0.5 * np.pi])


def test_get_angles():

    tree = MockTree(
        np.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ])
    )

    begs = [-1, 1]

    children = {1: [2, 3]}
    parents = {1: 0}

    angles = tested.PersistentAngles._get_angles(tree, begs, parents, children)
    npt.assert_allclose(angles, [[0., 0., 0., 0.], [0., 0., 0.5 * np.pi, -0.5 * np.pi]])

