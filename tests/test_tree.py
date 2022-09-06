"""Test tmd.Tree."""
import numpy as np
from numpy import testing as npt

from tmd.Tree import Tree

x1 = np.array([0.0, 3.0, 4.0, 5.0, 5.0])
y1 = np.array([0.0, 4.0, 5.0, 6.0, 6.0])
z1 = np.array([0.0, 5.0, 6.0, 7.0, 7.0])
d1 = np.array([12.0, 12.0, 14.0, 16.0, 16.0])
t1 = np.array([1, 1, 1, 1, 1])
p1 = np.array([-1, 0, 1, 2, 3])

x2 = np.array([0.0, 3.0, 4.0, 5.0, 4.0])


def test_tree_init_():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    npt.assert_allclose(tree1.x, x1)
    npt.assert_allclose(tree1.y, y1)
    npt.assert_allclose(tree1.z, z1)
    npt.assert_allclose(tree1.d, d1)
    npt.assert_allclose(tree1.t, t1)
    npt.assert_allclose(tree1.p, p1)


def test_tree_is_equal():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    assert tree1.is_equal(tree2)

    tree3 = Tree.Tree(x=x2, y=y1, z=z1, d=d1, t=t1, p=p1)
    assert not tree1.is_equal(tree3)


def test_copy_tree():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = tree1.copy_tree()
    assert tree1.is_equal(tree2)
    assert tree1 != tree2
