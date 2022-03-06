'''Test tmd.Tree'''
import numpy as np
from numpy import testing as npt
from tmd.Tree import Tree

x1 = np.array([0.,  3.,  4.,  5.,  5.])
y1 = np.array([0.,  4.,  5.,  6.,  6.])
z1 = np.array([ 0.,  5.,  6.,  7.,  7.])
d1 = np.array([12.,  12.,  14.,  16.,  16.])
t1 = np.array([1, 1, 1, 1, 1])
p1 = np.array([-1,  0,  1, 2, 3])

x2 = np.array([0.,  3.,  4.,  5.,  4.])

def test_tree_init_():
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    npt.assert_allclose(tree1.x, x1)
    npt.assert_allclose(tree1.y, y1)
    npt.assert_allclose(tree1.z, z1)
    npt.assert_allclose(tree1.d, d1)
    npt.assert_allclose(tree1.t, t1)
    npt.assert_allclose(tree1.p, p1)

def test_tree_is_equal():
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    assert tree1.is_equal(tree2)

    tree3 = Tree.Tree(x=x2, y=y1, z=z1, d=d1, t=t1, p=p1)
    assert not tree1.is_equal(tree3)

def test_copy_tree():
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = tree1.copy_tree()
    assert tree1.is_equal(tree2)
    assert tree1 != tree2
