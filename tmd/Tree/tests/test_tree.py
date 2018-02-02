'''Test tmd.Tree'''
from nose import tools as nt
import numpy as np
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
    nt.ok_(np.allclose(tree1.x, x1))
    nt.ok_(np.allclose(tree1.y, y1))
    nt.ok_(np.allclose(tree1.z, z1))
    nt.ok_(np.allclose(tree1.d, d1))
    nt.ok_(np.allclose(tree1.t, t1))
    nt.ok_(np.allclose(tree1.p, p1))

def test_tree_is_equal():
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    nt.ok_(tree1.is_equal(tree2))
    tree3 = Tree.Tree(x=x2, y=y1, z=z1, d=d1, t=t1, p=p1)
    nt.ok_(not tree1.is_equal(tree3))

def test_copy_tree():
    tree1 = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)
    tree2 = tree1.copy_tree()
    nt.ok_(tree1.is_equal(tree2))
    nt.ok_(tree1 != tree2)
