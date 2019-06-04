'''Test tmd.soma'''
from nose import tools as nt
import numpy as np
from tmd.Soma import Soma

x1 = np.array([0.,  3.,  4.,  5.,  5.])
y1 = np.array([0.,  4.,  5.,  6.,  6.])
z1 = np.array([ 0.,  5.,  6.,  7.,  7.])
d1 = np.array([12.,  12.,  14.,  16.,  16.])

x2 = np.array([0.,  3.,  4.,  5.,  4.])

def test_soma_init_():
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    nt.ok_(np.allclose(soma1.x, x1))
    nt.ok_(np.allclose(soma1.y, y1))
    nt.ok_(np.allclose(soma1.z, z1))
    nt.ok_(np.allclose(soma1.d, d1))

def test_soma_is_equal():
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    soma2 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    nt.ok_(soma1.is_equal(soma2))
    soma2 = Soma.Soma(x=x2, y=y1, z=z1, d=d1)
    nt.ok_(not soma1.is_equal(soma2))

def test_copy_soma():
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    soma2 = soma1.copy_soma()
    nt.ok_(soma1.is_equal(soma2))
    nt.ok_(soma1 != soma2)
