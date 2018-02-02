'''Test tmd.soma'''
from nose import tools as nt
import numpy as np
from tmd.Soma import Soma

soma = Soma.Soma([0.],[0.],[0.],[12.])
soma1 = Soma.Soma([0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.])

def test_get_center():
    nt.ok_(np.allclose(soma.get_center(), [0., 0., 0.]))
    nt.ok_(np.allclose(soma1.get_center(), [0., 1./3., 1./3.]))

def test_get_diameter():
    nt.ok_(soma.get_diameter() == 12.)
    nt.ok_(soma1.get_diameter() == 0.65403883526363049)
