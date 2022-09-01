"""Test tmd.soma"""
import numpy as np
from numpy import testing as npt

from tmd.Soma import Soma

soma = Soma.Soma([0.0], [0.0], [0.0], [12.0])
soma1 = Soma.Soma([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])


def test_get_center():
    npt.assert_allclose(soma.get_center(), [0.0, 0.0, 0.0])
    npt.assert_allclose(soma1.get_center(), [0.0, 1.0 / 3.0, 1.0 / 3.0])


def test_get_diameter():
    assert soma.get_diameter() == 12.0
    assert soma1.get_diameter() == 0.65403883526363049
