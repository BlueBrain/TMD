"""Test tmd.soma."""
import numpy as np
from numpy import testing as npt

from tmd.Soma import Soma

x1 = np.array([0.0, 3.0, 4.0, 5.0, 5.0])
y1 = np.array([0.0, 4.0, 5.0, 6.0, 6.0])
z1 = np.array([0.0, 5.0, 6.0, 7.0, 7.0])
d1 = np.array([12.0, 12.0, 14.0, 16.0, 16.0])

x2 = np.array([0.0, 3.0, 4.0, 5.0, 4.0])


def test_soma_init_():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    npt.assert_allclose(soma1.x, x1)
    npt.assert_allclose(soma1.y, y1)
    npt.assert_allclose(soma1.z, z1)
    npt.assert_allclose(soma1.d, d1)


def test_soma_is_equal():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    soma2 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    assert soma1.is_equal(soma2)

    soma2 = Soma.Soma(x=x2, y=y1, z=z1, d=d1)
    assert not soma1.is_equal(soma2)


def test_copy_soma():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    soma1 = Soma.Soma(x=x1, y=y1, z=z1, d=d1)
    soma2 = soma1.copy_soma()
    assert soma1.is_equal(soma2)
    assert soma1 != soma2
