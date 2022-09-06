"""Test tmd.soma."""
# pylint: disable=redefined-outer-name
import pytest
from numpy import testing as npt

from tmd.Soma import Soma


@pytest.fixture
def soma1():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    return Soma.Soma([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])


def test_get_center(soma_test, soma1):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    npt.assert_allclose(soma_test.get_center(), [0.0, 0.0, 0.0])
    npt.assert_allclose(soma1.get_center(), [0.0, 1.0 / 3.0, 1.0 / 3.0])


def test_get_diameter(soma_test, soma1):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert soma_test.get_diameter() == 12.0
    assert soma1.get_diameter() == 0.65403883526363049
