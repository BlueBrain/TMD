"""TMD class : Soma."""

# Copyright (C) 2022  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy

import numpy as np


class Soma:
    """Class of neuron soma.

    Args:
        x (list[float]): The x-coordinates of surface trace of neuron soma.
        y (list[float]): The y-coordinates of surface trace of neuron soma.
        z (list[float]): The z-coordinate of surface trace of neuron soma.
        d (list[float]): The diameters of surface trace of neuron soma.
    """

    # pylint: disable=import-outside-toplevel
    from tmd.Soma.methods import get_center
    from tmd.Soma.methods import get_diameter

    def __init__(self, x=None, y=None, z=None, d=None):
        """Constructor for tmd Soma Object."""
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = []
        if d is None:
            d = []
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.z = np.array(z, dtype=float)
        self.d = np.array(d, dtype=float)

    def copy_soma(self):
        """Returns a deep copy of the Soma."""
        return copy.deepcopy(self)

    def is_equal(self, soma):
        """Tests if all soma data are the same."""
        eq = np.all(
            [
                np.allclose(self.x, soma.x, atol=1e-4),
                np.allclose(self.y, soma.y, atol=1e-4),
                np.allclose(self.z, soma.z, atol=1e-4),
                np.allclose(self.d, soma.d, atol=1e-4),
            ]
        )
        return eq
