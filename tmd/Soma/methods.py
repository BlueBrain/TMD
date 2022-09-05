"""TMD Soma's methods."""

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

import numpy as np


def get_center(self):
    """Soma method to get the center of the soma."""
    x_center = np.mean(self.x)
    y_center = np.mean(self.y)
    z_center = np.mean(self.z)

    return np.array([x_center, y_center, z_center])


def get_diameter(self):
    """Soma method to get the diameter of the soma."""
    if len(self.x) == 1:
        diameter = self.d[0]
    else:
        center = self.get_center()
        diameter = np.mean(
            np.sqrt(
                np.power(self.x - center[0], 2)
                + np.power(self.y - center[1], 2)
                + np.power(self.z - center[2], 2)
            )
        )
    return diameter
