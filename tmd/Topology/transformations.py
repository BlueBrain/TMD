"""TMD transformation algorithms."""

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


def tmd_scale(barcode, thickness):
    """Scale the first two components according to the thickness parameter.

    Only these components are scaled because they correspond to spatial coordinates.
    """
    scaling_factor = np.ones(len(barcode[0]), dtype=float)
    scaling_factor[:2] = thickness
    return np.multiply(barcode, scaling_factor).tolist()
