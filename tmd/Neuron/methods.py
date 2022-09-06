"""TMD Neuron's methods."""

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


def size(self, neurite_type="all"):
    """Neuron method to get size."""
    if neurite_type == "all":
        neurite_list = ["basal_dendrite", "axon", "apical_dendrite"]

    s = np.sum([len(getattr(self, neu)) for neu in neurite_list])

    return int(s)


def get_bounding_box(self):
    """Get the bounding box of the neurites.

    Args:
        neuron: A TMD neuron.

    Returns:
        bounding_box: np.array
            ([xmin,ymin,zmin], [xmax,ymax,zmax])
    """
    x = []
    y = []
    z = []

    for tree in self.neurites:
        x = x + tree.x.tolist()
        y = y + tree.y.tolist()
        z = z + tree.z.tolist()

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
