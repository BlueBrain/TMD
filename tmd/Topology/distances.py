"""Topological distances algorithms."""

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

from .vectorizations import betti_curve
from .vectorizations import life_entropy_curve
from .vectorizations import persistence_image_data


def total_betti_diff(ph1, ph2, bins=None, num_bins=1000):
    """Total difference between betti curves."""
    e1 = betti_curve(ph1, bins=bins, num_bins=num_bins)[0]
    e2 = betti_curve(ph2, bins=bins, num_bins=num_bins)[0]
    return np.sum(np.abs(np.subtract(e1, e2)))


def max_betti_diff(ph1, ph2, bins=None, num_bins=1000):
    """Max difference between betti curves."""
    e1 = betti_curve(ph1, bins=bins, num_bins=num_bins)[0]
    e2 = betti_curve(ph2, bins=bins, num_bins=num_bins)[0]
    return np.max(np.abs(np.subtract(e1, e2)))


def total_entropy_diff(ph1, ph2, bins=None, num_bins=1000):
    """Total difference between entropy curves."""
    e1 = life_entropy_curve(ph1, bins=bins, num_bins=num_bins)[0]
    e2 = life_entropy_curve(ph2, bins=bins, num_bins=num_bins)[0]
    return np.sum(np.abs(np.subtract(e1, e2)))


def max_entropy_diff(ph1, ph2, bins=None, num_bins=1000):
    """Max difference between entropy curves."""
    e1 = life_entropy_curve(ph1, bins=bins, num_bins=num_bins)[0]
    e2 = life_entropy_curve(ph2, bins=bins, num_bins=num_bins)[0]
    return np.max(np.abs(np.subtract(e1, e2)))


def image_diff_data(Z1, Z2, normalized=True):
    """Get the difference of two persistence images."""
    if normalized:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()
    return Z1 - Z2


def total_persistence_image_diff(
    ph1, ph2, xlim=None, ylim=None, bw_method=None, weights=None, resolution=100
):
    """Total absolute difference of the respective persistence images."""
    p1 = persistence_image_data(
        ph1, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights, resolution=resolution
    )
    p2 = persistence_image_data(
        ph2, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights, resolution=resolution
    )
    return np.sum(np.abs(image_diff_data(p1, p2)))


def max_persistence_image_diff(
    ph1, ph2, xlim=None, ylim=None, bw_method=None, weights=None, resolution=100
):
    """Max absolute difference of the respective persistence images."""
    p1 = persistence_image_data(
        ph1, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights, resolution=resolution
    )
    p2 = persistence_image_data(
        ph2, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights, resolution=resolution
    )
    return np.max(np.abs(image_diff_data(p1, p2)))
