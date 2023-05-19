"""Topology vectorization algorithms."""

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
from scipy import stats

from .statistics import get_lengths


def get_limits(phs_list):
    """Returns the x-y coordinates limits (min, max) for a list of persistence diagrams."""
    if any((isinstance(ph[0], list) for ph in phs_list)):
        phs = [list(ph_bar) for ph in phs_list for ph_bar in ph]
    else:
        phs = phs_list
    xlim = [min(np.transpose(phs)[0]), max(np.transpose(phs)[0])]
    ylim = [min(np.transpose(phs)[1]), max(np.transpose(phs)[1])]
    return xlim, ylim


def persistence_image_data(
    ph, norm_factor=None, xlim=None, ylim=None, bw_method=None, weights=None, resolution=100
):
    """Create the data for the generation of the persistence image.

    Args:
        ph: persistence diagram.
        norm_factor: persistence image data are normalized according to this.
            If norm_factor is provided the data will be normalized based on this,
            otherwise they will be normalized to 1.
        xlim: The image limits on x axis.
        ylim: The image limits on y axis.
        bw_method: The method used to calculate the estimator bandwidth for the gaussian_kde.
        weights: weights of the diagram points
        resolution: number of pixels in each dimension

    If xlim, ylim are provided the data will be scaled accordingly.
    """
    if xlim is None or xlim is None:
        xlim, ylim = get_limits(ph)
    res = complex(0, resolution)
    X, Y = np.mgrid[xlim[0] : xlim[1] : res, ylim[0] : ylim[1] : res]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values, bw_method=bw_method, weights=weights)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = np.max(Z)

    return Z / norm_factor


def _index_bar(ph_bar, t):
    """Computes if a bar is present at time t."""
    if min(ph_bar) <= t <= max(ph_bar):
        return 1
    else:
        return 0


def betti_curve(ph_diagram, bins=None, num_bins=1000):
    """Computes the betti curves of a persistence diagram.

    Corresponding to the number of bars at each distance t.
    """
    if bins is None:
        t_list = np.linspace(np.min(ph_diagram), np.max(ph_diagram), num_bins)
    else:
        t_list = bins
    betti_c = [np.sum([_index_bar(p, t) for p in ph_diagram]) for t in t_list]
    return betti_c, t_list


def _total_lifetime(ph_diagram):
    """Sums the total lengths of all bars."""
    return np.sum(get_lengths(ph_diagram))


def _bar_entropy(ph_bar, lifetime):
    """Absolute difference of a bar divided by lifetime."""
    Zn = np.abs(ph_bar[0] - ph_bar[1]) / lifetime
    return Zn * np.log(Zn)


def life_entropy_curve(ph_diagram, bins=None, num_bins=1000):
    """The life entropy curve, computes life entropy at different t values."""
    lifetime = _total_lifetime(ph_diagram)
    # Compute the entropy of each bar
    entropy = [_bar_entropy(ph_bar, lifetime) for ph_bar in ph_diagram]
    if bins is None:
        t_list = np.linspace(np.min(ph_diagram), np.max(ph_diagram), num_bins)
    else:
        t_list = bins
    t_entropy = [
        -np.sum([_index_bar(ph_bar, t) * e for (e, ph_bar) in zip(entropy, ph_diagram)])
        for t in t_list
    ]
    return t_entropy, t_list
