"""TMD statistical analysis on PH diagrams algorithms implementation."""

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


def get_bifurcations(ph):
    """Return the bifurcations from the diagram."""
    return np.array(ph)[:, 1]


def get_terminations(ph):
    """Return the terminations from the diagram."""
    return np.array(ph)[:, 0]


def get_lengths(ph):
    """Return the lengths of the bars from the diagram."""
    return np.array([np.abs(i[0] - i[1]) for i in ph])


def get_total_length(ph):
    """Calculate the total length of a barcode.

    The total length is computed by summing the length of each bar.
    This should be equivalent to the total length of the tree if the barcode represents path
    distances.
    """
    return sum(np.abs(p[1] - p[0]) for p in ph)


def transform_ph_to_length(ph, keep_side="end"):
    """Transform a persistence diagram into a (end, length) equivalent diagram.

    If `keep_side == "start"`, return a (start_point, length) diagram.

    .. note::

        The direction of the diagram will be lost!
    """
    if keep_side == "start":
        # keeps the start point and the length of the bar
        return [[min(i), np.abs(i[1] - i[0])] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[max(i), np.abs(i[1] - i[0])] for i in ph]


def transform_ph_from_length(ph, keep_side="end"):
    """Transform a persistence diagram into a (end_point, length) equivalent diagram.

    If `keep_side == "start"`, return a (start_point, length) diagram.

    .. note::

        The direction of the diagram will be lost!
    """
    if keep_side == "start":
        # keeps the start point and the length of the bar
        return [[i[0], i[1] - i[0]] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[i[0] - i[1], i[0]] for i in ph]


def nosify(var, noise=0.1):
    r"""Adds noise to an instance of data.

    Can be used with a ph as follows:

    .. code-block:: Python

        noisy_pd = [add_noise(d, 1.0) if d[0] != 0.0
                    else [d[0],add_noise([d[1]],1.0)[0]] for d in pd]

    To output the new pd:

    .. code-block:: Python

        F = open(...)
        for d in noisy_pd:
            towrite = '%f, %f\n'%(d[0],d[1])
            F.write(towrite)
        F.close()
    """
    var_new = np.zeros(len(var))
    for i, v in enumerate(var):
        var_new[i] = stats.norm.rvs(v, noise)
    return var_new
