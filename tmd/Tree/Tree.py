"""TMD class : Tree."""

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
import scipy.sparse as sp
from cached_property import cached_property


class Tree:
    """Tree class.

    Args:
        x (list[float]): The x-coordinates of the tree segments.
        y (list[float]): The y-coordinates of the tree segments.
        z (list[float]): The z-coordinate of the tree segments.
        d (list[float]): The diameters of the tree segments.
        t (list[int]): The types (basal_dendrite, apical_dendrite, axon) of the tree segments.
        p (list[int]): The index of the parent of the tree segments.
    """

    # pylint: disable=import-outside-toplevel
    from tmd.Tree.methods import get_bif_term
    from tmd.Tree.methods import get_bifurcations
    from tmd.Tree.methods import get_bounding_box
    from tmd.Tree.methods import get_direction_between
    from tmd.Tree.methods import get_multifurcations
    from tmd.Tree.methods import get_pca
    from tmd.Tree.methods import get_point_path_distances
    from tmd.Tree.methods import get_point_projection
    from tmd.Tree.methods import get_point_radial_distances
    from tmd.Tree.methods import get_point_radial_distances_time
    from tmd.Tree.methods import get_point_section_branch_orders
    from tmd.Tree.methods import get_point_section_lengths
    from tmd.Tree.methods import get_point_weighted_radial_distances
    from tmd.Tree.methods import get_sections_2
    from tmd.Tree.methods import get_sections_only_points
    from tmd.Tree.methods import get_segments
    from tmd.Tree.methods import get_terminations
    from tmd.Tree.methods import get_trunk_length
    from tmd.Tree.methods import get_type

    def __init__(self, x, y, z, d, t, p):
        """Constructor of tmd Tree Object."""
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.z = np.array(z, dtype=np.float32)
        self.d = np.array(d, dtype=np.float32)
        self.t = np.array(t, dtype=np.int32)
        self.p = np.array(p, dtype=np.int64)
        self.dA = sp.csr_matrix(
            (np.ones(len(self.x) - 1), (range(1, len(self.x)), self.p[1:])),
            shape=(len(self.x), len(self.x)),
        )

    def copy_tree(self):
        """Returns a deep copy of the Tree."""
        return copy.deepcopy(self)

    def is_equal(self, tree):
        """Tests if all tree lists are the same."""
        eq = np.all(
            [
                np.allclose(self.x, tree.x, atol=1e-4),
                np.allclose(self.y, tree.y, atol=1e-4),
                np.allclose(self.z, tree.z, atol=1e-4),
                np.allclose(self.d, tree.d, atol=1e-4),
                np.allclose(self.t, tree.t, atol=1e-4),
                np.allclose(self.p, tree.p, atol=1e-4),
            ]
        )
        return eq

    def rotate_xy(self, angle):
        """Rotates the tree in the x-y plane by the defined angle."""
        new_x = self.x * np.cos(angle) - self.y * np.sin(angle)
        new_y = self.x * np.sin(angle) + self.y * np.cos(angle)

        self.x = new_x
        self.y = new_y

    def move_to_point(self, point=(0, 0, 0)):
        """Moves the tree in the x-y-z plane so that it starts from the selected point."""
        self.x = self.x - (self.x[0]) + point[0]
        self.y = self.y - (self.y[0]) + point[1]
        self.z = self.z - (self.z[0]) + point[2]

    def extract_simplified(self):
        """Returns a simplified tree that corresponds to the start - end of the sections points."""
        beg0, end0 = self.get_sections_2()
        sections = np.transpose([beg0, end0])

        x = np.zeros([len(sections) + 1])
        y = np.zeros([len(sections) + 1])
        z = np.zeros([len(sections) + 1])
        d = np.zeros([len(sections) + 1])
        t = np.zeros([len(sections) + 1])
        p = np.zeros([len(sections) + 1])

        x[0] = self.x[sections[0][0]]
        y[0] = self.y[sections[0][0]]
        z[0] = self.z[sections[0][0]]
        d[0] = self.d[sections[0][0]]
        t[0] = self.t[sections[0][0]]
        p[0] = -1

        for i, s in enumerate(sections):
            x[i + 1] = self.x[s[1]]
            y[i + 1] = self.y[s[1]]
            z[i + 1] = self.z[s[1]]
            d[i + 1] = self.d[s[1]]
            t[i + 1] = self.t[s[1]]
            p[i + 1] = np.where(beg0 == s[0])[0][0]

        return Tree(x, y, z, d, t, p)

    @cached_property
    def sections(self):
        """Get the sections of the current tree.

        Returns:
            tuple:
                section_beg_point_ids (np.ndarray):
                    The starting point ids of sections
                section_end_point_ids (np.ndarray)
                    The ending point ids of actions
        """
        return self.get_sections_2()

    @cached_property
    def parents_children(self):
        """Returns the parents and children nodes of each section.

        Returns:
            parents (dict): Each key corresponds to a section id
                and the respective values to the parent section ids
            children (dict): Each key corresponds to a section id
                and the respective values to the children section ids

        Notes:
            If 0 exists in starting nodes, the parent from tree is assigned
        """
        begs, ends = self.sections
        parents = {e: b for b, e in zip(begs, ends)}

        if 0 in begs:
            parents[0] = self.p[0]

        children = {b: ends[np.where(begs == b)[0]] for b in np.unique(begs)}

        return parents, children
