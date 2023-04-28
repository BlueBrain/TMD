"""Persistent properties classes."""

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

from abc import ABC
from abc import abstractmethod

import numpy as np


class PersistentProperty(ABC):
    """Abstract class for persistent properties that are defined on persistent components."""

    @abstractmethod
    def get(self, component_start):
        """Get component property."""

    @abstractmethod
    def infinite_component(self, component_start):
        """Get property for infinite component."""


class NoProperty(PersistentProperty):
    """Function class for extracting a barcode without properties."""

    def __init__(self, _):
        pass

    def get(self, _):
        """Returns empty list, does not contribute to component."""
        return []

    def infinite_component(self, _):
        """Returns empty list, does not contribute to component."""
        return []


class PersistentMeanRadius(PersistentProperty):
    """Component mean radii.

    Args:
        tree (Tree): A tree object.
    """

    def __init__(self, tree):
        section_begs, section_ends = tree.sections
        self._radii = self._section_mean_radii(0.5 * tree.d, section_begs, section_ends)

    def get(self, component_start):
        """Get one persistent mean radius.

        Args:
            component_start (int): The component start.

        Returns:
            component_angles (list): A list of 1 radius.
        """
        return [self._radii[component_start]]

    def infinite_component(self, component_start):
        """Returns mean radius corresponding to inf component."""
        return self.get(component_start)

    @staticmethod
    def _section_mean_radii(tree_radii, section_begs, section_ends):
        """Returns the mean radius per section."""
        return np.fromiter(
            (np.mean(tree_radii[b:e]) for b, e in zip(section_begs, section_ends)), dtype=float
        )


class PersistentAngles(PersistentProperty):
    """Bifurcation angles per component.

    Args:
        tree (Tree): A tree object.
    """

    def __init__(self, tree):
        section_begs, _ = tree.sections
        section_parents, section_children = tree.parents_children

        self._angles = self._get_angles(tree, section_begs, section_parents, section_children)

    def get(self, component_start):
        """Get one persistent angle.

        Args:
            component_start (int): The component start.

        Returns:
            component_angles (list): A list of 4 angles.
        """
        return self._angles[component_start]

    def infinite_component(self, _):
        """Given that there are not angles for the inf component, nans are returned."""
        return [np.nan, np.nan, np.nan, np.nan]

    @staticmethod
    def _phi_theta(u, v):
        """Compute the angles between vectors u, v in the plane x-y (phi) and the plane x-z (theta).

        Args:
            u (np.ndarray): (3,) First vector
            v (np.ndarray): (3,) Second vector

        Returns:
            delta_phi (float): Difference of phi_angles phi_v - phi_u
                on the x-y plane
            delta_theta (float): Difference of theta_angles th_v - th_u
                on the x-z plane
        """
        phi1 = np.arctan2(u[1], u[0])
        # pylint: disable=assignment-from-no-return
        theta1 = np.arccos(u[2] / np.linalg.norm(u))

        # pylint: disable=assignment-from-no-return
        phi2 = np.arctan2(v[1], v[0])
        theta2 = np.arccos(v[2] / np.linalg.norm(v))

        delta_phi = phi2 - phi1  # np.abs(phi1 - phi2)
        delta_theta = theta2 - theta1  # np.abs(theta1 - theta2)

        return delta_phi, delta_theta  # dphi, dtheta

    @staticmethod
    def _angles_tree(tree, parID, parEND, ch1ID, ch2ID):
        """Compute the x-y and x-z angles between parent and children within the given tree.

        Args:
            tree (Tree): Morphology tree
            parID (int): Id of parent section
            parEND (int): Id of parent section end
            ch1ID (int): ID of first child
            ch2ID (int): ID of section child

        Returns:
            list:
                dphi (float):
                    Absolute difference of phi_angles between parent and first child
                dtheta (float):
                    Absolute difference of theta_angles between parent and first child
                delta_phi (float): Difference of phi_angles phi_v - phi_u
                    on the x-y plane
                delta_theta (float): Difference of theta_angles th_v - th_u
                    on the x-z plane
        """
        parent_direction = tree.get_direction_between(start_id=parID, end_id=parEND)
        child1_direction = tree.get_direction_between(start_id=parEND, end_id=ch1ID)
        child2_direction = tree.get_direction_between(start_id=parEND, end_id=ch2ID)

        phi1, theta1 = PersistentAngles._phi_theta(parent_direction, child1_direction)
        phi2, theta2 = PersistentAngles._phi_theta(parent_direction, child2_direction)

        if np.abs(phi1) < np.abs(phi2):
            dphi = phi1
            dtheta = theta1
            delta_phi, delta_theta = PersistentAngles._phi_theta(child1_direction, child2_direction)
        else:
            dphi = phi2
            dtheta = theta2
            delta_phi, delta_theta = PersistentAngles._phi_theta(child2_direction, child1_direction)

        return [dphi, dtheta, delta_phi, delta_theta]

    @staticmethod
    def _get_angles(tree, beg, parents, children):
        """Return the angles between all the triplets (parent, child1, child2) of the tree."""
        angles = [
            [0, 0, 0, 0],
        ]  # Null angle for non bif point

        for b in beg[1:]:
            angleBetween = PersistentAngles._angles_tree(
                tree, parID=parents[b], parEND=b, ch1ID=children[b][0], ch2ID=children[b][1]
            )

            angles.append(angleBetween)

        return angles
