'''
tmd class : Tree
'''

import numpy as np
import scipy.sparse as sp


class Tree(object):
    '''Tree class'''
    from tmd.Tree.methods import (get_sections_2, get_sections_only_points, get_segments,
                                  get_bounding_box, get_pca, get_type, get_direction_between,
                                  get_point_radial_distances, get_point_radial_distances_time,
                                  get_point_weighted_radial_distances, get_point_path_distances,
                                  get_point_projection, get_point_section_lengths,
                                  get_point_section_branch_orders, get_bif_term,
                                  get_bifurcations, get_multifurcations, get_terminations)

    @classmethod
    def from_morphio(cls, root_section):
        '''Factory from the morphio object'''
        sections = list(root_section.iter())
        points = np.vstack([section.points for section in sections])
        diameters = np.hstack([section.diameters for section in sections])
        types = np.hstack([[section.type] * len(section.points) for section in sections])

        counter = -1
        parents = list()
        section_last_point_id = dict()
        for section in sections:
            for i, _ in enumerate(section.points):
                parent_id = counter
                counter += 1
                if i == 0:
                    parent_id = -1 if section.is_root else section_last_point_id[section.parent.id]
                parents.append(parent_id)
            section_last_point_id[section.id] = counter
        return Tree(points[:, 0], points[:, 1], points[:, 2],
                    diameters, types, parents)

    def __init__(self, x=np.array([]), y=np.array([]), z=np.array([]),
                 d=np.array([]), t=np.array([]), p=np.array([])):
        '''Constructor of tmd Tree Object

        Parameters
        ----------
        x : numpy array
            The x-coordinates of neuron's tree segments.
        y : numpy array
            The y-coordinates of neuron's tree segments.
        z : numpy array
            The z-coordinate of neuron's tree segments.
        d : numpy array
            The diameters of neuron's tree segments.
        t : numpy array
            The types (basal, apical, axon) of neuron's tree segments.
        p : numpy array
            The index of the parent of neuron's tree segments.

        Returns
        -------
        tree : Tree
            tmd Tree object
        '''
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.z = np.array(z, dtype=float)
        self.d = np.array(d, dtype=float)
        self.t = np.array(t, dtype=int)
        self.p = np.array(p, dtype=int)
        self.dA = sp.csr_matrix((np.ones(len(self.x) - 1),
                                 (range(1, len(self.x)), self.p[1:])),
                                shape=(len(self.x), len(self.x)))

    def copy_tree(self):
        """
        Returns a deep copy of the Tree.
        """

        import copy

        return copy.deepcopy(self)

    def is_equal(self, tree):
        '''Tests if all tree lists are the same'''
        eq = np.alltrue([np.allclose(self.x, tree.x, atol=1e-4),
                         np.allclose(self.y, tree.y, atol=1e-4),
                         np.allclose(self.z, tree.z, atol=1e-4),
                         np.allclose(self.d, tree.d, atol=1e-4),
                         np.allclose(self.t, tree.t, atol=1e-4),
                         np.allclose(self.p, tree.p, atol=1e-4)])
        return eq

    def rotate_xy(self, angle):
        """Rotates the tree in the x-y plane
        by the defined angle.
        """
        new_x = self.x * np.cos(angle) - self.y * np.sin(angle)
        new_y = self.x * np.sin(angle) + self.y * np.cos(angle)

        self.x = new_x
        self.y = new_y

    def move_to_point(self, point=(0, 0, 0)):
        """Moves the tree in the x-y-z plane
        so that it starts from the selected point.
        """
        self.x = self.x - (self.x[0]) + point[0]
        self.y = self.y - (self.y[0]) + point[1]
        self.z = self.z - (self.z[0]) + point[2]

    def extract_simplified(self):
        """Returns a simplified tree that corresponds
           to the start - end of the sections points
        """
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
        p[0] = - 1

        for i, s in enumerate(sections):
            x[i + 1] = self.x[s[1]]
            y[i + 1] = self.y[s[1]]
            z[i + 1] = self.z[s[1]]
            d[i + 1] = self.d[s[1]]
            t[i + 1] = self.t[s[1]]
            p[i + 1] = np.where(beg0 == s[0])[0][0]

        return Tree(x, y, z, d, t, p)
