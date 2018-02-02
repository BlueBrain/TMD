'''
tmd class : Tree
'''


class Tree(object):
    '''Tree class'''
    import numpy as _np
    from tmd.Tree.methods import size
    from tmd.Tree.methods import get_sections
    from tmd.Tree.methods import get_sections_2
    from tmd.Tree.methods import get_section_number
    from tmd.Tree.methods import get_section_lengths
    from tmd.Tree.methods import get_section_radial_distances
    from tmd.Tree.methods import get_section_path_distances
    from tmd.Tree.methods import get_section_start
    from tmd.Tree.methods import get_segment_lengths
    from tmd.Tree.methods import get_segment_radial_distances
    from tmd.Tree.methods import get_point_radial_distances
    from tmd.Tree.methods import get_point_radial_distances_time
    from tmd.Tree.methods import get_point_weighted_radial_distances
    from tmd.Tree.methods import get_point_path_distances
    from tmd.Tree.methods import get_point_projection
    from tmd.Tree.methods import get_point_section_lengths
    from tmd.Tree.methods import get_point_section_branch_orders
    from tmd.Tree.methods import get_bif_term
    from tmd.Tree.methods import get_trunk
    from tmd.Tree.methods import get_bifurcations
    from tmd.Tree.methods import get_multifurcations
    from tmd.Tree.methods import get_terminations
    from tmd.Tree.methods import get_way_to_root
    from tmd.Tree.methods import get_way_to_section_end
    from tmd.Tree.methods import get_way_to_section_start
    from tmd.Tree.methods import get_bounding_box
    from tmd.Tree.methods import get_segments
    from tmd.Tree.methods import get_type
    from tmd.Tree.methods import get_children
    from tmd.Tree.methods import get_bif_angles
    from tmd.Tree.methods import get_direction
    from tmd.Tree.methods import get_direction_between
    from tmd.Tree.methods import get_pca

    def __init__(self, x=_np.array([]), y=_np.array([]), z=_np.array([]),
                 d=_np.array([]), t=_np.array([]), p=_np.array([])):
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
        import numpy as _np
        import scipy.sparse as sp

        self.x = _np.array(x, dtype=float)
        self.y = _np.array(y, dtype=float)
        self.z = _np.array(z, dtype=float)
        self.d = _np.array(d, dtype=float)
        self.t = _np.array(t, dtype=int)
        self.p = _np.array(p, dtype=int)
        self.dA = sp.csr_matrix((_np.ones(len(self.x) - 1),
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
        import numpy as np

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
        import numpy as np

        new_x = self.x * np.cos(angle) - self.y * np.sin(angle)
        new_y = self.x * np.sin(angle) + self.y * np.cos(angle)

        self.x = new_x
        self.y = new_y


    def move_to_point(self, point=(0, 0, 0)):
        """Moves the tree in the x-y-z plane
        so that it starts from the selected point.
        """
        import numpy as np

        self.x = self.x - (self.x[0]) + point[0]
        self.y = self.y - (self.y[0]) + point[1]
        self.z = self.z - (self.z[0]) + point[2]

