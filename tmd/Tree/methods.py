'''
tmd Tree's methods
'''
import copy
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA


def _rd(point1, point2):
    '''Returns euclidean distance between point1 and point2
    '''
    return np.linalg.norm(np.subtract(point1, point2), 2)


def _rd_w(p1, p2, w=(1., 1., 1.), normed=True):
    '''Returns weighted euclidean distance between p1 and p2
    '''
    if normed:
        w = (np.array(w) / np.linalg.norm(w))
    return np.dot(w, (np.subtract(p1, p2)))


def size(tree):
    '''
    Tree method to get the size of the tree lists.

    Note: All the lists of the Tree should be
    of the same size, but this should be
    checked in the initialization of the Tree!
    '''
    return int(len(tree.x))


def get_type(self):
    '''Returns type of tree
    '''
    return int(np.median(self.t))


def get_bounding_box(self):
    """
    Input
    ------
    tree: tmd tree

    Returns
    ---------
    bounding_box: np.array
        ([xmin,ymin,zmin], [xmax,ymax,zmax])
    """
    xmin = np.min(self.x)
    xmax = np.max(self.x)
    ymin = np.min(self.y)
    ymax = np.max(self.y)
    zmin = np.min(self.z)
    zmax = np.max(self.z)

    return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])


# Segment features
def get_segments(self):
    """
    Input
    ------
    tree: tmd tree

    Returns
    ---------
    seg_list: np.array
        (child[x,y,z], parent[x,y,z])
    """
    seg_list = []

    for seg_id in range(1, size(self)):
        par_id = self.p[seg_id]
        child_coords = np.array([self.x[seg_id],
                                 self.y[seg_id],
                                 self.z[seg_id]])
        parent_coords = np.array([self.x[par_id],
                                  self.y[par_id],
                                  self.z[par_id]])
        seg_list.append(np.array([parent_coords, child_coords]))

    return seg_list


def get_segment_lengths(tree):
    '''Returns segment lengths
    '''
    seg_len = np.zeros(size(tree) - 1)
    segs = tree.get_segments()

    for iseg, seg in enumerate(segs):
        seg_len[iseg] = _rd(seg[0], seg[1])

    return seg_len


# Points features to be used for topological extraction
def get_point_radial_distances(self, point=None, dim='xyz'):
    '''Tree method to get radial distances from a point.
    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    '''
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])

    radial_distances = np.zeros(size(self), dtype=float)

    for i in range(size(self)):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])

        radial_distances[i] = _rd(point, point_dest)

    return radial_distances


def get_point_radial_distances_time(self, point=None, dim='xyz', zero_time=0, time=1):
    '''Tree method to get radial distances from a point.
    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    '''
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])
    point.append(zero_time)

    radial_distances = np.zeros(size(self), dtype=float)

    for i in range(size(self)):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])
        point_dest.append(time)

        radial_distances[i] = _rd(point, point_dest)

    return radial_distances


def get_point_weighted_radial_distances(self, point=None, dim='xyz', w=(1, 1, 1), normed=False):
    '''Tree method to get radial distances from a point.
    If point is None, the soma surface -defined by
    the initial point of the tree- will be used
    as a reference point.
    '''
    if point is None:
        point = []
        for d in dim:
            point.append(getattr(self, d)[0])

    radial_distances = np.zeros(size(self), dtype=float)

    for i in range(size(self)):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])

        radial_distances[i] = _rd_w(point, point_dest, w, normed)

    return radial_distances


def get_point_path_distances(self):
    '''Tree method to get path distances from the root.
    '''
    seg_len = get_segment_lengths(self)
    path_lengths = np.append(0, copy.deepcopy(seg_len))
    children = get_children(self)

    for i in children.keys():
        path_lengths[children[i]] = path_lengths[children[i]] + path_lengths[i]

    return path_lengths


def get_point_section_lengths(self):
    '''Tree method to get section lengths.
    '''
    lengths = np.zeros(size(self), dtype=float)
    ways, end = self.get_sections_only_points()
    seg_len = get_segment_lengths(self)

    for i, item in enumerate(end):
        lengths[item] = np.sum(seg_len[ways[i]:item])

    return lengths


def get_branch_order(tree, seg_id):
    '''Returns branch order of segment'''
    B = tree.get_multifurcations()
    return sum([1 if i in B else 0 for i in get_way_to_root(tree, seg_id)])


def get_point_section_branch_orders(self):
    '''Tree method to get section lengths.
    '''
    return np.array([get_branch_order(self, i) for i in range(size(self))])


def get_point_projection(self, vect=(0, 1, 0), point=None):
    """Projects each point in the tree (x,y,z) - input_point
       to a selected vector. This gives the orientation of
       each section according to a vector in space, if normalized,
       otherwise it returns the relative length of the section.
    """
    if point is None:
        point = [self.x[0], self.y[0], self.z[0]]

    xyz = np.transpose([self.x, self.y, self.z]) - point

    return np.dot(xyz, vect)


# Section features
def get_sections_2(self):
    '''Tree method to get the sections'
    begining and ending indices.
    '''
    end = np.array(sp.csr_matrix.sum(self.dA, 0) != 1)[0].nonzero()[0]

    if 0 in end:  # If first segment is a bifurcation
        end = end[1:]

    beg = np.append([0], self.p[np.delete(np.hstack([0, 1 + end]), len(end))][1:])

    return beg, end


def get_sections_only_points(self):
    '''Tree method to get the sections'
    begining and ending indices.
    '''
    end = np.array(sp.csr_matrix.sum(self.dA, 0) != 1)[0].nonzero()[0]

    if 0 in end:  # If first segment is a bifurcation
        end = end[1:]

    beg = np.delete(np.hstack([0, 1 + end]), len(end))

    return beg, end


def get_bif_term(self):
    '''Returns number of children per point
    '''
    return np.array(sp.csr_matrix.sum(self.dA, axis=0))[0]


def get_bifurcations(self):
    '''Returns bifurcations
    '''
    bif_term = get_bif_term(self)
    bif = np.where(bif_term == 2.)[0]
    return bif


def get_multifurcations(self):
    '''Returns bifurcations
    '''
    bif_term = get_bif_term(self)
    bif = np.where(bif_term >= 2.)[0]
    return bif


def get_terminations(self):
    '''Returns terminations
    '''
    bif_term = get_bif_term(self)
    term = np.where(bif_term == 0.)[0]
    return term


def get_direction_between(self, start_id=0, end_id=1):
    '''Returns direction of a branch
    defined as end point - start point
    normalized as a unit vector.
    '''
    # pylint: disable=assignment-from-no-return
    vect = np.subtract([self.x[end_id], self.y[end_id], self.z[end_id]],
                       [self.x[start_id], self.y[start_id], self.z[start_id]])

    if np.linalg.norm(vect) != 0.0:
        return vect / np.linalg.norm(vect)
    return vect


def _vec_angle(u, v):
    '''Returns the angle between v and u in 3D.
    '''
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return np.arccos(c)


def get_angle_between(tree, sec_id1, sec_id2):
    '''Returns local bifurcations angle
    between two sections, defined by their ids.
    sec_id1: the start point of the section #1
    sec_id2: the start point of the section #2
    '''
    beg, end = tree.get_sections_only_points()
    b1 = np.where(beg == sec_id1)[0][0]
    b2 = np.where(beg == sec_id2)[0][0]

    u = tree.get_direction_between(beg[b1],
                                   end[b1])
    v = tree.get_direction_between(beg[b2],
                                   end[b2])

    return _vec_angle(u, v)


def get_way_to_root(tree, sec_id=0):
    '''Returns way to root
    '''
    way = []
    tmp_id = sec_id

    while tmp_id != -1:
        way.append(tree.p[tmp_id])
        tmp_id = tree.p[tmp_id]

    return way


def get_children(tree):
    '''Returns a dictionary of children
       for each node of the tree
    '''
    return OrderedDict({i: np.where(tree.p == i)[0] for i in range(len(tree.p))})


def get_pca(self, plane='xy', component=0):
    '''Returns the i-th principal
    component of PCA on the points
    of the tree in the selected plane
    '''
    pca = PCA(n_components=2)
    pca.fit(np.transpose([getattr(self, plane[0]), getattr(self, plane[1])]))

    return pca.components_[component]
