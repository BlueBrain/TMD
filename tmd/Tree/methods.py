'''
tmd Tree's methods
'''
import numpy as _np


def _rd(point1, point2):
    '''Returns euclidean distance between point1 and point2
    '''
    return _np.linalg.norm(_np.array(point1) - _np.array(point2), 2)


def _rd_w(p1, p2, w=(1., 1., 1.), normed=True):
    '''Returns weighted euclidean distance between p1 and p2
    '''
    if normed:
        w = (_np.array(w) / _np.linalg.norm(w))

    return _np.dot(w, (_np.array(p1) - _np.array(p2)))


def size(self):
    '''
    Tree method to get the size of the tree lists.

    Note: All the lists of the Tree should be
    of the same size, but this should be
    checked in the initialization of the Tree!
    '''
    s = len(self.x)

    return int(s)


def get_type(self):
    '''Returns type of tree
    '''
    return int(_np.median(self.t))


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
    xmin = _np.min(self.x)
    xmax = _np.max(self.x)
    ymin = _np.min(self.y)
    ymax = _np.max(self.y)
    zmin = _np.min(self.z)
    zmax = _np.max(self.z)

    return _np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])


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

    for seg_id in range(1, self.size()):
        par_id = self.p[seg_id]
        child_coords = _np.array([self.x[seg_id],
                                  self.y[seg_id],
                                  self.z[seg_id]])
        parent_coords = _np.array([self.x[par_id],
                                   self.y[par_id],
                                   self.z[par_id]])
        seg_list.append(_np.array([parent_coords, child_coords]))

    return seg_list


def get_segment_lengths(self):
    '''Returns segment lengths
    '''
    seg_len = _np.zeros(self.size() - 1)
    segs = self.get_segments()

    for iseg, seg in enumerate(segs):
        seg_len[iseg] = _rd(seg[0], seg[1])

    return seg_len


def get_segment_radial_distances(self, point=None):
    '''Tree method to get radial distances from a point.
    If point is None, the soma surface -defined by
    the initial segment of the tree- will be used
    as a reference point.
    '''
    segs = self.get_segments()

    if point is None:
        point = [self.x[0], self.y[0], self.z[0]]

    radial_distances = _np.zeros(len(segs), dtype=float)

    for iseg, seg in enumerate(segs):
        radial_distances[iseg] = _rd(point, seg[1])

    return radial_distances


# Points features
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

    radial_distances = _np.zeros(self.size(), dtype=float)

    for i in range(self.size()):
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

    radial_distances = _np.zeros(self.size(), dtype=float)

    for i in range(self.size()):
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

    radial_distances = _np.zeros(self.size(), dtype=float)

    for i in range(self.size()):
        point_dest = []
        for d in dim:
            point_dest.append(getattr(self, d)[i])

        radial_distances[i] = _rd_w(point, point_dest, w, normed)

    return radial_distances


def get_point_path_distances(self):
    '''Tree method to get path distances from the root.
    '''
    seg_len = self.get_segment_lengths()

    def path_length(seg_id):
        '''Returns path length of segment'''
        return sum([seg_len[i] for i in self.get_way_to_root(seg_id)[1:]])

    return _np.array([path_length(i) for i in range(self.size())])


def get_point_section_lengths(self):
    '''Tree method to get section lengths.
    '''
    lengths = _np.zeros(self.size(), dtype=float)
    ways, end = self.get_sections_only_points()
    seg_len = self.get_segment_lengths()

    for i, item in enumerate(end):
        lengths[item] = _np.sum(seg_len[ways[i]:item])

    return lengths


def get_branch_order(self, seg_id):
    '''Returns branch order of segment'''
    B = self.get_multifurcations()
    return sum([1 if i in B else 0 for i in self.get_way_to_root(seg_id)])


def get_point_section_branch_orders(self):
    '''Tree method to get section lengths.
    '''
    return _np.array([self.get_branch_order(i) for i in range(self.size())])


def get_point_projection(self, vect=(0, 1, 0), point=None):
    """Projects each point in the tree (x,y,z) - input_point
       to a selected vector. This gives the orientation of
       each section according to a vector in space, if normalized,
       otherwise it returns the relative length of the section.
    """
    if point is None:
        point = [self.x[0], self.y[0], self.z[0]]

    xyz = _np.transpose([self.x, self.y, self.z]) - point

    return _np.dot(xyz, vect)


# Section features
def get_sections(self):
    '''Tree method to get the sections'
    begining and ending indices.
    '''
    beg = [0]
    end = [self.get_way_to_section_end(0)[-1]]

    for b in self.get_multifurcations():
        children = self.get_children(b)
        for ch in children:
            beg = beg + [b]
            end = end + [self.get_way_to_section_end(ch)[-1]]

    return beg, end


def get_sections_2(self):
    '''Tree method to get the sections'
    begining and ending indices.
    '''
    import scipy.sparse as sp
    end = _np.array(sp.csr_matrix.sum(self.dA, 0) != 1)[0].nonzero()[0]

    if 0 in end:  # If first segment is a bifurcation
        end = end[1:]

    beg = _np.append([0], self.p[_np.delete(_np.hstack([0, 1 + end]), len(end))][1:])

    return beg, end


def get_sections_only_points(self):
    '''Tree method to get the sections'
    begining and ending indices.
    '''
    import scipy.sparse as sp
    end = _np.array(sp.csr_matrix.sum(self.dA, 0) != 1)[0].nonzero()[0]

    if 0 in end:  # If first segment is a bifurcation
        end = end[1:]

    beg = _np.delete(_np.hstack([0, 1 + end]), len(end))

    return beg, end


def extract_simplified(self):
    """Returns a simplified tree that corresponds
       to the start - end of the sections points
    """
    from tmd import Tree
    beg0, end0 = self.get_sections_2()
    sections = _np.transpose([beg0, end0])

    x = _np.zeros([len(sections)+1])
    y = _np.zeros([len(sections)+1])
    z = _np.zeros([len(sections)+1])
    d = _np.zeros([len(sections)+1])
    t = _np.zeros([len(sections)+1])
    p = _np.zeros([len(sections)+1])

    x[0] = self.x[sections[0][0]]
    y[0] = self.y[sections[0][0]]
    z[0] = self.z[sections[0][0]]
    d[0] = self.d[sections[0][0]]
    t[0] = self.t[sections[0][0]]
    p[0] = -1

    for i,s in enumerate(sections):
        x[i+1] = self.x[s[1]]
        y[i+1] = self.y[s[1]]
        z[i+1] = self.z[s[1]]
        d[i+1] = self.d[s[1]]
        t[i+1] = self.t[s[1]]
        p[i+1] = _np.where(beg0 ==s[0])[0][0]

    return Tree.Tree(x,y,z,d,t,p)


def get_section_points(self, sec_start):
    '''Tree method to get the sections' points
    from the start to end indices.
    The sec_start must be the beginning (start) of a section.
    '''
    if sec_start in self.get_sections_only_points():
        return self.get_way_to_section_end(sec_start)
    else:
        return "The input sec_start is not a section start point"


def get_section_projection(self, vect=(0, 1, 0)):
    """Projects each section (i.e., end_point - start_point)
       to a selected vector. This gives the orientation of
       each section according to a vector in space, if normalized,
       otherwise it returns the relative length of the section.
    """
    beg, end = self.get_sections_2()
    xyz = _np.transpose([self.x[end], self.y[end], self.z[end]]) - \
        _np.transpose([self.x[beg], self.y[beg], self.z[beg]])
    return _np.dot(xyz, vect)


def get_section_number(self):
    '''Returns number of sections
    '''
    beg, _ = self.get_sections()
    return len(beg)


def get_section_lengths(self):
    """
    Tree method to get the sections'
    lengths. TODO: fix this one!!!
    """
    ways, end = self.get_sections_only_points()
    seg_len = self.get_segment_lengths()
    array_length = self.get_section_number()
    lengths = _np.zeros(array_length, dtype=float)

    for i in range(array_length):
        lengths[i] = _np.sum(seg_len[ways[i]:end[i]])

    return lengths


def get_section_branch_orders(self):
    """
    Tree method to get the sections'
    lengths. TODO: fix this one!!!
    """
    beg, end = self.get_sections()
    array_length = len(beg)
    branch_orders = _np.zeros(array_length, dtype=float)

    for i in range(array_length):
        branch_orders[i] = self.get_branch_order(end[i])

    return branch_orders


def get_section_path_distances(self):
    """
    Tree method to get path distances from a point.
    If point is None, the soma surface -defined by
    the initial segment of the tree- will be used
    as a reference point.
    """
    array_length = get_section_number(self)
    beg, _ = self.get_sections()
    lengths = self.get_section_lengths()
    path_distances = _np.zeros(array_length, dtype=float)
    path_distances[0] = lengths[0]

    for i in range(1, array_length):
        path_distances[i] = path_distances[beg[i]] + lengths[i]

    return path_distances


def get_section_radial_distances(self, point=None, initial=False):
    """
    Tree method to get radial distances from a point.
    If point is None, the soma surface -defined by
    the initial segment of the tree- will be used
    as a reference point.
    """
    beg, end = self.get_sections()
    array_length = get_section_number(self)

    if point is None:
        point = [self.x[0], self.y[0], self.z[0]]

    radial_distances = _np.zeros(array_length, dtype=float)

    for i in range(array_length):
        if initial:
            radial_distances[i] = _rd(point,
                                      [self.x[beg[i]],
                                       self.y[beg[i]],
                                       self.z[beg[i]]])
        else:
            radial_distances[i] = _rd(point,
                                      [self.x[end[i]],
                                       self.y[end[i]],
                                       self.z[end[i]]])

    return radial_distances


def get_bif_term(self):
    '''Returns number of children per point
    '''
    import scipy.sparse as sp
    return _np.array(sp.csr_matrix.sum(self.dA, axis=0))[0]


def get_bifurcations(self):
    '''Returns bifurcations
    '''
    bif_term = get_bif_term(self)
    bif = _np.where(bif_term == 2.)[0]
    return bif


def get_multifurcations(self):
    '''Returns bifurcations
    '''
    bif_term = get_bif_term(self)
    bif = _np.where(bif_term >= 2.)[0]
    return bif


def get_terminations(self):
    '''Returns terminations
    '''
    bif_term = get_bif_term(self)
    term = _np.where(bif_term == 0.)[0]
    return term


def get_children(self, sec_id=0):
    '''Returns children of section
    '''
    children = _np.where(self.p == sec_id)[0]

    return children


def get_direction(self, sec_id=0, child_id=0):
    '''Returns direction of a section
    defined as terminal point - initial point
    normalized as a unit vector.
    '''
    beg, end = self.get_sections_only_points()
    e = end[_np.where(beg == sec_id)[0]][child_id]

    vect = _np.subtract([self.x[e], self.y[e], self.z[e]],
                        [self.x[sec_id], self.y[sec_id], self.z[sec_id]])
    direction = vect / _np.linalg.norm(vect)

    return direction


def get_direction_between(self, start_id=0, end_id=1):
    '''Returns direction of a branch
    defined as end point - start point
    normalized as a unit vector.
    '''
    vect = _np.subtract([self.x[end_id], self.y[end_id], self.z[end_id]],
                        [self.x[start_id], self.y[start_id], self.z[start_id]])

    if _np.linalg.norm(vect) != 0.0:
        return vect / _np.linalg.norm(vect)
    return vect


def _vec_angle(u, v):
    '''Returns the angle between v and u in 3D.
    '''
    c = _np.dot(u, v) / _np.linalg.norm(u) / _np.linalg.norm(v)
    return _np.arccos(c)


def get_angle_between(self, sec_id1, sec_id2):
    '''Returns local bifurcations angle
    between two sections, defined by their ids.
    sec_id1: the start point of the section #1
    sec_id2: the start point of the section #2
    '''
    beg, end = self.get_sections_only_points()
    print sec_id1, sec_id2
    b1 = _np.where(beg == sec_id1)[0][0]
    b2 = _np.where(beg == sec_id2)[0][0]

    u = self.get_direction_between(beg[b1],
                                   end[b1])
    v = self.get_direction_between(beg[b2],
                                   end[b2])

    return _vec_angle(u, v)


def get_bif_angles(self):
    '''Returns local bifurcations angles
    '''
    angs = []
    bif = self.get_bifurcations()

    for b in bif:
        children = self.get_children(b)
        # Angle between two first children
        angs.append(self.get_angle_between(children[0], children[1]))

    return angs


def get_parent_child_angles(self, funct=_np.min):
    '''Returns the angles of a section and its children
    at each bifurcation
    '''
    angles = []
    beg, end = self.get_sections_only_points()

    for i,e1 in enumerate(end):
        angs = []
        children = self.get_children(e1)

        for ch in children:
            angs.append(self.get_angle_between(beg[i], ch))
        small = funct(angs)

        if len(children) > 1:
            angles.append(small)

    return angles


def get_way_to_root(self, sec_id=0):
    '''Returns way to root
    '''
    way = []
    tmp_id = sec_id

    while tmp_id != -1:
        way.append(self.p[tmp_id])
        tmp_id = self.p[tmp_id]

    return way


def get_way_to_section_end(self, sec_id=0):
    '''Returns way to leaf
    '''
    way = []
    tmp_id = sec_id
    bif = self.get_bifurcations()
    term = self.get_terminations()

    while (tmp_id not in term) and (tmp_id not in bif):
        way.append(tmp_id)
        tmp_id = get_children(self, tmp_id)[0]

    way.append(tmp_id)  # Add last section point: bif or term

    return way


def get_way_to_section_start(self, sec_id=0):
    '''Returns way to section start
    '''
    way = []
    tmp_id = sec_id
    bif = self.get_bifurcations()

    if sec_id in bif:
        way.append(tmp_id)
        tmp_id = self.p[tmp_id]

    while (tmp_id not in bif) and tmp_id != -1:
        way.append(tmp_id)
        tmp_id = self.p[tmp_id]

    way.append(tmp_id)  # Add last section point: bif or term

    return way


def get_section_start(self, sec_id=0):
    '''Returns way to section start
    '''
    tmp_id = sec_id
    bif = self.get_multifurcations()

    while (tmp_id not in bif) and tmp_id != -1:
        tmp_id = self.p[tmp_id]

    return tmp_id


# Trunk features
def get_trunk(self):
    '''Returns trunks ids
    '''
    trunk = _np.where(self.p == -1)[0][0]
    return trunk


# PCA
def get_pca(self, plane='xy', component=0):
    '''Returns the i-th principal
    component of PCA on the points
    of the tree in the selected plane
    '''
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(_np.transpose([getattr(self, plane[0]), getattr(self, plane[1])]))

    return pca.components_[component]
