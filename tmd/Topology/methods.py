'''
tmd Topology algorithms implementation
'''
import numpy as np
import scipy.spatial as sp
from tmd.Topology.analysis import sort_ph


def write_ph(ph, output_file='test.txt'):
    '''Writes a persistence diagram in
       an output file.
    '''
    wfile = open(output_file, 'w')

    for p in ph:
        wfile.write(str(p[0]) + ' ' + str(p[1]) + '\n')

    wfile.close()


def get_persistence_diagram(tree, feature='radial_distances', **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''
    ph = []

    rd = getattr(tree, 'get_point_' + feature)(**kwargs)

    active = tree.get_bif_term() == 0

    beg, end = tree.get_sections_2()

    beg = np.array(beg)
    end = np.array(end)

    parents = {e: b for b, e in zip(beg, end)}

    if 0 in beg:
        parents[0] = tree.p[0]

    children = {b: end[np.where(beg == b)[0]] for b in np.unique(beg)}

    while len(np.where(active)[0]) > 1:
        alive = list(np.where(active)[0])
        for l in alive:

            p = parents[l]
            c = children[p]

            if np.alltrue(active[c]):
                active[p] = True
                active[c] = False

                mx = np.argmax(abs(rd[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    ph.append([rd[ci], rd[p]])

                rd[p] = rd[mx_id]

    ph.append([rd[np.where(active)[0][0]], 0])  # Add the last alive component

    return ph


def _phi_theta(u, v):
    """Computes the angles between vectors u, v
    in the plane x-y (phi angle) and the plane x-z (theta angle).
    Returns phi, theta
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


def _angles_tree(tree, parID, parEND, ch1ID, ch2ID):
    '''Computes the x-y and x-z angles between parent
       and children within the given tree.
    '''

    dirP = tree.get_direction_between(start_id=parID, end_id=parEND)
    dirU = tree.get_direction_between(start_id=parEND, end_id=ch1ID)
    dirV = tree.get_direction_between(start_id=parEND, end_id=ch2ID)

    phi1, theta1 = _phi_theta(dirP, dirU)
    phi2, theta2 = _phi_theta(dirP, dirV)

    if np.abs(phi1) < np.abs(phi2):
        dphi = phi1
        dtheta = theta1
        delta_phi, delta_theta = _phi_theta(dirU, dirV)
    else:
        dphi = phi2
        dtheta = theta2
        delta_phi, delta_theta = _phi_theta(dirV, dirU)

    return [dphi, dtheta, delta_phi, delta_theta]


def get_angles(tree, beg, parents, children):
    """Returns the angles between all the triplets (parent, child1, child2)
    of the tree"""
    angles = [[0, 0, 0, 0], ]  # Null angle for non bif point

    for b in beg[1:]:

        angleBetween = _angles_tree(tree, parID=parents[b], parEND=b,
                                    ch1ID=children[b][0],
                                    ch2ID=children[b][1])
        angles.append(angleBetween)

    return angles


def get_ph_angles(tree, feature='radial_distances', **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''
    ph = []

    rd = getattr(tree, 'get_point_' + feature)(**kwargs)

    active = tree.get_bif_term() == 0

    beg, end = tree.get_sections_2()

    beg = np.array(beg)
    end = np.array(end)

    parents = {e: b for b, e in zip(beg, end)}

    if 0 in beg:
        parents[0] = tree.p[0]

    children = {b: end[np.where(beg == b)[0]] for b in np.unique(beg)}

    angles = get_angles(tree, beg, parents, children)

    while len(np.where(active)[0]) > 1:
        alive = list(np.where(active)[0])
        for l in alive:

            p = parents[l]
            c = children[p]

            if np.alltrue(active[c]):
                active[p] = True
                active[c] = False

                mx = np.argmax(abs(rd[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    angID = np.array(angles)[np.where(beg == p)[0][0]]
                    ph.append([rd[ci], rd[p], angID[0], angID[1], angID[2], angID[3]])

                rd[p] = rd[mx_id]

    ph.append([rd[np.where(active)[0][0]], 0, np.nan, np.nan, np.nan, np.nan])

    return ph


def get_ph_radii(tree, feature='radial_distances', **kwargs):
    """Returns the ph diagram enhanced with the corresponding encoded radii"""
    def get_section_mean_radii(tree, beg, end):
        """Returns the mean radii of a section"""
        return [np.mean(tree.d[beg[i]:end[i]]) for i in range(len(beg))]

    ph = []

    rd = getattr(tree, 'get_point_' + feature)(**kwargs)

    active = tree.get_bif_term() == 0

    beg, end = tree.get_sections_2()

    beg = np.array(beg)
    end = np.array(end)

    parents = {e: b for b, e in zip(beg, end)}
    children = {b: end[np.where(beg == b)[0]] for b in np.unique(beg)}

    radii = get_section_mean_radii(tree, beg, end)

    while len(np.where(active)[0]) > 1:
        alive = list(np.where(active)[0])
        for l in alive:

            p = parents[l]
            c = children[p]

            if np.alltrue(active[c]):
                active[p] = True
                active[c] = False

                mx = np.argmax(abs(rd[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    radiiID = np.array(radii)[np.where(beg == p)[0][0]]
                    ph.append([rd[ci], rd[p], radiiID])

                rd[p] = rd[mx_id]

    ph.append([rd[np.where(active)[0][0]], 0, radii[beg[0]]])  # Add the last alive component

    return ph


def get_ph_neuron(neuron, feature='radial_distances', neurite_type='all', **kwargs):
    '''Method to extract ph from a neuron that contains mutlifurcations'''

    ph_all = []

    if neurite_type == 'all':
        neurite_list = ['neurites']
    else:
        neurite_list = [neurite_type]

    for t in neurite_list:
        for tr in getattr(neuron, t):
            ph_all = ph_all + get_persistence_diagram(tr, feature=feature, **kwargs)

    return ph_all


def extract_ph(tree, feature='radial_distances', output_file='test.txt',
               sort=False, **kwargs):
    '''Extracts persistent homology from tree'''
    ph = get_persistence_diagram(tree, feature=feature, **kwargs)

    if sort:
        p = sort_ph(ph)
    else:
        p = ph

    write_ph(p, output_file)


def extract_ph_neuron(neuron, feature='radial_distances', output_file=None,
                      neurite_type='all', sort=False, **kwargs):
    '''Extracts persistent homology from tree'''
    ph = get_ph_neuron(neuron, feature=feature, neurite_type='all', **kwargs)

    if sort:
        sort_ph(ph)
    else:
        p = ph

    if output_file is None:
        output_file = 'PH_' + neuron.name + '_' + neurite_type + '.txt'

    write_ph(p, output_file)


def get_lifetime(tree, feature='point_radial_distances'):
    '''Returns the sequence of birth - death times for each section.
    This can be used as the first step for the approximation of P.H.
    of the radial distances of the neuronal branches.
    '''
    begs, ends = tree.get_sections_2()
    rd = getattr(tree, 'get_' + feature)()
    lifetime = np.array(len(begs) * [np.zeros(2)])

    for i, (beg, end) in enumerate(zip(begs, ends)):
        lifetime[i] = np.array([rd[beg], rd[end]])

    return lifetime


def extract_connectivity_from_points(tree, threshold=1.0):
    '''Extract connectivity from list of points'''
    coords = np.transpose([tree.x, tree.y, tree.z])
    distances_matrix = sp.distance.cdist(coords, coords)
    mat = distances_matrix < threshold

    return mat
