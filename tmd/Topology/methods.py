'''
tmd Topology algorithms implementation
'''

import numpy as np


def sort_ph(ph):
    '''Sorts the persistence diagram
    so that birth is always before death.
    '''
    for i in ph:
        i.sort()


def write_ph(ph, output_file='test.txt'):
    '''Writes a persistence diagram in
       an output file.
    '''
    wfile = open(output_file, 'w')

    for p in ph:

        wfile.write(str(p[0]) + ' ' + str(p[1]) + '\n')

    wfile.close()


def get_graph(tree):
    '''Generate tree graph'''
    from collections import OrderedDict

    graph = OrderedDict()

    section_points = np.transpose(tree.get_sections())

    for sp in section_points:

        if sp[0] != sp[1]:
            if sp[0] in graph.keys():
                graph[sp[0]].append(sp[1])
            else:
                graph[sp[0]] = [sp[1]]

    return graph


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

    ph.append([rd[np.where(active)[0][0]], 0]) # Add the last alive component

    return ph


def get_persistence_diagram_rotation(tree, feature='radial_distances', **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''

    ph = get_persistence_diagram(tree, feature=feature, **kwargs)

    tr_pca = tree.get_pca()

    def rotation(x, y, angle=0.0):
        '''Rotates coordinates x-y to the selected angle'''
        return [np.cos(angle) * x - np.sin(angle) * y,
                np.sin(angle) * x + np.cos(angle) * y]

    ph_rot = [rotation(i[0], i[1], angle=np.arctan2(*tr_pca)) for i in ph]

    return ph_rot


def phi_theta(u, v):
    """Computes the angles between vectors u, v
    in the plane x-y (phi angle) and the plane x-z (theta angle).
    Returns phi, theta
    """
    phi1 = np.arctan2(u[1], u[0])
    theta1 = np.arccos(u[2] / np.linalg.norm(u))

    phi2 = np.arctan2(v[1], v[0])
    theta2 = np.arccos(v[2] / np.linalg.norm(v))

    delta_phi = phi2 - phi1 # np.abs(phi1 - phi2)
    delta_theta = theta2 - theta1 # np.abs(theta1 - theta2)

    return delta_phi, delta_theta # dphi, dtheta


def get_angles(tree, beg, parents, children):
    """Returns the angles between all the triplets (parent, child1, child2)
    of the tree"""

    angles = [[0, 0, 0, 0], ]

    for b in beg[1:]:

        dirP = tree.get_direction_between(start_id=parents[b], end_id=b)

        dirU = tree.get_direction_between(start_id=b,
                                          end_id=children[b][0])

        dirV = tree.get_direction_between(start_id=b,
                                          end_id=children[b][1])

        phi1, theta1 = phi_theta(dirP, dirU)
        phi2, theta2 = phi_theta(dirP, dirV)

        if np.abs(phi1) < np.abs(phi2):
            dphi = phi1
            dtheta = theta1
            delta_phi, delta_theta = phi_theta(dirU, dirV)
        else:
            dphi = phi2
            dtheta = theta2
            delta_phi, delta_theta = phi_theta(dirV, dirU)

        angles.append([dphi, dtheta, delta_phi, delta_theta])

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

    ph.append([rd[np.where(active)[0][0]], 0, None, None, None, None])

    return ph


def get_section_radii(tree, beg, end):
    """Returns the mean radii of a section"""
    return [np.mean(tree.d[beg[i]:end[i]]) for i in xrange(len(beg))]


def get_ph_radii(tree, feature='radial_distances', **kwargs):
    """Returns the ph diagram enhanced with the corresponding encoded radii"""
    ph = []

    rd = getattr(tree, 'get_point_' + feature)(**kwargs)

    active = tree.get_bif_term() == 0

    beg, end = tree.get_sections_2()

    beg = np.array(beg)
    end = np.array(end)

    parents = {e: b for b, e in zip(beg, end)}
    children = {b: end[np.where(beg == b)[0]] for b in np.unique(beg)}

    radii = get_section_radii(tree, beg, end)

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

    ph.append([rd[np.where(active)[0][0]], 0, radii[beg[0]]]) # Add the last alive component

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


def get_ph_neuron_rot(neuron, feature='radial_distances', neurite_type='all', **kwargs):
    '''Method to extract ph from a neuron that contains mutlifurcations'''

    ph_all = []

    if neurite_type == 'all':
        neurite_list = ['basal', 'apical', 'axon']
    else:
        neurite_list = [neurite_type]

    for t in neurite_list:
        for tr in getattr(neuron, t):
            ph_all = ph_all + get_persistence_diagram_rotation(tr, feature=feature, **kwargs)

    return ph_all


def extract_ph(tree, feature='radial_distances', output_file='test.txt',
               sort=False, **kwargs):
    '''Extracts persistent homology from tree'''

    ph = get_persistence_diagram(tree, feature=feature, **kwargs)

    if sort:
        sort_ph(ph)

    write_ph(ph, output_file)

    print 'File ' + output_file + ' completed!'


def extract_ph_neuron(neuron, feature='radial_distances', output_file=None,
                      neurite_type='all', sort=False, **kwargs):
    '''Extracts persistent homology from tree'''

    ph = get_ph_neuron(neuron, feature=feature, neurite_type='all', **kwargs)

    if sort:
        sort_ph(ph)

    if output_file is None:
        output_file = 'PH_' + neuron.name + '_' + neurite_type + '.txt'

    write_ph(ph, output_file)

    print 'File ' + output_file + ' completed!'


def get_lifetime(tree, feature='point_radial_distances'):
    '''Returns the sequence of birth - death times for each section.
    This can be used as the first step for the approximation of P.H.
    of the radial distances of the neuronal branches.
    '''

    beg, end = tree.get_sections()

    rd = getattr(tree, 'get_' + feature)()

    lifetime = np.array(len(beg) * [np.zeros(2)])

    for i in xrange(len(beg)):

        lifetime[i] = np.array([rd[beg[i]], rd[end[i]]])

    return lifetime


def extract_connectivity_from_points(tree, threshold=1.0):
    '''Extract connectivity from list of points'''
    import scipy.spatial as sp

    coords = np.transpose([tree.x, tree.y, tree.z])

    distances_matrix = sp.distance.cdist(coords, coords)

    mat = distances_matrix < threshold

    return mat
