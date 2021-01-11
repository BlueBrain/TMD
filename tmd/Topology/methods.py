'''
tmd Topology algorithms implementation
'''
import numpy as np
import scipy.spatial as sp
from tmd.Topology.persistent_properties import NoProperty
from tmd.Topology.persistent_properties import PersistentAngles
from tmd.Topology.persistent_properties import PersistentMeanRadius
from tmd.Topology.analysis import sort_ph


def write_ph(ph, output_file='test.txt'):
    '''Writes a persistence diagram in
       an output file.
    '''
    wfile = open(output_file, 'w')

    for p in ph:
        wfile.write(str(p[0]) + ' ' + str(p[1]) + '\n')

    wfile.close()


def tree_to_property_barcode(tree, filtration_function, property_class=NoProperty):
    """Decompose a tree data structure into a barcode, where each bar in the barcode
    is optionally linked with a property determined by property_class.

    Args:

        filtration_function (Callable[tree] -> np.ndarray):
            The filtration function to apply on the tree

        property_class (PersistentProperty, optional): A PersistentProperty class.By
            default the NoProperty is used which does not add entries in the barcode.

    Returns:
        barcode (list): A list of bars [bar1, bar2, ..., barN], where each bar is a
            list of:
                - filtration value start
                - filtration value end
                - property_value1
                - property_value2
                - ...
                - property_valueN
    """
    point_values = filtration_function(tree)

    beg, _ = tree.sections
    parents, children = tree.parents_children

    prop = property_class(tree)

    active = tree.get_bif_term() == 0
    alives = np.where(active)[0]

    ph = []
    while len(alives) > 1:
        for alive in alives:

            p = parents[alive]
            c = children[p]

            if np.alltrue(active[c]):

                active[p] = True
                active[c] = False

                mx = np.argmax(abs(point_values[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    component_id = np.where(beg == p)[0][0]
                    ph.append(
                        [point_values[ci], point_values[p]] + prop.get(component_id)
                    )

                point_values[p] = point_values[mx_id]
        alives = np.where(active)[0]

    ph.append(
        [point_values[alives[0]], 0] + prop.infinite_component(beg[0])
    )  # Add the last alive component

    return ph


def _filtration_function(feature, **kwargs):
    """Returns filtration function lambda that will be applied point-wise
    on the tree"""
    return lambda tree: getattr(tree, 'get_point_' + feature)(**kwargs)


def get_persistence_diagram(tree, feature='radial_distances', **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''
    return tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(feature, **kwargs),
        property_class=NoProperty
    )


def get_ph_angles(tree, feature='radial_distances', **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''
    return tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(feature, **kwargs),
        property_class=PersistentAngles
    )


def get_ph_radii(tree, feature='radial_distances', **kwargs):
    """Returns the ph diagram enhanced with the corresponding encoded radii"""
    return tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(feature, **kwargs),
        property_class=PersistentMeanRadius
    )


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
