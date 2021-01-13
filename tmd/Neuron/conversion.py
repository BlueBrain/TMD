"""Functions for converting morphio to tmd Neuron
"""
from collections import namedtuple
import numpy as np
from tmd.Soma.Soma import Soma
from tmd.Tree import Tree


SectionData = namedtuple('SectionData', ['points', 'diameters', 'section_type', 'parents'])


def convert_morphio_soma(morphio_soma):
    """Converts a morphio's morphology
    Args:
        morphio_soma (Union[morphio.Soma])
    Returns:
        tmd_soma (Soma)
    """
    points = morphio_soma.points
    return Soma(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        d=morphio_soma.diameters
    )


def _section_to_data(section, tree_length, start, parent):
    """Extract data from morphio section

    Args:
        section (morphio.Section): A morphio section object
        tree_length (int): Number of nodes in the tree so far

    Returns:
        tuple:
            - n (int): Number of nodes extracted
            - section_data (SectionData): namedtuple with section data
    """
    points = section.points

    # start is either 0 or 1 if the section is root or not respectively
    # dropping the first node in the latter case
    n = len(points) - start

    # each node has the previous node as a parent, except for the first
    # one, which is given from the parent
    parents = np.arange(tree_length - 1, tree_length + n - 1, dtype=np.int)
    parents[0] = parent

    return n, SectionData(
        points[start:],
        section.diameters[start:],
        int(section.type),
        parents)


def convert_morphio_trees(morphio_neuron):
    """Convert morphio morphology's trees to tmd ones

    Args:
        morphio_neuron (Union[morphio.Morphology, morphio.mut.Morphology]):
            morphio neuron object

    Yields:
        tuple:
            - tree (Tree): The constructed tmd Tree
            - tree_types (dict): The neuron tree types
    """
    total_points = len(morphio_neuron.diameters)

    x = np.empty(total_points, dtype=np.float32)
    y = np.empty(total_points, dtype=np.float32)
    z = np.empty(total_points, dtype=np.float32)
    d = np.empty(total_points, dtype=np.float32)
    t = np.empty(total_points, dtype=np.int32)
    p = np.empty(total_points, dtype=np.int64)

    section_final_nodes = np.empty(total_points, dtype=np.int)

    tree_end = 0
    for root in morphio_neuron.root_sections:

        tree_length = 0
        tree_beg = tree_end

        for section in root.iter():

            # root sections have parent -1
            if section.is_root:
                start = 0
                parent = -1
            else:
                # tmd does not use a duplicate point representation
                # thus the first point of the section is dropped
                start = 1
                parent = section_final_nodes[section.parent.id]

            n, data = _section_to_data(section, tree_length, start, parent)

            x[tree_end: n + tree_end] = data.points[:, 0]
            y[tree_end: n + tree_end] = data.points[:, 1]
            z[tree_end: n + tree_end] = data.points[:, 2]
            d[tree_end: n + tree_end] = data.diameters
            t[tree_end: n + tree_end] = data.section_type
            p[tree_end: n + tree_end] = data.parents

            tree_end += n
            tree_length += n

            # keep track of the last node in the section because we need
            # to establish the correct connectivity when we ommit the first
            # point from the children sections
            section_final_nodes[section.id] = tree_length - 1

        yield Tree.Tree(
            x=x[tree_beg: tree_end],
            y=y[tree_beg: tree_end],
            z=z[tree_beg: tree_end],
            d=d[tree_beg: tree_end],
            t=t[tree_beg: tree_end],
            p=p[tree_beg: tree_end]
        )
