'''
Python module that contains the functions
about reading h5 files.
'''

import numpy as np
import h5py
from tmd.io.swc import SWC_DCT

# Definition of h5 data container
# separated in points (PX,PY,PZ,PD)
# and groups (GPFIRST, GTYPE, GPID)
h5_dct = {'PX': 0,
          'PY': 1,
          'PZ': 2,
          'PD': 3,
          'GPFIRST': 0,
          'GTYPE': 1,
          'GPID': 2}


def _find_group(point_id, groups):
    '''Find the structure group a points id belongs to.
       Return: group or section point_id belongs to. Last group if
               point_id out of bounds.
    '''
    bs = np.searchsorted(groups[:, h5_dct['GPFIRST']], point_id, side='right')
    bs = max(bs - 1, 0)
    return groups[bs]


def _find_parent_id(point_id, groups):
    '''Find the parent ID of a point'''
    group = _find_group(point_id, groups)
    if point_id != group[h5_dct['GPFIRST']]:
        # point is not first point in section
        # so parent is previous point
        return point_id - 1
    # parent is last point in parent group
    parent_group_id = group[h5_dct['GPID']]
    # get last point in parent group
    return groups[parent_group_id + 1][h5_dct['GPFIRST']] - 1


def _find_last_point(group_id, groups, points):
    ''' Identifies and returns the id of the last point of a group'''
    group_initial_ids = np.sort(np.transpose(groups)[0])

    if group_id != len(group_initial_ids) - 1:
        return group_initial_ids[np.where(group_initial_ids == groups[group_id][0])[0][0] + 1] - 1
    return len(points) - 1


def remove_duplicate_points(points, groups):
    ''' Removes the duplicate points from the beginning of a section,
    if they are present in points-groups representation.
    Returns:
       points, groups with unique points.
    '''

    group_initial_ids = groups[:, h5_dct['GPFIRST']]

    to_be_reduced = np.zeros(len(group_initial_ids))
    to_be_removed = []

    for ig, g in enumerate(groups):
        iid, typ, pid = g[h5_dct['GPFIRST']], g[h5_dct['GTYPE']], g[h5_dct['GPID']]
        # Remove first point from sections that are
        # not the root section, a soma, or a child of a soma
        if pid != -1 and typ != 1 and groups[pid][h5_dct['GTYPE']] != 1:
            # Remove duplicate from list of points
            to_be_removed.append(iid)
            # Reduce the id of the following sections
            # in groups structure by one
            to_be_reduced[ig + 1:] += 1

    groups[:, h5_dct['GPFIRST']] = groups[:, h5_dct['GPFIRST']] - to_be_reduced
    points = np.delete(points, to_be_removed, axis=0)

    return points, groups


def _unpack_data(points, groups):
    '''Unpack data from h5 data groups into internal format'''
    return np.array([(i, _find_group(i, groups)[h5_dct['GTYPE']],
                     p[h5_dct['PX']], p[h5_dct['PY']],
                     p[h5_dct['PZ']], p[h5_dct['PD']],
                     _find_parent_id(i, groups))
                     for i, p in enumerate(points)])


def _unpack_v1(h5file):
    '''Unpacks data of h5_v1 file
       in a simplified data structure.
    '''
    points = np.array(h5file['points'])
    groups = np.array(h5file['structure'])
    return points, groups


def _unpack_v2(h5file, stage):
    '''Unpack groups from HDF5 v2 file'''
    if stage == 'unraveled':
        stage1 = 'raw'
    else:
        stage1 = stage
    points = np.array(h5file['neuron1/%s/points' % stage])
    groups = np.array(h5file['neuron1/structure/%s' % stage1])
    stypes = np.array(h5file['neuron1/structure/sectiontype'])
    groups = np.hstack([groups, stypes])
    groups[:, [1, 2]] = groups[:, [2, 1]]
    return points, groups


def _get_h5_version(h5file):
    '''Determine whether an HDF5 file is v1 or v2.

    Returns:
        1, 2 or None
    '''
    if 'points' in h5file and 'structure' in h5file:
        return 1
    if 'neuron1/structure' in h5file:
        return 2
    return None


def read_h5(input_file, remove_duplicates=True):
    '''Function to properly load sn h5 file,
       of v1 or v2 format.
    '''
    h5file = h5py.File(input_file, mode='r')
    version = _get_h5_version(h5file)

    if version == 1:
        points, groups = _unpack_v1(h5py.File(input_file, mode='r'))
    elif version == 2:
        stg = next(s for s in ('repaired', 'unraveled', 'raw', '0', '1', '2')
                   if s in h5file['neuron1'])
        points, groups = _unpack_v2(h5py.File(input_file, mode='r'),
                                    stage=stg)
    else:
        raise Exception("Not recognized h5 version")

    h5file.close()

    if remove_duplicates:
        return _unpack_data(*remove_duplicate_points(points, groups))
    return _unpack_data(points, groups)


def h5_data_to_lists(data):
    """
    Transforms data as loaded from read_h5
    into a set of 'meaningful' lists:

    x: list of floats
        x-coordinates

    y: list of floats
        y-coordinates

    z: list of floats
        z-coordinates

    d: list of floats
        diameters

    t: list of ints
        tree type

    p: list of ints
        parent id

    ch: dictionary
        children id(s)

    """
    length = len(data)

    x = np.transpose(data)[SWC_DCT['x']]
    y = np.transpose(data)[SWC_DCT['y']]
    z = np.transpose(data)[SWC_DCT['z']]
    d = np.transpose(data)[SWC_DCT['radius']]
    t = np.transpose(data)[SWC_DCT['type']]
    p = np.transpose(data)[SWC_DCT['parent']]
    ch = {}

    for enline in range(length):

        ch[enline] = list(np.where(p == enline)[0])

    return x, y, z, d, t, p, ch
