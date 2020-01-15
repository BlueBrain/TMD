'''
tmd statistical analysis on PH diagrams algorithms implementation
'''
import numpy as np
from scipy import stats


def get_bifurcations(ph):
    """
    Returns the bifurcations from the diagram
    """
    return np.array(ph)[:, 1]


def get_terminations(ph):
    """
    Returns the terminations from the diagram
    """
    return np.array(ph)[:, 0]


def get_lengths(ph):
    """
    Returns the lengths of the bars from the diagram
    """
    return np.array([np.abs(i[0] - i[1]) for i in ph])


def get_total_length(ph):
    """Calculates the total length of a barcode
       by summing the length of each bar. This should
       be equivalent to the total length of the tree
       if the barcode represents path distances.
    """
    return sum([np.abs(p[1] - p[0]) for p in ph])


def transform_ph_to_length(ph, keep_side='end'):
    '''Transforms a persistence diagram into a
    (start_point, length) equivalent diagram or a
    (end, length) diagram depending on keep_side option.
    Note: the direction of the diagram will be lost!
    '''
    if keep_side == 'start':
        # keeps the start point and the length of the bar
        return [[min(i), np.abs(i[1] - i[0])] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[max(i), np.abs(i[1] - i[0])] for i in ph]


def transform_ph_from_length(ph, keep_side='end'):
    '''Transforms a persistence diagram into a
    (start_point, length) equivalent diagram or a
    (end, length) diagram depending on keep_side option.
    Note: the direction of the diagram will be lost!
    '''
    if keep_side == 'start':
        # keeps the start point and the length of the bar
        return [[i[0], i[1] - i[0]] for i in ph]
    else:
        # keeps the end point and the length of the bar
        return [[i[0] - i[1], i[0]] for i in ph]


def nosify(var, noise=0.1):
    '''Adds noise to an instance of data
    Can be used with a ph as follows:
    noisy_pd = [add_noise(d, 1.0) if d[0] != 0.0
                else [d[0],add_noise([d[1]],1.0)[0]] for d in pd]

    To output the new pd:
    F = open(...)
    for d in noisy_pd:
        towrite = '%f, %f\n'%(d[0],d[1])
        F.write(towrite)
    F.close()
    '''
    var_new = np.zeros(len(var))
    for i, v in enumerate(var):
        var_new[i] = stats.norm.rvs(v, noise)
    return var_new
