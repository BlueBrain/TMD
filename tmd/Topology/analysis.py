'''
tmd Topology analysis algorithms implementation
'''
# pylint: disable=invalid-slice-index

import numpy as np


def collapse(ph_list):
    '''Collapses a list of ph diagrams
       into a single instance for plotting.
    '''
    return [list(pi) for p in ph_list for pi in p]


def sort_ph(ph):
    """
    Sorts barcode according to decreasing length of bars.
    """
    return np.array(ph)[np.argsort([p[0] - p[1] for p in ph])].tolist()


def load_file(filename, delimiter=' '):
    """Load PH file in a np.array
    """
    f = open(filename, 'r')
    ph = np.array([np.array(line.split(delimiter), dtype=np.float) for line in f])
    f.close()
    return ph


def get_limits(phs_list, coll=True):
    '''Returns the x-y coordinates limits (min, max)
    for a list of persistence diagrams
    '''
    import copy
    if coll:
        ph = collapse(phs_list)
    else:
        ph = copy.deepcopy(phs_list)
    xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]
    return xlims, ylims


def get_persistence_image_data(ph, norm_factor=None, xlims=None, ylims=None):
    '''Create the data for the generation of the persistence image.
    If norm_factor is provided the data will be normalized based on this,
    otherwise they will be normalized to 1.
    If xlims, ylims are provided the data will be scaled accordingly.
    '''
    from scipy import stats

    if xlims is None or xlims is None:
        xlims, ylims = get_limits(ph, coll=False)

    X, Y = np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = np.max(Z)

    return Z / norm_factor


def get_image_diff_data(Z1, Z2, norm=True):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their difference: diff(Z1 - Z2)
    """
    if norm:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()
    return Z1 - Z2


def get_image_add_data(Z1, Z2, norm=True):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their sum: add(Z1 - Z2)
    """
    if norm:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()
    return Z1 + Z2


def histogram_horizontal(ph, num_bins=100, min_bin=None, max_bin=None):
    """Calculate how many barcode lines are found in each bin.
    """
    import math
    ph1 = [p[:2] for p in ph]  # simplify to ensure ph corresponds to 2d barcode

    if min_bin is None:
        min_bin = np.min(ph1)
    if max_bin is None:
        max_bin = np.max(ph1)

    bins = np.linspace(min_bin, max_bin, num_bins, dtype=float)
    binsize = (max_bin - min_bin) / (num_bins - 1.)
    results = np.zeros(num_bins - 1)

    for p in ph1:
        ph_decompose = np.linspace(np.min(p), np.max(p),
                                   math.ceil((np.max(p) - np.min(p)) / binsize),
                                   dtype=float)

        bin_ph = np.histogram(ph_decompose, bins=bins)[0]
        results = np.add(results, bin_ph)

    return bins, results


def histogram_stepped(ph1):
    '''Calculate step distance of ph data'''
    from itertools import chain

    bins = np.unique(list(chain(*ph1)))
    results = np.zeros(len(bins) - 1)

    for ph in ph1:
        for it, _ in enumerate(bins[:-1]):
            if min(ph) <= bins[it + 1] and max(ph) > bins[it]:
                results[it] = results[it] + 1

    return bins, results


def distance_stepped(ph1, ph2, order=1):
    '''Calculate step distance difference between two ph'''
    from itertools import chain
    from numpy.linalg import norm

    bins1 = np.unique(list(chain(*ph1)))
    bins2 = np.unique(list(chain(*ph2)))
    bins = np.unique(np.append(bins1, bins2))
    results1 = np.zeros(len(bins) - 1)
    results2 = np.zeros(len(bins) - 1)

    for ph in ph1:
        for it, _ in enumerate(bins[:-1]):
            if min(ph) <= bins[it + 1] and max(ph) > bins[it]:
                results1[it] = results1[it] + 1

    for ph in ph2:
        for it, _ in enumerate(bins[:-1]):
            if min(ph) <= bins[it + 1] and max(ph) > bins[it]:
                results2[it] = results2[it] + 1

    return norm(np.abs(np.subtract(results1, results2)) * (bins[1:] + bins[:-1]) / 2, order)


def distance_horizontal(ph1, ph2, norm=1, bins=100):
    """Calculate distance between two ph diagrams.
       Distance definition:
    """
    _, data_1 = histogram_horizontal(ph1, num_bins=bins)
    _, data_2 = histogram_horizontal(ph2, num_bins=bins)
    return np.linalg.norm(np.abs(np.subtract(data_1, data_2)), norm)


def distance_horizontal_unnormed(ph1, ph2, norm=1, bins=100):
    """Calculate unnormed distance between two ph diagrams.
    """
    maxb = np.max([np.max(ph1), np.max(ph2)])
    minb = np.min([np.min(ph1), np.min(ph2)])
    _, results1 = histogram_horizontal(ph1, num_bins=bins, min_bin=minb, max_bin=maxb)
    _, results2 = histogram_horizontal(ph2, num_bins=bins, min_bin=minb, max_bin=maxb)
    return np.linalg.norm(np.abs(np.subtract(results1, results2)), norm)


def get_average_persistence_image(ph_list, xlims=None, ylims=None, norm_factor=None):
    '''Plots the gaussian kernel of a population of cells
       as an average of the ph diagrams that are given.
    '''
    im_av = False
    k = 1

    for p in ph_list:
        if not isinstance(im_av, np.ndarray):
            try:
                im = get_persistence_image_data(p, norm_factor=norm_factor,
                                                xlims=xlims, ylims=ylims)
                if not np.isnan(np.sum(im)):
                    im_av = im
            except BaseException:
                pass
        else:
            try:
                im = get_persistence_image_data(p, norm_factor=norm_factor,
                                                xlims=xlims, ylims=ylims)
                if not np.isnan(np.sum(im)):
                    im_av = np.add(im_av, im)
                    k = k + 1
            except BaseException:
                pass
    return im_av / k


def find_apical_point_distance(ph):
    '''
    Finds the apical distance (measured in radial distance from soma)
    based on the variation of the barcode.
    '''
    n_bins, counts = histogram_horizontal(ph, num_bins=3 * len(ph))
    der1 = counts[1:] - counts[:-1]  # first derivative
    der2 = der1[1:] - der1[:-1]  # second derivative
    # Find all points that take minimum value, and have first derivative zero == no variation
    inters = np.intersect1d(np.where(counts == min(counts))[0], np.where(der1 == 0)[0])
    # Find all points that are also below a positive second derivative
    # The definition of how positive the second derivative should be is arbitrary,
    # but it is the only value that works nicely for cortical cells
    best_all = inters[np.where(inters <= np.max(np.where(der2 > len(n_bins) / 100)[0]))]

    if len(best_all) > 0:
        return n_bins[np.max(best_all)]
    else:
        return 0.0


def _symmetric(p):
    '''Returns the symmetric point of a PD point on the diagonal
    '''
    return [(p[0] + p[1]) / 2., (p[0] + p[1]) / 2]


def matching_munkress_modified(p1, p2, use_diag=True):
    '''Returns a list of matching components
       and the corresponding distance between
       the two input diagrams
    '''
    from scipy.spatial.distance import cdist
    import munkres
    from tmd.view import common as _cm

    if use_diag:
        p1_enh = p1 + [_symmetric(i) for i in p2]
        p2_enh = p2 + [_symmetric(i) for i in p1]
    else:
        p1_enh = p1
        p2_enh = p2

    D = cdist(p1_enh, p2_enh)
    m = munkres.Munkres()
    indices = m.compute(np.copy(D))
    ssum = np.sum([D[i][j] for (i, j) in indices])

    return indices, ssum
