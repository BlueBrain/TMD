'''
tmd Topology analysis algorithms implementation
'''

import numpy as np


def sort_ph(ph, reverse=True):
    """
    Sorts barcode according to decreasing length of bars.
    """
    ph_sort = []

    for ip, p in enumerate(ph):
        ph_sort.append([p[0], p[1], np.abs(p[0] - p[1])])

    ph_sort.sort(key=lambda x: x[2], reverse=reverse)

    return ph_sort


def load_file(filename):
    """Load PH file in a np.array
    """
    f = open(filename, 'r')

    ph = []

    for line in f:

        line = np.array(line.split(), dtype=np.float)
        ph.append(line)

    f.close()

    return np.array(ph)


def horizontal_hist(ph1, num_bins=100, min_bin=None, max_bin=None):
    """Calculate how many barcode lines are found in each bin.
    """
    import math

    if min_bin is None:
        min_bin = np.min(ph1)
    if max_bin is None:
        max_bin = np.max(ph1)

    bins = np.linspace(min_bin, max_bin, num_bins, dtype=float)

    binsize = (max_bin - min_bin) / (num_bins - 1.)

    results = np.zeros(num_bins - 1)

    for ph in ph1:

        ph_decompose = np.linspace(np.min(ph), np.max(ph),
                                   math.ceil((np.max(ph) - np.min(ph)) / binsize),
                                   dtype=float)

        bin_ph = np.histogram(ph_decompose,
                              bins=bins)[0]

        results = np.add(results, bin_ph)

    return bins, results


def step_hist(ph1):
    '''Calculate step distance of ph data'''
    from itertools import chain

    bins = np.unique(list(chain(*ph1)))

    results = np.zeros(len(bins) - 1)

    for ph in ph1:
        for it, _ in enumerate(bins[:-1]):
            if min(ph) <= bins[it + 1] and max(ph) > bins[it]:
                results[it] = results[it] + 1

    return bins, results


def step_dist(ph1, ph2, order=1):
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


def hist_distance(ph1, ph2, norm=1, bins=100):
    """Calculate distance between two ph diagrams.
       Distance definition:
    """
    _, data_1 = horizontal_hist(ph1, num_bins=bins)
    _, data_2 = horizontal_hist(ph2, num_bins=bins)

    return np.linalg.norm(np.abs(np.subtract(data_1, data_2)), norm)


def hist_distance_un(ph1, ph2, norm=1, bins=100):
    """Calculate unnormed distance between two ph diagrams.
    """
    maxb = np.max([np.max(ph1), np.max(ph2)])

    minb = np.min([np.min(ph1), np.min(ph2)])

    _, results1 = horizontal_hist(ph1, num_bins=bins, min_bin=minb,
                                  max_bin=maxb)

    _, results2 = horizontal_hist(ph2, num_bins=bins, min_bin=minb,
                                  max_bin=maxb)

    return np.linalg.norm(np.abs(np.subtract(results1, results2)), norm)


def get_total_length(ph):
    """Calculates the total length of a barcode
       by summing the length of each bar. This should
       be equivalent to the total length of the tree
       if the barcode represents path distances.
    """
    return sum([np.abs(p[1] - p[0]) for p in ph])


def get_image_diff(Z1, Z2):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their absolute difference:
       diff(abs(Z1 - Z2))
    """
    img1 = Z1[0] / Z1[0].max()
    img2 = Z2[0] / Z2[0].max()

    diff = np.sum(np.abs(img2 - img1))

    # Normalize the difference to % of #pixels
    diff = diff / np.prod(np.shape(img1))

    return diff


def get_image_max_diff(Z1, Z2):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their absolute difference:
       diff(abs(Z1 - Z2))
    """
    img1 = Z1[0] / Z1[0].max()
    img2 = Z2[0] / Z2[0].max()

    diff = np.max(np.abs(img2 - img1))

    return diff
