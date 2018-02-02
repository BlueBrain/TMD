'''
tmd statistical analysis on PH diagrams algorithms implementation
'''
import numpy as np


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


def get_apical_point(ph):
    """
    Returns the apical point of the tree from the diagram
    """
    B = np.sort(get_bifurcations(ph))

    fig = plt.figure()
    plt.hist(B, bins=len(ph) / 4)

    heights, bins = np.histogram(B, bins=len(ph) / 4)
    empty_bins = np.where(heights == 0)[0]
    consecutive_empty_bins = np.split(empty_bins, np.where(np.diff(empty_bins) != 1)[0]+1)

    max_separation = np.argmax([len(i) for i in consecutive_empty_bins])

    #print consecutive_empty_bins[max_separation]

    separation = consecutive_empty_bins[max_separation][-1]

    return B[np.where(B > bins[separation + 1])[0][0]]


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
    for i,v in enumerate(var):
        var_new[i] = stats.norm.rvs(v,noise)
    return var_new
