# Example script to compute the diversity index from a set of classes.


def diversity_index(perc, simil, q):
    """Computes the generalized diversity index
       as described in http://onlinelibrary.wiley.com/doi/10.1890/10-2402.1/abstract
       Inputs:
            perc: list of percentages of species distribution
            of size S.
            simil: confusion matrix indicating the similarity
            between species of size SxS.
            q: the order of diversity index.
    """
    import numpy as np

    perc = np.array(perc, dtype=float) / sum(perc)

    diq = 0.0

    for i in range(len(perc)):
        zpi = np.dot(simil[i], perc)
        diq = diq + np.power(perc[i], q) * np.power(zpi, q - 1.)

    return np.power(diq, 1. / (1. - q))


def diversity_index_inf(perc, simil):
    """Computes the generalized diversity index
       as described in http://onlinelibrary.wiley.com/doi/10.1890/10-2402.1/abstract
       Inputs:
            perc: list of percentages of species distribution
            of size S.
            simil: confusion matrix indicating the similarity
            between species of size SxS.
            q: the order of diversity index is set to inf
    """
    import numpy as np

    diq = np.inf

    perc = np.array(perc, dtype=float) / sum(perc)

    for i in range(len(perc)):
        zpi = np.dot(simil[i], perc)
        diq = min(diq, zpi)

    return np.float(1.) / diq


def diversity_index_one(perc, simil):
    """Computes the generalized diversity index
       as described in http://onlinelibrary.wiley.com/doi/10.1890/10-2402.1/abstract
       Inputs:
            perc: list of percentages of species distribution
            of size S.
            simil: confusion matrix indicating the similarity
            between species of size SxS.
            q: the order of diversity index is set to one
    """
    import numpy as np

    diq = 1.0

    perc = np.array(perc, dtype=float) / sum(perc)

    for i in range(len(perc)):
        zpi = np.dot(simil[i], perc)
        diq = diq * np.power(zpi, perc[i])

    return np.float(1.) / diq


def diversity_index_zero(perc, simil):
    """Computes the generalized diversity index
       as described in http://onlinelibrary.wiley.com/doi/10.1890/10-2402.1/abstract
       Inputs:
            perc: list of percentages of species distribution
            of size S.
            simil: confusion matrix indicating the similarity
            between species of size SxS.
            q: the order of diversity index is set to one
    """
    import numpy as np

    diq = 1.0

    perc = np.array(perc, dtype=float) / sum(perc)

    for i in range(len(perc)):
        zpi = np.dot(simil[i], perc)
        diq = diq * np.power(zpi, perc[i])

    return np.float(1.) / diq

    perc = np.array(perc, dtype=float) / sum(perc)

    diq = 0.0

    for i in range(len(perc)):
        zpi = np.dot(simil[i], perc)
        if not np.close(zpi, 0.0):
            diq = diq + perc[i] / zpi

    return diq


def diversity_vary_q(perc, simil, dep=np.linspace(0.05, 0.95, 20).tolist() + range(2, 10)):
    """Computes the diversity index
       with different q values:
       from q=0 to q=infinity
    """
    # div_index = [diversity_index_zero(perc, simil), diversity_index_one(perc, simil)]

    div_index = [diversity_index(perc, simil, q) for q in dep]

    # div_index = div_index + [diversity_index_inf(perc, simil)]

    return dep, div_index
