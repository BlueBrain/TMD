'''
tmd matching algorithms implementation
'''


def marriage_problem(women_preferences, men_preferences):
    '''Matches N women to M men so that max(M, N)
    are coupled to their preferred choice that is available
    See https://en.wikipedia.org/wiki/Stable_marriage_problem
    '''
    N = len(women_preferences)
    M = len(men_preferences)

    swapped = False

    if M > N:
        swap = women_preferences
        women_preferences = men_preferences
        men_preferences = swap
        N = len(women_preferences)
        M = len(men_preferences)
        swapped = True

    free_women = range(N)
    free_men = range(M)

    couples = {x: None for x in range(N)}  # woman first, then current husband

    while len(free_men) > 0:
        m = free_men.pop()
        choice = men_preferences[m].pop(0)

        if choice in free_women:
            couples[choice] = m
            free_women.remove(choice)
        else:
            current = np.where(np.array(women_preferences)[choice] == couples[choice])[0][0]
            tobe = np.where(np.array(women_preferences)[choice] == m)[0][0]
            if current < tobe:
                free_men.append(couples[choice])
                couples[choice] = m
            else:
                free_men.append(m)

    if swapped:
        return [(couples[k], k) for k in couples]

    return [(k, couples[k]) for k in couples]


def symmetric(p):
    '''Returns the symmetric point of a PD point on the diagonal
    '''
    return [(p[0] + p[1]) / 2., (p[0] + p[1]) / 2]


def matching_diagrams(p1, p2, plot=False, method='munkres', use_diag=True, new_fig=True, subplot=(111)):
    '''Returns a list of matching components
    Possible matching methods:
    - munkress
    - marriage problem
    '''
    from scipy.spatial.distance import cdist
    import munkres
    from tmd.view import common as _cm

    def plot_matching(p1, p2, indices, new_fig=True, subplot=(111)):
        '''Plots matching between p1, p2
        for the corresponding indices
        '''
        import pylab as plt
        fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)
        for i, j in indices:
            ax.plot((p1[i][0], p2[j][0]), (p1[i][1], p2[j][1]), color='black')
            ax.scatter(p1[i][0], p1[i][1], c='r')
            ax.scatter(p2[j][0], p2[j][1], c='b')

    if use_diag:
        p1_enh = p1 + [symmetric(i) for i in p2]
        p2_enh = p2 + [symmetric(i) for i in p1]
    else:
        p1_enh = p1
        p2_enh = p2

    D = cdist(p1_enh, p2_enh)

    if method == 'munkres':
        m = munkres.Munkres()
        indices = m.compute(np.copy(D))
    elif method == 'marriage':
        first_pref = [np.argsort(k).tolist() for k in cdist(p1_enh, p2_enh)]
        second_pref = [np.argsort(k).tolist() for k in cdist(p2_enh, p1_enh)]
        indices = marriage_problem(first_pref, second_pref)

    if plot:
        plot_matching(p1_enh, p2_enh, indices, new_fig=new_fig, subplot=subplot)

    ssum = np.sum([D[i][j] for (i, j) in indices])

    return indices, ssum

