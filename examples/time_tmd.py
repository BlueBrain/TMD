
def find_matching(p1, p2):
    '''Finds the best matching between two
    persistent diagrams and returns a list
    of matched indices. The difference is
    computed between p2 from p1, and
    the matching starts from the longest
    components.    
    '''
    from scipy.spatial import distance
    from view import plot

    p2 = sort_ph(p2)
    p2_symm = [((p[0] + p[1]) / 2., (p[0] + p[1]) / 2.) for p in p2]
    avail = range(len(p1))
    avail_symm = range(len(p2_symm))

    plot.ph_diagram(p1)
    plot.ph_diagram(p2, new_fig=False, color='r')

    for p in p2:
        if avail:
            index = np.argmin( [np.linalg.norm(np.subtract(p[:2], pi)) for pi in np.array(p1)[avail]])
            value = np.array(p1)[avail][index]
            print p[:2], value, np.where(p1 == value)[0][0]
            plt.plot([p[:2][0], value[0]], [p[:2][1], value[1]], c='b')
            avail.remove(np.where(p1 == value)[0][0])
        else:
            index = np.argmin( [np.linalg.norm(np.subtract(p[:2], pi)) for pi in np.array(p2_symm)[avail_symm]])
            value = np.array(p2_symm)[avail_symm][index]
            print p[:2], value, np.where(p2_symm == value)[0][0]
            plt.plot([p[:2][0], value[0]], [p[:2][1], value[1]], c='b')
            avail_symm.remove(np.where(p2_symm == value)[0][0])

    
def find_matching_time_series(p_list):
    '''Finds the best matching between two
    persistent diagrams and returns a list
    of matched indices. The difference is
    computed between p2 from p1, and
    the matching starts from the longest
    components.    
    '''
    from scipy.spatial import distance
    from view import plot

    p2 = sort_ph(p2)
    p1_symm = [((p[0] + p[1]) / 2., (p[0] + p[1]) / 2.) for p in p1]
    avail = range(len(p1))
    avail_symm = range(len(p1_symm))

    plot.ph_diagram(p1)
    plot.ph_diagram(p2, new_fig=False, color='r')

    for p in p2:
        if avail:
            index = np.argmin( [np.linalg.norm(np.subtract(p[:2], pi)) for pi in np.array(p1)[avail]])
            value = np.array(p1)[avail][index]
            print p[:2], value, np.where(p1 == value)[0][0]
            plt.plot([p[:2][0], value[0]], [p[:2][1], value[1]], c='b')
            avail.remove(np.where(p1 == value)[0][0])
        else:
            index = np.argmin( [np.linalg.norm(np.subtract(p[:2], pi)) for pi in np.array(p1_symm)[avail_symm]])
            value = np.array(p1_symm)[avail_symm][index]
            print p[:2], value, np.where(p1_symm == value)[0][0]
            plt.plot([p[:2][0], value[0]], [p[:2][1], value[1]], c='b')
            avail_symm.remove(np.where(p1_symm == value)[0][0])


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

    couples = {x: None for x in xrange(N)} # woman first, then current husband

    count = 0

    while len(free_men) > 0:
        m = free_men.pop()
        choice = men_preferences[m].pop(0)

        if choice in free_women:
            couples[choice] = m
            free_women.remove(choice)
        else:
            current = np.where(np.array(women_preferences)[choice] == couples[choice])[0][0]
            tobe = np.where(np.array(women_preferences)[choice] == m)[0][0]
            if  current < tobe:
                free_men.append(couples[choice])
                couples[choice] = m
            else:
                free_men.append(m)

    if swapped:
        return {couples[k]: k for k in couples}

    return couples


def marry_components(p1, p2, ax=None, z1=0, z2=0.2):
    '''Returns a list of matching components
    '''
    from scipy.spatial.distance import cdist
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    #if ax is None:
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

    first_distances = cdist(p1, p2)
    second_distances = cdist(p2, p1)

    first_pref = [np.argsort(k).tolist() for k in first_distances]
    second_pref = [np.argsort(k).tolist() for k in second_distances]

    church = marriage_problem(first_pref, second_pref)

    #ax.scatter3D(np.transpose(p1)[0], np.transpose(p1)[1], z1)
    #ax.scatter3D(np.transpose(p2)[0], np.transpose(p2)[1], z2, c='r')
    #ax.plot3D([0,200], [0,200], z1, c='g')
    #ax.plot3D([0,200], [0,200], z2, c='g')

    def symmetric(i, j):
        return ((i + j) / 2., (i + j) / 2.)

    speed = []

    for c in church:
        if c is None:
            x, y = symmetric(p2[church[c]][0], p2[church[c]][1])
            #ax.plot3D([x, p2[church[c]][0]], [y, p2[church[c]][1]], [z1,z2], c='b')
            sp = np.linalg.norm(np.subtract([x, p2[church[c]][0]], [y, p2[church[c]][1]]))
        elif church[c] is not None:
            #ax.plot3D([p1[c][0], p2[church[c]][0]], [p1[c][1], p2[church[c]][1]], [z1,z2], c='b')
            sp = np.linalg.norm(np.subtract([p1[c][0], p2[church[c]][0]], [p1[c][1], p2[church[c]][1]]))
        else:
            x, y = symmetric(p1[c][0], p1[c][1])
            #ax.plot3D([p1[c][0], x], [p1[c][1], y], [z1,z2], c='b')
            sp = np.linalg.norm(np.subtract([p1[c][0], x], [p1[c][1], y]))

        speed.append(sp)

    return speed

def vineyards(ph_list):
    from matplotlib import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    speeds = []

    for ip, p in enumerate(ph_list[:-1]):
        sp = marry_components(ph_list[ip], ph_list[ip+1], z1=ip*0.2, z2=(ip + 1)*0.2)
        speeds.append(sp)
        #ax = plt.gca()

    return speeds


def get_persistence_diagram_timelapse(trees, **kwargs):
    '''Method to extract ph from tree that contains mutlifurcations'''
    ph = []

    for itr, tree in enumerate(trees):
        rd = getattr(tree, 'get_point_radial_distances_time')(time=itr, **kwargs)

        active = tree.get_bif_term() == 0

        beg, end = tree.get_sections_3()

        beg = np.array(beg)
        end = np.array(end)

        parents = {e: b for b, e in zip(beg, end)}
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

                    alive.remove(mx_id)
                    c = np.delete(c, mx)

                    for ci in c:
                        ph.append([rd[ci], rd[p]])
                        alive.remove(ci)

                    alive.append(p)

                    rd[p] = rd[mx_id]

        ph.append([rd[np.where(active)[0][0]], 0]) # Add the last alive component

    return ph
