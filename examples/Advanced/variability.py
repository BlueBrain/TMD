import tmd
import view

### Defining average images

def img_diff_distance(Z1, Z2, norm=True):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their absolute difference:
       diff(abs(Z1 - Z2))
    """
    if norm:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()

    diff = np.sum(np.abs(Z2 - Z2))

    # Normalize the difference to % of #pixels
    diff = diff / np.prod(np.shape(Z1))

    return diff

def img_diff(Z1, Z2, norm=True):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and returns
       their absolute difference:
       diff(abs(Z1 - Z2))
    """
    if norm:
        Z1 = Z1 / Z1.max()
        Z2 = Z2 / Z2.max()

    return Z1 - Z2


def number_of_bars(phs_list):
    '''Returns the length of each ph_diagram of a list
    '''
    return [len(p) for p in phs_list]


def weight_bars(phs_list, max_norm=None):
    '''Returns weights according to the number of branches.
    If max not defined the maximum within a group will be used.
    '''
    lengths = number_of_bars(phs_list)
    if max_norm is None:
        max_norm = np.max(lengths)
    return np.array(lengths, dtype=np.float) / max_norm


def persistence_image(ph, norm_factor=None, xlims=None, ylims=None):
    '''Create the data for the generation of the persistence image.
    If norm_factor is provided the data will be normalized based on this,
    otherwise they will be normalized to 1.
    If xlims, ylims are provided the data will be scaled accordingly.   
    '''
    from scipy import stats
    import numpy as np

    if xlims is None:
        xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    if ylims is None:
        ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    X, Y = np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = np.max(Z)

    Zn = Z / norm_factor

    return Zn


def average_ph_image(images_list):
    '''Generates a normalized average image
    from a list of images. Careful: images should be
    at the same scale (x-y) for appropriate comparison.
    '''
    average_imgs = images_list[0]

    for im in images_list[1:]:
        average_imgs = np.add(average_imgs, im)

    average_imgs = average_imgs / len(images_list)

    return average_imgs


def average_ph_image_weighted(images_list, weights):
    '''Generates a normalized average image
    from a list of images. Careful: images should be
    at the same scale (x-y) for appropriate comparison.
    '''
    average_imgs = weights[0] * images_list[0]

    for i,im in enumerate(images_list[1:]):
        average_imgs = np.add(average_imgs, weights[i] * im)

    average_imgs = average_imgs / len(images_list)

    return average_imgs / np.max(average_imgs)

def average_ph_from_list(phs_list, xlims=None, ylims=None, norm_factor=None):
    '''Generates average image from list of phs'''
    imgs = [persistence_image(p, norm_factor=norm_factor, xlims=xlims, ylims=ylims) for p in phs_list]
    return average_ph_image(imgs)


def average_weighted_ph_from_list(phs_list, xlims=None, ylims=None, norm_factor=None, max_norm=None):
    '''Generates average image from list of phs'''
    weights = weight_bars(phs_list, max_norm=max_norm)
    imgs = [persistence_image(p, norm_factor=norm_factor, xlims=xlims, ylims=ylims) for p in phs_list]
    return average_ph_image_weighted(imgs, weights=weights)


def define_limits(phs_list):
    '''Returns the x-y coordinates limits (min, max)
    for a list of persistence diagrams
    '''
    ph = view.plot.collapse(phs_list)
    xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    return xlims, ylims


### Ploting functions

def plot_imgs(img, new_fig=True, subplot=111, title='', xlims=None, ylims=None, cmap=cm.jet, vmin=0, vmax=1., masked=False, threshold=0.01):
    '''Plots the gaussian kernel of the input image.
    '''
    from scipy import stats
    from view import common
    import numpy as np

    if xlims is None:
        xlims = (0,100)
    if ylims is None:
        ylims = (0,100)

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        img = np.ma.masked_where((threshold > np.abs(img)), img)

    cax= ax.imshow(np.rot90(img), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear', extent=xlims+ylims)

    kwargs = {}

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = title

    plt.colorbar(cax)

    ax.set_aspect('equal')

    return common.plot_style(fig=fig, ax=ax, aspect='equal', **kwargs)


def plot_av(phs1, title=''):
    '''Generates and plots the average images from input phs'''
    xlims, ylims = define_limits(phs1)
    imgs1 = [persistence_image(p, xlims=xlims, ylims=ylims) for p in phs1]
    IMG = average_ph_image(imgs1)
    return plot_imgs(IMG, xlims=xlims, ylims=ylims, title=title)


def multiplot(phs1, title=''):
    '''Plots distances, average image and an example from the population'''

    xlims, ylims = define_limits(phs1)
    imgs1 = [persistence_image(p, xlims=xlims, ylims=ylims) for p in phs1]

    IMG = average_ph_image(imgs1)

    distances = [img_diff_distance(IMG, im) for im in imgs1]
    ID = np.argmax(distances)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.hist(distances, bins=20)
    plt.title(title)

    plot_imgs(IMG, xlims=xlims, ylims=ylims, new_fig=False, subplot=132)
    plot_imgs(imgs1[ID], xlims=xlims, ylims=ylims, new_fig=False, subplot=133)

    return distances


def multiplot_outliers(phs1, title=''):
    '''Plots distances, average image and an example from the population'''

    xlims, ylims = define_limits(phs1)
    imgs1 = [persistence_image(p, xlims=xlims, ylims=ylims) for p in phs1]

    IMG = average_ph_image(imgs1)

    distances = [img_diff_distance(IMG, im) for im in imgs1]

    d_mean = np.mean(distances)
    d_std = np.std(distances)
    outliers = np.where((np.array(distances) > d_mean + 2*d_std))[0]
    no_outliers = np.delete(np.arange(len(distances)), outliers)

    IMG_out = average_ph_image(np.array(imgs1)[outliers])
    IMG_without = average_ph_image(np.array(imgs1)[no_outliers])

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.hist(distances, bins=20)
    plt.title(title)

    plot_imgs(IMG, xlims=xlims, ylims=ylims, new_fig=False, subplot=222)
    plt.title('All')
    plot_imgs(IMG_out, xlims=xlims, ylims=ylims, new_fig=False, subplot=223)
    plt.title('Outliers')
    plot_imgs(IMG_without, xlims=xlims, ylims=ylims, new_fig=False, subplot=224)
    plt.title('No_outliers')

    return distances


def outliers_distances(dist_list, n=2):
    '''Outputs the number of cells
    that are outside the n*sigma regime.
    '''
    d_mean = np.mean(dist_list)
    d_std = np.std(dist_list)

    outliers = np.where((np.array(dist_list) > d_mean + n*d_std))[0]

    return outliers, len(outliers), len(dist_list)


def representatives_distances(dist_list, n=3):
    '''Outputs the number of cells
    that are outside the n*sigma regime.
    '''
    return np.argsort(dist_list)[:n]

# Functions using the previous basics, customized for Sandra's datasets

def example_run_outliers(filename='./Female/control 4h/IPL/', title=''):
    '''Special funct'''
    pop = tmd.io.load_population(filename, tree_types={0:'basal'})

    phs1 = []
    pids = []

    for i,n in enumerate(pop.neurons):
        try:
            p = tmd.methods.get_ph_neuron(n)
            if len(p) > 4:
                phs1.append(p)
                pids.append(i)
        except:
            print n.name

    distances = multiplot_outliers(phs1, title=title)

    return distances


def example_run(filename='./Female/control 4h/IPL/', title=''):
    '''Special funct'''
    pop = tmd.io.load_population(filename, tree_types={0:'basal'})

    phs1 = []

    for i,n in enumerate(pop.neurons):
        try:
            p = tmd.methods.get_ph_neuron(n)
            if len(p) > 4:
                phs1.append(p)
        except:
            #print n.name
            pass

    distances = multiplot(phs1, title=title)

    return distances


def get_phs_clean(filename='./Female/control 4h/IPL/'):
    '''Special funct'''
    pop = tmd.io.load_population(filename, tree_types={0:'basal'})

    phs1 = []

    for i,n in enumerate(pop.neurons):
        try:
            p = tmd.methods.get_ph_neuron(n)
            if len(p) > 4:
                phs1.append(p)
        except:
            #print n.name
            pass

    return phs1


def norm_limits(ph_list):
    '''Normalizes the limits and the intensity of ph_images
    in order to compare all the ph of a list.
    '''
    # Normalization of limits
    pp = []
    for ph in ph_list:
        pp = pp + ph
    xlims, ylims = define_limits(pp)

    return xlims, ylims


def norm_intensity(ph_list):
    '''Normalizes the limits and the intensity of ph_images
    in order to compare all the ph of a list.
    '''
    # Normalization of intensity
    intensities = []
    for ph in ph_list:
        Z = average_ph_from_list(ph, norm_factor=1.0, xlims=xlims, ylims=ylims)
        intensities.append(np.max(Z))
    M = np.max(intensities)

    return M


def multiplot_comparison(ph_list, norm=True, diff=1., thresh=0.1, masked=True):

    xlims, ylims = norm_limits(ph_list)

    if norm:
        M = norm_intensity(ph_list)
    else:
        M = None

    # Generate normalized averages
    Zns = []
    for ph in ph_list:
        Z = average_ph_from_list(ph, norm_factor=M, xlims=xlims, ylims=ylims)
        Zns.append(Z)

    plot_imgs(Zns[0], new_fig=True, subplot=331, xlims=xlims, ylims=ylims)
    plot_imgs(Zns[1], new_fig=False, subplot=332, xlims=xlims, ylims=ylims)
    plot_imgs(Zns[2], new_fig=False, subplot=334, xlims=xlims, ylims=ylims)
    plot_imgs(Zns[3], new_fig=False, subplot=335, xlims=xlims, ylims=ylims)

    D1 = img_diff(Zns[0], Zns[2], norm=False)
    D2 = img_diff(Zns[1], Zns[3], norm=False)
    D3 = img_diff(Zns[0], Zns[1], norm=False)
    D4 = img_diff(Zns[2], Zns[3], norm=False)

    plot_imgs(D1, new_fig=False, subplot=337, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh)
    plot_imgs(D2, new_fig=False, subplot=338, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh)
    plot_imgs(D3, new_fig=False, subplot=333, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh)
    plot_imgs(D4, new_fig=False, subplot=336, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh)

    return Zns


def multiplot_weighted_comparison_naive(ph_list, diff=1., thresh=0.1, masked=True, title='', xlims=None, ylims=None):

    if xlims is None and ylims is None:    
        xlims, ylims = norm_limits(ph_list)

    # Generate normalized averages
    Zns = []
    for ph in ph_list:
        Z = average_weighted_ph_from_list(ph, norm_factor=None, xlims=xlims, ylims=ylims)
        Zns.append(Z)

    plot_imgs(Zns[0], new_fig=True, subplot=331, xlims=xlims, ylims=ylims, title='Control 4h')
    plot_imgs(Zns[1], new_fig=False, subplot=332, xlims=xlims, ylims=ylims, title='Control 48h')
    plot_imgs(Zns[2], new_fig=False, subplot=334, xlims=xlims, ylims=ylims, title='Naive 4h')
    plot_imgs(Zns[3], new_fig=False, subplot=335, xlims=xlims, ylims=ylims, title='Naive 48h')

    D1 = img_diff(Zns[0], Zns[2], norm=False)
    D2 = img_diff(Zns[1], Zns[3], norm=False)
    D3 = img_diff(Zns[0], Zns[1], norm=False)
    D4 = img_diff(Zns[2], Zns[3], norm=False)

    plot_imgs(D1, new_fig=False, subplot=337, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Naive 4h')
    print 'Control - Naive 4h :', np.sum(np.abs(D1))
    plot_imgs(D2, new_fig=False, subplot=338, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Naive 48h')
    print 'Control - Naive 48h :', np.sum(np.abs(D2))
    plot_imgs(D3, new_fig=False, subplot=333, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control 4h-48h')
    print 'Control 4h-48h :', np.sum(np.abs(D3))
    plot_imgs(D4, new_fig=False, subplot=336, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Naive 4h-48h')
    print 'Naive 4h-48h :', np.sum(np.abs(D4))

    plt.suptitle(title)
    plt.tight_layout(True)

    return Zns



def multiplot_weighted_comparison(ph_list, diff=1., thresh=0.1, masked=True, title='', xlims=None, ylims=None):

    if xlims is None and ylims is None:    
        xlims, ylims = norm_limits(ph_list)

    # Generate normalized averages
    Zns = []
    for ph in ph_list:
        Z = average_weighted_ph_from_list(ph, norm_factor=None, xlims=xlims, ylims=ylims)
        Zns.append(Z)

    plot_imgs(Zns[0], new_fig=True, subplot=331, xlims=xlims, ylims=ylims, title='Control 4h')
    plot_imgs(Zns[1], new_fig=False, subplot=332, xlims=xlims, ylims=ylims, title='Control 48h')
    plot_imgs(Zns[2], new_fig=False, subplot=334, xlims=xlims, ylims=ylims, title='Drug 4h')
    plot_imgs(Zns[3], new_fig=False, subplot=335, xlims=xlims, ylims=ylims, title='Drug 48h')

    D1 = img_diff(Zns[0], Zns[2], norm=False)
    D2 = img_diff(Zns[1], Zns[3], norm=False)
    D3 = img_diff(Zns[0], Zns[1], norm=False)
    D4 = img_diff(Zns[2], Zns[3], norm=False)

    plot_imgs(D1, new_fig=False, subplot=337, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Drug 4h')
    print 'Control - Drug 4h :', np.sum(np.abs(D1))
    plot_imgs(D2, new_fig=False, subplot=338, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Drug 48h')
    print 'Control - Drug 48h :', np.sum(np.abs(D2))
    plot_imgs(D3, new_fig=False, subplot=333, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control 4h-48h')
    print 'Control 4h-48h :', np.sum(np.abs(D3))
    plot_imgs(D4, new_fig=False, subplot=336, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Drug 4h-48h')
    print 'Drug 4h-48h :', np.sum(np.abs(D4))

    plt.suptitle(title)
    plt.tight_layout(True)

    return Zns


def multiplot_weighted_comparison_male_female(ph_list, diff=1., thresh=0.1, masked=True, title='', xlims=None, ylims=None):

    if xlims is None and ylims is None:    
        xlims, ylims = norm_limits(ph_list)

    # Generate normalized averages
    Zns = []
    for ph in ph_list:
        Z = average_weighted_ph_from_list(ph, norm_factor=None, xlims=xlims, ylims=ylims)
        Zns.append(Z)

    plot_imgs(Zns[0], new_fig=True, subplot=331, xlims=xlims, ylims=ylims, title='Control 4h')
    plot_imgs(Zns[1], new_fig=False, subplot=332, xlims=xlims, ylims=ylims, title='Control 48h')
    plot_imgs(Zns[2], new_fig=False, subplot=334, xlims=xlims, ylims=ylims, title='Drug 4h')
    plot_imgs(Zns[3], new_fig=False, subplot=335, xlims=xlims, ylims=ylims, title='Drug 48h')

    D1 = img_diff(Zns[0], Zns[2], norm=False)
    D2 = img_diff(Zns[1], Zns[3], norm=False)
    D3 = img_diff(Zns[0], Zns[1], norm=False)
    D4 = img_diff(Zns[2], Zns[3], norm=False)

    plot_imgs(D1, new_fig=False, subplot=337, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Drug 4h')
    print 'Control - Drug 4h :', np.sum(np.abs(D1))
    plot_imgs(D2, new_fig=False, subplot=338, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control - Drug 48h')
    print 'Control - Drug 48h :', np.sum(np.abs(D2))
    plot_imgs(D3, new_fig=False, subplot=333, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Control 4h-48h')
    print 'Control 4h-48h :', np.sum(np.abs(D3))
    plot_imgs(D4, new_fig=False, subplot=336, xlims=xlims, ylims=ylims, vmin=-diff, vmax=diff, masked=masked, threshold=thresh, title='Drug 4h-48h')
    print 'Drug 4h-48h :', np.sum(np.abs(D4))

    plt.suptitle(title)
    plt.tight_layout(True)

    return Zns


def distance_number_of_cells(ph_list, step_size=10, samples=10, xlims=None, ylims=None, title=''):
    #import seaborn
    min_max_size = np.min([len(p) for p in ph_list])
    intervals = np.linspace(step_size, min_max_size, (min_max_size-step_size)/step_size)

    distancesCD4 = []
    distancesCD4std = []
    distancesCD48 = []
    distancesCD48std = []

    for i in intervals:

        d4 = []
        d48 = []

        for s in np.arange(samples):
            Zns = []
            for ph in ph_list:
                ph_random_indices = random.choice(np.arange(len(ph)), int(i), replace=False)
                ph_random = np.array(ph)[ph_random_indices]
                Z = average_weighted_ph_from_list(ph_random, norm_factor=None, xlims=xlims, ylims=ylims)
                Zns.append(Z)

            DiffCD4 = np.sum(np.abs(img_diff(Zns[0], Zns[1], norm=False)))
            DiffCD48 = np.sum(np.abs(img_diff(Zns[0], Zns[2], norm=False)))

            d4.append(DiffCD4/ (DiffCD4+DiffCD48))
            d48.append(DiffCD48/ (DiffCD4+DiffCD48))

        distancesCD4.append(np.mean(d4))
        distancesCD48.append(np.mean(d48))
        distancesCD4std.append(np.std(d4))
        distancesCD48std.append(np.std(d48))

    fig = plt.figure(figsize=(15,10))
    plt.plot(intervals, distancesCD4, c='b', label='Dist: C-D4')
    plt.plot(intervals, distancesCD48, c='r', label='Dist: C-D48')

    plt.fill_between(intervals, np.subtract(distancesCD4, distancesCD4std), np.add(distancesCD4, distancesCD4std), color='b', alpha=0.2)
    plt.fill_between(intervals, np.subtract(distancesCD48, distancesCD48std), np.add(distancesCD48, distancesCD48std), color='r', alpha=0.2)

    plt.xlabel('Number of cells', fontsize=18)
    plt.ylabel('Normalized distance', fontsize=18)
    plt.legend()
    plt.title(title)

    return intervals, distancesCD4, distancesCD48, distancesCD4std, distancesCD48std


def check_input_data(filename):

    import os
    L = os.listdir(filename)

    counter_process = 0
    counter_load = 0
    
    for c in L:
        try:
            n = tmd.io.load_neuron(filename+c, tree_types={0:'basal'})
            try:
                p = tmd.methods.get_ph_neuron(n)
            except:
                counter_process = counter_process + 1
        except:
            counter_load = counter_load + 1
            # print c

    return len(L), counter_load, counter_process

