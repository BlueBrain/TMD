def define_limits(phs_list):
    '''Returns the x-y coordinates limits (min, max)
    for a list of persistence diagrams
    '''
    ph = view.plot.collapse(phs_list)
    xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    return xlims, ylims


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
