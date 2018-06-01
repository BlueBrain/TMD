"""Plotting functions of tmd"""

import numpy as _np
from tmd import Topology as _tm
import view as _view
import common as _cm
from tmd.Topology.analysis import collapse

def _sort_ph(ph):
    """
    Sorts barcode according to length of bars.
    """
    ph_sort = []

    for ip, p in enumerate(ph):
        ph_sort.append([p[0], p[1], _np.abs(p[0] - p[1])])

    ph_sort.sort(key=lambda x: x[2])

    return ph_sort


def barcode(ph, new_fig=True, subplot=False, color='b', linewidth=1.2, **kwargs):
    """
    Generates a 2d figure (barcode) of the persistent homology
    of a tree as it has been computed by
    Topology.get_persistent_homology method.

    Parameters
    ----------
    ph: 2d array
        persistent homology array

    Options
    -------
    color: str or None
        Defines the color of the barcode.
        Default value is 'red'.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    Returns
    --------
    A 2D matplotlib figure with a barcode.

    """
    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    ph_sort = _sort_ph(ph)

    for ip, p in enumerate(ph_sort):

        ax.plot(p[:2], [ip, ip], c=color, linewidth=linewidth)

    kwargs['title'] = kwargs.get('title', 'Persistence barcode')
    kwargs['xlabel'] = kwargs.get('xlabel', 'Lifetime: radial distance from soma')

    _cm.plt.ylim([-1, len(ph_sort)])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def ph_diagram(ph, new_fig=True, subplot=False, color='b', alpha=1.0, **kwargs):
    """
    Generates a 2d figure (ph diagram) of the persistent homology
    of a tree as it has been computed by
    Topology.get_persistent_homology method.

    Parameters
    ----------
    ph: 2d array
        persistent homology array

    Options
    -------
    color: str or None
        Defines the color of the barcode.
        Default value is 'red'.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    Returns
    --------
    A 2D matplotlib figure with a barcode.

    """
    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    bounds_max = _np.max(_np.max(ph))
    bounds_min = _np.min(_np.min(ph))

    for p in ph:

        ax.scatter(p[0], p[1], c=color, edgecolors='black', alpha=alpha)

    _cm.plt.plot([bounds_min, bounds_max], [bounds_min, bounds_max], c='black')

    kwargs['title'] = kwargs.get('title', 'Persistence diagram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def ph_image(ph, new_fig=True, subplot=111, xlims=None, ylims=None, masked=False, colorbar=False,
             norm_factor=None, threshold=0.01, vmin=None, vmax=None, cmap=_cm.plt.cm.jet, **kwargs):
    '''Plots the gaussian kernel
       of the ph diagram that is given.
    '''
    from tmd.Topology.analysis import persistence_image_data

    if xlims is None:
        xlims = [min(_np.transpose(ph)[0]), max(_np.transpose(ph)[0])]
    if ylims is None:
        ylims = [min(_np.transpose(ph)[1]), max(_np.transpose(ph)[1])]

    Zn = persistence_image_data(ph, norm_factor=norm_factor,
                                xlims=xlims, ylims=ylims)

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        Zn = _np.ma.masked_where((threshold > Zn), Zn)

    cax = ax.imshow(_np.rot90(Zn), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear', extent=xlims+ylims)

    if colorbar:
        _cm.plt.colorbar(cax)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = kwargs.get('title', 'Persistence image')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return Zn, _cm.plot_style(fig=fig, ax=ax, **kwargs)


def tree_all(tree, plane='xy', feature='radial_distances', title='',
             diameter=True, treecol='b', xlims=None, ylims=None, **kwargs):
    '''Subplot with ph, barcode and tree
    '''
    from tmd import utils as _utils
    from matplotlib.collections import LineCollection

    kwargs['output_path'] = kwargs.get('output_path', None)

    fig1, ax1 = _view.tree(tree, new_fig=True, subplot=221, plane='xy',
                           title=title, treecolor=treecol, diameter=diameter)

    feat = getattr(tree, 'get_section_' + feature)()
    segs = tree.get_segments()

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment"""

        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ['xy', 'yx', 'zx', 'xz', 'yz', 'zy']:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception('Plane value not recognised')

    bounds = max(max(ph))

    fig1, ax2 = ph_diagram(ph, new_fig=False, subplot=222, color=treecol)

    fig1, ax3 = barcode(ph, new_fig=False, subplot=223, color=treecol)

    fig1, ax4 = ph_image(ph, new_fig=False, subplot=224, xlims=xlims, ylims=ylims)

    _cm.plt.tight_layout(True)

    if kwargs['output_path'] is not None:
        fig = _cm.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def neu_all(neuron, plane='xy', feature='radial_distances', title='',
            diameter=True, treecol='b', xlims=None, ylims=None, neurite_type='basal', **kwargs):
    '''Subplot with ph, barcode
       and tree within spheres
    '''
    from tmd import utils as _utils
    from matplotlib.collections import LineCollection

    kwargs['output_path'] = kwargs.get('output_path', None)

    fig1, ax1 = _view.neuron(neuron, new_fig=True, subplot=221, plane='xy', neurite_type=[neurite_type],
                             title=title, treecolor=treecol, diameter=diameter)

    if plane in ['xy', 'yx', 'zx', 'xz', 'yz', 'zy']:
        ph = _tm.methods.get_ph_neuron(neuron, feature=feature, neurite_type=neurite_type)
    else:
        raise Exception('Plane value not recognised')

    bounds = max(max(ph))

    fig1, ax2 = ph_diagram(ph, new_fig=False, subplot=222, color=treecol)

    fig1, ax3 = barcode(ph, new_fig=False, subplot=223, color=treecol)

    fig1, ax4 = ph_image(ph, new_fig=False, subplot=224, xlims=xlims, ylims=ylims)

    _cm.plt.tight_layout(True)

    if kwargs['output_path'] is not None:
        fig = _cm.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def stepped_hist(ph, new_fig=True, subplot=False, color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the stepped histogram of a persistent
       homology array.
    '''
    from tmd.Topology.analysis import step_hist

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    hist_data = step_hist(ph)

    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def stepped_hist_population(ph_list, new_fig=True, subplot=False, color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the stepped histogram of a list of persistence diagrams.
       The histogram is normalized acording to the number of persistence diagrams.
    '''
    from tmd.Topology.analysis import step_hist

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    hist_data = step_hist(collapse(ph_list))

    ax.fill_between(hist_data[0][:-1], 0, hist_data[1]/len(ph_list), color=color, alpha=alpha)

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def horizontal_hist(ph, new_fig=True, subplot=False, bins=100, color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the binned histogram of a persistent
       homology array.
    '''
    from tmd.Topology.analysis import horizontal_hist

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    hist_data = horizontal_hist(ph, num_bins=bins)

    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def image_diff(Z1, Z2, new_fig=True, subplot=111, xlims=None, ylims=None, norm=True,
               vmin=-1., vmax=1., cmap=_cm.plt.cm.jet, **kwargs):
    """Takes as input two images as exported from the gaussian kernel
       plotting function, and plots their difference.
    """
    from tmd.Topology.analysis import img_diff_data

    difference = img_diff_data(Z1, Z2, norm=norm)

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    ax.imshow(_np.rot90(difference), vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='bilinear', extent=xlims+ylims)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims

    #_cm.plt.colorbar()

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def image_add(Z2, Z1, new_fig=True, subplot=111, xlims=None, ylims=None, **kwargs):
    """Takes as input two images
       as exported from the gaussian kernel
       plotting function, and plots
       their difference.
    """
    if xlims is None:
        xmin = min(Z1[1][1].get_xlim() + Z2[1][1].get_xlim())
        xmax = max(Z1[1][1].get_xlim() + Z2[1][1].get_xlim())
        xlims = [xmin, xmax]
    if ylims is None:
        ymin = min(Z1[1][1].get_ylim() + Z2[1][1].get_ylim())
        ymax = max(Z1[1][1].get_ylim() + Z2[1][1].get_ylim())
        ylims = [ymin, ymax]

    X, Y = _np.mgrid[xlims[0]:xlims[1]:100j, ylims[0]:ylims[1]:100j]

    add_img = Z1[0] / Z1[0].max() + Z2[0] / Z2[0].max()

    add_img = add_img / add_img.max()

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    _cm.plt.pcolor(X, Y, add_img, vmin=0.0, vmax=1.0)

    _cm.plt.colorbar()

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def plot_average(ph_list, new_fig=True, subplot=111, xlims=None, ylims=None, bins=100j, 
                norm_factor=1.0, masked=False, vmin=0., vmax=1., cmap=_cm.plt.cm.jet, **kwargs):
    """Merges ph diagrams and plots them.
    """
    from tmd.Topology.analysis import average_image

    av_imgs = average_image(ph_list, xlims=xlims, ylims=ylims, bins=bins, **kwargs)

    if xlims is None:
        xlims = [min(_np.transpose(collapse(ph_list))[0]), max(_np.transpose(collapse(ph_list))[0])]
    if ylims is None:
        ylims = [min(_np.transpose(collapse(ph_list))[1]), max(_np.transpose(collapse(ph_list))[1])]

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    ax.imshow(_np.rot90(av_imgs), vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='bilinear', extent=xlims+ylims)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = kwargs.get('title', 'Average persistence image')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return av_imgs, _cm.plot_style(fig=fig, ax=ax, **kwargs)


def start_length_plot(ph, direction=False, new_fig=True, subplot=False, color='b', alpha=1.0, **kwargs):
    '''Plots the transformed ph diagram that
    represents lengths and starting points of
    a component.
    '''
    from tmd.Topology.analysis import transform_to_length

    ph_transformed = transform_to_length(ph, direction=direction)

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    for p in ph:
        ax.scatter(p[0], p[1], c=color, edgecolors='black', alpha=alpha)

    kwargs['title'] = kwargs.get('title', 'Transformed Persistence diagram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'Start of the component')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Length of the component')

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def plot_img_basic(img, new_fig=True, subplot=111, title='', xlims=None, ylims=None,
                   cmap=_cm.plt.cm.jet, vmin=None, vmax=None, masked=False, threshold=0.01):
    '''Plots the gaussian kernel of the input image.
    '''
    if xlims is None:
        xlims = (0,100)
    if ylims is None:
        ylims = (0,100)

    if vmin is None:
        vmin = _np.min(img)
    if vmax is None:
        vmax = _np.max(img)

    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        img = _np.ma.masked_where((threshold > _np.abs(img)), img)

    cax = ax.imshow(_np.rot90(img), vmin=vmin, vmax=vmax, cmap=cmap,
                    interpolation='bilinear', extent=xlims+ylims)

    kwargs = {}

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = title

    _cm.plt.colorbar(cax)

    ax.set_aspect('equal')

    return _cm.plot_style(fig=fig, ax=ax, aspect='equal', **kwargs)
