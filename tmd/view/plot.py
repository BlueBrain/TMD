"""Plotting functions of tmd"""

import numpy as np
from matplotlib import pylab as plt
from tmd.view import common as cm
from tmd.Topology import analysis
from tmd.view.common import jet_map
from tmd.Topology.statistics import transform_ph_to_length


def barcode(ph, new_fig=True, subplot=False, color='b', linewidth=1.2, **kwargs):
    """
    Generates a 2d figure (barcode) of the persistent homology
    of a tree as it has been computed by
    Topology.get_persistent_homology method.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ph_sort = analysis.sort_ph(ph)

    for ip, p in enumerate(ph_sort):
        ax.plot(p[:2], [ip, ip], c=color, linewidth=linewidth)

    kwargs['title'] = kwargs.get('title', 'Persistence barcode')
    kwargs['xlabel'] = kwargs.get('xlabel', 'Lifetime: radial distance from soma')

    plt.ylim([-1, len(ph_sort)])
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def barcode_enhanced(ph, new_fig=True, subplot=False, linewidth=1.2,
                     valID=2, cmap=jet_map, **kwargs):
    """
    Generates a 2d figure (barcode) of the persistent homology
    of a tree enhanced by a parameter encodes in ph[valID]
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    Z = [[0, 0], [0, 0]]
    levels = np.linspace(0.0, val_max, 200)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    def sort_ph_enhanced(ph, valID):
        """
        Sorts barcode according to length.
        """
        ph_sort = [p[:valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort.sort(key=lambda x: x[valID + 1])
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    for ip, p in enumerate(ph_sort):
        ax.plot(p[:2], [ip, ip], c=cmap(p[valID] / val_max), linewidth=linewidth)

    kwargs['title'] = kwargs.get('title', 'Barcode of p.h.')
    kwargs['xlabel'] = kwargs.get('xlabel', 'Lifetime')
    plt.ylim([-1, len(ph_sort)])
    plt.colorbar(CS3)

    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def diagram(ph, new_fig=True, subplot=False, color='b', alpha=1.0,
            edgecolors='black', s=30, **kwargs):
    """
    Generates a 2d figure (ph diagram) of the persistent homology of a tree.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    bounds_max = np.max(np.max(ph))
    bounds_min = np.min(np.min(ph))
    plt.plot([bounds_min, bounds_max], [bounds_min, bounds_max], c='black')

    ax.scatter(np.array(ph)[:, 0], np.array(ph)[:, 1], c=color, alpha=alpha,
               edgecolors=edgecolors, s=s)

    kwargs['title'] = kwargs.get('title', 'Persistence diagram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def diagram_enhanced(ph, new_fig=True, subplot=False, alpha=1.0, valID=2,
                     cmap=jet_map, edgecolors='black', s=30, **kwargs):
    """
    Generates a 2d figure (diagram) of the persistent homology
    of a tree enhanced by a parameter encodes in ph[valID]
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    Z = [[0, 0], [0, 0]]
    levels = np.linspace(0.0, val_max, 200)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    def sort_ph_enhanced(ph, valID):
        """
        Sorts barcode according to length.
        """
        ph_sort = [p[:valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort.sort(key=lambda x: x[valID + 1])
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    bounds_max = np.max(np.max(ph_sort))
    bounds_min = np.min(np.min(ph_sort))
    plt.plot([bounds_min, bounds_max], [bounds_min, bounds_max], c='black')

    colors = [cmap(p[valID] / val_max) for p in ph_sort]

    ax.scatter(np.array(ph_sort)[:, 0], np.array(ph_sort)[:, 1],
               c=colors, alpha=alpha, edgecolors=edgecolors, s=s)

    kwargs['title'] = kwargs.get('title', 'Persistence diagram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    plt.ylim([-1, len(ph_sort)])
    plt.colorbar(CS3)

    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image(ph, new_fig=True, subplot=111, xlims=None, ylims=None,
                      masked=False, colorbar=False, norm_factor=None, threshold=0.01,
                      vmin=None, vmax=None, cmap=jet_map, bw_method=None, **kwargs):
    '''Plots the gaussian kernel
       of the ph diagram that is given.
    '''
    if xlims is None or xlims is None:
        xlims, ylims = analysis.get_limits(ph, coll=False)

    # pylint: disable=unexpected-keyword-arg
    Zn = analysis.get_persistence_image_data(ph, norm_factor=norm_factor, bw_method=bw_method,
                                             xlims=xlims, ylims=ylims)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        Zn = np.ma.masked_where((threshold > Zn), Zn)

    cax = ax.imshow(np.rot90(Zn), vmin=vmin, vmax=vmax, cmap=cmap,
                    interpolation='bilinear', extent=xlims + ylims)

    if colorbar:
        plt.colorbar(cax)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = kwargs.get('title', 'Persistence image')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return Zn, cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image_diff(Z1, Z2, new_fig=True, subplot=111, xlims=None, ylims=None,
                           norm=True, vmin=-1., vmax=1., cmap=jet_map, **kwargs):
    """Takes as input two images as exported from the gaussian kernel
       plotting function, and plots their difference.
    """
    if xlims is None or xlims is None:
        xlims, ylims = ((0, 100), (0, 100))

    difference = analysis.get_image_diff_data(Z1, Z2, normalized=norm)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(np.rot90(difference), vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='bilinear', extent=xlims + ylims)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image_add(Z2, Z1, new_fig=True, subplot=111, xlims=None, ylims=None,
                          norm=True, vmin=0, vmax=2., cmap=jet_map, **kwargs):
    """Takes as input two images as exported from the gaussian kernel
       plotting function, and plots their addition.
    """
    if xlims is None or xlims is None:
        xlims, ylims = ((0, 100), (0, 100))

    addition = analysis.get_image_add_data(Z1, Z2, normalized=norm)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(np.rot90(addition), vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='bilinear', extent=xlims + ylims)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image_average(ph_list, new_fig=True, subplot=111, xlims=None,
                              ylims=None, norm_factor=1.0, vmin=None, vmax=None,
                              cmap=jet_map, weighted=False, **kwargs):
    """Merges a list of ph diagrams and plots their respective average image.
    """
    # pylint: disable=unexpected-keyword-arg
    av_imgs = analysis.get_average_persistence_image(ph_list, xlims=xlims, ylims=ylims,
                                                     norm_factor=norm_factor, weighted=weighted)
    if xlims is None or xlims is None:
        xlims, ylims = analysis.get_limits(ph_list)

    if vmin is None:
        vmin = np.min(av_imgs)
    if vmax is None:
        vmax = np.max(av_imgs)

    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(np.rot90(av_imgs), vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='bilinear', extent=xlims + ylims)

    kwargs['xlim'] = xlims
    kwargs['ylim'] = ylims
    kwargs['title'] = kwargs.get('title', 'Average persistence image')
    kwargs['xlabel'] = kwargs.get('xlabel', 'End radial distance from soma')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Start radial distance from soma')

    return av_imgs, cm.plot_style(fig=fig, ax=ax, **kwargs)


def start_length_diagram(ph, new_fig=True, subplot=False, color='b', alpha=1.0, **kwargs):
    '''Plots the transformed ph diagram that represents lengths and starting points
       of a component.
    '''
    ph_transformed = transform_ph_to_length(ph)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    for p in ph_transformed:
        ax.scatter(p[0], p[1], c=color, edgecolors='black', alpha=alpha)

    kwargs['title'] = kwargs.get('title', 'Transformed Persistence diagram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'Start of the component')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Length of the component')
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def histogram_stepped(ph, new_fig=True, subplot=False, color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the stepped histogram of a persistent
       homology array.
    '''
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_stepped(ph)
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def histogram_stepped_population(ph_list, new_fig=True, subplot=False,
                                 color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the stepped histogram of a list of persistence diagrams.
       The histogram is normalized acording to the number of persistence diagrams.
    '''
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_stepped(analysis.collapse(ph_list))
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1] / len(ph_list), color=color, alpha=alpha)
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def histogram_horizontal(ph, new_fig=True, subplot=False, bins=100, color='b', alpha=0.7, **kwargs):
    '''Extracts and plots the binned histogram of a persistent
       homology array.
    '''
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_horizontal(ph, num_bins=bins)
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)

    return cm.plot_style(fig=fig, ax=ax, **kwargs)
