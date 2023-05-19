"""Plotting functions of TMD."""

# Copyright (C) 2022  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from matplotlib import pylab as plt

from tmd.Topology import analysis
from tmd.Topology import distances
from tmd.Topology import vectorizations
from tmd.Topology.statistics import transform_ph_to_length
from tmd.view import common as cm
from tmd.view.common import jet_map


def barcode(ph, new_fig=True, subplot=False, color="b", linewidth=1.2, **kwargs):
    """Generate a 2d figure (barcode) of the persistent homology of a tree.

    The persistent homology should have been computed by Topology.get_persistent_homology method.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ph_sort = analysis.sort_ph(ph)

    for ibar, pbar in enumerate(ph_sort):
        bar_color = color[ibar] if isinstance(color, list) else color
        ax.plot(pbar[:2], [ibar, ibar], c=bar_color, linewidth=linewidth)

    all_kwargs = {
        "title": "Persistence barcode",
        "xlabel": "Lifetime: radial distance",
    }
    all_kwargs.update(kwargs)

    plt.ylim([-1, len(ph_sort)])
    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def barcode_enhanced(
    ph, new_fig=True, subplot=False, linewidth=1.2, valID=2, cmap=jet_map, **kwargs
):
    """Generate a 2d figure (barcode) of the persistent homology of an enhanced tree.

    The tree is enhanced by a parameter encoded in ph[valID].
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    Z = [[-100, -100], [-100, -100]]
    levels = np.linspace(0.0, val_max, 200)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    def sort_ph_enhanced(ph, valID):
        """Sorts barcode according to length."""
        ph_sort = [p[: valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort.sort(key=lambda x: x[valID + 1])
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    for ibar, pbar in enumerate(ph_sort):
        ax.plot(pbar[:2], [ibar, ibar], c=cmap(pbar[valID] / val_max), linewidth=linewidth)

    all_kwargs = {
        "title": "Persistence barcode",
        "xlabel": "Lifetime",
    }
    all_kwargs.update(kwargs)

    plt.ylim([-1, len(ph_sort)])
    plt.colorbar(CS3)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def diagram(
    ph, new_fig=True, subplot=False, color="b", alpha=1.0, edgecolors="black", s=30, **kwargs
):
    """Generate a 2d figure (ph diagram) of the persistent homology of a tree."""
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    bounds_max = np.max(np.max(ph))
    bounds_min = np.min(np.min(ph))
    plt.plot([bounds_min, bounds_max], [bounds_min, bounds_max], c="black")

    ax.scatter(
        np.array(ph)[:, 0], np.array(ph)[:, 1], c=color, alpha=alpha, edgecolors=edgecolors, s=s
    )

    all_kwargs = {
        "title": "Persistence diagram",
        "xlabel": "End radial distance",
        "ylabel": "Start radial distance",
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def diagram_enhanced(
    ph,
    new_fig=True,
    subplot=False,
    alpha=1.0,
    valID=2,
    cmap=jet_map,
    edgecolors="black",
    s=30,
    **kwargs,
):
    """Generate a 2d figure (diagram) of the persistent homology of a enhanced tree.

    The tree is enhanced by a parameter encodes in ph[valID].
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    Z = [[0, 0], [0, 0]]
    levels = np.linspace(0.0, val_max, 200)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    def sort_ph_enhanced(ph, valID):
        """Sorts barcode according to length."""
        ph_sort = [p[: valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort.sort(key=lambda x: x[valID + 1])
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    bounds_max = np.max(np.max(ph_sort))
    bounds_min = np.min(np.min(ph_sort))
    plt.plot([bounds_min, bounds_max], [bounds_min, bounds_max], c="black")

    colors = [cmap(p[valID] / val_max) for p in ph_sort]

    ax.scatter(
        np.array(ph_sort)[:, 0],
        np.array(ph_sort)[:, 1],
        c=colors,
        alpha=alpha,
        edgecolors=edgecolors,
        s=s,
    )

    all_kwargs = {
        "title": "Persistence diagram",
        "xlabel": "End radial distance",
        "ylabel": "Start radial distance",
    }
    all_kwargs.update(kwargs)

    plt.ylim([-1, len(ph_sort)])
    plt.colorbar(CS3)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def persistence_image(  # pylint: disable=too-many-arguments
    ph,
    new_fig=True,
    subplot=111,
    xlim=None,
    ylim=None,
    masked=False,
    colorbar=False,
    norm_factor=None,
    threshold=0.01,
    vmin=None,
    vmax=None,
    cmap=jet_map,
    bw_method=None,
    weights=None,
    resolution=100,
    **kwargs,
):
    """Plot the gaussian kernel of the ph diagram that is given."""
    if xlim is None or xlim is None:
        xlim, ylim = vectorizations.get_limits(ph)

    # pylint: disable=unexpected-keyword-arg
    Zn = vectorizations.persistence_image_data(
        ph,
        norm_factor=norm_factor,
        bw_method=bw_method,
        xlim=xlim,
        ylim=ylim,
        weights=weights,
        resolution=resolution,
    )
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        Zn = np.ma.masked_where((threshold > Zn), Zn)

    cax = ax.imshow(
        np.rot90(Zn),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        extent=xlim + ylim,
    )

    if colorbar:
        plt.colorbar(cax)

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim

    all_kwargs = {
        "title": "Persistence image",
        "xlabel": "End radial distance",
        "ylabel": "Start radial distance",
    }
    all_kwargs.update(kwargs)

    return Zn, cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def persistence_image_diff(
    Z1,
    Z2,
    new_fig=True,
    subplot=111,
    xlim=None,
    ylim=None,
    norm=True,
    vmin=-1.0,
    vmax=1.0,
    cmap=jet_map,
    **kwargs,
):
    """Plot the difference of 2 images from the gaussian kernel plotting function.

    The difference is computed as: diff(Z1 - Z2))
    """
    if xlim is None or xlim is None:
        xlim, ylim = ((0, 100), (0, 100))

    difference = distances.image_diff_data(Z1, Z2, normalized=norm)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(
        np.rot90(difference),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        extent=xlim + ylim,
    )

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image_add(
    Z2,
    Z1,
    new_fig=True,
    subplot=111,
    xlim=None,
    ylim=None,
    norm=True,
    vmin=0,
    vmax=2.0,
    cmap=jet_map,
    **kwargs,
):
    """Plot the sum of 2 images from the gaussian kernel plotting function."""
    if xlim is None or xlim is None:
        xlim, ylim = ((0, 100), (0, 100))

    addition = analysis.get_image_add_data(Z1, Z2, normalized=norm)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(
        np.rot90(addition),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        extent=xlim + ylim,
    )

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def persistence_image_average(
    ph_list,
    new_fig=True,
    subplot=111,
    xlim=None,
    ylim=None,
    norm_factor=1.0,
    vmin=None,
    vmax=None,
    cmap=jet_map,
    weighted=False,
    **kwargs,
):
    """Merge a list of ph diagrams and plot their respective average image."""
    # pylint: disable=unexpected-keyword-arg
    av_imgs = analysis.get_average_persistence_image(
        ph_list, xlim=xlim, ylim=ylim, norm_factor=norm_factor, weighted=weighted
    )
    if xlim is None or xlim is None:
        xlim, ylim = vectorizations.get_limits(ph_list)

    if vmin is None:
        vmin = np.min(av_imgs)
    if vmax is None:
        vmax = np.max(av_imgs)

    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    ax.imshow(
        np.rot90(av_imgs),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        extent=xlim + ylim,
    )

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim

    all_kwargs = {
        "title": "Average persistence image",
        "xlabel": "End radial distance",
        "ylabel": "Start radial distance",
    }
    all_kwargs.update(kwargs)

    return av_imgs, cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def start_length_diagram(ph, new_fig=True, subplot=False, color="b", alpha=1.0, **kwargs):
    """Plot a transformed ph diagram that represents lengths and starting points of a component."""
    ph_transformed = transform_ph_to_length(ph)
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    for p in ph_transformed:
        ax.scatter(p[0], p[1], c=color, edgecolors="black", alpha=alpha)

    all_kwargs = {
        "title": "Transformed persistence diagram",
        "xlabel": "Start of the component",
        "ylabel": "Length of the component",
    }
    all_kwargs.update(kwargs)
    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def histogram_stepped(ph, new_fig=True, subplot=False, color="b", alpha=0.7, **kwargs):
    """Extract and plot the stepped histogram of a persistent homology array."""
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_stepped(ph)
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def histogram_stepped_population(
    ph_list, new_fig=True, subplot=False, color="b", alpha=0.7, **kwargs
):
    """Extract and plot the stepped histogram of a list of persistence diagrams.

    The histogram is normalized according to the number of persistence diagrams.
    """
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_stepped(analysis.collapse(ph_list))
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1] / len(ph_list), color=color, alpha=alpha)
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def histogram_horizontal(ph, new_fig=True, subplot=False, bins=100, color="b", alpha=0.7, **kwargs):
    """Extract and plot the binned histogram of a persistent homology array."""
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)
    hist_data = analysis.histogram_horizontal(ph, num_bins=bins)
    ax.fill_between(hist_data[0][:-1], 0, hist_data[1], color=color, alpha=alpha)

    return cm.plot_style(fig=fig, ax=ax, **kwargs)
