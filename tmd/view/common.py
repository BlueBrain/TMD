"""Module containing the common functionality to be used by view-plot modules."""

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

import os

import matplotlib.pyplot as plt
import numpy as np

jet_map = plt.cm.get_cmap("jet")
blues_map = plt.cm.get_cmap("Blues")
reds_map = plt.cm.get_cmap("Reds")


def figure_naming(pretitle=None, posttitle=None, prefile=None, postfile=None):
    """Helping function to define the strings that handle pre-post conventions.

    These strings can be used for viewing - plotting title and saving options.

    Args:
        pretitle : str
            String to include before the general title of the figure.
            Default is None.

        posttitle : str
            String to include after the general title of the figure.
            Default is None.

        prefile : str
            String to include before the general filename of the figure.
            Default is None.

        postfile : str
            String to include after the general filename of the figure.
            Default is None.

    Returns:
        string_output : string
            String to include in the figure name and title, in a suitable form.
    """
    if not pretitle:
        pretitle = ""
    else:
        pretitle = f"{pretitle} -- "

    if not posttitle:
        posttitle = ""
    else:
        posttitle = f" -- {posttitle}"

    if not prefile:
        prefile = ""
    else:
        prefile = f"{prefile}_"

    if not postfile:
        postfile = ""
    else:
        postfile = f"_{postfile}"

    return pretitle, posttitle, prefile, postfile


def get_color(treecolor, tree_type):
    """If treecolor is none returns the color depending on the type.

    The mapping is the following:
    * Basal dendrite: "red"
    * Axon : "blue"
    * Apical dendrite: "purple"
    * Soma tree: "black"
    * Undefined-Any other unrecognised type: "green"

    Otherwise return the treecolor.
    """
    if treecolor is None:
        if tree_type == "basal_dendrite":
            treecolor = "red"
        elif tree_type == "apical_dendrite":
            treecolor = "purple"
        elif tree_type == "axon":
            treecolor = "blue"
        elif tree_type == "soma":
            treecolor = "black"
        else:
            treecolor = "green"

    return treecolor


def get_figure(new_fig=True, new_axes=False, subplot=False, params=None, no_axes=False):
    """Function to be used for viewing - plotting, to initialize the matplotlib figure - axes.

    Args:
        new_fig : boolean
            Defines if a new figure will be created.
            If False the current figure is returned.
            Default is True.

        subplot : subplot or False
            Defines if a subplot will be generated.
            If False the subplot is the standard 111.
            Default is False.

        params: dict
            If dict not empty the parameters will
            be passed to the initialization of the axes.
            Default value is {}

        no_axes: boolean
            Defines the output of the function:
            If False the axes is returned.
            If True the figure is returned.
            Default value is False.

    Returns:
        figure or axes
            Returns a figure if no_axes is True,
            or axes if no_axes is False.
    """
    if new_fig:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    if no_axes:
        return fig

    if not subplot:
        subplot = 111

    if params is None:
        params = {}

    if new_axes:
        if isinstance(subplot, (tuple, list)):
            ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], **params)
        else:
            ax = fig.add_subplot(subplot, **params)
    else:
        ax = plt.gca()

    return fig, ax


def save_plot(fig, **kwargs):
    """Function to be used for viewing - plotting to save a matplotlib figure.

    Args:
        fig: matplotlib figure

    Keyword args:
        prefile : str
            String to include before the general filename of the figure.
            Default is None.

        postfile : str
            String to include after the general filename of the figure.
            Default is None.

        output_path : str
            String to define the path to the output directory.
            Default value is './'

        output_name : str
            String to define the name of the output figure.
            Default value is 'Figure'

        output_format : str
            String to define the format of the output figure.
            Default value is 'png'

        dpi: int
            Define the DPI (Dots per Inch) of the figure.
            Default value is 300.

        transparent: boolean
            If True the saved figure will have a transparent background.
            Default value is False.

    Returns:
        Generates a figure file in the selected directory.
    """
    prefile = kwargs.get("prefile", "")
    postfile = kwargs.get("postfile", "")
    output_path = kwargs.get("output_path", "./")
    output_name = kwargs.get("output_name", "Figure")
    output_format = kwargs.get("output_format", "png")
    dpi = kwargs.get("dpi", 300)
    transparent = kwargs.get("transparent", False)

    if not os.path.exists(output_path):
        os.makedirs(output_path)  # Make output directory if non-exsiting

    output = os.path.join(output_path, prefile + output_name + postfile + "." + output_format)

    plt.savefig(output, dpi=dpi, transparent=transparent)

    return fig


def plot_style(fig, ax, **kwargs):
    """Function to set the basic options of a matplotlib figure.

    Args:
        ax: matplotlib axes

    Keyword args:
        pretitle : str
            String to include before the general title of the figure.
            Default value is None.

        posttitle : str
            String to include after the general title of the figure.
            Default value is None.

        title : str
            The title for the figure
            Default value is "Figure".

        no_title : boolean
            Defines the presence of a title in the figure.
            If True the title will be set to an empty string "".
            Default value is False.

        title_fontsize : int
            Defines the size of the title's font.
            Default value is 14.

        title_arg : dict
            Defines the arguments that will be passed
            into matplotlib as title arguments.
            Default value is None.

        xlabel : str
            The xlabel for the figure
            Default value is "X".

        ylabel : str
            The xlabel for the figure
            Default value is "Y".

        no_labels : boolean
            Defines the presence of the labels in the figure.
            If True the labels will be set to an empty string "".
            Default value is False.

        no_xlabel : boolean
            Defines the presence of the xlabel in the figure.
            If True the xlabel will be set to an empty string "".
            Default value is False.

        no_ylabel : boolean
            Defines the presence of the ylabel in the figure.
            If True the ylabel will be set to an empty string "".
            Default value is False.

        label_fontsize : int
            Defines the size of the labels' font.
            Default value is 14.

        xlabel_arg : dict
            Defines the arguments that will be passed
            into matplotlib as xlabel arguments.
            Default value is None.

        ylabel_arg : dict
            Defines the arguments that will be passed
            into matplotlib as ylabel arguments.
            Default value is None.

        no_ticks : boolean
            Defines the presence of x-y ticks in the figure.
            If True the ticks will be empty.
            Default value is False.

        no_xticks : boolean
            Defines the presence of x ticks in the figure.
            If True the ticks will be empty.
            Default value is False.

        no_yticks : boolean
            Defines the presence of y ticks in the figure.
            If True the ticks will be empty.
            Default value is False.

        xticks : list of ticks
            Defines the values of x ticks in the figure.
            Default value is None.

        yticks : list of ticks
            Defines the values of y ticks in the figure.
            Default value is None.

        tick_fontsize : int
            Defines the size of the ticks' font.
            Default value is 12.

        xticks_arg : dict
            Defines the arguments that will be passed
            into matplotlib as xticks arguments.
            Default value is None.

        yticks_arg : dict
            Defines the arguments that will be passed
            into matplotlib as yticks arguments.
            Default value is None.

        no_limits : boolean
            Defines the presence of plot limits in the figure.
            Default value is False.

        no_xlim : boolean
            Defines the presence of plot x-limits in the figure.
            Default value is False.

        no_ylim : boolean
            Defines the presence of plot y-limits in the figure.
            Default value is False.

        xlim: list of two floats
            Defines the min and the max values in x-axis.
            Default in None

        ylim: list of two floats
            Defines the min and the max values in y-axis.
            Default in None

        no_legend : boolean
            Defines the presence of a legend in the figure.
            If True the legend will not be included in the Figure.
            Default value is True.

        legend_arg : dict
            Defines the arguments that will be passed
            into matplotlib as legend arguments.
            Default value is None.

        prefile : str
            String to include before the general filename of the figure.
            Default is None.

        postfile : str
            String to include after the general filename of the figure.
            Default is None.

        output_path : str
            String to define the path to the output directory.
            Default value is './'

        output_name : str
            String to define the name of the output figure.
            Default value is 'Figure'

        output_format : str
            String to define the format of the output figure.
            Default value is 'png'

        dpi: int
            Define the DPI (Dots per Inch) of the figure.
            Default value is 300.

        transparent: boolean
            If True the saved figure will have a transparent background.
            Default value is False.

        show_plot : boolean
            If True the figure is displayed.
            Default value is True.

        no_axes: boolean
            If True the labels and the frame will be set off.
            Default value is False.

        tight: boolean
            If True the set layout of matplotlib will be activated.
            Default value is False.

    Returns:
        figure_output : figure, axes
    """
    # Definition of title/file naming variables
    prefile = kwargs.get("prefile", "")
    postfile = kwargs.get("postfile", "")
    pretitle = kwargs.get("pretitle", "")
    posttitle = kwargs.get("posttitle", "")

    # Definition of global options
    no_axes = kwargs.get("no_axes", False)
    show_plot = kwargs.get("show_plot", True)
    tight = kwargs.get("tight", False)
    aspect = kwargs.get("aspect", "auto")

    # Definition of save options
    output_path = kwargs.get("output_path", None)

    pretitle, posttitle, prefile, postfile = figure_naming(pretitle, posttitle, prefile, postfile)

    fig, ax = plot_title(fig, ax, **kwargs)

    fig, ax = plot_labels(fig, ax, **kwargs)

    fig, ax = plot_ticks(fig, ax, **kwargs)

    fig, ax = plot_limits(fig, ax, **kwargs)

    fig, ax = plot_legend(fig, ax, **kwargs)

    if no_axes:
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if tight:
        fig.set_tight_layout(True)

    ax.set_aspect(aspect)

    if output_path is not None:
        fig = save_plot(fig=ax, **kwargs)

    if not show_plot:
        plt.close()
        return (None, None)

    return fig, ax


def plot_title(fig, ax, **kwargs):
    """Function that defines the title options of a matplotlib plot.

    Args:
        fig: matplotlib figure

        ax: matplotlib axes

    Keyword args:
        pretitle : str
            String to include before the general title of the figure.
            Default value is None.

        posttitle : str
            String to include after the general title of the figure.
            Default value is None.

        title : str
            Set the title for the figure.
            If "" no title will be added.
            Default value is "Figure".

        title_fontsize : int
            Defines the size of the title's font.
            Default value is 14.

        title_arg : dict
            Defines the arguments that will be passed
            into matplotlib as title arguments.
            Default value is None.

    Returns:
        figure_output : figure, axes
    """
    # Definition of title options
    pretitle = kwargs.get("pretitle", "")
    posttitle = kwargs.get("posttitle", "")
    title = kwargs.get("title", "Figure")
    title_fontsize = kwargs.get("titlefontsize", 14)
    title_arg = kwargs.get("titlearg", None)

    if title_arg is None:
        title_arg = {}

    ax.set_title(pretitle + title + posttitle, fontsize=title_fontsize, **title_arg)

    return fig, ax


def plot_labels(fig, ax, **kwargs):
    """Function that defines the labels options of a matplotlib plot.

    Args
        fig: matplotlib figure

        ax: matplotlib axes

    Keyword args:
        xlabel : str
            The xlabel for the figure.
            For no_xlabel set to ''.
            Default value is "X".

        ylabel : str
            The xlabel for the figure.
            For no_ylabel set to ''.
            Default value is "Y".

        zlabel : str
            The zlabel for the figure.
            For no_zlabel set to ''.
            Default value is "Z".

        label_fontsize : int
            Defines the size of the labels' font.
            Default value is 14.

        xlabel_arg : dict
            Defines the arguments that will be passed
            into matplotlib as xlabel arguments.
            Default value is None.

        ylabel_arg : dict
            Defines the arguments that will be passed
            into matplotlib as ylabel arguments.
            Default value is None.

        zlabel_arg : dict
            Defines the arguments that will be passed
            into matplotlib as zlabel arguments.
            Default value is None.

    Returns:
        figure_output : figure, axes
    """
    # Definition of label options
    xlabel = kwargs.get("xlabel", "X")
    ylabel = kwargs.get("ylabel", "Y")
    zlabel = kwargs.get("zlabel", "Z")
    label_fontsize = kwargs.get("labelfontsize", 14)
    xlabel_arg = kwargs.get("xlabel_arg", None)
    ylabel_arg = kwargs.get("ylabel_arg", None)
    zlabel_arg = kwargs.get("zlabel_arg", None)

    if xlabel_arg is None:
        xlabel_arg = {}

    if ylabel_arg is None:
        ylabel_arg = {}

    if zlabel_arg is None:
        zlabel_arg = {}

    ax.set_xlabel(xlabel, fontsize=label_fontsize, **xlabel_arg)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, **ylabel_arg)

    if hasattr(ax, "zaxis"):
        ax.set_zlabel(zlabel, fontsize=label_fontsize, **zlabel_arg)

    return fig, ax


def plot_ticks(fig, ax, **kwargs):
    """Function that defines the labels options of a matplotlib plot.

    Args
        fig: matplotlib figure

        ax: matplotlib axes

    Keyword args:
        xticks : list of ticks
            Defines the values of x ticks in the figure.
            If None the xticks will not be modified.
            For no_xticks set to [].
            Default value is None.

        yticks : list of ticks
            Defines the values of y ticks in the figure.
            If None the yticks will not be modified.
            For no_yticks set to [].
            Default value is None.

        zticks : list of ticks
            Defines the values of z ticks in the figure.
            If None the zticks will not be modified.
            For no_zticks set to [].
            Default value is None.

        tick_fontsize : int
            Defines the size of the ticks' font.
            Default value is 12.

        xticks_arg : dict
            Defines the arguments that will be passed
            into matplotlib as xticks arguments.
            Default value is None.

        yticks_arg : dict
            Defines the arguments that will be passed
            into matplotlib as yticks arguments.
            Default value is None.

        zticks_arg : dict
            Defines the arguments that will be passed
            into matplotlib as zticks arguments.
            Default value is None.

    Returns:
        figure_output : figure, axes
    """
    # Definition of tick options
    xticks = kwargs.get("xticks", None)
    yticks = kwargs.get("yticks", None)
    zticks = kwargs.get("zticks", None)
    tick_fontsize = kwargs.get("tickfontsize", 12)
    xticks_arg = kwargs.get("xticksarg", None)
    yticks_arg = kwargs.get("yticksarg", None)
    zticks_arg = kwargs.get("zticksarg", None)

    if xticks_arg is None:
        xticks_arg = {}

    if yticks_arg is None:
        yticks_arg = {}

    if zticks_arg is None:
        zticks_arg = {}

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.xaxis.set_tick_params(labelsize=tick_fontsize, **xticks_arg)

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_tick_params(labelsize=tick_fontsize, **yticks_arg)

    if zticks is not None:
        ax.set_zticks(zticks)
        ax.zaxis.set_tick_params(labelsize=tick_fontsize, **zticks_arg)

    return fig, ax


def plot_limits(fig, ax, **kwargs):
    """Function that defines the limit options of a matplotlib plot.

    Args:
        fig: matplotlib figure

        ax: matplotlib axes

    Keyword args:
        no_limits : boolean
            Defines the presence of plot limits in the figure.
            Default value is False.

        no_xlim : boolean
            Defines the presence of plot x-limits in the figure.
            Default value is False.

        no_ylim : boolean
            Defines the presence of plot y-limits in the figure.
            Default value is False.

        xlim: list of two floats
            Defines the min and the max values in x-axis.
            Default in None

        ylim: list of two floats
            Defines the min and the max values in y-axis.
            Default in None

    Returns:
        figure_output : figure, axes
    """
    # Definition of limit options
    no_xlim = kwargs.get("no_xlim", False)
    no_ylim = kwargs.get("no_ylim", False)
    no_zlim = kwargs.get("no_zlim", False)
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    zlim = kwargs.get("zlim", None)

    if not no_xlim:
        ax.set_xlim(xlim)
    if not no_ylim:
        ax.set_ylim(ylim)
    if hasattr(ax, "zaxis") and not no_zlim:
        ax.set_zlim(zlim)

    return fig, ax


def plot_legend(fig, ax, **kwargs):
    """Function that defines the legend options of a matplotlib plot.

    Args:
        fig: matplotlib figure

        ax: matplotlib axes

    Keyword args:
        no_legend : boolean
            Defines the presence of a legend in the figure.
            If True the legend will not be included in the Figure.
            Default value is True.

        legend_arg : dict
            Defines the arguments that will be passed
            into matplotlib as legend arguments.
            Default value is None.

    Returns:
        figure_output : figure, axes
    """
    # Definition of legend options
    no_legend = kwargs.get("no_legend", True)
    legend_arg = kwargs.get("legendarg", None)

    if legend_arg is None:
        legend_arg = {}

    if not no_legend:
        ax.legend(**legend_arg)

    return fig, ax


def plot_sphere(fig, ax, center, radius, color="black", alpha=1.0):
    """Plots a 3d sphere, given the center and the radius."""
    u = np.linspace(0, 2 * np.pi, 300)
    v = np.linspace(0, np.pi, 300)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, linewidth=0.0, color=color, alpha=alpha)

    return fig, ax


def plot_img_basic(
    img,
    new_fig=True,
    subplot=111,
    title="",
    xlim=None,
    ylim=None,
    colorbar=False,
    cmap=jet_map,
    vmin=None,
    vmax=None,
    masked=False,
    threshold=0.01,
    **kwargs,
):
    """Plots the gaussian kernel of the input image."""
    if xlim is None:
        xlim = (0, 100)
    if ylim is None:
        ylim = (0, 100)

    if vmin is None:
        vmin = np.min(img)
    if vmax is None:
        vmax = np.max(img)

    fig, ax = get_figure(new_fig=new_fig, subplot=subplot)

    if masked:
        img = np.ma.masked_where((threshold > np.abs(img)), img)

    cax = ax.imshow(
        np.rot90(img),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        extent=xlim + ylim,
    )

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim
    kwargs["title"] = title

    if colorbar:
        plt.colorbar(cax)
    ax.set_aspect("equal")

    return plot_style(fig=fig, ax=ax, aspect="equal", **kwargs)
