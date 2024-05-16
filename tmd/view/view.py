"""Module for viewing neuronal morphologies."""

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

# pylint: disable=too-many-lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection as _LC

from tmd.Topology.methods import _filtration_function
from tmd.Topology.methods import tree_to_property_barcode as tp_barcode
from tmd.utils import TREE_TYPE_DICT
from tmd.utils import term_dict
from tmd.view import common as cm
from tmd.view import plot
from tmd.view.common import blues_map


def _get_default(variable, **kwargs):
    """Returns default variable or kwargs variable if it exists."""
    default = {
        "linewidth": 1.2,
        "alpha": 0.8,
        "treecolor": None,
        "diameter": True,
        "diameter_scale": 1.0,
        "white_space": 30.0,
    }

    return kwargs.get(variable, default[variable])


def trunk(tr, plane="xy", new_fig=True, subplot=False, hadd=0.0, vadd=0.0, N=10, **kwargs):
    """Generate a 2d figure of the trunk = first N segments of the tree.

    Args:
        tr (Tree): A Tree object.
        plane (str): The plane to consider.
        new_fig (bool): Create a new figure if set to `True`.
        subplot (bool): Create a subplot if set to `True`.
        hadd (float): X shift.
        vadd (float): Y shift.
        N (int): Number of segments.
    """
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    # bounding_box = tr.get_bounding_box()

    def _seg_2d(seg, x_add=0.0, y_add=0.0):
        """2d coordinates required for the plotting of a segment."""
        horz = term_dict[plane[0]]
        vert = term_dict[plane[1]]

        horz1 = seg[0][horz] + x_add
        horz2 = seg[1][horz] + x_add
        vert1 = seg[0][vert] + y_add
        vert2 = seg[1][vert] + y_add

        return ((horz1, vert1), (horz2, vert2))

    N = min(N, len(tr.get_segments()))

    segs = [_seg_2d(seg, hadd, vadd) for seg in tr.get_segments()[:N]]

    linewidth = _get_default("linewidth", **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default("diameter", **kwargs):
        scale = _get_default("diameter_scale", **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = cm.get_color(_get_default("treecolor", **kwargs), TREE_TYPE_DICT[tr.get_type()])

    # Plot the collection of lines.
    collection = _LC(
        segs, color=treecolor, linewidth=linewidth, alpha=_get_default("alpha", **kwargs)
    )

    ax.add_collection(collection)

    all_kwargs = {
        "title": "Tree view",
        "xlabel": plane[0],
        "ylabel": plane[1],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def tree(tr, plane="xy", new_fig=True, subplot=False, hadd=0.0, vadd=0.0, **kwargs):
    """Generate a 2d figure of the tree.

    Args:
        tr (Tree): A Tree object.
        plane (str): The plane to consider.
        new_fig (bool): Create a new figure if set to `True`.
        subplot (bool): Create a subplot if set to `True`.
        hadd (float): X shift.
        vadd (float): Y shift.
    """
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_2d(seg, x_add=0.0, y_add=0.0):
        """2d coordinates required for the plotting of a segment."""
        horz = term_dict[plane[0]]
        vert = term_dict[plane[1]]

        horz1 = seg[0][horz] + x_add
        horz2 = seg[1][horz] + x_add
        vert1 = seg[0][vert] + y_add
        vert2 = seg[1][vert] + y_add

        return ((horz1, vert1), (horz2, vert2))

    segs = [_seg_2d(seg, hadd, vadd) for seg in tr.get_segments()]

    linewidth = _get_default("linewidth", **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default("diameter", **kwargs):
        scale = _get_default("diameter_scale", **kwargs)
        linewidth = [d * scale for d in tr.d]

    if tr.get_type() not in TREE_TYPE_DICT:
        treecolor = "black"
    else:
        treecolor = cm.get_color(_get_default("treecolor", **kwargs), TREE_TYPE_DICT[tr.get_type()])

    # Plot the collection of lines.
    collection = _LC(
        segs, color=treecolor, linewidth=linewidth, alpha=_get_default("alpha", **kwargs)
    )

    ax.add_collection(collection)

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": "Tree view",
        "xlabel": plane[0],
        "ylabel": plane[1],
        "xlim": [
            bounding_box[0][term_dict[plane[0]]] - white_space,
            bounding_box[1][term_dict[plane[0]]] + white_space,
        ],
        "ylim": [
            bounding_box[0][term_dict[plane[1]]] - white_space,
            bounding_box[1][term_dict[plane[1]]] + white_space,
        ],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def soma(sm, plane="xy", new_fig=True, subplot=False, hadd=0.0, vadd=0.0, **kwargs):
    """Generate a 2d figure of the soma.

    Args:
        sm (Soma): A Soma object.
        plane (str): The plane to consider.
        new_fig (bool): Create a new figure if set to `True`.
        subplot (bool): Create a subplot if set to `True`.
        hadd (float): X shift.
        vadd (float): Y shift.
    """
    treecolor = kwargs.get("treecolor", None)
    outline = kwargs.get("outline", True)

    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Definition of the tree color depending on the tree type.
    treecolor = cm.get_color(treecolor, tree_type="soma")

    # Plot the outline of the soma as a circle, if outline is selected.
    if not outline:
        soma_circle = plt.Circle(
            sm.get_center() + [hadd, vadd, 0.0],
            sm.get_diameter() / 2.0,
            color=treecolor,
            alpha=_get_default("alpha", **kwargs),
        )
        ax.add_artist(soma_circle)
    else:
        horz = getattr(sm, plane[0]) + hadd
        vert = getattr(sm, plane[1]) + vadd

        horz = np.append(horz, horz[0])  # To close the loop for a soma
        vert = np.append(vert, vert[0])  # To close the loop for a soma
        plt.plot(
            horz,
            vert,
            color=treecolor,
            alpha=_get_default("alpha", **kwargs),
            linewidth=_get_default("linewidth", **kwargs),
        )

    all_kwargs = {
        "title": "Soma view",
        "xlabel": plane[0],
        "ylabel": plane[1],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def neuron(
    nrn,
    plane="xy",
    new_fig=True,
    subplot=False,
    hadd=0.0,
    vadd=0.0,
    neurite_type="all",
    apical_alignment=False,
    plot_soma=True,
    new_axes=True,
    **kwargs,
):
    """Generate a 2d figure of the neuron that contains a soma and a list of trees.

    Args:
        nrn (Neuron): A neuron object.

        plane (str):
            Accepted values: Any sorted pair of xyz.
            Default value is 'xy'.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).
            Default value is True.

        subplot (bool):
            Create a subplot if set to `True`.

        hadd (float):
            X shift.
            Default value is 0.

        vadd (float):
            Y shift.
            Default value is 0.

        neurite_type (str):
            The types of neurites that should be plotted.
            Default value is 'all'.

        apical_alignment (bool):
            Defines if the neuron will be automatically aligned toward the pia.
            Default value is False.

        plot_soma (bool):
            Defines if the soma will be plotted.
            Default value is True.

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).
            Default value is True.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.view.soma`,
            :func:`tmd.view.view.tree` and :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view, at the selected plane.
    """
    # pylint: disable=too-many-locals
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot, new_axes=new_axes)

    kwargs["new_fig"] = False
    kwargs["subplot"] = subplot

    if plot_soma:
        soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    h = []
    v = []

    to_plot = []

    if apical_alignment:
        if not nrn.apical_dendrite:
            raise ValueError(
                "The 'apical_alignment' is set to True but the neuron has no apical dendrite."
            )
        angle = np.arctan2(nrn.apical_dendrite[0].get_pca()[0], nrn.apical_dendrite[0].get_pca()[1])
    else:
        angle = None

    if neurite_type == "all":
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:
        if angle is not None:
            temp_tree.rotate_xy(angle)

        bounding_box = temp_tree.get_bounding_box()

        h.append([bounding_box[0][term_dict[plane[0]]], bounding_box[1][term_dict[plane[0]]]])
        v.append([bounding_box[0][term_dict[plane[1]]], bounding_box[1][term_dict[plane[1]]]])

        tree(temp_tree, hadd=hadd, vadd=vadd, **kwargs)

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": nrn.name,
        "xlabel": plane[0],
        "ylabel": plane[1],
        "xlim": [np.min(h) - white_space + hadd, np.max(h) + white_space + hadd],
        "ylim": [np.min(v) - white_space + vadd, np.max(v) + white_space + vadd],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def all_trunks(
    nrn,
    plane="xy",
    new_fig=True,
    subplot=False,
    hadd=0.0,
    vadd=0.0,
    neurite_type="all",
    N=10,
    **kwargs,
):
    """Generate a 2d figure of the neuron, that contains a soma and a list of trees.

    Args:
        nrn (Neuron): A Neuron object.

        plane (str):
            Accepted values: Any sorted pair of xyz.
            Default value is 'xy'.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).
            Default value is True.

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

        hadd (float):
            X shift.

        vadd (float):
            Y shift.

        neurite_type (str):
            The types of neurites that should be plotted.

        N (float):
            Half of the window size used if xlim and ylim are not given.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view, at the selected plane.
    """
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs["new_fig"] = False
    kwargs["subplot"] = subplot

    soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    to_plot = []

    if neurite_type == "all":
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:
        trunk(temp_tree, N=N, **kwargs)

    all_kwargs = {
        "title": nrn.name,
        "xlabel": plane[0],
        "ylabel": plane[1],
        "xlim": [nrn.soma.get_center()[0] - 2.0 * N, nrn.soma.get_center()[0] + 2.0 * N],
        "ylim": [nrn.soma.get_center()[1] - 2.0 * N, nrn.soma.get_center()[1] + 2.0 * N],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def population(
    pop, plane="xy", new_fig=True, subplot=False, hadd=0.0, vadd=0.0, neurite_type="all", **kwargs
):
    """Generate a 2d figure of the population, that contains a soma and a list of trees.

    Args:
        pop (Population):
            A Population object.

        plane (str):
            Accepted values: Any sorted pair of xyz.
            Default value is 'xy'.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

        hadd (float):
            X shift.

        vadd (float):
            Y shift.

        neurite_type (str):
            The types of neurites that should be plotted.

        N (float):
            Half of the window size used if xlim and ylim are not given.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`soma`, :func:`tree` and
            :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view, at the selected plane.
    """
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs["new_fig"] = False
    kwargs["subplot"] = subplot

    h = []
    v = []

    for nrn in pop.neurons:
        soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

        if neurite_type == "all":
            neurite_list = ["basal_dendrite", "apical_dendrite", "axon"]
        else:
            neurite_list = [neurite_type]

        for nt in neurite_list:
            for temp_tree in getattr(nrn, nt):
                bounding_box = temp_tree.get_bounding_box()

                h.append(
                    [
                        bounding_box[0][term_dict[plane[0]]] + hadd,
                        bounding_box[1][term_dict[plane[1]]] + vadd,
                    ]
                )
                v.append(
                    [
                        bounding_box[0][term_dict[plane[0]]] + hadd,
                        bounding_box[1][term_dict[plane[1]]] + vadd,
                    ]
                )

                tree(temp_tree, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    all_kwargs = {
        "title": "Neuron view",
        "xlabel": plane[0],
        "ylabel": plane[1],
        "xlim": [np.min(h), np.max(h)],
        "ylim": [np.min(v), np.max(v)],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def tree3d(tr, new_fig=True, new_axes=True, subplot=False, **kwargs):
    """Generate a figure of the tree in 3d.

    Args:
        tr (Tree):
            A Tree object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

    Keyword Args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view.
    """
    # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_3d(seg):
        """3d coordinates needed for the plotting of a segment."""
        horz = term_dict["x"]
        vert = term_dict["y"]
        depth = term_dict["z"]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]
        depth1 = seg[0][depth]
        depth2 = seg[1][depth]

        return ((horz1, vert1, depth1), (horz2, vert2, depth2))

    segs = [_seg_3d(seg) for seg in tr.get_segments()]

    linewidth = _get_default("linewidth", **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default("diameter", **kwargs):
        scale = _get_default("diameter_scale", **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = cm.get_color(_get_default("treecolor", **kwargs), TREE_TYPE_DICT[tr.get_type()])

    # Plot the collection of lines.
    collection = Line3DCollection(
        segs, color=treecolor, linewidth=linewidth, alpha=_get_default("alpha", **kwargs)
    )

    ax.add_collection3d(collection)

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": "Tree 3d-view",
        "xlabel": "X",
        "ylabel": "Y",
        "zlabel": "Z",
        "xlim": [
            bounding_box[0][term_dict["x"]] - white_space,
            bounding_box[1][term_dict["x"]] + white_space,
        ],
        "ylim": [
            bounding_box[0][term_dict["y"]] - white_space,
            bounding_box[1][term_dict["y"]] + white_space,
        ],
        "zlim": [
            bounding_box[0][term_dict["z"]] - white_space,
            bounding_box[1][term_dict["z"]] + white_space,
        ],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def trunk3d(tr, new_fig=True, new_axes=True, subplot=False, N=10, **kwargs):
    """Generate a figure of the trunk in 3d.

    Args:
        tr (Tree):
            A Tree object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

        N (int):
            The number of segments to plot.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view.
    """
    # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    def _seg_3d(seg):
        """3d coordinates needed for the plotting of a segment."""
        horz = term_dict["x"]
        vert = term_dict["y"]
        depth = term_dict["z"]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]
        depth1 = seg[0][depth]
        depth2 = seg[1][depth]

        return ((horz1, vert1, depth1), (horz2, vert2, depth2))

    N = min(N, len(tr.get_segments()))

    segs = [_seg_3d(seg) for seg in tr.get_segments()[:N]]

    linewidth = _get_default("linewidth", **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default("diameter", **kwargs):
        scale = _get_default("diameter_scale", **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = cm.get_color(_get_default("treecolor", **kwargs), TREE_TYPE_DICT[tr.get_type()])

    # Plot the collection of lines.
    collection = Line3DCollection(
        segs, color=treecolor, linewidth=linewidth, alpha=_get_default("alpha", **kwargs)
    )

    ax.add_collection3d(collection)

    all_kwargs = {
        "title": "Tree 3d-view",
        "xlabel": "X",
        "ylabel": "Y",
        "zlabel": "Z",
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def soma3d(sm, new_fig=True, new_axes=True, subplot=False, **kwargs):
    """Generate a 3d figure of the soma.

    Args:
        soma (Soma):
            A Soma object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view.
    """
    treecolor = kwargs.get("treecolor", None)

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    # Definition of the tree color depending on the tree type.
    treecolor = cm.get_color(treecolor, tree_type="soma")

    center = sm.get_center()

    xs = center[0]
    ys = center[1]
    zs = center[2]

    # Plot the soma as a circle.
    fig, ax = cm.plot_sphere(
        fig,
        ax,
        center=[xs, ys, zs],
        radius=sm.get_diameter(),
        color=treecolor,
        alpha=_get_default("alpha", **kwargs),
    )

    all_kwargs = {
        "title": "Soma view",
        "xlabel": "X",
        "ylabel": "Y",
        "zlabel": "Z",
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def neuron3d(nrn, new_fig=True, new_axes=True, subplot=False, neurite_type="all", **kwargs):
    """Generate a figure of the neuron, that contains a soma and a list of trees.

    Args:
        neuron (Neuron):
            A Neuron object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

        neurite_type (str):
            The types of neurites that should be plotted.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    kwargs["new_fig"] = False
    kwargs["new_axes"] = False
    kwargs["subplot"] = subplot

    soma3d(nrn.soma, **kwargs)

    h = []
    v = []
    d = []

    to_plot = []

    if neurite_type == "all":
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:
        bounding_box = temp_tree.get_bounding_box()

        h.append([bounding_box[0][term_dict["x"]], bounding_box[1][term_dict["x"]]])
        v.append([bounding_box[0][term_dict["y"]], bounding_box[1][term_dict["y"]]])
        d.append([bounding_box[0][term_dict["z"]], bounding_box[1][term_dict["z"]]])

        tree3d(temp_tree, **kwargs)

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": nrn.name,
        "xlim": [np.min(h) - white_space, np.max(h) + white_space],
        "ylim": [np.min(v) - white_space, np.max(v) + white_space],
        "zlim": [np.min(d) - white_space, np.max(d) + white_space],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def all_trunks3d(
    nrn, new_fig=True, new_axes=True, subplot=False, neurite_type="all", N=10, **kwargs
):
    """Generate a figure of the neuron, that contains a soma and a list of trees.

    Args:
        neuron (Neuron):
            A Neuron object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

        neurite_type (str):
            The types of neurites that should be plotted.

        N (float):
            Half of the window size used if xlim and ylim are not given.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a tree view.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    kwargs["new_fig"] = False
    kwargs["new_axes"] = False
    kwargs["subplot"] = subplot

    soma3d(nrn.soma, **kwargs)

    to_plot = []

    if neurite_type == "all":
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:
        trunk3d(temp_tree, N=N, **kwargs)

    all_kwargs = {
        "title": nrn.name,
        "xlim": [nrn.soma.get_center()[0] - 2.0 * N, nrn.soma.get_center()[0] + 2.0 * N],
        "ylim": [nrn.soma.get_center()[1] - 2.0 * N, nrn.soma.get_center()[1] + 2.0 * N],
        "zlim": [nrn.soma.get_center()[2] - 2.0 * N, nrn.soma.get_center()[2] + 2.0 * N],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def population3d(pop, new_fig=True, new_axes=True, subplot=False, **kwargs):
    """Generate a figure of the population, that contains the pop somata and a set of list of trees.

    Args:
        pop (Population):
            A Population object.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 3D matplotlib figure with a population view.
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(
        new_fig=new_fig, new_axes=new_axes, subplot=subplot, params={"projection": "3d"}
    )

    kwargs["new_fig"] = False
    kwargs["new_axes"] = False
    kwargs["subplot"] = subplot

    h = []
    v = []
    d = []

    for nrn in pop.neurons:
        # soma3d(nrn.soma, **kwargs)
        for temp_tree in nrn.neurites:
            bounding_box = temp_tree.get_bounding_box()
            h.append([bounding_box[0][term_dict["x"]], bounding_box[1][term_dict["x"]]])
            v.append([bounding_box[0][term_dict["y"]], bounding_box[1][term_dict["y"]]])
            d.append([bounding_box[0][term_dict["z"]], bounding_box[1][term_dict["z"]]])
            tree3d(temp_tree, **kwargs)

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": "",
        "xlim": [np.min(h) - white_space, np.max(h) + white_space],
        "ylim": [np.min(v) - white_space, np.max(v) + white_space],
        "zlim": [np.min(d) - white_space, np.max(d) + white_space],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


# pylint: disable=too-many-locals
def density_cloud(
    obj,
    new_fig=True,
    subplot=111,
    new_axes=True,
    neurite_type="all",
    bins=100,
    plane="xy",
    color_map=blues_map,
    alpha=0.8,
    centered=True,
    colorbar=True,
    plot_neuron=False,
    **kwargs,
):
    """View the neuron morphologies of a population as a density cloud."""
    x1 = []
    y1 = []

    if neurite_type == "all":
        ntypes = "neurites"
    else:
        ntypes = neurite_type

    fig, ax = cm.get_figure(new_fig=new_fig, new_axes=new_axes, subplot=subplot)

    for neu in obj.neurons:
        for tr in getattr(neu, ntypes):
            if centered:
                dx = neu.soma.get_center()[0]
                dy = neu.soma.get_center()[1]
            else:
                dx = 0
                dy = 0
            x1 = x1 + list(getattr(tr, plane[0]) - dx)
            y1 = y1 + list(getattr(tr, plane[1]) - dy)

    H1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=(bins, bins))
    mask = H1 < 0.05
    H2 = np.ma.masked_array(H1, mask)
    color_map.set_bad(color="white", alpha=None)

    plots = ax.contourf(
        (xedges1[:-1] + xedges1[1:]) / 2,
        (yedges1[:-1] + yedges1[1:]) / 2,
        np.transpose(H2),
        cmap=color_map,
        alpha=alpha,
    )

    kwargs["new_fig"] = False
    kwargs["subplot"] = subplot

    if plot_neuron:
        nrn = obj.neurons[0]
        h, v, _ = nrn.soma.get_center()
        soma(nrn.soma, plane="xy", hadd=-h, vadd=-v, **kwargs)
        for temp_tree in getattr(nrn, ntypes):
            tree(temp_tree, plane="xy", hadd=-h, vadd=-v, treecolor="r", **kwargs)

    if colorbar:
        plt.colorbar(plots)
    # soma(neu.soma, new_fig=False)
    return cm.plot_style(fig=fig, ax=ax, **kwargs)


def _get_polar_data(pop, neurite_type="neurites", bins=20):
    """Extract the data to plot the polar length distribution of a population of neurons."""

    def seg_angle(seg):
        """Angle between mean x, y coordinates of a seg."""
        mean_x = np.mean([seg[0][0], seg[1][0]])
        mean_y = np.mean([seg[0][1], seg[1][1]])
        return np.arctan2(mean_y, mean_x)

    def seg_length(seg):
        """Compute the length of a seg."""
        return np.linalg.norm(np.subtract(seg[1], seg[0]))

    segs = []
    for tr in getattr(pop, neurite_type):
        segs = segs + tr.get_segments()

    angles = np.array([seg_angle(s) for s in segs])
    lens = np.array([seg_length(s) for s in segs])
    ranges = [
        [i * 2 * np.pi / bins - np.pi, (i + 1) * 2 * np.pi / bins - np.pi] for i in range(bins)
    ]
    results = [r + [np.sum(lens[np.where((angles > r[0]) & (angles < r[1]))[0]])] for r in ranges]

    return results


def polar_plot(pop, neurite_type="neurites", bins=20):
    """Generate a polar plot of a neuron or population."""
    input_data = _get_polar_data(pop, neurite_type=neurite_type, bins=bins)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    theta = np.array(input_data)[:, 0]
    norm_radii = np.array(input_data)[:, 2] / np.max(input_data)
    width = 2 * np.pi / len(input_data)
    ax.bar(theta, norm_radii, width=width, bottom=0.0, alpha=0.8)


def _tree_colors(
    tr,
    ph_graph,
    colors=None,
    new_fig=True,
    subplot=111,
    new_axes=True,
    plane="xy",
    cmap=plt.jet,
    **kwargs,
):
    """Generate a 2d pic of the tree, each branch has a unique color.

    Args:
        tr (Tree):
            A Tree object.

        plane (str):
            Accepted values: Any sorted pair of xyz:
            (xy, xz, yx, yz, zx, zy)
            Default value is 'xy'.

        ph_graph (list): The list of bars corresponding to tree branches
            as extracted as the second output
            from tree_to_property_barcode function.

        colors (list of matplotlib colors): If None,
            a list of colors will be generated.

        new_fig (bool):
            Defines if the neuron will be plotted
            in the current figure (False)
            or in a new figure (True).

        new_axes (bool):
            Defines if the neuron will be plotted
            in the current axes (False)
            or in new axes (True).

        subplot (matplotlib subplot value or False):
            If False the default subplot 111 will be used.
            For any other value a matplotlib subplot
            will be generated.
            Default value is False.

    Keyword args:
        **kwargs:
            All keyword arguments will be passed to :func:`tmd.view.common.plot_style`.

    Returns:
        A 2D matplotlib figure with a tree view.
    """
    if plane not in ("xy", "yx", "xz", "zx", "yz", "zy"):
        raise ValueError("No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.")

    # Initialization of matplotlib figure and axes.
    fig, ax = cm.get_figure(new_fig=new_fig, new_axes=new_axes, subplot=subplot)

    # Initialization of colors to be used.
    if colors is None:
        ordered_nums = np.arange(len(ph_graph)) / len(ph_graph)
        colors_random = cmap(ordered_nums)
    else:
        colors_random = colors

    # Definition of tree branch colors based on persistence
    def get_colors_ph(tr, ph_graph, colors):
        """Assigns colors to each tree branch according to the persistence levels."""
        beg, end = tr.get_sections_2()
        end_graph = {}
        for j, graph_id in enumerate(ph_graph):
            for gid in graph_id:
                end_id = np.where(end == gid)[0][0]
                end_graph[end_id] = j

        sec_ids = [1] * (end[0] - beg[0])
        for i, e in enumerate(end[1:]):
            sec_ids += [i + 2] * (e - end[i])

        colors_select = [colors[end_graph[s - 1]] for s in sec_ids]
        return colors_select

    treecolors = get_colors_ph(tr, ph_graph, colors_random)

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = term_dict[plane[0]]
        vert = term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    segs = [_seg_2d(seg) for seg in tr.get_segments()]

    # Definition of the linewidth according to diameter, if diameter is True.

    linewidth = list(tr.d)

    # Plot the collection of lines.
    collection = _LC(segs, color=treecolors, linewidth=linewidth, alpha=1.0)

    ax.add_collection(collection)

    ax.set_xlim(bounding_box[0][term_dict[plane[0]]], bounding_box[1][term_dict[plane[0]]])
    ax.set_ylim(bounding_box[0][term_dict[plane[1]]], bounding_box[1][term_dict[plane[1]]])

    white_space = _get_default("white_space", **kwargs)
    all_kwargs = {
        "title": "Tree structure",
        "xlabel": plane[0],
        "ylabel": plane[1],
        "xlim": [
            bounding_box[0][term_dict[plane[0]]] - white_space,
            bounding_box[1][term_dict[plane[0]]] + white_space,
        ],
        "ylim": [
            bounding_box[0][term_dict[plane[1]]] - white_space,
            bounding_box[1][term_dict[plane[1]]] + white_space,
        ],
    }
    all_kwargs.update(kwargs)

    return cm.plot_style(fig=fig, ax=ax, **all_kwargs)


def tree_barcode_colors(tr, plane="xy", feature="path_distances", cmap=cm.jet_map):
    """Generates a two panel figure with color-coded branches, bars.

    Generates a 2d pic of the tree, each branch has a unique color.
    A persistence barcode with the same colors at each bar.

    Args:
        tr (Tree):
            A Tree object.

        plane (str):
            Accepted values: Any sorted pair of xyz:
            (xy, xz, yx, yz, zx, zy)
            Default value is 'xy'.

        feature (str):
            Accepted values: path_distances, radial_distances.
            Default value is 'path_distances'.

        cmap (matplotlib colormap):
            Default value is jet.

    Returns:
        2d matplotlib figure, axes.
    """
    # Extract ph and ph_graph
    ph, ph_graph = tp_barcode(tr, filtration_function=_filtration_function(feature))
    colors_random = [cmap(i / len(ph)) for i in np.arange(len(ph))]

    fig, ax = _tree_colors(
        tr, ph_graph, colors=colors_random, plane=plane, new_fig=True, subplot=211
    )

    fig.add_subplot(212)
    plot.barcode(ph, color=colors_random, new_fig=False, subplot=212)

    return fig, ax


def tree_full_persistence_colors(tr, plane="xy", feature="path_distances", cmap=cm.jet_map):
    """Generates a four panel figure with color-coded branches, bars.

    Generates a 2d pic of the tree, each branch has a unique color.
    A persistence barcode with the same colors at each bar,
    A persistence diagram with the same colors,
    A persistence image with the same colormap.

    Args:
        tr (Tree):
            A Tree object.

        plane (str):
            Accepted values: Any sorted pair of xyz:
            (xy, xz, yx, yz, zx, zy)
            Default value is 'xy'.

        feature (str):
            Accepted values: path_distances, radial_distances.
            Default value is 'path_distances'.

        cmap (matplotlib colormap):
            Default value is jet.

    Returns:
        2d matplotlib figure, axes.
    """
    # Extract ph and ph_graph
    ph, ph_graph = tp_barcode(tr, filtration_function=_filtration_function(feature))
    colors_random = [cmap(i / len(ph)) for i in np.arange(len(ph))]

    fig, _ = _tree_colors(
        tr, ph_graph, colors=colors_random, plane=plane, new_fig=True, subplot=221
    )

    fig.add_subplot(222)
    plot.barcode(ph, color=colors_random, new_fig=False, subplot=222)

    bounds_max = np.max(ph)

    fig.add_subplot(223)
    plot.diagram(ph, color=colors_random, new_fig=False, subplot=223)

    ax = fig.add_subplot(224)
    plot.persistence_image(
        ph,
        cmap=cmap,
        new_fig=False,
        subplot=224,
        xlim=(-10, bounds_max),
        ylim=(-10, bounds_max),
    )

    return fig, ax
