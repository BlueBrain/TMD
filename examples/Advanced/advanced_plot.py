"""Example script of plotting functions of tmd (untested and more complex plotting)."""
import common as _cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as _np
import view as _view
from scipy import stats

from tmd import Topology as _tm
from tmd import utils as _utils
from tmd import view
from tmd.analysis import sort_ph
from tmd.view.plot import barcode
from tmd.view.plot import diagram
from tmd.view.plot import persistence_image


def _sort_ph_radii(ph):
    """Sorts barcode according to length of bars."""
    ph_sort = [[p[0], p[1], p[2], _np.abs(p[0] - p[1])] for p in ph]
    ph_sort.sort(key=lambda x: x[3])
    return ph_sort


def _sort_ph_initial(ph):
    """Sorts barcode according to length of bars."""
    ph_sort = [[p[0], p[1], p[2], _np.abs(p[0] - p[1])] for p in ph]
    ph_sort.sort(key=lambda x: x[2])
    return [p[:2] for p in ph_sort]


def _mirror_ph(ph):
    """Sorts barcode according to length of bars."""
    ph_mirror = [[p[0], p[1], p[0] - p[1]] for p in ph]
    ph_mirror.sort(key=lambda x: x[2])
    return ph_mirror


def tree_all(
    tree,
    plane="xy",
    feature="radial_distances",
    title="",
    diameter=True,
    treecol="b",
    xlim=None,
    ylim=None,
    **kwargs,
):
    """Subplot with ph, barcode and tree."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.tree(
        tree,
        new_fig=True,
        subplot=221,
        plane="xy",
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    # feat = getattr(tree, "get_section_" + feature)()
    # segs = tree.get_segments()

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    # bounds = max(max(ph))
    fig1, ax2 = diagram(ph, new_fig=False, subplot=222, color=treecol)
    fig1, ax3 = barcode(ph, new_fig=False, subplot=223, color=treecol)
    fig1, ax4 = persistence_image(ph, new_fig=False, subplot=224, xlim=xlim, ylim=ylim)
    _cm.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _cm.save_plot(fig=ax1, **kwargs)

    return fig1, ax1, ax2, ax3, ax4


def neu_all(
    neuron,
    plane="xy",
    feature="radial_distances",
    title="",
    diameter=True,
    treecol="b",
    xlim=None,
    ylim=None,
    neurite_type="basal_dendrite",
    **kwargs,
):
    """Subplot with ph, barcode and tree within spheres."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.neuron(
        neuron,
        new_fig=True,
        subplot=221,
        plane="xy",
        neurite_type=[neurite_type],
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_ph_neuron(neuron, feature=feature, neurite_type=neurite_type)
    else:
        raise Exception("Plane value not recognised")

    # bounds = max(max(ph))
    fig1, ax2 = diagram(ph, new_fig=False, subplot=222, color=treecol)
    fig1, ax3 = barcode(ph, new_fig=False, subplot=223, color=treecol)
    fig1, ax4 = persistence_image(ph, new_fig=False, subplot=224, xlim=xlim, ylim=ylim)
    _cm.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _cm.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def barcode_radii(ph, new_fig=True, subplot=False, linewidth=1.2, diam_max=2.0, **kwargs):
    """Generate a 2d figure (barcode) of the persistent homology."""
    from pylab import cm

    # Initialization of matplotlib figure and axes.
    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    # Colorbar creation
    Z = [[0, 0], [0, 0]]
    levels = _np.linspace(0.0, diam_max, 200)
    CS3 = _view.common.plt.contourf(Z, levels, cmap=cm.jet)
    ph_sort = _sort_ph_radii(ph)

    for ip, p in enumerate(ph_sort):
        ax.plot(p[:2], [ip, ip], c=cm.jet(p[2] / diam_max), linewidth=linewidth)

    all_kwargs = {
        "title": "Barcode of p.h.",
        "xlabel": "Lifetime",
    }
    all_kwargs.update(kwargs)

    _view.common.plt.ylim([-1, len(ph_sort)])
    _view.common.plt.colorbar(CS3)

    return _view.common.plot_style(fig=fig, ax=ax, **all_kwargs)


def barcode_mirror(ph, new_fig=True, subplot=False, color="b", **kwargs):
    """Generate a mirrored 2d figure (barcode) of the persistent homology."""
    # Initialization of matplotlib figure and axes.
    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    ph_mirror = _mirror_ph(ph)

    for ip, p in enumerate(ph_mirror):
        if p[2] >= 0:
            ax.plot(p[:2], [ip, ip], c=color)
        if p[2] < 0:
            ax.plot(_np.subtract([0, 0], p[:2]), [ip, ip], c=color)

    all_kwargs = {
        "title": "Mirror persistence barcode",
        "xlabel": "Lifetime",
    }
    all_kwargs.update(kwargs)
    _view.common.plt.ylim([-len(ph_mirror), len(ph_mirror)])

    return _view.common.plot_style(fig=fig, ax=ax, **all_kwargs)


def ph_birth_length(ph, new_fig=True, subplot=False, color="b", **kwargs):
    """Generate a 2d figure (ph diagram) of the persistent homology."""
    # Initialization of matplotlib figure and axes.
    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    ph_sort = sort_ph(ph)

    for p in ph_sort:
        ax.scatter(p[0], p[2], c=color)

    # bounds = _np.max(_np.max(ph))
    # _view.common.plt.plot([0, bounds], [0, bounds], c=color)

    all_kwargs = {
        "title": "Birth-Length diagram",
        "xlabel": "Birth",
        "ylabel": "Length",
    }
    all_kwargs.update(kwargs)

    return _view.common.plot_style(fig=fig, ax=ax, **all_kwargs)


def ph_on_tree(tree, new_fig=True, subplot=False, plane="xy", alpha=0.05, **kwargs):
    """Generate a 3d figure of the tree.

    Also adds the corresponding spheres that represent important events in the persistent homology
    diagram (birth and death of components).
    """
    # Initialization of matplotlib figure and axes.
    fig, ax = _view.tree(tree, new_fig=new_fig, subplot=subplot, plane=plane, **kwargs)

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, dim=plane)
    else:
        raise Exception("Plane value not recognised")

    for p in ph:
        c1 = _view.common.plt.Circle([tree.x[0], tree.y[0], tree.z[0]], p[0], alpha=alpha)
        c2 = _view.common.plt.Circle([tree.x[0], tree.y[0], tree.z[0]], p[1], alpha=alpha)

        ax.add_patch(c1)  # pylint: disable=no-member
        ax.add_patch(c2)  # pylint: disable=no-member

    return _view.common.plot_style(fig=fig, ax=ax, **kwargs)


def barcode_tree(tree, new_fig=True, plane="xy", output_dir=None, **kwargs):
    """Generate a 2d figure (barcode) of the persistent homology of a tree.

    This persistent homology is supposed to be computed by the Topology.get_persistent_homology()
    method.
    """
    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, dim=plane)
    else:
        raise Exception("Plane value not recognised")

    for ip, p in enumerate(ph):
        ph[ip].append(_np.abs(p[0] - p[1]))

    ph.sort(key=lambda x: x[2])

    for ip, p in enumerate(ph):
        fig, ax = _view.tree(tree, new_fig=new_fig, subplot=121, plane=plane)

        c1 = _view.common.plt.Circle([tree.x[0], tree.y[0]], p[0], alpha=0.2)
        c2 = _view.common.plt.Circle([tree.x[0], tree.y[0]], p[1], alpha=0.2)

        ax.add_patch(c1)  # pylint: disable=no-member
        ax.add_patch(c2)  # pylint: disable=no-member

        fig, ax = _view.common.get_figure(new_fig=False, subplot=122)

        for ip1, p1 in enumerate(ph):
            if ip1 != ip:
                ax.plot(p1[:2], [ip1, ip1], c="b")
            else:
                ax.plot(p1[:2], [ip1, ip1], c="r")

        all_kwargs = {
            "title": "Persistence barcode",
            "xlabel": "Lifetime",
            "ylabel": "",
        }
        all_kwargs.update(kwargs)

        _view.common.plt.ylim([-1, len(ph)])

        _view.common.plot_style(fig, ax, **all_kwargs)

        if output_dir is not None:
            all_kwargs["output_path"] = output_dir
            all_kwargs["output_name"] = (
                "barcode_"
                + "0" * (2 - len(str(len(tree.get_bifurcations()) - ip)))
                + str(len(tree.get_bifurcations()) - ip)
            )

        _view.common.save_plot(fig, **all_kwargs)


def ph_diagram_tree(tree, new_fig=True, plane="xy", output_dir=None, **kwargs):
    """Generate a 2d figure (barcode) of the persistent homology of a tree.

    This persistent homology is supposed to be computed by the Topology.get_persistent_homology()
    method.
    """
    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, dim=plane)
    else:
        raise Exception("Plane value not recognised")

    bounds = max(max(ph))

    for ip, p in enumerate(ph):
        ph[ip].append(_np.abs(p[0] - p[1]))

    ph.sort(key=lambda x: x[2])

    for ip, p in enumerate(ph):
        fig, ax = _view.tree(tree, new_fig=new_fig, subplot=121, plane=plane)

        c1 = _view.common.plt.Circle([tree.x[0], tree.y[0]], p[0], alpha=0.2)
        c2 = _view.common.plt.Circle([tree.x[0], tree.y[0]], p[1], alpha=0.2)

        ax.add_patch(c1)  # pylint: disable=no-member
        ax.add_patch(c2)  # pylint: disable=no-member

        fig, ax = _view.common.get_figure(new_fig=False, subplot=122)

        for ip1, p1 in enumerate(ph):
            if ip1 != ip:
                ax.scatter(p1[0], p1[1], c="b")
            else:
                ax.scatter(p1[0], p1[1], c="r", s=50)

        all_kwargs = {
            "title": "P.H. diagram",
            "xlabel": "Birth",
            "ylabel": "Death",
        }
        all_kwargs.update(kwargs)

        _view.common.plot_style(fig, ax, **all_kwargs)

        _view.common.plt.plot([0, bounds], [0, bounds])

        if output_dir is not None:
            all_kwargs["output_path"] = output_dir
            all_kwargs["output_name"] = (
                "ph_"
                + "0" * (2 - len(str(len(tree.get_bifurcations()) - ip)))
                + str(len(tree.get_bifurcations()) - ip)
                + ".png"
            )

        _view.common.save_plot(fig, **all_kwargs)


def tree_instance(
    tree,
    new_fig=True,
    plane="xy",
    component_num=1,
    feature="radial_distances",
    diameter=True,
    col="r",
    treecol="b",
    **kwargs,
):
    """Subplot with ph, barcode and tree within spheres."""
    from matplotlib.collections import LineCollection

    if new_fig:
        fig1, ax1 = _view.tree(
            tree,
            new_fig=new_fig,
            subplot=121,
            plane="xy",
            title="",
            treecolor=treecol,
            diameter=diameter,
        )
    else:
        fig1, ax1 = _view.common.get_figure(new_fig=new_fig, subplot=121)
    feat = getattr(tree, "get_section_" + feature)()
    segs = tree.get_segments()

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    if new_fig:
        fig1, ax2 = barcode(ph, new_fig=False, subplot=222, color=treecol)
        fig1, ax3 = diagram(ph, new_fig=False, subplot=224, color=treecol)
    else:
        fig1, ax2 = _view.common.get_figure(new_fig=new_fig, subplot=222)
        fig1, ax3 = _view.common.get_figure(new_fig=new_fig, subplot=224)

    ph = sort_ph(ph)
    select_section = ph[component_num]

    ax2.plot(select_section[:-1], [component_num, component_num], color=col, linewidth=2.0)
    ax3.scatter(select_section[0], select_section[1], color=col, s=50.0)

    initial = _np.transpose(tree.get_sections())[_np.where(feat == select_section[0])[0]]
    all_way = _np.array(tree.get_way_to_root(initial[0][1]))

    if select_section[1] != -1:
        final = _np.transpose(tree.get_sections())[_np.where(feat == select_section[1])[0]]
        until = _np.array(tree.get_way_to_root(final[0][1]))
    else:
        until = _np.array(tree.get_way_to_root(0))

    between = _np.setxor1d(all_way, until)

    tmp_segs = _np.array(segs)[between]
    toplot_segs = [_seg_2d(seg) for seg in tmp_segs]
    linewidth = [2 * d * 2 for d in _np.array(tree.d)[between]]

    collection = LineCollection(toplot_segs, color=col, linewidth=linewidth, alpha=1.0)
    ax1.add_collection(collection)
    _view.common.plt.tight_layout(True)

    return ax1, collection


def customized_cmap():
    """Returns a custom cmap."""

    def make_colormap(seq):
        """Return a LinearSegmentedColormap.

        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {"red": [], "green": [], "blue": []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict["red"].append([item, r1, r2])
                cdict["green"].append([item, g1, g2])
                cdict["blue"].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap("CustomMap", cdict)

    rvb = make_colormap(
        [
            (0.9, 0.9, 0.98),
            0.01,
            (0.8, 0.8, 0.98),
            0.05,
            (0.7, 0.7, 0.98),
            0.10,
            (0.6, 0.6, 0.98),
            0.12,
            (0.5, 0.5, 0.98),
            0.15,
            (0.2, 0.2, 1.0),
            0.18,
            (0.0, 0.5, 1.0),
            0.20,
            (0.0, 0.8, 0.8),
            0.25,
            (0.0, 0.9, 0.5),
            0.33,
            (0.0, 0.95, 0.2),
            0.35,
            (0.4, 0.95, 0.2),
            0.4,
            (0.6, 0.95, 0.1),
            0.45,
            (0.98, 0.95, 0.0),
            0.5,
            (0.99, 0.99, 0.0),
            0.55,
            (0.99, 0.8, 0.0),
            0.60,
            (0.99, 0.65, 0.0),
            0.67,
            (0.99, 0.45, 0.0),
            0.7,
            (0.99, 0.15, 0.0),
            0.75,
            (1.0, 0.0, 0.0),
            0.8,
            (0.9, 0.0, 0.0),
            0.85,
            (0.75, 0.0, 0.0),
            0.999,
            (0.55, 0.0, 0.0),
        ]
    )

    return rvb


def gaussian_kernel_resample(ph, num_samples=None, scale_persistent_comp=10):
    """Plot the gaussian kernel of the ph diagram that is given."""
    values = _np.transpose(ph)
    kernel = stats.gaussian_kde(values)

    if num_samples is None:
        num_samples = len(ph)

    return _np.transpose(kernel.resample(size=num_samples))


def gaussian_kernel_rot(ph, new_fig=True, subplot=111, xlim=None, ylim=None, angle=0, **kwargs):
    """Plot the gaussian kernel of the ph diagram that is given."""

    def rotation(x, y, angle=0.0):
        return [_np.cos(angle) * x - _np.sin(angle) * y, _np.sin(angle) * x + _np.cos(angle) * y]

    ph_r = [rotation(i[0], i[1], angle=angle) for i in ph]

    xmin = min(_np.transpose(ph_r)[0])
    xmax = max(_np.transpose(ph_r)[0])
    ymin = min(_np.transpose(ph_r)[1])
    ymax = max(_np.transpose(ph_r)[1])

    if xlim is None:
        xlim = [xmin, xmax]
    if ylim is None:
        ylim = [ymin, ymax]

    X, Y = _np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]

    values = _np.transpose(ph_r)
    kernel = stats.gaussian_kde(values)
    positions = _np.vstack([X.ravel(), Y.ravel()])
    Z = _np.reshape(kernel(positions).T, X.shape)
    Zn = Z / _np.max(Z)

    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.pcolor(Zn, vmin=0.0, vmax=1.0, cmap=_view.common.plt.cm.inferno)

    return Z, _view.common.plot_style(fig=fig, ax=ax, **kwargs)


def gaussian_kernel_weighted(ph, new_fig=True, subplot=111, xlim=None, ylim=None, **kwargs):
    """Plot the gaussian kernel of the ph diagram that is given."""
    xmin = min(_np.transpose(ph)[0])
    xmax = max(_np.transpose(ph)[0])
    ymin = min(_np.transpose(ph)[1])
    ymax = max(_np.transpose(ph)[1])

    if xlim is None:
        xlim = [xmin, xmax]
    if ylim is None:
        ylim = [ymin, ymax]

    X, Y = _np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]

    values = _np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = _np.vstack([X.ravel(), Y.ravel()])
    Z = _np.reshape(kernel(positions).T, X.shape)
    Zn = Z / _np.max(Z)

    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.pcolor(Zn, vmin=0.0, vmax=1.0, cmap=_view.common.plt.cm.inferno)

    return Z, _view.common.plot_style(fig=fig, ax=ax, **kwargs)


def gaussian_kernel_superposition(
    ph, new_fig=True, subplot=111, xlim=None, ylim=None, color="r", **kwargs
):
    """Plot the gaussian kernel of the ph diagram that is given."""
    xmin = min(_np.transpose(ph)[0])
    xmax = max(_np.transpose(ph)[0])
    ymin = min(_np.transpose(ph)[1])
    ymax = max(_np.transpose(ph)[1])

    if xlim is None:
        xlim = [xmin, xmax]
    if ylim is None:
        ylim = [ymin, ymax]

    X, Y = _np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]

    values = _np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = _np.vstack([X.ravel(), Y.ravel()])
    Z = _np.reshape(kernel(positions).T, X.shape)
    Zn = Z / _np.max(Z)

    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.pcolor(Zn, vmin=0.0, vmax=1.0, cmap=_view.common.plt.cm.inferno)

    for p in ph:
        ax.scatter(p[0], p[1], c=color)

    return _view.common.plot_style(fig=fig, ax=ax, xlim=xlim, ylim=ylim, **kwargs)


def tree_br(
    tree, plane="xy", feature="radial_distances", title="", diameter=True, treecol="b", **kwargs
):
    """Subplot with ph, barcode and tree within spheres."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.tree(
        tree,
        new_fig=True,
        subplot=121,
        plane="xy",
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    fig1, ax2 = barcode(ph, new_fig=False, subplot=122, color=treecol, ylabel="")

    _view.common.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _view.common.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def tree_gaussian_kernel(
    tree,
    plane="xy",
    feature="radial_distances",
    title="",
    diameter=True,
    treecol="b",
    xlim=None,
    ylim=None,
    **kwargs,
):
    """Subplot with ph, barcode and tree within spheres."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.tree(
        tree,
        new_fig=True,
        subplot=121,
        plane="xy",
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    fig1, ax2 = gaussian_kernel_weighted(ph, new_fig=False, subplot=122, xlim=xlim, ylim=ylim)

    _view.common.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _view.common.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def tree_ph(
    tree,
    plane="xy",
    feature="radial_distances",
    title="",
    diameter=True,
    treecol="b",
    **kwargs,
):
    """Subplot with ph, barcode and tree within spheres."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.tree(
        tree,
        new_fig=True,
        subplot=121,
        plane="xy",
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    fig1, ax2 = diagram(ph, new_fig=False, subplot=122, color=treecol)

    _view.common.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _view.common.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def tree_evol(
    tree,
    plane="xy",
    feature="radial_distances",
    title="",
    diameter=True,
    treecol="b",
    barcode_xlim=None,
    barcode_ylim=None,
    kernel_xlim=None,
    kernel_ylim=None,
    **kwargs,
):
    """Subplot with ph, barcode and tree within spheres."""
    kwargs["output_path"] = kwargs.get("output_path", None)

    fig1, ax1 = _view.tree(
        tree,
        new_fig=True,
        subplot=121,
        plane="xy",
        title=title,
        treecolor=treecol,
        diameter=diameter,
    )

    def _seg_2d(seg):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]

        return ((horz1, vert1), (horz2, vert2))

    if plane in ["xy", "yx", "zx", "xz", "yz", "zy"]:
        ph = _tm.methods.get_persistence_diagram(tree, feature=feature)
    else:
        raise Exception("Plane value not recognised")

    fig1, ax3 = barcode(
        ph, new_fig=False, subplot=222, color=treecol, xlim=barcode_xlim, ylim=barcode_ylim
    )

    fig1, ax4 = gaussian_kernel_weighted(
        ph, new_fig=False, subplot=224, xlim=kernel_xlim, ylim=kernel_ylim
    )

    _view.common.plt.tight_layout(True)

    if kwargs["output_path"] is not None:
        _view.common.save_plot(fig=ax1, **kwargs)

    return fig1, ax1


def image_diff_time(
    Z1, Z2, time_steps=100, new_fig=True, subplot=111, xlim=None, ylim=None, **kwargs
):
    """Take as input a set of images and plots their difference."""
    xmin = min(Z1[1][1].get_xlim() + Z2[1][1].get_xlim())
    xmax = max(Z1[1][1].get_xlim() + Z2[1][1].get_xlim())
    ymin = min(Z1[1][1].get_ylim() + Z2[1][1].get_ylim())
    ymax = max(Z1[1][1].get_ylim() + Z2[1][1].get_ylim())

    if xlim is None:
        xlim = [xmin, xmax]
    if ylim is None:
        ylim = [ymin, ymax]

    X, Y = _np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]

    img1 = Z1[0] / Z1[0].max()
    img2 = Z2[0] / Z2[0].max()

    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    _view.common.plt.pcolor(X, Y, img2 - img1, vmin=-1.0, vmax=1.0)

    _view.common.plt.colorbar()

    kwargs["xlim"] = xlim
    kwargs["ylim"] = ylim

    return _view.common.plot_style(fig=fig, ax=ax, **kwargs)


def plot_simple_tree(
    tr,
    plane="xy",
    new_fig=True,
    subplot=False,
    hadd=0.0,
    vadd=0.0,
    treecolor="b",
    alpha=1.0,
    **kwargs,
):
    """Generate a 2d figure of the tree."""
    from matplotlib.collections import LineCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = _view.common.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_2d(seg, x_add=0.0, y_add=0.0):
        """2d coordinates required for the plotting of a segment."""
        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz] + x_add
        horz2 = seg[1][horz] + x_add
        vert1 = seg[0][vert] + y_add
        vert2 = seg[1][vert] + y_add

        return ((horz1, vert1), (horz2, vert2))

    segs = [_seg_2d(seg, hadd, vadd) for seg in tr.get_segments()]

    linewidth = [2 * d * 1.0 for d in tr.d]

    # Plot the collection of lines.
    collection = LineCollection(segs, color=treecolor, linewidth=linewidth, alpha=alpha)

    ax.add_collection(collection)

    all_kwargs = {
        "xlim": [
            bounding_box[0][_utils.term_dict[plane[0]]] - 20,
            bounding_box[1][_utils.term_dict[plane[0]]] + 20,
        ],
        "ylim": [
            bounding_box[0][_utils.term_dict[plane[1]]] - 20,
            bounding_box[1][_utils.term_dict[plane[1]]] + 20,
        ],
    }
    all_kwargs.update(kwargs)

    return _view.common.plot_style(fig=fig, ax=ax, **all_kwargs)


def plot_intermediate(ph_all, colors_bar, tree, colors, counter, linewidth=1.0, output_path="./"):
    """Plot the tree and the barcode with defined colors."""
    fig, ax1 = _view.common.get_figure(new_fig=True, subplot=122)

    for ipers, pers in enumerate(ph_all):
        ax1.plot(pers[:2], [ipers, ipers], c=colors_bar[ipers], linewidth=linewidth)
        ax1.set_title("Barcode", fontsize=14)

    ax1.set_ylim([-1, len(ph_all)])
    ax1.set_xlim([-10, _np.max(ph_all)])

    ax1.set_xlabel("Distance from soma", fontsize=14)

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    # ax1.spines['left'].set_visible(False)
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.set_yticks([])

    fig, ax2 = plot_simple_tree(tree, new_fig=False, subplot=121, treecolor=colors, diameter=False)

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.set_ticks_position("left")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.set_yticks([])

    _view.common.plot_style(
        fig=fig,
        ax=ax2,
        output_path=output_path,
        title="Morphology",
        tight=True,
        xlabel="Length (um)",
        ylabel="",
        output_name=str(0) * (3 - len(str(counter))) + str(counter),
    )


def plot_persistent_homology_video(
    tree,
    feature="radial_distances",
    linewidth=1.0,
    c1="b",
    c2="w",
    c3="w",
    c4="b",
    c5="r",
    output_path="./",
    **kwargs,
):
    """Extract ph from tree that contains mutlifurcations and generate a video of the process."""
    ph_all = _tm.methods.get_persistence_diagram(tree)

    ph_all = _sort_ph_initial(ph_all)

    colors_bar = _np.array([c4 for s in range(len(ph_all))])

    ph = []

    rd = getattr(tree, "get_point_" + feature)(**kwargs)

    colors = _np.array([c1 for s in range(len(tree.x) - 1)])

    counter = 5

    plot_intermediate(ph_all, colors_bar, tree, colors, counter, output_path=output_path)

    counter = counter + 1

    active = tree.get_bif_term() == 0

    beg, end = tree.get_sections_2()

    beg = _np.array(beg)
    end = _np.array(end)

    parents = {e: b for b, e in zip(beg, end)}
    children = {b: end[_np.where(beg == b)[0]] for b in _np.unique(beg)}

    colors_bar = _np.array([c3 for s in range(len(ph_all))])

    plot_intermediate(ph_all, colors_bar, tree, colors, counter, output_path=output_path)

    counter = counter + 1

    active_paths = {}

    while len(_np.where(active)[0]) > 1:
        alive = list(_np.where(active)[0])

        for i in alive:
            p = parents[i]
            c = children[p]

            if _np.all(active[c]):
                to_modify = []
                to_modify_bars = []

                active[p] = True
                active[c] = False

                mx = _np.argmax(abs(rd[c]))
                mx_id = c[mx]

                alive.remove(mx_id)
                c = _np.delete(c, mx)

                for ci in c:
                    ph.append([rd[ci], rd[p]])
                    # colors_bar[_np.where(_np.array(ph_all) == [rd[ci], rd[p]])[0][1]] = c4
                    to_modify_bars = to_modify_bars + [
                        _np.where(_np.array(ph_all) == [rd[ci], rd[p]])[0][1]
                    ]

                    alive.remove(ci)

                    # colors[tree.get_way_to_section_start(ci - 1)] = c2
                    to_modify = to_modify + tree.get_way_to_section_start(ci - 1)

                    if ci < len(colors):
                        # colors[ci] = c2
                        to_modify = to_modify + [ci]

                    key = int(ci)

                    while key in active_paths:
                        # colors[tree.get_way_to_section_start(active_paths[key] - 1)] = c2
                        to_modify = to_modify + tree.get_way_to_section_start(active_paths[key] - 1)
                        if active_paths[key] < len(colors):
                            # colors[active_paths[key]] = c2
                            to_modify = to_modify + [active_paths[key]]
                        key = active_paths.pop(key)

                active_paths[int(p)] = int(mx_id)

                alive.append(p)

                rd[p] = rd[mx_id]

                colors[to_modify] = c5
                colors_bar[to_modify_bars] = c5

                plot_intermediate(
                    ph_all, colors_bar, tree, colors, counter, output_path=output_path
                )

                counter = counter + 1

                colors[to_modify] = c2
                colors_bar[to_modify_bars] = c4

                plot_intermediate(
                    ph_all, colors_bar, tree, colors, counter, output_path=output_path
                )

                counter = counter + 1

    to_modify = []

    ph.append([rd[_np.where(active)[0][0]], 0])  # Add the last alive component

    key = int(_np.where(active)[0][0])
    way = tree.get_way_to_section_start(key - 1)
    way.remove(-1)
    # colors[way] = c2
    to_modify = to_modify + way

    if key < len(colors):
        # colors[key] = c2
        to_modify = to_modify + [key]

    while key in active_paths:
        # colors[tree.get_way_to_section_start(active_paths[key] - 1)] = c2
        to_modify = to_modify + tree.get_way_to_section_start(active_paths[key] - 1)
        if active_paths[key] < len(colors):
            # colors[active_paths[key]] = c2
            to_modify = to_modify + [active_paths[key]]
        key = active_paths.pop(key)

    colors_bar[_np.where(_np.array(ph_all) == [rd[_np.where(active)[0][0]], 0])[0][1]] = c5

    colors[to_modify] = c5

    plot_intermediate(ph_all, colors_bar, tree, colors, counter, output_path=output_path)

    counter = counter + 1

    colors[to_modify] = c2

    colors_bar[_np.where(_np.array(ph_all) == [rd[_np.where(active)[0][0]], 0])[0][1]] = c4

    plot_intermediate(ph_all, colors_bar, tree, colors, counter, output_path=output_path)

    # plot_intermediate(ph_all, colors_bar, tree, colors, counter)

    return ph


def multiple_trees_plot(trees, phs, title_1="Asymmetry"):
    """Plot multiple trees."""
    plt.figure()
    N = len(trees)
    lims = [_np.min(_np.min(phs)) - 5, _np.max(_np.max(phs)) + 5]

    for i, tr in enumerate(trees):
        view.view.tree(
            tr,
            new_fig=False,
            subplot=(3, N, i + 1),
            title=title_1,
            plane="xy",
            diameter=False,
            linewidth=0.8,
            treecolor="black",
        )

        view.plot.barcode(phs[i], new_fig=False, subplot=(3, N, i + 1 + N), title="", xlim=lims)
        view.plot.persistence_image(
            phs[i], new_fig=False, subplot=(3, N, i + 1 + 2 * N), title="", xlim=lims, ylim=lims
        )


def polar_plot_custom_color(
    population, bins=25, apical_color="purple", basal_color="r", edgecolor=None, alpha=0.8
):
    """Plot a population."""
    fig = _cm.plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    input_data1 = view.view._get_polar_data(population, neurite_type="basal_dendrite", bins=bins)
    input_data2 = view.view._get_polar_data(population, neurite_type="apical_dendrite", bins=bins)

    maximum = _np.max(_np.array(input_data1)[:, 2].tolist() + _np.array(input_data2)[:, 2].tolist())

    theta = _np.array(input_data1)[:, 0]
    radii = _np.array(input_data1)[:, 2] / maximum
    width = 2 * _np.pi / len(input_data1)
    ax.bar(
        theta, radii, width=width, edgecolor=edgecolor, bottom=0.0, alpha=alpha, color=basal_color
    )

    theta = _np.array(input_data2)[:, 0]
    radii = _np.array(input_data2)[:, 2] / maximum
    width = 2 * _np.pi / len(input_data2)
    ax.bar(
        theta, radii, width=width, edgecolor=edgecolor, bottom=0.0, alpha=alpha, color=apical_color
    )
