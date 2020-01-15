'''Module for viewing neuronal morphologies
'''

import numpy as _np
from matplotlib.collections import LineCollection as _LC
from tmd.view import common as _cm
from tmd import utils as _utils
from tmd.view.common import blues_map


def _get_default(variable, **kwargs):
    '''Returns default variable or kwargs variable if it exists.
    '''
    default = {'linewidth': 1.2,
               'alpha': 0.8,
               'treecolor': None,
               'diameter': True,
               'diameter_scale': 1.0,
               'white_space': 30.}

    return kwargs.get(variable, default[variable])


def trunk(tr, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0, N=10, **kwargs):
    '''Generates a 2d figure of the trunk = first N segments of the tree.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    # bounding_box = tr.get_bounding_box()

    def _seg_2d(seg, x_add=0.0, y_add=0.0):

        """2d coordinates required for the plotting of a segment"""

        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz] + x_add
        horz2 = seg[1][horz] + x_add
        vert1 = seg[0][vert] + y_add
        vert2 = seg[1][vert] + y_add

        return ((horz1, vert1), (horz2, vert2))

    if len(tr.get_segments()) < N:
        N = len(tr.get_segments())

    segs = [_seg_2d(seg, hadd, vadd) for seg in tr.get_segments()[:N]]

    linewidth = _get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default('diameter', **kwargs):

        scale = _get_default('diameter_scale', **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = _cm.get_color(_get_default('treecolor', **kwargs),
                              _utils.tree_type[tr.get_type()])

    # Plot the collection of lines.
    collection = _LC(segs, color=treecolor, linewidth=linewidth,
                     alpha=_get_default('alpha', **kwargs))

    ax.add_collection(collection)

    kwargs['title'] = kwargs.get('title', 'Tree view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def tree(tr, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0, **kwargs):
    '''Generates a 2d figure of the tree.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_2d(seg, x_add=0.0, y_add=0.0):

        """2d coordinates required for the plotting of a segment"""

        horz = _utils.term_dict[plane[0]]
        vert = _utils.term_dict[plane[1]]

        horz1 = seg[0][horz] + x_add
        horz2 = seg[1][horz] + x_add
        vert1 = seg[0][vert] + y_add
        vert2 = seg[1][vert] + y_add

        return ((horz1, vert1), (horz2, vert2))

    segs = [_seg_2d(seg, hadd, vadd) for seg in tr.get_segments()]

    linewidth = _get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default('diameter', **kwargs):

        scale = _get_default('diameter_scale', **kwargs)
        linewidth = [d * scale for d in tr.d]

    if tr.get_type() not in _utils.tree_type:
        treecolor = 'black'
    else:
        treecolor = _cm.get_color(_get_default('treecolor', **kwargs),
                                  _utils.tree_type[tr.get_type()])

    # Plot the collection of lines.
    collection = _LC(segs, color=treecolor, linewidth=linewidth,
                     alpha=_get_default('alpha', **kwargs))

    ax.add_collection(collection)

    kwargs['title'] = kwargs.get('title', 'Tree view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    white_space = _get_default('white_space', **kwargs)

    kwargs['xlim'] = kwargs.get('xlim', [bounding_box[0][_utils.term_dict[plane[0]]] - white_space,
                                         bounding_box[1][_utils.term_dict[plane[0]]] + white_space])
    kwargs['ylim'] = kwargs.get('ylim', [bounding_box[0][_utils.term_dict[plane[1]]] - white_space,
                                         bounding_box[1][_utils.term_dict[plane[1]]] + white_space])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def soma(sm, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0, **kwargs):
    '''Generates a 2d figure of the soma.

    Parameters
    ----------
    soma: Soma
        neurom.Soma object
    '''
    treecolor = kwargs.get('treecolor', None)
    outline = kwargs.get('outline', True)

    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    # Definition of the tree color depending on the tree type.
    treecolor = _cm.get_color(treecolor, tree_type='soma')

    # Plot the outline of the soma as a circle, is outline is selected.
    if not outline:
        soma_circle = _cm.plt.Circle(sm.get_center() + [hadd, vadd, 0.0],
                                     sm.get_diameter() / 2., color=treecolor,
                                     alpha=_get_default('alpha', **kwargs))
        ax.add_artist(soma_circle)
    else:
        horz = getattr(sm, plane[0]) + hadd
        vert = getattr(sm, plane[1]) + vadd

        horz = _np.append(horz, horz[0]) + hadd  # To close the loop for a soma viewer.
        vert = _np.append(vert, vert[0]) + vadd  # To close the loop for a soma viewer.

        _cm.plt.plot(horz, vert, color=treecolor,
                     alpha=_get_default('alpha', **kwargs),
                     linewidth=_get_default('linewidth', **kwargs))

    kwargs['title'] = kwargs.get('title', 'Soma view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def neuron(nrn, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0,
           neurite_type='all', rotation=None, nosoma=False, new_axes=True, **kwargs):
    '''Generates a 2d figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object

    Options
    -------
    plane: str
        Accepted values: Any pair of of xyz
        Default value is 'xy'

    linewidth: float
        Defines the linewidth of the tree and soma
        of the neuron, if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the neuron.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the trees.
        If None the default values will be used,
        depending on the type of tree:
        Soma: "black"
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the neuron will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    Returns
    --------
    A 3D matplotlib figure with a tree view, at the selected plane.
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot, new_axes=new_axes)

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot

    if not nosoma:
        soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    h = []
    v = []

    to_plot = []

    if rotation == 'apical':
        angle = _np.arctan2(nrn.apical[0].get_pca()[0], nrn.apical[0].get_pca()[1])
        angle = _np.arctan2(rotation[1], rotation[0])

    if neurite_type == 'all':
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:
        if rotation is not None:
            temp_tree.rotate_xy(angle)

        bounding_box = temp_tree.get_bounding_box()

        h.append([bounding_box[0][_utils.term_dict[plane[0]]],
                  bounding_box[1][_utils.term_dict[plane[0]]]])
        v.append([bounding_box[0][_utils.term_dict[plane[1]]],
                  bounding_box[1][_utils.term_dict[plane[1]]]])

        tree(temp_tree, hadd=hadd, vadd=vadd, **kwargs)

    kwargs['title'] = kwargs.get('title', nrn.name)
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    white_space = _get_default('white_space', **kwargs)
    kwargs['xlim'] = kwargs.get('xlim', [_np.min(h) - white_space + hadd,
                                         _np.max(h) + white_space + hadd])
    kwargs['ylim'] = kwargs.get('ylim', [_np.min(v) - white_space + vadd,
                                         _np.max(v) + white_space + vadd])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def all_trunks(nrn, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0,
               neurite_type='all', N=10, **kwargs):
    '''Generates a 2d figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot

    soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    to_plot = []

    if neurite_type == 'all':
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:

        trunk(temp_tree, N=N, **kwargs)

    kwargs['title'] = kwargs.get('title', nrn.name)
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    kwargs['xlim'] = kwargs.get('xlim', [nrn.soma.get_center()[0] - 2. * N,
                                         nrn.soma.get_center()[0] + 2. * N])

    kwargs['ylim'] = kwargs.get('ylim', [nrn.soma.get_center()[1] - 2. * N,
                                         nrn.soma.get_center()[1] + 2. * N])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def population(pop, plane='xy', new_fig=True, subplot=False, hadd=0.0, vadd=0.0,
               neurite_type='all', **kwargs):
    '''Generates a 2d figure of the population,
    that contains a soma and a list of trees.

    Parameters
    ----------
    pop: Population
        neurom.Population object

    Options
    -------
    plane: str
        Accepted values: Any pair of of xyz
        Default value is 'xy'

    linewidth: float
        Defines the linewidth of the tree and soma
        of the neuron, if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the neuron.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the trees.
        If None the default values will be used,
        depending on the type of tree:
        Soma: "black"
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the neuron will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    Returns
    --------
    A 3D matplotlib figure with a tree view, at the selected plane.
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot

    h = []
    v = []

    for nrn in pop.neurons:

        soma(nrn.soma, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

        if neurite_type == 'all':
            neurite_list = ['basal', 'apical', 'axon']
        else:
            neurite_list = [neurite_type]

        for nt in neurite_list:
            for temp_tree in getattr(nrn, nt):

                bounding_box = temp_tree.get_bounding_box()

                h.append([bounding_box[0][_utils.term_dict[plane[0]]] + hadd,
                          bounding_box[1][_utils.term_dict[plane[1]]] + vadd])
                v.append([bounding_box[0][_utils.term_dict[plane[0]]] + hadd,
                          bounding_box[1][_utils.term_dict[plane[1]]] + vadd])

                tree(temp_tree, plane=plane, hadd=hadd, vadd=vadd, **kwargs)

    kwargs['title'] = kwargs.get('title', 'Neuron view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])
    kwargs['xlim'] = kwargs.get('xlim', [_np.min(h), _np.max(h)])
    kwargs['ylim'] = kwargs.get('ylim', [_np.min(v), _np.max(v)])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def tree3d(tr, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''Generates a figure of the tree in 3d.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object
    '''
    # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    # Data needed for the viewer: x,y,z,r
    bounding_box = tr.get_bounding_box()

    def _seg_3d(seg):

        """3d coordinates needed for the plotting of a segment"""

        horz = _utils.term_dict['x']
        vert = _utils.term_dict['y']
        depth = _utils.term_dict['z']

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]
        depth1 = seg[0][depth]
        depth2 = seg[1][depth]

        return ((horz1, vert1, depth1), (horz2, vert2, depth2))

    segs = [_seg_3d(seg) for seg in tr.get_segments()]

    linewidth = _get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default('diameter', **kwargs):

        scale = _get_default('diameter_scale', **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = _cm.get_color(_get_default('treecolor', **kwargs),
                              _utils.tree_type[tr.get_type()])

    # Plot the collection of lines.
    collection = Line3DCollection(segs, color=treecolor, linewidth=linewidth,
                                  alpha=_get_default('alpha', **kwargs))

    ax.add_collection3d(collection)

    kwargs['title'] = kwargs.get('title', 'Tree 3d-view')
    kwargs['xlabel'] = kwargs.get('xlabel', 'X')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Y')
    kwargs['zlabel'] = kwargs.get('zlabel', 'Z')

    white_space = _get_default('white_space', **kwargs)

    kwargs['xlim'] = kwargs.get('xlim', [bounding_box[0][_utils.term_dict['x']] - white_space,
                                         bounding_box[1][_utils.term_dict['x']] + white_space])
    kwargs['ylim'] = kwargs.get('ylim', [bounding_box[0][_utils.term_dict['y']] - white_space,
                                         bounding_box[1][_utils.term_dict['y']] + white_space])
    kwargs['zlim'] = kwargs.get('zlim', [bounding_box[0][_utils.term_dict['z']] - white_space,
                                         bounding_box[1][_utils.term_dict['z']] + white_space])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def trunk3d(tr, new_fig=True, new_axes=True, subplot=False, N=10, **kwargs):
    '''Generates a figure of the trunk in 3d.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object
    '''
    # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    def _seg_3d(seg):

        """3d coordinates needed for the plotting of a segment"""

        horz = _utils.term_dict['x']
        vert = _utils.term_dict['y']
        depth = _utils.term_dict['z']

        horz1 = seg[0][horz]
        horz2 = seg[1][horz]
        vert1 = seg[0][vert]
        vert2 = seg[1][vert]
        depth1 = seg[0][depth]
        depth2 = seg[1][depth]

        return ((horz1, vert1, depth1), (horz2, vert2, depth2))

    if len(tr.get_segments()) < N:
        N = len(tr.get_segments())

    segs = [_seg_3d(seg) for seg in tr.get_segments()[:N]]

    linewidth = _get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.

    if _get_default('diameter', **kwargs):

        scale = _get_default('diameter_scale', **kwargs)
        linewidth = [d * scale for d in tr.d]

    treecolor = _cm.get_color(_get_default('treecolor', **kwargs),
                              _utils.tree_type[tr.get_type()])

    # Plot the collection of lines.
    collection = Line3DCollection(segs, color=treecolor, linewidth=linewidth,
                                  alpha=_get_default('alpha', **kwargs))

    ax.add_collection3d(collection)

    kwargs['title'] = kwargs.get('title', 'Tree 3d-view')
    kwargs['xlabel'] = kwargs.get('xlabel', 'X')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Y')
    kwargs['zlabel'] = kwargs.get('zlabel', 'Z')

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def soma3d(sm, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''Generates a 3d figure of the soma.

    Parameters
    ----------
    soma: Soma
        neurom.Soma object
    '''
    treecolor = kwargs.get('treecolor', None)

    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    # Definition of the tree color depending on the tree type.
    treecolor = _cm.get_color(treecolor, tree_type='soma')

    center = sm.get_center()

    xs = center[0]
    ys = center[1]
    zs = center[2]

    # Plot the soma as a circle.
    fig, ax = _cm.plot_sphere(fig, ax, center=[xs, ys, zs],
                              radius=sm.get_diameter(), color=treecolor,
                              alpha=_get_default('alpha', **kwargs))

    kwargs['title'] = kwargs.get('title', 'Soma view')
    kwargs['xlabel'] = kwargs.get('xlabel', 'X')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Y')
    kwargs['zlabel'] = kwargs.get('zlabel', 'Z')

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def neuron3d(nrn, new_fig=True, new_axes=True, subplot=False, neurite_type='all', **kwargs):
    '''Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object
    '''
    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    kwargs['new_fig'] = False
    kwargs['new_axes'] = False
    kwargs['subplot'] = subplot

    soma3d(nrn.soma, **kwargs)

    h = []
    v = []
    d = []

    to_plot = []

    if neurite_type == 'all':
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:

        bounding_box = temp_tree.get_bounding_box()

        h.append([bounding_box[0][_utils.term_dict['x']],
                  bounding_box[1][_utils.term_dict['x']]])
        v.append([bounding_box[0][_utils.term_dict['y']],
                  bounding_box[1][_utils.term_dict['y']]])
        d.append([bounding_box[0][_utils.term_dict['z']],
                  bounding_box[1][_utils.term_dict['z']]])

        tree3d(temp_tree, **kwargs)

    kwargs['title'] = kwargs.get('title', nrn.name)
    white_space = _get_default('white_space', **kwargs)
    kwargs['xlim'] = kwargs.get('xlim', [_np.min(h) - white_space,
                                         _np.max(h) + white_space])
    kwargs['ylim'] = kwargs.get('ylim', [_np.min(v) - white_space,
                                         _np.max(v) + white_space])
    kwargs['zlim'] = kwargs.get('zlim', [_np.min(d) - white_space,
                                         _np.max(d) + white_space])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def all_trunks3d(nrn, new_fig=True, new_axes=True, subplot=False, neurite_type='all',
                 N=10, **kwargs):
    '''Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object
    '''
    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    kwargs['new_fig'] = False
    kwargs['new_axes'] = False
    kwargs['subplot'] = subplot

    soma3d(nrn.soma, **kwargs)

    to_plot = []

    if neurite_type == 'all':
        to_plot = nrn.neurites
    else:
        for neu_type in neurite_type:
            to_plot = to_plot + getattr(nrn, neu_type)

    for temp_tree in to_plot:

        trunk3d(temp_tree, N=N, **kwargs)

    kwargs['title'] = kwargs.get('title', nrn.name)
    kwargs['xlim'] = kwargs.get('xlim', [nrn.soma.get_center()[0] - 2. * N,
                                         nrn.soma.get_center()[0] + 2. * N])

    kwargs['ylim'] = kwargs.get('ylim', [nrn.soma.get_center()[1] - 2. * N,
                                         nrn.soma.get_center()[1] + 2. * N])

    kwargs['zlim'] = kwargs.get('zlim', [nrn.soma.get_center()[2] - 2. * N,
                                         nrn.soma.get_center()[2] + 2. * N])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def population3d(pop, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''Generates a figure of the population,
    that contains the pop somata and a set of list of trees.

    Parameters
    ----------
    pop: Population
        neurom.Population object
    '''
    # Initialization of matplotlib figure and axes.
    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes,
                             subplot=subplot, params={'projection': '3d'})

    kwargs['new_fig'] = False
    kwargs['new_axes'] = False
    kwargs['subplot'] = subplot

    h = []
    v = []
    d = []

    for nrn in pop.neurons:
        # soma3d(nrn.soma, **kwargs)
        for temp_tree in nrn.neurites:
            bounding_box = temp_tree.get_bounding_box()
            h.append([bounding_box[0][_utils.term_dict['x']],
                      bounding_box[1][_utils.term_dict['x']]])
            v.append([bounding_box[0][_utils.term_dict['y']],
                      bounding_box[1][_utils.term_dict['y']]])
            d.append([bounding_box[0][_utils.term_dict['z']],
                      bounding_box[1][_utils.term_dict['z']]])
            tree3d(temp_tree, **kwargs)

    kwargs['title'] = kwargs.get('title', '')
    white_space = _get_default('white_space', **kwargs)
    kwargs['xlim'] = kwargs.get('xlim', [_np.min(h) - white_space,
                                         _np.max(h) + white_space])
    kwargs['ylim'] = kwargs.get('ylim', [_np.min(v) - white_space,
                                         _np.max(v) + white_space])
    kwargs['zlim'] = kwargs.get('zlim', [_np.min(d) - white_space,
                                         _np.max(d) + white_space])

    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


# pylint: disable=too-many-locals
def density_cloud(obj, new_fig=True, subplot=111, new_axes=True, neurite_type='all',
                  bins=100, plane='xy', color_map=blues_map, alpha=0.8,
                  centered=True, colorbar=True, plot_neuron=False, **kwargs):
    """
    View the neuron morphologies of a population as a density cloud.
    """
    x1 = []
    y1 = []

    if neurite_type == 'all':
        ntypes = 'neurites'
    else:
        ntypes = neurite_type

    fig, ax = _cm.get_figure(new_fig=new_fig, new_axes=new_axes, subplot=subplot)

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

    H1, xedges1, yedges1 = _np.histogram2d(x1, y1, bins=(bins, bins))
    mask = H1 < 0.05
    H2 = _np.ma.masked_array(H1, mask)
    color_map.set_bad(color='white', alpha=None)

    plots = ax.contourf((xedges1[:-1] + xedges1[1:]) / 2,
                        (yedges1[:-1] + yedges1[1:]) / 2,
                        _np.transpose(H2), cmap=color_map,
                        alhpa=alpha)

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot

    if plot_neuron:
        nrn = obj.neurons[0]
        h, v, _ = nrn.soma.get_center()
        soma(nrn.soma, plane='xy', hadd=-h, vadd=-v, **kwargs)
        for temp_tree in getattr(nrn, ntypes):
            tree(temp_tree, plane='xy', hadd=-h, vadd=-v, treecolor='r', **kwargs)

    if colorbar:
        _cm.plt.colorbar(plots)
    # soma(neu.soma, new_fig=False)
    return _cm.plot_style(fig=fig, ax=ax, **kwargs)


def _get_polar_data(pop, neurite_type='neurites', bins=20):
    '''Extracts the data to plot the polar length distribution
    of a neuron or a population of neurons.
    '''
    def seg_angle(seg):
        '''angle between mean x, y coordinates of a seg'''
        mean_x = _np.mean([seg[0][0], seg[1][0]])
        mean_y = _np.mean([seg[0][1], seg[1][1]])
        return _np.arctan2(mean_y, mean_x)

    def seg_length(seg):
        '''compute the length of a seg'''
        return _np.linalg.norm(_np.subtract(seg[1], seg[0]))

    segs = []
    for tr in getattr(pop, neurite_type):
        segs = segs + tr.get_segments()

    angles = _np.array([seg_angle(s) for s in segs])
    lens = _np.array([seg_length(s) for s in segs])
    ranges = [[i * 2 * _np.pi / bins - _np.pi, (i + 1) * 2 * _np.pi / bins - _np.pi]
              for i in range(bins)]
    results = [r + [_np.sum(lens[_np.where((angles > r[0]) & (angles < r[1]))[0]])]
               for r in ranges]

    return results


def polar_plot(pop, neurite_type='neurites', bins=20):
    '''
    Generates a polar plot of a neuron or population
    '''
    input_data = _get_polar_data(pop, neurite_type=neurite_type, bins=bins)

    fig = _cm.plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    theta = _np.array(input_data)[:, 0]
    radii = _np.array(input_data)[:, 2] / _np.max(input_data)
    width = 2 * _np.pi / len(input_data)
    ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.8)
