import odl
from odl.util import safe_int_conv
import mayavi
import numpy as np


def show_planes(elem, planes=('x', 'y', 'z'), plane_indcs=None, cmap='bone',
                **kwargs):
    """Show orthogonal slices of ``elem``.

    This function requires the `mayavi`_ package, installable via
    ``conda install -c conda-forge mayavi``.

    Parameters
    ----------
    elem : 3-dim. `Tensor`
        Element of a 3-dimensional space that is to be visualized.
    planes : sequence of strings, optional
        Sequence defining the planes to be shown. It may only contain
        the strings ``'x'``, ``'y'`` and ``'z'``. If an entry occurs
        more than once, the corresponding ``plane_indcs`` must be
        defined to avoid identical planes.
    plane_indcs : sequence of ints, optional
        Index positions of the planes as specified in ``planes``.
        For ``None`` entries, the middle index along that axis is
        selected, and a single ``None`` results in that behavior in
        all axes.
    axis_labels : sequence of strings, optional
        Labels for the 3 coordinate axes. If labels are provided, axes
        are shown in the figure. By default, no axes are shown.
    axes_kwargs : dict, optional
        Pass these additional arguments to the ``mlab.axes``
        constructor.
    cmap : optional
        Type of colormap to use for the
        `mayavi.modules.image_plane_widget.ImagePlaneWidget`'s.
    clim : sequence of 2 floats, optional
        Manually set the colormap's minimum and maximum to these values.
        For ``None``, the data min/max are used.
    cbar : bool, optional
        If ``True``, display a colorbar. Default: ``False``
    cbar_kwargs : dict, optional
        Pass these additional arguments to the
        `mayavi.tools.decorations.colorbar` constructor.
    outline : bool, optional
        If ``True``, display an outline. Default: ``False``
    outline_kwargs : dict, optional
        Pass these additional arguments to the ``mlab.outline``
        constructor.
    force_show : bool, optional
        If ``True``, force the figure to be drawn immediately. Otherwise,
        defer showing until `mayavi.tools.show.show` is called.

    Returns
    -------
    scene : `mayavi.core.scene.Scene`
        Handle to the figure that is used for display of this
        plot.

    References
    ----------
    .. _mayavi: http://code.enthought.com/projects/mayavi/
    """
    # Takes a while to import, therefore do it lazily
    from mayavi import mlab

    # Handle elem
    assert isinstance(elem, odl.discr.lp_discr.DiscreteLpElement)
    assert elem.ndim == 3

    # Handle planes
    assert all(plane in ('x', 'y', 'z') for plane in planes)
    n_planes = len(planes)

    # Handle plane_indcs
    if plane_indcs is not None:
        plane_indcs = tuple(safe_int_conv(i) for i in plane_indcs)
    else:
        plane_indcs = tuple(n // 2 for n in elem.shape)[:n_planes]
    assert len(plane_indcs) == n_planes

    # Handle other params (we don't want to fail when plot is already
    # half done)
    clim = kwargs.pop('clim', None)
    if clim is not None:
        cmin, cmax = map(float, clim)
        assert cmin <= cmax

    cbar = bool(kwargs.pop('cbar', False))
    cbar_kwargs = dict(kwargs.pop('cbar_kwargs', {}))
    outline = bool(kwargs.pop('outline', False))
    outline_kwargs = dict(kwargs.pop('outline_kwargs', {}))
    force_show = kwargs.pop('force_show', False)
    force_show = bool(force_show)
    axis_labels = kwargs.pop('axis_labels', ())
    axes_kwargs = dict(kwargs.pop('axes_kwargs', {}))
    assert len(axis_labels) in (0, 3)

    elem_arr = np.transpose(elem.asarray(), (2, 0, 1))
    data_src = mlab.pipeline.scalar_field(elem_arr)

    for i in range(n_planes):
        plane_idx = plane_indcs[i]
        # Check validity of indices, again to avoid drawing half-done figures
        if plane_idx < 0:
            plane_idx = plane_idx + elem.shape[i]
        assert 0 <= plane_idx < elem.shape[i]

        orientation = planes[i] + '_axes'
        mlab.pipeline.image_plane_widget(data_src,
                                         plane_orientation=orientation,
                                         slice_index=plane_idx,
                                         colormap=cmap)

    # Need colorbar object to set clim values. Better solution?
    colorbar = mlab.colorbar(**cbar_kwargs)
    if not cbar:
        colorbar.show_scalar_bar = False
    if clim is not None:
        colorbar.lut.table_range = np.array([cmin, cmax])
    if outline:
        outline_kwargs.pop('figure', None)
        mlab.outline(**outline_kwargs)
    if axis_labels:
        axes_kwargs.pop('figure', None)
        data_ranges = np.zeros(6)
        data_ranges[::2] = elem.space.min_pt
        data_ranges[1::2] = elem.space.max_pt
        ranges = axes_kwargs.pop('ranges', data_ranges)
        mlab.axes(xlabel=axis_labels[0],
                  ylabel=axis_labels[1],
                  zlabel=axis_labels[2],
                  ranges=ranges,
                  **axes_kwargs)
    if force_show:
        mlab.show()

    return mlab.gcf()


def update_planes_scene(scene, new_data, **kwargs):
    """Update the data in a scene created by `show_planes`.

    Parameters
    ----------
    scene : `mayavi.core.scene.Scene`
        The scene to be updated. It must have an
        `mayavi.sources.array_source.ArraySource` whose ``scalar_data``
        is shape- and dtype-compatible with ``data``.
    new_data : `numpy.ndarray`
        The new data to be used in the ``scene``.
    clim : sequence of 2 floats, optional
        Manually set the colormap's minimum and maximum to these values.
        For ``None``, the data min/max are used.
    cbar : bool, optional
        If ``True``, display a colorbar. Default: ``False``
    cbar_kwargs : dict, optional
        Pass these additional arguments to the
        `mayavi.tools.decorations.colorbar` constructor.
    """
    # TODO: this doesn't work. Find a way to update figures in a non-blocking
    # way!
    assert isinstance(scene, mayavi.core.scene.Scene)
    assert len(scene.children) == 1
    assert isinstance(scene.children[0],
                      mayavi.sources.array_source.ArraySource)
    array_source = scene.children[0]
    old_data = array_source.scalar_data
    new_data = np.transpose(np.asarray(new_data), (2, 0, 1))
    assert new_data.shape == old_data.shape
    assert new_data.dtype == old_data.dtype
    # TODO: perhaps rather assign values? In that case, use can_cast
    array_source.scalar_data = new_data
    array_source.update()
    clim = kwargs.pop('clim', None)
    if clim is not None:
        cmin, cmax = map(float, clim)
        assert cmin <= cmax

    cbar = bool(kwargs.pop('cbar', False))
    cbar_kwargs = dict(kwargs.pop('cbar_kwargs', {}))
    colorbar = mlab.colorbar(**cbar_kwargs)
    if not cbar:
        colorbar.show_scalar_bar = False
    if clim is not None:
        assert colorbar is not None
        colorbar.lut.table_range = np.array([cmin, cmax])



# %% Testing ground

from mayavi import mlab
space = odl.uniform_discr([0] * 3, [1] * 3, [200] * 3)
phantom = odl.phantom.shepp_logan(space, modified=True)
scene = show_planes(phantom, planes=['x', 'y'], cbar=True, clim=[-1, 1],
                    axis_labels=['x', 'y', 'z'])
array_source = scene.children[0]
import time
for i in range(20):
    time.sleep(0.2)
    new_data = phantom * np.sin(i * np.pi / 5)
    new_data = np.transpose(np.asarray(new_data), (2, 0, 1))
    scene.children[0].scalar_data = new_data
