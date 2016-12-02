# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Functions and classes for n-dimensional extra- and interpolation."""

from itertools import product
import numpy as np

from odl.discr.grid import sparse_meshgrid
from odl.discr.partition import RectPartition
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_meshgrid)


__all__ = ()


def remap_points(points, partition, pad_mode):
    """Map points to the partition domain according to ``pad_mode``.

    Parameters
    ----------
    points : `numpy.ndarray` or `meshgrid` sequence
        Points that are to be mapped. The number of entries along the
        first axis (for array) or the length of the sequence (for meshgrid)
        must be equal to ``partition.ndim``.
        The points are updated in-place.
    partition : `RectPartition`
        Domain partition defining the valid domain and a sampling grid.
    pad_mode : str
        Padding mode defining how the points are mapped back.
        Possible values:

        ``'constant', 'periodic', 'symmetric', 'order0', 'order1'``

    Returns
    -------
    outside_left : tuple of `numpy.ndarray`'s
        Sequence of index arrays for the point coordinates that are equal
        to or below ``partition.min_pt`` in the respective axes.
    outside_right : tuple of `numpy.ndarray`'s
        Sequence of index arrays for the point coordinates that are equal
        to or above ``partition.max_pt`` in the respective axes.
    inside : tuple of `numpy.ndarray`'s
        Sequence of index arrays for the point coordinates that are
        strictly between ``partition.min_pt`` and ``partition.max_pt``
        in the respective axes.
    """
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` must be a `RectPartition` instance, '
                        'got {!r}'.format(partition))

    if is_valid_input_meshgrid(points, partition.ndim):
        remapped = tuple(p.copy().ravel() for p in points)
    elif is_valid_input_array(points, partition.ndim):
        remapped = np.copy(points)
        if remapped.ndim == 1:
            # Make extra axis 0 so remapped[0] gives whole array
            remapped = remapped[None, :]
    else:
        txt_1d = ' or (n,)' if partition.ndim == 1 else ''
        raise TypeError('`points` {!r} not a valid input. Expected '
                        'a `numpy.ndarray` with shape ({ndim}, n){} or a '
                        'length-{ndim} meshgrid tuple.'
                        ''.format(points, txt_1d, ndim=partition.ndim))

    pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
    squeeze_out = (partition.ndim == 1 and isinstance(points, np.ndarray))

    outside_left, outside_right, inside = [], [], []
    for axis, new_pts in enumerate(remapped):
        xmin = partition.min_pt[axis]
        xmax = partition.max_pt[axis]
        grid_pts = partition.grid.coord_vectors[axis]

        # Store index arrays downstream reuse
        # TODO: perhaps we need the boolean arrays instead?
        out_left = (new_pts <= xmin)
        out_right = (new_pts >= xmax)
        outside_left.append(np.where(out_left)[0])
        outside_right.append(np.where(out_right)[0])
        inside.append(np.where(~(out_left | out_right))[0])

        # Remap the points to the domain
        if pad_mode == 'constant':
            new_pts[out_left] = xmin
            new_pts[out_right] = xmax
        elif pad_mode == 'periodic':
            # out <-- xmin + (x - xmin) mod (xmax - xmin)
            np.mod(new_pts - xmin, xmax - xmin, out=new_pts)
            new_pts += xmin
        elif pad_mode == 'symmetric':
            #         { y,             if y <= xmax
            # out <-- {
            #         { 2 * xmax - y,  if y > xmax
            #
            # where y = xmin + (x - xmin) mod (2 * (xmax - xmin))
            np.mod(new_pts - xmin, 2 * (xmax - xmin), out=new_pts)
            new_pts += xmin
            right_half = (new_pts > xmax)
            new_pts[right_half] = 2 * xmax - new_pts[right_half]
        elif pad_mode in ('order0', 'order1'):
            new_pts[out_left] = grid_pts[0]
            new_pts[out_right] = grid_pts[-1]
        else:
            raise ValueError("invalid `pad_mode` '{}'".format(pad_mode_in))

    if squeeze_out:
        remapped = remapped.squeeze()
    return remapped, outside_left, outside_right, inside


def indices_weights(points, partition, pad_mode):
    """Return indices and weights for interpolating at ``points``.

    The weights are computed with respect to ``partition.grid``.
    All points are assumed to lie in ``partition.set``, which is
    **not** checked for efficiency reasons. To ensure a correct point
    layout, use `remap_points`.

    Parameters
    ----------
    points : `numpy.ndarray` or `meshgrid` sequence
        Points whose indices and weights are to be computed. The number of
        entries along the first axis (for array) or the length of the
        sequence (for meshgrid) must be equal to ``partition.ndim``.
    partition : `RectPartition`
        Domain partition defining the valid domain and a sampling grid.
    pad_mode : str
        Padding mode defining how the points close to the boundary are
        to be handled.
        Possible values:

        ``'constant', 'periodic', 'symmetric', 'order0', 'order1'``

    Returns
    -------
    indices : tuple of `numpy.ndarray`
        Sequence of index arrays specifying the closest *left* neighbors
        of ``points`` in each axis. Due to grid extension, indices
        ``-1`` and ``n`` can occur, where ``n`` is the number of grid
        points in a given axis.
    weights : tuple of `numpy.ndarray`
        Sequence of arrays with values 0.0 <= weights <= 1 specifying
        the interpolation weight of the *right* neighbors of ``points``
        in each axis.

    Notes
    -----
    Usually, the coordinates of ``partition.min_pt`` and
    ``partition.grid.min_pt`` are different, i.e. there are parts of
    the partitioned set that only have neighboring grid points on one
    side. To handle those points, the grid is extended by 1 point at
    all boundaries according to the padding mode, such that 2-sided
    interpolation is possible also in those cases.
    """
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` must be a `RectPartition` instance, '
                        'got {!r}'.format(partition))

    if not (is_valid_input_array(points, partition.ndim) or
            is_valid_input_meshgrid(points, partition.ndim)):
        txt_1d = ' or (n,)' if partition.ndim == 1 else ''
        raise TypeError('`points` {!r} not a valid input. Expected '
                        'a `numpy.ndarray` with shape ({ndim}, n){} or a '
                        'length-{ndim} meshgrid tuple.'
                        ''.format(points, txt_1d, ndim=partition.ndim))

    pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode

    indices, weights = [], []
    for axis in range(partition.ndim):
        if partition.ndim == 1 and isinstance(points, np.ndarray):
            pts = points.ravel()
        else:
            pts = points[axis].ravel()
        xmin = partition.min_pt[axis]
        xmax = partition.max_pt[axis]
        grid_pts = partition.grid.coord_vectors[axis]

        if pad_mode in ('constant', 'order0', 'order1'):
            xleft = xmin
            xright = xmax
        elif pad_mode == 'periodic':
            xleft = xmin - (xmax - grid_pts[-1])
            xright = xmax + (grid_pts[0] - xmin)
        elif pad_mode == 'symmetric':
            xleft = 2 * xmin - grid_pts[0]
            xright = 2 * xmax - grid_pts[-1]
        else:
            raise ValueError("invalid `pad_mode` '{}'".format(pad_mode_in))

        extended_gridpts = np.concatenate([[xleft], grid_pts, [xright]])
        n_ext = len(extended_gridpts)
        indcs = np.searchsorted(extended_gridpts, pts, side='right') - 1
        indcs[indcs == n_ext - 1] = n_ext - 2
        xi = extended_gridpts[indcs]
        xip1 = extended_gridpts[indcs + 1]
        wgts = (pts - xi) / (xip1 - xi)

        indices.append(indcs - 1)  # Compensate for added element left
        weights.append(wgts)

    return indices, weights


def interp_linear(fvals, indices, weights, bounds_check=False, grid_shape=None,
                  pad_mode='constant', pad_const=0, out=None):
    """Evaluate linear interpolation using function values ``fvals``.

    Parameters
    ----------
    fvals : `array-like`
        The function values to interpolate.
    indices, weights : sequence of `np.ndarray`
        Arrays of the indices and weights of the points relative to
        the grid where the function values are defined. Usually the
        result of the `indices_weights` helper function.
    bounds_check : bool, optional
        If ``True``, check for all axes if there are ``indices``
        below 0 (out-of-bounds left) or above ``n - 2``, where ``n`` is
        the entry of ``grid_shape`` in the corresponding axis.
        For such indices, extra values are used according to ``pad_mode``.
        Apparently, ``grid_shape`` is required in this case.

        For ``False``, no such check is performed. If in this case, there
        are out-of-bounds indices, they are wrapped (negative) and/or
        lead to an ``IndexError``.
    grid_shape : sequence of ints, optional
        Shape of the interpolation grid. This parameter is required if
        ``bounds_check=True`` and not used otherwise.
    pad_mode : str, optional
        Padding mode defining which function values are used to extend
        ``fvals`` for points close to the boundary.
        Possible values:

        ``'constant', 'periodic', 'symmetric', 'order0', 'order1'``

    pad_const : scalar, optional
        Constant value for extension of ``fvals`` in the case of
        ``pad_mode='constant'``. Has no effect otherwise.
    out : `numpy.ndarray`, optional
        Array to which the interpolated values should be written. Its
        size must match the number of interpolation points, and its data
        type must be compatible with (i.e. safely castable from) the one of
        ``fvals``.

    Returns
    -------
    out : `numpy.ndarray`
        The array of interpolated values. If ``out`` was given, the
        returned object is a reference to it.
    """
    fvals = np.asarray(fvals)
    if len(indices) != fvals.ndim:
        raise ValueError('lenght of `indices` does not match number of '
                         'dimensions in `fvals` ({} != {})'
                         ''.format(len(indices), fvals.ndim))
    if len(weights) != fvals.ndim:
        raise ValueError('lenght of `weights` does not match number of '
                         'dimensions in `fvals` ({} != {})'
                         ''.format(len(weights), fvals.ndim))
    if bounds_check:
        if grid_shape is None:
            raise TypeError('`grid_shape` cannot be `None` for for '
                            '`bounds_check=True`')
        elif len(grid_shape) != fvals.ndim:
            raise ValueError('lenght of `grid_shape` does not match number of '
                             'dimensions in `fvals` ({} != {})'
                             ''.format(len(grid_shape), fvals.ndim))

    out_shape = out_shape_from_meshgrid(weights)
    out_dtype = fvals.dtype
    if out is None:
        out = np.empty(out_shape, dtype=out_dtype)
    else:
        if out.shape != out_shape:
            raise ValueError('`out` has wrong shape, expected {}, got {}'
                             ''.format(out_shape, out.shape))
        if not np.can_cast(fvals.dtype, out.dtype):
            raise ValueError('cannot safely cast from data type {} of '
                             '`fvals` to out.dtype {}'
                             ''.format(fvals.dtype, out.dtype))

    # TODO: this must go inside the loop
    if pad_mode == 'constant':
        fleft = fright = pad_const
    elif pad_mode == 'periodic':
        fleft = fvals[-1]
        fright = fvals[0]
    elif pad_mode in ('symmetric', 'order0', 'order1'):
        fleft = fvals[0]
        fright = fvals[-1]
    else:
        raise RuntimeError('bad mode')

    edge_indices = [[i, i + 1] for i in indices]
    weights = sparse_meshgrid(*weights)
    low_weights = [1.0 - w for w in weights]
    high_weights = weights
    axis_order = np.argsort([np.prod(w.strides) for w in weights])[::-1]

    # Iterate over all possible combinations of [i, i+1] for each
    # axis (i.e. corners), resulting in a loop of length 2**ndim
    for corner, edges in zip(product(*([['lo', 'hi']] * len(indices))),
                             product(*edge_indices)):
        # Determine the weights for this particular corner setup
        weight = 1.0
        for axis in axis_order:
            # We don't multiply in-place to exploit the cheap operations
            # in the beginning: sizes grow gradually as following:
            # (n, 1, 1, ...) -> (n, m, 1, ...) -> ...
            # Hence, it is faster to build up the weight array instead
            # of doing full-size operations from the beginning.
            #
            # The ordering of axes is chosen optimally with respect to the
            # array strides (largest first)
            if corner[axis] == 'lo':
                weight = weight * low_weights[axis]
            elif corner[axis] == 'hi':
                weight = weight * high_weights[axis]
            else:
                raise RuntimeError('invalid corner')

        # TODO: here somewhere, the OOB indices must be handled. Instead of
        # fvals[edges], we need to take the `fleft` and `fright` values
        out += np.asarray(fvals[edges]) * weight

    return np.array(out, copy=False, ndmin=1)
