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

import numpy as np

from odl.discr.partition import RectPartition
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid)


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

    if not (is_valid_input_array(points, partition.ndim) or
            is_valid_input_meshgrid(points, partition.ndim)):
        txt_1d = ' or (n,)' if partition.ndim == 1 else ''
        raise TypeError('`points` {!r} not a valid input. Expected '
                        'a `numpy.ndarray` with shape ({ndim}, n){} or a '
                        'length-{ndim} meshgrid tuple.'
                        ''.format(points, txt_1d, ndim=partition.ndim))

    pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode

    outside_left, outside_right, inside = [], [], []
    for axis in range(partition.ndim):
        if partition.ndim == 1 and isinstance(points, np.ndarray):
            pts = points
        else:
            pts = points[axis]
        xmin = partition.min_pt[axis]
        xmax = partition.max_pt[axis]
        grid_pts = partition.grid.coord_vectors[axis]

        # Store index arrays downstream reuse
        # TODO: perhaps we need the boolean arrays instead?
        out_left = (pts <= xmin)
        out_right = (pts >= xmax)
        outside_left.append(np.where(out_left)[0])
        outside_right.append(np.where(out_right)[0])
        inside.append(np.where(~(out_left | out_right))[0])

        # Remap the points in-place
        if pad_mode == 'constant':
            pts[out_left] = xmin
            pts[out_right] = xmax
        elif pad_mode == 'periodic':
            # pts <-- xmin + (pts - xmin) mod (xmax - xmin)
            np.mod(pts - xmin, xmax - xmin, out=pts)
            pts += xmin
        elif pad_mode == 'symmetric':
            #         { y,             if y <= xmax
            # pts <-- {
            #         { 2 * xmax - y,  if y > xmax
            #
            # where y = xmin + (pts - xmin) mod (2 * (xmax - xmin))
            np.mod(pts - xmin, 2 * (xmax - xmin), out=pts)
            pts += xmin
            right_half = (pts > xmax)
            pts[right_half] = 2 * xmax - pts[right_half]
        elif pad_mode in ('order0', 'order1'):
            pts[out_left] = grid_pts[0]
            pts[out_right] = grid_pts[-1]
        else:
            raise ValueError("invalid `pad_mode` '{}'".format(pad_mode_in))

    return outside_left, outside_right, inside
