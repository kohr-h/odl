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

"""Single-slice FBP in 3D cone beam geometry for tomopackets.

This example computes the FBP in a single slice by pre-computing the
filtered data and then back-projecting to an arbitrary slice.
The slice is defined by a slice visualizer using a set of matrix entries
for the transformation from slice coordinates to world coordinates in
the following way::

    [x]     [a  d  g]     [u]
    [y]  =  [b  e  h]  *  [v]
    [z]     [c  f  i]     [1]
    [1]     [0  0  1]

The ``set_callback`` method of a ``tomop.server`` expects a callback as
prototyped below.
"""

import numpy as np
import odl
from odl.tomo.util import rotation_matrix_from_to


def callback_proto(slice_spec):
    """Prototypical callback for ``tomop.server``.

    Parameters
    ----------
    slice_spec : `numpy.ndarray`, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from slice coordinates to world coordinates.
        See Notes for details.

    Returns
    -------
    shape_array : `numpy_ndarray`, ``dtype='int32', shape=(2,)``
        Number of sent values per axis, usually the shape of the full slice.
    values : `numpy.ndarray`, ``dtype='uint32', shape=shape_array``
        Values to send. The size is determined by ``shape_array``.

    Notes
    -----
    The transformation from normalized slice coordinates to normalized
    world coordinates is as follows:

        .. math::
            :nowrap:

            \\begin{equation*}

            \\begin{pmatrix}
              x \\\\
              y \\\\
              z \\\\
              1
            \end{pmatrix}
            %
            =
            %
            \\begin{pmatrix}
              a & d & g \\\\
              b & e & h \\\\
              c & f & i \\\\
              0 & 0 & 1
            \end{pmatrix}
            %
            \\begin{pmatrix}
              u \\\\
              v \\\\
              1
            \end{pmatrix}

            \end{equation*}

    In the following we define the exact meaning of that transformation.
    Throughout the explanation we use :math:`u, v` as slice coordinates
    and as subscripts of quantities defined in the 2D slice coordinate
    system.
    Likewise, :math:`x, y, z` stand for 3D world coordinates or quantities
    defined there.

    Let :math:`N = (N_x, N_y, N_z)`, :math:`U = (U_x, U_y, U_z)` and
    :math:`V = (V_x, V_y, V_z)` be 3D unit vectors that are perpendicular
    to each other. Let further :math:`T = (T_x, T_y, T_z)` be an arbitrary
    vector. :math:`N` will be the normal vector of the slice, :math:`T`
    a translation and :math:`U` and :math:`V` the vectors spanning the
    slice (when shifted back by :math:`-T`).

    Consider now a slice defined by an origin :math:`O = (O_x, O_y, O_z)`
    and side lengths :math:`l = (l_u, l_v) > (0, 0)`:

        .. math::

            S = O + \{ u\, l_u\, U + v\, l_v\, V\, |\, 0 \leq u, v \leq 1 \}

    Then the pairs :math:`(u, v)` are called **normalized slice
    coordinates**.

    Similarly, consider a 3D cube of side lengths :math:`W = (W_x, W_y, W_z)`
    and origin :math:`P = (P_x, P_y, P_z)`, i.e.,

        .. math::

            C = p + \{ (x\, W_x, y\, W_y, z\, W_z)\, |\,
                      0 \leq x, y, z \leq 1 \}.

    Here, we refer to :math:`(x, y, z)` as **normalized volume
    coordinates**.

    Thus, the transformation by the coefficients as passed to the callback
    is defined in terms of these normalized coordinates.
    """
    return (np.array([2, 2], dtype='int32'),
            np.array([1, 2, 3, 4], dtype='uint32'))


def slice_spec_to_geom_matrix(slice_spec, vol_extent=(1.0, 1.0, 1.0)):
    """Convert a slice specification vector to a 3x4 matrix.

    The return value of this function is to be fed into the
    ``geometry.frommatrix()`` constructor.

    Parameters
    ----------
    slice_spec : array-like, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from slice coordinates to world coordinates.
    vol_extent : sequence of 3 real numbers, optional
        Side lengths of the volume. These numbers determine how to translate
        the normalized shift ``(g, h, i)`` to the actual shift of the
        slice in world coordinates.

    Returns
    -------
    matrix : `numpy.ndarray`, shape ``(3, 4)``
        Matrix that translates and rotates the world system such that the
        slice unit vectors ``(a, b, c)`` and ``(d, e, f)`` are transformed
        to ``(1, 0, 0)`` and ``(0, 1, 0)``, respectively, and the slice
        becomes centered around ``(0, 0, 0)``.

    Notes
    -----
    The transformation from normalized slice coordinates to normalized
    world coordinates is as follows:

        .. math::
            :nowrap:

            \\begin{equation*}

            \\begin{pmatrix}
              x \\\\
              y \\\\
              z \\\\
              1
            \end{pmatrix}
            %
            =
            %
            \\begin{pmatrix}
              a & d & g \\\\
              b & e & h \\\\
              c & f & i \\\\
              0 & 0 & 1
            \end{pmatrix}
            %
            \\begin{pmatrix}
              u \\\\
              v \\\\
              1
            \end{pmatrix}

            \end{equation*}

    Thus, the vectors :math:`U = (a, b, c)` and :math:`V = (d, e, f)`
    are the coordinate vectors of the local slice system, and
    ``(g, h, i)`` is the normalized shift of the slice center relative to
    the volume center.
    """
    a, b, c, d, e, f, g, h, i = np.asarray(slice_spec, dtype=float)
    vec_u = np.array([a, b, c])
    vec_v = np.array([d, e, f])
    extent = np.asarray(lengths, dtype=float).reshape((3,))
    transl = np.array([g, h, i]) * extent

    # TODO: do stuff with the vectors


# %% Setup code for the full problem


vol_min_pt = np.array([-20, -20, -20], dtype=float)
vol_max_pt = np.array([20, 20, 20], dtype=float)
vol_shape = (256, 256, 256)

# Create reconstruction space and geometry for the full 3D problem
full_reco_space = odl.uniform_discr(vol_min_pt, vol_max_pt, vol_shape,
                                    dtype='float32')

angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
detector_partition = odl.uniform_partition([-40, -40], [40, 40], [500, 500])
full_geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40,
    axis=[0, 0, 1])

# Create ray transform and filtering operator
full_ray_trafo = odl.tomo.RayTransform(full_reco_space, full_geometry,
                                       impl='astra_cuda')
filter_op = odl.tomo.analytic.filtered_back_projection.fbp_filter_op(
    full_ray_trafo, filter_type='Shepp-Logan', frequency_scaling=0.8)


# %% Create raw and filtered data


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(full_reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = full_ray_trafo(phantom)
filtered_data = filter_op(proj_data)


# %% Define slice and reconstruct it


# Define the slice by a normal and a shift in the following way:
#   plane(t, n) = t + perp(n)
# Here, `t` is the (absolute) translation vector, `n` the normal vector and
# `perp(n)` the plane perpendicular to `n`.
# The finite-extent version of this is
#   slice(t, n) = {u in plane(t, n) | T(t, n)u in R x {0} },
# where `R` is a 2D rectangle,
#   T(t, n)u = M(n)^(-1)(u - t),
# is an affine transformation and `M(n)` is the matrix rotating
# `(0, 0, 1)` to `n`.
slice_normal = np.array([0, 1, 0], dtype=float)
slice_shift = np.array([5, 0, 5], dtype=float)

# Compute M(n) and M(n)^(-1)t
rot_z_axis_to_normal = rotation_matrix_from_to(from_vec=[0, 0, 1],
                                               to_vec=slice_normal)
slc_shift_rot = rot_z_axis_to_normal.T.dot(slice_shift)

# Construct the slice space, which is a discretized version of
# `R = T(t, n) slice(t, n)`.
# TODO: let user specify extent and shape
slc_min_pt = full_reco_space.min_pt.copy()
slc_min_pt[2] = -full_reco_space.cell_sides[2] / 2
slc_max_pt = full_reco_space.max_pt.copy()
slc_max_pt[2] = full_reco_space.cell_sides[2] / 2
slice_reco_space = odl.uniform_discr(
    min_pt=slc_min_pt, max_pt=slc_max_pt, shape=[500, 500, 1],
    dtype='float32',
    axis_labels=['$x^*$', '$y^*$', '$z^*$'])

# The matrix given to geometry.frommatrix is built up as `[M  v]`, where
# `M` is the transformation matrix to be applied to the geometry-defining
# vectors, and `v` is a translation to be applied *after* applying `M`.
# Thus, in our case we need to set `M = M(n)^(-1)` and `v = -M(n)^(-1)t`
# in order to effectively apply `T(t, n)` to the default geometry.
# TODO: adapt this in case the original geometry is not a standard one.
matrix = np.hstack([rot_z_axis_to_normal.T, -slc_shift_rot[:, None]])

# Construct the geometry transformed by `T(t, n)`
trafo_geometry = odl.tomo.CircularConeFlatGeometry.frommatrix(
    angle_partition, detector_partition, src_radius=40, det_radius=40,
    init_matrix=matrix)

# Construct ray transform for slice space and transformed geometry,
# and cast the filtered data
ray_trafo_slice = odl.tomo.RayTransform(slice_reco_space, trafo_geometry,
                                        impl='astra_cuda')
trafo_filtered_data = ray_trafo_slice.range.element(filtered_data)

# Compute the slice reconstruction and show it
fbp_reconstruction = ray_trafo_slice.adjoint(trafo_filtered_data)
fig_title = 'FBP, slice normal = {}, slice shift = {}'.format(slice_normal,
                                                              slice_shift)
fbp_reconstruction.show(title=fig_title)
print('Axes in the image:')
print('x* = ', np.array2string(rot_z_axis_to_normal.T[0], precision=2))
print('y* = ', np.array2string(rot_z_axis_to_normal.T[1], precision=2))
