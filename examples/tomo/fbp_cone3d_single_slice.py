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

"""Single-slice FBP in 3D cone beam geometry.

This example computes the FBP in a single slice by pre-computing the
filtered data and then back-projecting to an arbitrary slice defined by a
normal vector and and an absolute shift.
The whole system is rotated in such a way that the slice becomes
perpendicular to the third coordinate axis. In this situation, the
standard BP can be used with a volume of shape ``(N, M, 1)``.
"""

# %% Setup code for the full problem

import numpy as np
import odl
from odl.tomo.util import rotation_matrix_from_to


# Create reconstruction space and geometry for the full 3D problem
full_reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[500, 500, 500],
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
slice_shift = np.array([5, 10, 5], dtype=float)

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
