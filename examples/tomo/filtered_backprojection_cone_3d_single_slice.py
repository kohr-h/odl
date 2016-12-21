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

"""
Example using a filtered back-projection (FBP) in cone-beam 3d using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complicated
methods.
"""

import numpy as np
import odl
from odl.tomo.util.utility import axis_rotation_matrix


# --- Set-up geometry of the problem --- #


full_reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[500, 500, 500],
    dtype='float32')

angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
detector_partition = odl.uniform_partition([-40, -40], [40, 40], [500, 500])
full_geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40,
    axis=[0, 0, 1])


# --- Create ray transform and filtering operator --- #


full_ray_trafo = odl.tomo.RayTransform(full_reco_space, full_geometry,
                                       impl='astra_cuda')
filter_op = odl.tomo.analytic.filtered_back_projection.fbp_filter_op(
    full_ray_trafo, filter_type='Shepp-Logan', frequency_scaling=0.8)


# %% --- Create raw and filtered data --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(full_reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = full_ray_trafo(phantom)
filtered_data = filter_op(proj_data)


# %% --- Define slice and reconstruct it --- #

def rotation_matrix(from_vec, to_vec):
    from_vec = np.array(from_vec, dtype=float, copy=True)
    from_vec /= np.linalg.norm(from_vec)
    to_vec = np.array(to_vec, dtype=float, copy=True)
    to_vec /= np.linalg.norm(to_vec)
    rot_axis = np.cross(from_vec, to_vec)
    cross_norm = np.linalg.norm(rot_axis)
    if cross_norm < 1e-6:
        return np.eye(len(from_vec))
    else:
        rot_axis /= cross_norm
        rot_angle = np.arccos(np.dot(from_vec, to_vec))
        return axis_rotation_matrix(rot_axis, rot_angle)


# Define the slice by a normal and a shift
slice_normal = np.array([0, 0, 1], dtype=float)
slice_shift = np.array([0, 0, 0], dtype=float)

# Compute geometric quantities derived from the slice definition
rot_z_axis_to_normal = rotation_matrix(from_vec=[0, 0, 1], to_vec=slice_normal)
geometry_axis = rot_z_axis_to_normal.T.dot([1, 0, 0])

# Reconstruction space of the slice
non_shifted_min_pt = np.append(full_reco_space.min_pt[:2],
                               -full_reco_space.cell_sides[2] / 2)
non_shifted_max_pt = np.append(full_reco_space.max_pt[:2],
                               full_reco_space.cell_sides[2] / 2)

slc_min_pt = rot_z_axis_to_normal.T.dot(slice_shift) + non_shifted_min_pt
slc_max_pt = rot_z_axis_to_normal.T.dot(slice_shift) + non_shifted_max_pt
slice_reco_space = odl.uniform_discr(
    min_pt=slc_min_pt, max_pt=slc_max_pt, shape=[500, 500, 1],
    dtype='float32')

# Rotated geometry
rot_det_init_axes = [rot_z_axis_to_normal.T.dot(a)
                     for a in full_geometry.det_init_axes]
rot_src_to_det_init = rot_z_axis_to_normal.T.dot(full_geometry.src_to_det_init)
rot_geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40,
    axis=rot_z_axis_to_normal.T.dot([0, 0, 1]),
    src_to_det_init=rot_src_to_det_init,
    det_init_axes=rot_det_init_axes)

# Ray transforms in the rotated geometry
rot_ray_trafo = odl.tomo.RayTransform(full_reco_space, rot_geometry,
                                      impl='astra_cuda')
rotated_filtered_data = rot_ray_trafo.range.element(filtered_data)

slice_ray_trafo = odl.tomo.RayTransform(slice_reco_space, rot_geometry,
                                        impl='astra_cuda')

# Compute the reconstruction and show it
fbp_reconstruction = slice_ray_trafo.adjoint(rotated_filtered_data)
fig_title = 'FBP, slice normal = {}, slice shift = {}'.format(slice_normal,
                                                              slice_shift)
fbp_reconstruction.show(title=fig_title)
