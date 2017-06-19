"""Cone beam 2D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

The back-projected data should be a blurry version of the phantom, with
all features in the correct positions, not flipped or rotated.

This example is best run in Spyder section-by-section (CTRL-Enter).
"""

# %% Set up the things that never change

import numpy as np
import odl

# Set back-end here
impl = 'astra_cuda'
# Set a volume shift. This should not have any influence on the back-projected
# data.
shift = (0, 25)

img_shape = (100, 150)
img_max_pt = np.array(img_shape, dtype=float) / 2
img_min_pt = -img_max_pt
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

# Take 1 degree increments, full angular range
grid = odl.RectGrid(np.linspace(0, 2 * np.pi, 360, endpoint=False))
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
src_radius = 500
det_radius = 1000
fan_angle = np.arctan(img_max_pt[1] / src_radius)
det_size = np.floor(2 * (src_radius + det_radius) * np.sin(fan_angle))
det_shape = int(det_size)
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)


# %% Test back-projection


geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius, det_radius)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)
back_proj = ray_trafo.adjoint(proj_data)
back_proj.show('Back-projection')
phantom.show('Phantom')
