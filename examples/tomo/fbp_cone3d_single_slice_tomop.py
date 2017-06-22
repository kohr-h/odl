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
import time

import sys
tomop_path = '/export/scratch1/kohr/git/slicevis/ext/tomopackets/python'
if tomop_path not in sys.path:
    sys.path.append(tomop_path)
import tomop

DEBUG = True


def callback_proto(slice_spec):
    """Prototypical callback for ``tomop.server``.

    Parameters
    ----------
    slice_spec : `numpy.ndarray`, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from normalized slice coordinates to
        normalized world coordinates. See Notes for details.

    Returns
    -------
    shape_array : `numpy_ndarray`, ``dtype='int32', shape=(2,)``
        Number of sent values per axis, usually the shape of the full slice.
    values : `numpy.ndarray`, ``dtype='uint32', shape=(np.prod(shape_array),)``
        Flattened array of reconstructed values to send. The order of
        flattening is row-major (``'C'``).

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
    to each other. Let further :math:`O = (O_x, O_y, O_z)` be an arbitrary
    vector. :math:`N` will be the normal vector of the slice, :math:`O`
    the origin of the slice and :math:`U` and :math:`V` the vectors spanning
    the slice (when shifted back by :math:`-O`).

    Consider now a slice defined by origin :math:`O` and side lengths
    :math:`l = (l_u, l_v) > (0, 0)`:

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


def callback_null(slice_spec):
    return (np.array([0, 0], dtype='int32'), np.array([], dtype='uint32'))


def slice_spec_to_rot_matrix(slice_spec):
    """Convert a slice specification vector to a 3x3 rotation matrix.

    Parameters
    ----------
    slice_spec : array-like, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from slice coordinates to world coordinates.

    Returns
    -------
    matrix : `numpy.ndarray`, shape ``(3, 3)``
        Matrix that rotates the world system such that the slice vectors
        ``(a, b, c)`` and ``(d, e, f)`` are after normalization transformed
        to ``(1, 0, 0)`` and ``(0, 1, 0)``, respectively.

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
    span the local slice coordinate system.
    """
    a, b, c, d, e, f, g, h, i = np.asarray(slice_spec, dtype=float)

    vec_u = np.array([a, b, c])
    vec_u_norm = np.linalg.norm(vec_u)
    if vec_u_norm == 0:
        raise ValueError('`[a, b, c]` vector is zero')
    else:
        vec_u /= vec_u_norm

    vec_v = np.array([d, e, f])
    vec_v_norm = np.linalg.norm(vec_v)
    if vec_v_norm == 0:
        raise ValueError('`[d, e, f]` vector is zero')
    else:
        vec_v /= vec_v_norm

    # Complete matrix to a rotation matrix
    normal = np.cross(vec_u, vec_v)
    rot_matrix = np.vstack([vec_u, vec_v, normal])
    return slicevis_to_odl_axes.dot(rot_matrix)


# %% Variables defining the problem geometry

# Volume
vol_min_pt = np.array([-20, -20, -20], dtype=float)
vol_extent = [40, 40, 40]
vol_max_pt = vol_min_pt + vol_extent
vol_shape = (256, 256, 256)
vol_half_extent = np.array(vol_extent, dtype=float) / 2

# Projection angles
num_angles = 360
min_angle = 0
max_angle = 2 * np.pi
angle_partition = odl.nonuniform_partition(
    np.linspace(min_angle, max_angle, num_angles, endpoint=False))

# Detector
det_min_pt = np.array([-40, -40], dtype=float)
det_extent = [80, 80]
det_max_pt = det_min_pt + det_extent
det_shape = (512, 512)
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

# Further geometry parameters
src_radius = 40
det_radius = 40
axis = [0, 0, 1]

# Constructor for geometries and arguments by keyword
geometry_type = odl.tomo.CircularConeFlatGeometry
geometry_kwargs_base = {
    'apart': angle_partition,
    'dpart': detector_partition,
    'src_radius': src_radius,
    'det_radius': det_radius
    }
geometry_kwargs_full = geometry_kwargs_base.copy()
geometry_kwargs_full['axis'] = axis

# Filter parameters
padding = False
filter_type = 'Shepp-Logan'
relative_freq_cutoff = 0.8

# Parameters for the slice (given in slice coordinates)
slice_min_pt = np.array([-20, -20])
slice_extent = [40, 40]
slice_max_pt = slice_min_pt + slice_extent
slice_shape = (256, 256)
min_val = 0.0
max_val = 1.0

# Convert axis conventions
slicevis_to_odl_axes = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=float)


# %% Define the full problem


# Full reconstruction space (volume) and projection geometry
reco_space_full = odl.uniform_discr(vol_min_pt, vol_max_pt, vol_shape,
                                    dtype='float32')
geometry_full = geometry_type(**geometry_kwargs_full)

# Ray transform and filtering operator
ray_trafo_full = odl.tomo.RayTransform(reco_space_full, geometry_full,
                                       impl='astra_cuda')
filter_op = odl.tomo.analytic.filtered_back_projection.fbp_filter_op(
    ray_trafo_full, padding, filter_type,
    frequency_scaling=relative_freq_cutoff)


# %% Create raw and filtered data


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space_full, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo_full(phantom)
proj_data_filtered = filter_op(proj_data)


# %% Define callback that reconstructs the slice specified by the server


def callback_fbp(slice_spec):
    """Reconstruct the slice given by ``slice_spec``.

    Parameters
    ----------
    slice_spec : `numpy.ndarray`, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from normalized slice coordinates to
        normalized world coordinates. See Notes for details.

    Returns
    -------
    shape_array : `numpy_ndarray`, ``dtype='int32', shape=(2,)``
        Number of sent values per axis. Always equal to ``slice_shape``
        from global scope.
    values : `numpy.ndarray`, ``dtype='uint32', shape=(np.prod(shape_array),)``
        Flattened array of reconstructed values to send. The order of
        flattening is row-major (``'C'``).

    Notes
    -----
    The callback uses the following variables from global scope:

        - ``slice_shape`` : Number of points per axis in the slice that is
          to be reconstructed
        - ``vol_extent`` : Physical extent of the reconstruction volume,
          used to scale the normalized translation to a physical shift
        - ``geometry_type`` : Geometry class used in this problem
        - ``proj_data_filtered`` : Pre-filtered projection data
    """
    # TODO: use logging to log events
    print('')
    print('--- New slice requested ---')
    print('')
    time_at_start = time.time()
    slice_spec = np.asarray(slice_spec, dtype=float)
    a, b, c, d, e, f, g, h, i = slice_spec

    # Just return empty stuff if shape is wrong, don't crash
    # TODO: maybe it's better to crash?
    try:
        slice_spec = slice_spec.reshape((9,))
    except ValueError as err:
        print('Malformed slice specification: expected shape (9,), got'
              'shape {}'.format(slice_spec.shape))
        return (np.array([0, 0], dtype='int32'), np.array([], dtype='uint32'))

    # Construct rotated & translated geometry as defined by slice_spec
    geom_kwargs = geometry_kwargs_base.copy()
    rot_world = slice_spec_to_rot_matrix(slice_spec)
    if DEBUG:
        print('world rotation to align slice with x-y plane:')
        print(rot_world)

    vec_u = np.array([a, b, c])
    vec_v = np.array([d, e, f])
    orig_norm = np.array([g, h, i])
    if DEBUG:
        print('U vector:', vec_u)
        print('V vector:', vec_v)
        print('origin:', orig_norm)

    # Scale by half extent since normalized sizes are between -1 and 1
    slc_pt1 = orig_norm * vol_half_extent
    slc_pt2 = (orig_norm + vec_u + vec_v) * vol_half_extent
    slc_mid_pt = (slc_pt1 + slc_pt2) / 2
    translation = reco_space_full.partition.mid_pt - slc_mid_pt
    translation_in_slc_coords = rot_world.dot(translation)
    if DEBUG:
        print('slice mid_pt:', slc_mid_pt)
        print('translation (world sys):', translation)
        print('translation (slice sys):', translation_in_slc_coords)

    init_matrix = np.hstack([rot_world, translation_in_slc_coords[:, None]])
    if DEBUG:
        print('slice geometry init_matrix:')
        print(init_matrix)
    geom_kwargs['init_matrix'] = init_matrix
    geometry_slice = geometry_type.frommatrix(**geom_kwargs)

    # Construct slice reco space with size 1 in the z axis
    slc_pt1_rot = rot_world.dot(slc_pt1)
    slc_pt2_rot = rot_world.dot(slc_pt2)
    slc_min_pt_rot = np.minimum(slc_pt1_rot, slc_pt2_rot)
    slc_max_pt_rot = np.maximum(slc_pt1_rot, slc_pt2_rot)
    slc_spc_min_pt = (list(slc_min_pt_rot)[:2] +
                      [-reco_space_full.cell_sides[2] / 2])
    slc_spc_max_pt = (list(slc_max_pt_rot)[:2] +
                      [reco_space_full.cell_sides[2] / 2])
    slc_spc_shape = list(slice_shape) + [1]
    if DEBUG:
        print('slice pt1 (slice sys):', slc_pt1_rot)
        print('slice pt2 (slice sys):', slc_pt2_rot)
        print('slice min_pt (slice sys):', slc_min_pt_rot)
        print('slice max_pt (slice sys):', slc_max_pt_rot)
        print('slice space min_pt:', slc_spc_min_pt)
        print('slice space max_pt:', slc_spc_max_pt)
        print('slice spcae shape:', slc_spc_shape)
    reco_space_slice = odl.uniform_discr(
        slc_spc_min_pt, slc_spc_max_pt, slc_spc_shape, dtype='float32',
        axis_labels=['$x^*$', '$y^*$', '$z^*$'])

    if DEBUG:
        print('slice space:')
        print(reco_space_slice)

    # Define ray trafo on the slice, using the transformed geometry
    # TODO: use specialized back-end when available
    ray_trafo_slice = odl.tomo.RayTransform(reco_space_slice, geometry_slice,
                                            impl='astra_cuda')

    time_after_setup = time.time()
    setup_time_ms = 1e3 * (time_after_setup - time_at_start)
    if DEBUG:
        print('time for setup: {:7.3f} ms'.format(setup_time_ms))

    # Compute back-projection with this ray transform
    fbp_reco_slice = ray_trafo_slice.adjoint(proj_data_filtered)
    if DEBUG:
        # fbp_reco_slice.show()
        pass

    time_after_comp = time.time()
    comp_time_ms = 1e3 * (time_after_comp - time_after_setup)
    if DEBUG:
        print('time for computation: {:7.3f} ms'.format(comp_time_ms))

    # Output must be a numpy.ndarray with 'uint32' data type, we first clip
    # to `[min_val, max_val]` and then rescale to [0, uint32_max - 1]
    reco_clipped = np.clip(np.asarray(fbp_reco_slice), min_val, max_val)
    reco_clipped *= np.iinfo(np.uint32).max - 1

    # Returning the shape as an 'int32' array and the flattened values
    return (np.array(slice_shape, dtype='int32'),
            reco_clipped.astype('uint32').ravel())


# %% Connect to local slicevis server and register the callback

# Connect to server at localhost. The second argument would be the URI(?).
serv = tomop.server('Slice FBP')
print('Server started')

reco_space_preview = odl.uniform_discr(vol_min_pt, vol_max_pt, (64, 64, 64))

# Define a preview (coarse resolution volume) and quantize to uint32
# TODO: replace with FBP
preview = odl.phantom.shepp_logan(reco_space_preview, modified=True)
preview_quant = np.clip(np.asarray(preview), 0, 1)
preview_quant *= np.iinfo(np.uint32).max - 1
preview_quant = preview_quant.astype('uint32')

# Define volume packet and send it
preview_packet = tomop.volume_data_packet(
    serv.scene_id(),
    np.array(preview_quant.shape, dtype='int32'),
    preview_quant.ravel())
serv.send(preview_packet)
print('Preview volume sent')

# Register the callback
# serv.set_callback(callback_null)
# print('NULL callback registered')
serv.set_callback(callback_fbp)
print('FBP callback registered')

# Do it
serv.serve()
