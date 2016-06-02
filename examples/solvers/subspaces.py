# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:08:10 2016

@author: Ozan Öktem
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import math
from numbers import Integral
import numpy as np
from scipy import misc
import odl

from odl.operator.pspace_ops import (
    BroadcastOperator, DiagonalOperator)

# TODO: Move this to utilities
def sequence_regrouping(seq, grouping):
    """Regroup a given ``seq`` according to ``grouping``.

    Parameters
    ----------
    seq : sequence
        Sequence whose elements are to be re-grouped
    grouping : index expression or tuple of index expressions
        Grouping scheme specifying the re-arrangement of the elements
        of ``seq``.
        If ``grouping`` is a single index expression (e.g. an integer
        or a slice), the elements corresponding to ``seq[grouping]``
        are put into a first tuple, and the remaining ones, if existing,
        into a second tuple.
        If ``grouping`` is a sequence of index expressions, each such
        expression defines a tuple of elements according to the above
        rule. The remaining elements are, if existing, are put into
        a last tuple.

    Returns
    -------
    regrouped : tuple of tuples
        Tuple containing the regrouped elements from ``seq`` according
        to ``grouping``. The number of entries in the tuple is equal
        to the number of given index expression in ``grouping``, plus
        one if there are remaining elements.
    """
    # Make an array from the sequence to allow for advanced slicing
    seq_list = list(seq)
    seq_array = np.array(seq_list)
    print(seq_array.shape)
    unused = np.ones(len(seq_array), dtype=bool)
    regrouped = []
    if isinstance(grouping, tuple):
        for idx in grouping:
            # TODO: think harder about correct embedding in tuples for
            # single integers
            sliced = np.atleast_1d(seq_array[idx])
            regrouped.append(tuple(sliced))
            unused[idx] = False
    else:
        regrouped.append(tuple(seq_array[grouping]))
        unused[grouping] = False
    # Handle the unsued indices
    if np.any(unused):
        unused_indices = np.arange(len(seq_array))[unused]
        regrouped.append(tuple(seq_array[unused_indices]))

    return tuple(regrouped)


#--- ART method --#
# TODO: Allow rules to set relaxation parameter as input
# TODO: Allow stopping rule as input
# TODO: Allow to optionally ensure intermediate iterates tget
#       projected to fufill bound constraints.
def generlized_art(fwd_ops, inv_normal_ops, data, rec, relax_param, max_iter,
                   callback=None):
    """Perform block ART type of iteration for solving inverse problems.

    Implementation of the generalized Kaczmarz's method given in eq. (5.4.9)
    in the book "Mathematical methods in image reconstruction"
    by F. Wubbeling and F. Natterer, SIAM, 2001.
    The normal operator mentioned below is the C_j operator in eq. (5.4.9).

    Parameters
    ----------
    fwd_ops : `sequence` of `Operator` or `BroadcastOperator`
        Vectorial forward operator

    inv_normal_ops : `sequence` of `Operator` or `BroadcastOperator`
        Vectorial normal operator inverse. Mathematically, it should
        represent a positive definite approximation of the inverse
        of the normal operator that is associated with the forward
        operator. It should be a data-to-data mapping whose range equals
        the domain of the vectorial forward operator

    data : ``fwd_ops.range`` `element-like`
        The data. If the vectorial forward operator has ``k`` components
        where the correspondign ranges have shape ``(n, m)``, then
        data can be given as ``(k, n, m)`` `arrary-like`

    rec : ``fwd_ops.domain`` `element-like`
      The starting iterate and also the place holder for the reconstruction

    relax_param : positive ``float``
      real number that is the relaxation parameter. Should be
      between 0 and 2 and value 1 corresponds to unregularized
      Kaczmarz's method

    max_iter :  positive ``int``
      Maximum number of iterations
    """

    # Make sure the pfwd_op is really a vectorial operator
    if isinstance(fwd_ops, BroadcastOperator):
        pfwd_op = fwd_ops
    else:
        pfwd_op = BroadcastOperator(*fwd_ops)

    # Make sure the inv_normal_op is really a vectorial operator
    if isinstance(inv_normal_ops, DiagonalOperator):
        pinv_normal_op = inv_normal_ops
    else:
        pinv_normal_op = DiagonalOperator(*inv_normal_ops)

    # Make sure domain of adjoint of pfwd_op equals the range of inv_normal_op.
    if pfwd_op.adjoint.domain != pinv_normal_op.range:
        raise ValueError('Range of forward operator needs to be the same as '
                        'the domain of the normal inverse operator.')

    # Make sure domain and of inv_normal_op are the same.
    if pinv_normal_op.range != pinv_normal_op.domain:
        raise ValueError('Range and domain of the normal inverse operator '
                        'to be the same.')

    # Convert data into the corersponding element in the range of the
    # vectorial forward operator.
    # TODO: Make this work properly for CUDA
    if data in pfwd_op.range:
        pspace_data = data
    else:
        pspace_data = pfwd_op.range.element(np.asarray(data))

    # Create reusable temporary place holders in the reconstruction and data
    # spaces for the iteration. This is solely for efficieny purposes.
    tmp_rec = pfwd_op.domain.element()
    tmp_data = pfwd_op.range.element()

    # Outer loop
    for _ in range(max_iter):
        # Inner loop taken over all sub-spaces
        for fwd_op, normal_op, pdata, dtmp in zip(pfwd_op.operators,
                                                  pinv_normal_op.operators,
                                                  pspace_data,
                                                  tmp_data):
            # ART iterate done in an efficient manner
            fwd_op(rec, out=dtmp)
            dtmp -= pdata
            # TODO: use another temporary?
            modified_residual = normal_op(dtmp)
            fwd_op.adjoint(modified_residual, out=tmp_rec)
            rec.lincomb(1, rec, -relax_param, tmp_rec)

        if callback is not None:
            callback(rec)



# --- Examples --#
# Set range of grey-scale values to visualize
min_range = 0
max_range= 300

#--- Read phantom from an image file -- #
# A png-file "face.png" is a color image that we import as grey scale
# and store as a numpy array. Note that we need to flip x- and y-axis
# when we read in the image.
image = np.fliplr(
            misc.imread('/home/ozan/dev/odl_scripts/face.png',flatten=True).T)

#--- Define reconstruction space -- #
# The discretized functions in the reconstruction space shopuld have
# same number of sample points as the number of pixels in the image.
# We also have a rectangular image domain of size 40 along the x-axis,
# 30 along the y-axis, and with the origin at the center. This gives a
# ration of 4:3 that matches the ratio in terms of pixels, so elements
# in the reconstruction space are discretized with square pixels. Theis is
# unfourtenately required if we seek to use ASTRA as back-end in calculating
# the forward operator later on.
im_domain_x=4*10
im_domain_y=3*10
reco_space = odl.uniform_discr(
    min_corner=[-im_domain_x/2, -im_domain_y/2],
    max_corner=[im_domain_x/2,im_domain_y/2],
    nsamples=image.shape,dtype='float32')
# Create phantom
phantom = reco_space.element(image)
# Show phantom
phantom.show('Phantom',clim=[min_range,max_range])

#--- Specify detector and data acquisition geometry -- #
# We assume parallel beam geometry with a centered detector that covers
# the image domain. This implies that the 1D detector is centered around
# origin and the 2D origin in image domain projects to the 1D origin in
# the detector along the beam direction.
#
# Specification of detector: 512 pixels and its size ensures it covers
# the entire reconstruction space, i.e., the length is
# sqrt(2)* max-side-length of image domain.
detector_size = math.sqrt(2) * max(im_domain_x,im_domain_y)
detector_partition = odl.uniform_partition(-detector_size/2,
                                           detector_size/2, 512)
# Specification of the data acquisition geometry: Parallel beam geometry
# with 360 angles uniformly spaced from 0 to 180 degrees
start_angle = 0
end_angle = np.pi
number_of_directions = 360
angle_partition = odl.uniform_partition(start_angle, end_angle,
                                        number_of_directions)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

#--- Create projection data -- #
# Here we use the non-vectorial forward operator to generate data from
# the phantom and add some additive Gaussian noise.
# The non-vectorial forward operator
forward_operator = odl.tomo.RayTransform(reco_space, geometry,
                                         impl='astra_cuda')
# Ideal data
data = forward_operator(phantom)
# Add additive Gaussian noise
sigma = 0.2
data += sigma * np.std(data) * np.random.random(data.shape)
# Show data
data.show('Noisy parallel beam sinogram')


#--- Assemble the vectorial forward operator. --#
# Partition the geometry: Assemble a list of geometries (sub-geometries)
# where each sub-geometry corresponds to a single view.
singleview_geom_list = list(geometry)

# Assemble a list of forward operators where each forward operator
# in the list is the restriction of the forward operator to a
# single view geometry.
singleview_forward_op_list = [odl.tomo.RayTransform(reco_space, sub_geom,
                                                    impl='astra_cuda')
                              for sub_geom in singleview_geom_list]

# Partition data into a list where each element is data corresponding to
# a single view.
singleview_data_list = list(data.asarray())

#--- Create the sub-spaces and their ordering. --#
# The sub-spaces and their ordering is given by specifying how the single
# views are grouped and ordered. Below are some examples.

# Single sub-spaces with all views
#grouping = slice(None)

# Sequential sub-spaces: Each sub-space is a single view ordered sequentially.
# This is actually unnecessary, we add the code for pedagogical reasons.
num_views = len(singleview_data_list)
grouping = tuple(range(num_views))

# Two sub-spaces: One is all even single view elements and the other
# all odd single view elements.
#num_views = len(singleview_data_list)
#last_odd = num_views if num_views % 2 == 1 else num_views - 1
#grouping = (np.s_[::2], np.s_[last_odd::-2])

# Symmetric: Sub-spaces are all single view but they are ordered symmetrically
# around the 'middle' single-view.
#num_views = len(singleview_data_list)
#...

# Blocks of k-elements: Group sub-spaces sequentially into blocks of k single
# views. Last sub-space will contain single views left out by this procedure.
#num_views = len(singleview_data_list)
#block_size = 4
#last_odd = num_views if num_views % 2 == 1 else num_views - 1
#grouping = tuple(np.s_[start:start + block_size]
#                 for start in range(0, num_views - block_size, block_size))

#--- Generate sub-space gmoetries, data, and forward operators --#
# Subdivide geometris, data and forward operator accoding to the sub spaces.
geometry_grouping = sequence_regrouping(singleview_geom_list, grouping)
data_grouping = sequence_regrouping(singleview_data_list, grouping)
#data_grouping = tuple(d[0] if len(d) == 1 else d for d in data_grouping)
forward_op_grouping = sequence_regrouping(singleview_forward_op_list,
                                            grouping)
# Transform the list of forward operators into an vectorial forward operator.
# This is not necessary, we could also use forward_op_list directly in
# the ART procedure.
pspace_forward_op_list = [odl.BroadcastOperator(*op_list)
                          for op_list in forward_op_grouping]
pspace_forward_op = odl.BroadcastOperator(*pspace_forward_op_list)


# --- ART reconstructions --#
# Ensure intermediate iterates are shown.
callback = odl.solvers.CallbackShow() & odl.solvers.CallbackPrintIteration()

# To speed up, we pre-decompose data (not necessary)
pdata = pspace_forward_op.range.element(data_grouping)

# ART with identity as inverse normal operators
# Set starting value, relaxation parameter, max iterations
rec = pspace_forward_op.domain.zero()
relax_param=0.1
max_iter=10
# Inverse normal operator: Approximated by the identity.
ident_ops = [odl.IdentityOperator(spc) for spc in pspace_forward_op.range]
# ART iterates
generlized_art(pspace_forward_op, ident_ops, pdata, rec, relax_param,
               max_iter, callback=callback)

# ART with "proper" inverse normal operators.
# Set starting value, relaxation parameter, max iterations
rec = pspace_forward_op.domain.zero()
relax_param = 1.0
max_iter=10
# Inverse normal operator: Here we seek the inverse of
# C_j = T_j T_j^* where T_j is the j:th compnent of the vectorial forward
# operator. The inverse is given by the operator that takes pointwise
# division with T_j(1) where "1" is is the unit image.
# We start by forming the unit function in reconstruction space.
unit_volume = forward_operator.domain.one()
# Calculate the multiplictive factor that will be used to define the normal
# inverse operator. This will in the end be 1/T_i(1) for entries that are
# too small and zero otherwsie.
# We start by forming the multiplicative factors T_j(1)
inv_normal_mfac = pspace_forward_op(unit_volume)
# Here we form 1/T_j(1) for large enough entries and zero otherwise.
# Large enough means at least epsilon * maximum absolute value.
epsilon = 0.01
for mfac in inv_normal_mfac:
    # Extract out the values as an array
    mfac_array = mfac.asarray()
    # Find out the indices for non-zero elements
    abs_array = np.abs(mfac_array)
    maxval = np.max(abs_array)
    idcs_large = np.greater(abs_array, maxval * epsilon)
    # Do in-place replacement in inv_normal_mfac where we insert
    # division by T_j(1).
    mfac_array[idcs_large] = (1 / mfac_array[idcs_large])
    # To ensure the above in-place replacement also works if the arrays are
    # CUDA arrays instead of numpy arrays. Brings no overhead for numpy arrays.
    mfac[:] = mfac_array
# The inverse normal operator is now the operator that is mutliplaction with
# the multiplicate factors in inv_normal_mfac.
inv_normal_ops = [odl.MultiplyOperator(fac) for fac in inv_normal_mfac]
# ART iterates
generlized_art(pspace_forward_op, inv_normal_ops, pdata, rec, relax_param,
               max_iter, callback=callback)
