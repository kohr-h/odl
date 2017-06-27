# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Geometry base and mixin classes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np

from odl.discr import RectPartition
from odl.tomo.geometry.detector import Detector
from odl.tomo.util import axis_rotation_matrix


__all__ = ('Geometry', 'DivergentBeamGeometry', 'AxisOrientedGeometry')


class Geometry(object):

    """Abstract geometry class.

    A geometry is described by

    * a detector,
    * a set of detector motion parameters,
    * a function mapping motion parameters to the location of a
      reference point (e.g. the center of the detector surface),
    * a rotation applied to the detector surface, depending on the motion
      parameters,
    * a mapping from the motion and surface parameters to the detector pixel
      direction to the source,
    * optionally a mapping from the motion parameters to the source position,
    * optionally a global translation of the geometry (shift of the origin)

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    def __init__(self, ndim, motion_part, detector, translation=None,
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : positive int
            Number of dimensions of this geometry, i.e. dimensionality
            of the physical space in which this geometry is embedded.
        motion_part : `RectPartition`
            Partition for the set of "motion" parameters.
        detector : `Detector`
            The detector of this geometry.
        translation : `array-like`, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`.
        check_bounds : bool, optional
            If ``True``, check if provided parameters for query functions
            like `det_refpoint` are in the valid range.
            Default: ``True``
        """
        ndim, ndim_in = int(ndim), ndim
        if ndim != ndim_in or ndim <= 0:
            raise ValueError('`ndim` must be a positive integer, got {}'
                             ''.format(ndim_in))
        if not isinstance(motion_part, RectPartition):
            raise TypeError('`motion_part` must be a `RectPartition`, '
                            'instance, got {!r}'.format(motion_part))
        if not isinstance(detector, Detector):
            raise TypeError('`detector` must be a `Detector` instance, '
                            'got {!r}'.format(detector))

        self.__ndim = ndim
        self.__motion_partition = motion_part
        self.__detector = detector
        self.__check_bounds = bool(kwargs.pop('check_bounds', True))

        if translation is None:
            self.__translation = np.zeros(self.ndim)
        else:
            translation = np.asarray(translation, dtype=float)
            if translation.shape != (self.ndim,):
                raise ValueError('`translation` must have shape ({},), got {}'
                                 ''.format(self.ndim, translation.shape))
            self.__translation = translation

        self.__implementation_cache = {}

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @property
    def ndim(self):
        """Number of dimensions of the geometry."""
        return self.__ndim

    @property
    def motion_partition(self):
        """Partition of the motion parameter set into subsets."""
        return self.__motion_partition

    @property
    def motion_params(self):
        """Continuous motion parameter range, an `IntervalProd`."""
        return self.motion_partition.set

    @property
    def motion_grid(self):
        """Sampling grid of `motion_params`."""
        return self.motion_partition.grid

    @property
    def detector(self):
        """Detector representation of this geometry."""
        return self.__detector

    @property
    def det_partition(self):
        """Partition of the detector parameter set into subsets."""
        return self.detector.partition

    @property
    def det_params(self):
        """Continuous detector parameter range, an `IntervalProd`."""
        return self.detector.params

    @property
    def det_grid(self):
        """Sampling grid of `det_params`."""
        return self.detector.grid

    @property
    def partition(self):
        """Joined parameter set partition for motion and detector.

        A `RectPartition` with `det_partition` appended to `motion_partition`.
        """
        return self.motion_partition.append(self.det_partition)

    @property
    def params(self):
        """Joined parameter set for motion and detector.

        By convention, the motion parameters come before the detector
        parameters.
        """
        return self.partition.set

    @property
    def grid(self):
        """Joined sampling grid for motion and detector.

        By convention, the motion grid comes before the detector grid.
        """
        return self.partition.grid

    @property
    def translation(self):
        """Shift of the origin of this geometry."""
        return self.__translation

    @property
    def check_bounds(self):
        """Whether to check if method parameters are in the valid range."""
        return self.__check_bounds

    def det_refpoint(self, mpar):
        """Detector reference point function.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the detector
            reference point.

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point, an `ndim`-dimensional vector.
        """
        raise NotImplementedError('abstract method')

    def rotation_matrix(self, mpar):
        """Return the rotation matrix from initial state to state at ``mpar``.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the rotation matrix.

        Returns
        -------
        rot : `numpy.ndarray`, shape (`ndim`, `ndim`)
            The rotation matrix mapping vectors at the initial state
            to the ones in the state defined by ``mpar``. The rotation
            is extrinsic, i.e., defined in the fixed ("world") coordinate
            system.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, mpar, dpar, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate.
        dpar : `det_params` element
            Detector parameter at which to evaluate.
        normalized : bool, optional
            If ``True``, return a normalized (unit) vector.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        raise NotImplementedError('abstract method')

    def det_point_position(self, mpar, dpar):
        """Return the detector point at ``(mpar, dpar)``.

        The motion parameter ``mpar`` is used to move the detector
        reference point, and the detector parameter ``dpar`` defines
        an intrinsic shift that is added to the reference point.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate.
        dpar : `det_params` element
            Detector parameter at which to evaluate.

        Returns
        -------
        pos : `numpy.ndarray`, shape ``(ndim,)``
            Source position, an `ndim`-dimensional vector.
        """
        # Offset relative to the detector reference point
        offset = self.rotation_matrix(mpar).dot(self.detector.surface(dpar))
        return self.det_refpoint(mpar) + offset

    @property
    def implementation_cache(self):
        """Dictionary acting as a cache for this geometry.

        Intended for reuse of computations. Implementations that use this
        storage should take care of unique naming.

        Returns
        -------
        implementations : dict
        """
        return self.__implementation_cache


class DivergentBeamGeometry(Geometry):

    """Abstract divergent beam geometry class.

    A divergent beam geometry is characterized by the presence of a
    point source.

    Special cases include fan beam in 2d and cone beam in 3d.
    """

    def src_position(self, mpar):
        """Source position function.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the source position.

        Returns
        -------
        pos : `numpy.ndarray` (shape (`ndim`,))
            Source position, an `ndim`-dimensional vector.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, mparams, dparams, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        The default implementation uses the `det_point_position` and
        `src_position` functions. Implementations can override this, for
        example if no source position is given.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate.
        dpar : `det_params` element
            Detector parameter at which to evaluate.
        normalized : bool, optional
            If ``True``, return a normalized (unit) vector.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source.
        """
        if self.check_bounds:
            if not self.motion_params.contains_all(mparams):
                raise ValueError('`mparams` {} not in the valid range {}'
                                 ''.format(mparams, self.motion_params))
            if not self.det_params.contains_all(dparams):
                raise ValueError('`dparams` {} not in the valid range {}'
                                 ''.format(dparams, self.det_params))

        det_to_src_vec = (self.src_position(mparams) -
                          self.det_point_position(mparams, dparams))

        if normalized:
            # axis = -1 allows this to be vectorized
            det_to_src_vec /= np.linalg.norm(det_to_src_vec, axis=-1)

        return det_to_src_vec


class AxisOrientedGeometry(object):

    """Mixin class for 3d geometries oriented along an axis."""

    def __init__(self, axis):
        """Initialize a new instance.

        Parameters
        ----------
        axis : `array-like`, shape ``(3,)``
            Vector defining the fixed rotation axis of this geometry.
        """
        axis, axis_in = np.asarray(axis, dtype=float), axis
        if axis.shape != (3,):
            raise ValueError('`axis.shape` must be (3,), got {}'
                             ''.format(axis.shape))

        if np.linalg.norm(axis) <= 1e-10:
            raise ValueError('`axis` {} too close to zero'.format(axis_in))

        self.__axis = axis / np.linalg.norm(axis)

    @property
    def axis(self):
        """Normalized axis of rotation, a 3d vector."""
        return self.__axis

    def rotation_matrix(self, angle):
        """Return the matrix for rotating around `axis` by ``angle``.

        The matrix is computed according to
        `Rodrigues' rotation formula
        <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

        Parameters
        ----------
        angle : float
            Motion parameter given in radians. It must be
            contained in this geometry's `motion_params`.

        Returns
        -------
        rot_mat : `numpy.ndarray`, shape ``(3, 3)``
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """
        if self.check_bounds and not self.motion_params.contains_all(angle):
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))

        return axis_rotation_matrix(self.axis, angle)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
