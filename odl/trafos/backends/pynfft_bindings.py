# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bindings to the ``pynfft`` back-end for non-uniform Fourier transforms.

The `pynfft <https://pythonhosted.org/pyNFFT/>`_  package is a Python
wrapper around the well-known
 `NFFT <https://www-user.tu-chemnitz.de/~potts/nfft/>`_ library for
non-uniform fast Fourier transforms.
"""

import numpy as np

try:
    import pynfft
    PYNFFT_AVAILABLE = True
except ImportError:
    PYNFFT_AVAILABLE = False

__all__ = ('PYNFFT_AVAILABLE', 'normalize_samples')


def normalize_samples(samples, grid_spacing, out=None):
    r"""Normalize samples to [-0.5; 0.5[.

    Parameters
    ----------
    samples : array-like
        Array of unnormalized samples. Its shape is expected to be
        ``(M, d)``, where ``M`` is the number of samples, and ``d`` the
        dimension of the space.
    grid_spacing : positive float or sequence of positive float
        Step between grid points in the space on which the non-uniform Fourier
        transform is applied. A sequence must have length ``d``. A single
        float is interpreted as the same spacing in each dimension.
    out : numpy.ndarray, optional
        Array in which the result should be stored. Must have the same shape
        as ``samples``.

    Returns
    -------
    norm_samples : numpy.ndarray
        Normalized samples. See Notes for details. If ``out`` was given, the
        returned array is a reference to it.

    Notes
    -----
    For a uniformly discretized space with grid spacing
    :math:`s > 0 \in \mathbb{R}^d`, samples :math:`\xi_k \in \mathbb{R}^d`
    are normalized to

    .. math::
        \bar \xi_k = \frac{s \odot \xi_k}{2\pi}.

    Since the reciprocal grid extends from :math:`-\pi s^{-1}` to
    :math:`\pi s^{-1}`, this transformation normalized the samples to lie
    between :math:`-0.5` and :math:`0.5`. Due to implementation details,
    the value :math:`0.5` has to be replaced with :math:`-0.5`, which gives
    the same result for periodic inputs.
    """
    samples = np.array(samples, copy=False, ndmin=2)
    if samples.ndim != 2:
        raise ValueError(
            '`samples` must have 2 dimensions, got array with `ndim={}`'
            ''.format(samples.ndim)
        )
    ndim = samples.shape[1]

    grid_spacing_in = grid_spacing
    try:
        iter(grid_spacing)
    except TypeError:
        grid_spacing = [float(grid_spacing)] * ndim
    else:
        grid_spacing = [float(s) for s in grid_spacing]

    if len(grid_spacing) != ndim:
        raise ValueError(
            '`grid_spacing` must have length of `samples.shape[1]`, but '
            '{} != {}'.format(len(grid_spacing), ndim)
        )

    if any(s <= 0 for s in grid_spacing):
        raise ValueError(
            '`grid_spacing` must be positive, got {}'.format(grid_spacing_in)
        )

    if out is not None and out.shape != samples.shape:
        raise ValueError(
            '`out.shape` must be equal to `samples.shape`, but {} != {}'
            ''.format(out.shape, samples.shape)
        )

    grid_spacing = np.array(grid_spacing, dtype=samples.dtype)
    norm_samples = np.multiply(
        samples, (grid_spacing[None, :] / 2 * np.pi), out=out
    )
    norm_samples[norm_samples == 0.5] = -0.5
    return norm_samples
