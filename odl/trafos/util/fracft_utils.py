# Copyright 2014, 2015 The ODL development group
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

"""Utility functions for fractional Fourier transforms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range

# External
from math import pi
import numpy as np

# Internal
from odl.discr.grid import RegularGrid
from odl.discr.lp_discr import DiscreteLp, dspace_type
from odl.discr.partition import uniform_partition_fromgrid
from odl.set.sets import RealNumbers
from odl.space.base_ntuples import _TYPE_MAP_R2C
from odl.space.fspace import FunctionSpace
from odl.util.numerics import fast_1d_tensor_mult
from odl.util.utility import (
    is_real_dtype, is_scalar_dtype, is_real_floating_dtype,
    is_complex_floating_dtype, dtype_repr, conj_exponent)


__all__ = ('fractional_ft',)


def fractional_ft(x, alpha, dft, padded_len=None, axes=None, out=None,
                  **kwargs):
    """Compute the fractional FT of x with parameter alpha.

    The fractional FT of ``x`` with parameter ``alpha`` is defined as::

        G[k] = sum_j( x[j] * exp(-1j * 2*pi * j*k*alpha) )

    If ``alpha == 1 / len(x)``, this is the usual DFT. The sum can be
    written as a circular convolution of length ``2 * p``::

        G[k] = conj(z[k]) * sum_j (y[j] * z[k - j]), 0 <= k < len(x),

    where ``z[j] = exp(1j * pi * alpha * j**2)`` and
    ``y[j] = x[j] * conj(z[j])`` for ``0 <= j < len(x)`` and
    appropriately padded for ``len(x) <= j < 2 * p``. The parameter
    ``p`` can be chosen freely from the integers larger than or equal
    to ``len(x) - 1``.

    For higher dimensions, the described transform is applied per
    dimension.

    Parameters
    ----------
    x : `array-like`
        Input array to be transformed
    alpha : nonzero `float` or `sequence` of `float`
        Parameter in the complex exponential of the transform.
        If a sequence is given, it must have the same length as
        ``axes`` if provided, otherwise length ``dom.ndim``.
        A single float is interpreted as global value for all
        axes.
    dft : `Operator`
        Discrete Fourier transform operator for the evaluation of the
        forward and backward transforms in the convolution
    padded_len : `sequence` of `int`, optional
        Length of the padded arrays per axis. By default, in each axis,
        ``padded_len = 2 * (n - 1)`` is chosen, where ``n`` is the
        length of the array in this axis. This is the smallest
        possible choice. Selecting a power of 2 may speed up the
        computation.
    precomp_z : `sequence` of `array-like`, optional
        Array of precomputed factors (one per axis)
        ``z[j] = exp(1j * pi * alpha * j**2)`` as used in the
        transform. Their lengths must be at least the length of ``x``
        in the corresponding axes. Values at indices beyond the lenght
        of ``x`` in the respective axis are ignored.
    precomp_zhat : tuple of `array-like`, optional
        Array of precomputed factors ``zhat`` (one per axis) which is
        the Fourier transform of the factors ``z``. Its lengths must be
        equal to ``padded_len`` in the respective axes.
    out : `numpy.ndarray`, optional
        Array to store the values in. Must have the same shape and
        data type as ``x``.

    Returns
    -------
    out : `numpy.ndarray`
        The fractional FT of ``x``. The returned array has the same
        shape as ``x`` (padded values are discarded). If ``out`` was
        given, the returned object is a reference to it.
    precomp_z : `tuple` of `numpy.ndarray`
        The precomputed values of the DFT of ``z``. If ``precomp_z``
        was given as a tuple of ndarray, the returned object is a
        reference to it.
    precomp_zhat : `tuple` of `numpy.ndarray`
        The precomputed values of the DFT of ``zhat``. If ``precomp_zhat``
        was given as a tuple of ndarray, the returned object is a
        reference to it.
    """
    # Check and precompute parameters to the algorithm, in order
    # TODO: other data types
    x = np.asarray(x, dtype='complex')
    ndim = x.ndim
    shape = x.shape

    try:
        alpha = [float(alpha)] * ndim
    except TypeError:
        alpha = list(alpha)

    if padded_len is None:
        padded_len = [2 * (n - 1) for n in shape]
    else:
        padded_len_ = []
        for i, (plen, n) in enumerate(zip(padded_len, shape)):
            padded_len_.append(int(plen))

            if plen < 2 * (n - 1):
                raise ValueError('padded length in axis {} must be at least '
                                 '{}, got {}.'.format(i, 2 * (n - 1), plen))
            if plen % 2:
                raise ValueError('padded length in axis {} must be even, '
                                 'got {}.'.format(i, plen))
        padded_len = padded_len_

    precomp_z = kwargs.pop('precomp_z', None)
    if precomp_z is None:
        precomp_z_ = []
        for n, a in zip(shape, alpha):
            # Initialize the precomputed z values. These are
            # exp(1j * pi * alpha * j**2) for 0 <= j < len(x)
            arr = np.exp(1j * np.pi * a * np.arange(n) ** 2)
            precomp_z_.append(arr)
        precomp_z = tuple(precomp_z_)
    else:
        # We accept also larger (padded) versions of z, using only the
        # part up to index len(x) - 1.
        for i, (arr, n) in enumerate(zip(precomp_z, shape)):
            if np.ndim(arr) != 1:
                raise ValueError('precomp_z array {} has {} dimensions, '
                                 'expected 1.'.format(np.ndim(arr)))
            if len(arr) < n:
                raise ValueError('precomp_z has length {}, expected at least {}.'
                                 ''.format(len(precomp_z), len(x)))

    precomp_zhat = kwargs.pop('precomp_zhat', None)
    if precomp_zhat is None:
        for n, a, z in zip(shape, alpha, precomp_z):
            pass
        # Initialize the padded FT of the precomputed z values. These are
        # o exp(1j * pi * alpha * j**2) for 0 <= j < len(x)
        # o exp(1j * pi * alpha * (2*p - j)**2) for 2*p - m <= j < 2*p
        # o 0, otherwise
        # o followed by a discrete FT.
        # Here, 2*p refers to the even padded length of the arrays.
#        precomp_zhat = np.empty(padded_len, dtype='complex')
#        precomp_zhat[:len(x)] = precomp_z[:len(x)]
#        precomp_zhat[-len(x) + 1:] = precomp_z[1:len(x)][::-1]
#        # Here we get a copy, no way around since fft has no in-place method
#        precomp_zhat = np.fft.fft(precomp_zhat)
    else:
        precomp_zhat = np.asarray(precomp_zhat, dtype='complex')
        if precomp_zhat.ndim != 1:
            raise ValueError('precomp_zhat has {} dimensions, expected 1.'
                             ''.format(precomp_zhat.ndim))
        if len(precomp_zhat) != padded_len:
            raise ValueError('precomp_zhat has length {}, expected {}.'
                             ''.format(len(precomp_zhat), padded_len))

    # TODO: axis order
    if out is None:
        out = np.empty_like(x)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('out is not a numpy.ndarray instance.'.format(out))
        if out.shape != x.shape:
            raise ValueError('out has shape {}, expected {}.'
                             ''.format(out.shape, x.shape))
        # TODO: adapt this once other dtypes are considered
        if out.dtype != x.dtype:
            raise ValueError('out has dtype {}, expected {}.'
                             ''.format(out.dtype, x.dtype))

    # Now the actual computation. First the input array x needs to be padded
    # with zeros up to padded_len (in a new array).

    y = np.zeros(padded_len, dtype='complex')
    y[:len(x)] = x
    y[:len(x)] *= precomp_z[:len(x)]
    yhat = np.fft.fft(y)
    yhat *= precomp_zhat
    y = np.fft.ifft(yhat)
    out[:] = y[:len(x)]
    out *= precomp_z[:len(x)].conj()

    return out, precomp_z, precomp_zhat


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
