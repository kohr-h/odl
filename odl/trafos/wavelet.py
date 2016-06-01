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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

import numpy as np
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

from odl.discr import DiscreteLp
from odl.operator import Operator


__all__ = ('WaveletTransform', 'WaveletTransformInverse',
           'PYWAVELETS_AVAILABLE')


_SUPPORTED_IMPL = ('pywt',)


def coeff_size_list_axes(shape, wbasis, mode, axes):
    """Construct a size list from given wavelet coefficient and given axes.

    Related to 1D, 2D and 3D wavelet transforms along specified axes using
    `PyWavelets
    <http://www.pybytes.com/pywavelets/>`_.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels/voxels in the image. Its length must be 1, 2 or 3.

    wbasis : ``pywt.Wavelet``
        Selected wavelet basis. For more information see the
        `PyWavelets documentation on wavelet bases
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_.

    mode : `str`
        Signal extention mode. Possible extension modes are

        'zpd': zero-padding -- signal is extended by adding zero samples

        'cpd': constant padding -- border values are replicated

        'sym': symmetric padding -- signal extension by mirroring samples

        'ppd': periodic padding -- signal is trated as a periodic one

        'sp1': smooth padding -- signal is extended according to the
        first derivatives calculated on the edges (straight line)

        'per': periodization -- like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    axes : sequence of `int`, optional
           Dimensions in which to calculate the wavelet transform.
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('shape must have length 1, 2 or 3, got {}.'
                         ''.format(len(shape)))

    size_list = [shape]

    if len(shape) == 1:
        size_list = coeff_size_list(shape, 1, wbasis, mode)
    elif len(shape) == 2:
        x_ax = axes.count(0)
        y_ax = axes.count(1)
        nx = shape[0]
        ny = shape[1]
        if x_ax != 0:
            for _ in range(x_ax):
                shp_x = pywt.dwt_coeff_len(nx, filter_len=wbasis.dec_len,
                                           mode=mode)
                nx = shp_x
        else:
            shp_x = nx

        if y_ax != 0:
            for _ in range(y_ax):
                shp_y = pywt.dwt_coeff_len(ny, filter_len=wbasis.dec_len,
                                           mode=mode)
                ny = shp_y
        else:
            shp_y = ny

        size_list.append((shp_x, shp_y))
        size_list.append(size_list[-1])
        size_list.reverse()

    else:
        x_ax = axes.count(0)
        y_ax = axes.count(1)
        z_ax = axes.count(2)
        nx = shape[0]
        ny = shape[1]
        nz = shape[2]

        if x_ax != 0:
            for _ in range(x_ax):
                shp_x = pywt.dwt_coeff_len(nx, filter_len=wbasis.dec_len,
                                           mode=mode)
                nx = shp_x
        else:
            shp_x = nx

        if y_ax != 0:
            for _ in range(y_ax):
                shp_y = pywt.dwt_coeff_len(ny, filter_len=wbasis.dec_len,
                                           mode=mode)
                ny = shp_y
        else:
            shp_y = ny
        if z_ax != 0:
            for _ in range(z_ax):
                shp_z = pywt.dwt_coeff_len(nz, filter_len=wbasis.dec_len,
                                           mode=mode)
                nz = shp_z
        else:
            shp_z = nz

        size_list.append((shp_x, shp_y, shp_z))
        size_list.append(size_list[-1])
        size_list.reverse()

    return size_list


def coeff_size_list(shape, nscales, wbasis, pad_mode):
    """Construct a size list from given wavelet coefficients.

    Related to 1D, 2D and 3D multidimensional wavelet transforms that utilize
    `PyWavelets
    <http://www.pybytes.com/pywavelets/>`_.

    Parameters
    ----------
    shape : tuple
        Number of pixels/voxels in the image. Its length must be 1, 2 or 3.

    nscales : int
        Number of scales in the multidimensional wavelet
        transform.  This parameter is checked against the maximum number of
        scales returned by ``pywt.dwt_max_level``. For more information
        see the `PyWavelets documentation on the maximum level of scales
        <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_.

    wbasis : ``pywt.Wavelet``
        Selected wavelet basis. For more information see the
        `PyWavelets documentation on wavelet bases
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_.

    pad_mode : string
        Signal extention mode. Possible extension modes are

        'zpd': zero-padding -- signal is extended by adding zero samples

        'cpd': constant padding -- border values are replicated

        'sym': symmetric padding -- signal extension by mirroring samples

        'ppd': periodic padding -- signal is trated as a periodic one

        'sp1': smooth padding -- signal is extended according to the
        first derivatives calculated on the edges (straight line)

        'per': periodization -- like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    Returns
    -------
    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels:

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level

        ``size_list[1]`` = size of the detail coefficients at the
        coarsest level

        ...

        ``size_list[N]`` = size of the detail coefficients at the
        finest level

        ``size_list[N+1]`` = size of the original image

        ``N`` = number of scaling levels = nscales
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('shape must have length 1, 2 or 3, got {}'
                         ''.format(len(shape)))

    max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
    if nscales > max_level:
        raise ValueError('too many scaling levels, got {}, maximum useful '
                         'level is {}'
                         ''.format(nscales, max_level))

    # dwt_coeff_len calculates the number of coefficients at the next
    # scaling level given the input size, the length of the filter and
    # the applied mode.
    # We use this in the following way (per dimension):
    # - length[0] = original data length
    # - length[n+1] = dwt_coeff_len(length[n], ...)
    # - until n = nscales
    size_list = [shape]
    for scale in range(nscales):
        shp = tuple(pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                       mode=pad_mode)
                    for n in size_list[scale])
        size_list.append(shp)

    # Add a duplicate of the last entry for the approximation coefficients
    size_list.append(size_list[-1])

    # We created the list in reversed order compared to what pywt expects
    size_list.reverse()
    return size_list


def pywt_dict_to_array(coeffs, size_list):
    """Convert a PyWavelet coefficient dictionary into a flat array.

    Related to 1D, 2D and 3D discrete wavelet transforms with `axes` option.

    Parameters
    ----------
    coeff : ordered `dict`
        Coefficients are organized in the dictionary with the following keys:

        In 1D:
        ``a`` = approximation,

        ``d`` = detail

        In 2D:

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:
        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients when `axes` option is used.

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
    """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]

    if ndim == 1:
        a = coeffs['a']
        d = coeffs['d']
        flat_coeff[start:stop] = a.ravel()
        start, stop = stop, stop + flat_sizes[1]
        flat_coeff[start:stop] = d.ravel()

    elif ndim == 2:
        keys = list(coeffs.keys())
        keys.sort()
        details = tuple(coeffs[key] for key in keys if 'd' in key)
        coeff_list = []
        coeff_list.append(details)
        coeff_list.append(coeffs['aa'])
        coeff_list.reverse()
        flat_coeff = pywt_list_to_array(coeff_list, size_list)

    elif ndim == 3:
        keys = list(coeffs.keys())
        keys.sort()
        details = tuple(coeffs[key] for key in keys if 'd' in key)
        coeff_list = []
        coeff_list.append(details)
        coeff_list.append(coeffs['aaa'])
        coeff_list.reverse()
        flat_coeff[start:stop] = coeffs['aaa'].ravel()
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff_list[1:]):
            for dcoeff in detail_coeffs:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def pywt_list_to_array(coeff, size_list):
    """Convert a Pywavelets coefficient list into a flat array.

    Related to 1D, 2D and 3D multilevel discrete wavelet transforms.

    Parameters
    ----------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ..., (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail

        In 2D:

        ``[aaN, (adN, daN, ddN), ..., (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level,

        ``size_list[1]`` = size of the detailed coefficients at
        the coarsest level,

        ``size_list[N]`` = size of the detailed coefficients at
        the finest level,

        ``size_list[N+1]`` = size of original image,

        ``N`` =  the number of scaling levels

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
      """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = coeff[0].ravel()

    if dcoeffs_per_scale == 1:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = detail_coeffs.ravel()
    elif dcoeffs_per_scale == 3:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            for dcoeff in detail_coeffs:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()
    elif dcoeffs_per_scale == 7:
        for ind in range(1, len(size_list) - 1):
            detail_coeffs_dict = coeff[ind]
            keys = list(detail_coeffs_dict.keys())
            keys.sort()
            details = tuple(detail_coeffs_dict[key] for key in
                            keys if 'd' in key)
            fsize = flat_sizes[ind]
            for dcoeff in details:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def array_to_pywt_dict(coeff, size_list):
    """Convert a flat array into a PyWavelet coefficient dictionary.

    For 1D, 2D and 3D discrete wavelet transform with `axes` option.

    Parameters
    ----------
    coeff : `DiscreteLpVector`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of wavelet coefficient sizes.

    Returns
    -------
    coeff : ordered `dict` with the following key words:

        In 1D:
        ``a`` = approximation,

        ``d`` = detail

        In 2D:

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:
        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

    """
    ndim = len(size_list[0])

    if ndim == 1:
        flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
        start = 0
        stop = flat_sizes[0]
        a = np.asarray(coeff)[start:stop].reshape(size_list[0])
        start, stop = stop, stop + flat_sizes[1]
        d = np.asarray(coeff)[start:stop].reshape(size_list[1])

        coeff_dict = {'a': a, 'd': d}

    elif ndim == 2:
        flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
        start = 0
        stop = flat_sizes[0]
        aa = np.asarray(coeff)[start:stop].reshape(size_list[0])
        start, stop = stop, stop + flat_sizes[1]
        ad = np.asarray(coeff)[start:stop].reshape(size_list[1])
        start, stop = stop, stop + flat_sizes[1]
        da = np.asarray(coeff)[start:stop].reshape(size_list[1])
        start, stop = stop, stop + flat_sizes[1]
        dd = np.asarray(coeff)[start:stop].reshape(size_list[1])
        coeff_dict = {'aa': aa, 'ad': ad, 'da': da, 'dd': dd}

    elif ndim == 3:
        coeff_list = array_to_pywt_list(coeff, size_list)
        aaa = coeff_list[0]
        coeff_dict = coeff_list[1]
        coeff_dict['aaa'] = aaa

    return coeff_dict


def array_to_pywt_list(coeff, size_list):
    """Convert a flat array into a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient list.

    For multilevel 1D, 2D and 3D discrete wavelet transforms.

    Parameters
    ----------
    coeff : `DiscreteLpVector`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of coefficient sizes such that,

       ``size_list[0]`` = size of approximation coefficients at the coarsest
                          level,

       ``size_list[1]`` = size of the detailedetails at the coarsest level,

       ``size_list[N]`` = size of the detailed coefficients at the finest
                          level,

       ``size_list[N+1]`` = size of original image,

       ``N`` =  the number of scaling levels

    Returns
    -------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ... (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail,

        In 2D:

        ``[aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    start = 0
    stop = flat_sizes[0]
    coeff_list = [np.asarray(coeff)[start:stop].reshape(size_list[0])]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    if dcoeffs_per_scale == 1:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = np.asarray(coeff)[start:stop]
            coeff_list.append(detail_coeffs)
    elif ndim == 2:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            coeff_list.append(detail_coeffs)
    elif ndim == 3:
        for ind in range(1, len(size_list) - 1):
            fsize = flat_sizes[ind]
            shape = size_list[ind]
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            (aad, ada, add, daa, dad, dda, ddd) = detail_coeffs
            coeff_dict = {'aad': aad, 'ada': ada, 'add': add,
                          'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
            coeff_list.append(coeff_dict)

    return coeff_list


class WaveletTransform(Operator):

    """Discrete wavelet trafo between discrete L2 spaces."""

    def __init__(self, domain, nscales, wbasis, pad_mode, axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        nscales : int
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .

        wbasis :  {string, ``pywt.Wavelet``}
            If a string is given, converts to a ``pywt.Wavelet``.
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zpd': zero-padding -- signal is extended by adding zero samples

            'cpd': constant padding -- border values are replicated

            'sym': symmetric padding -- signal extension by mirroring samples

            'ppd': periodic padding -- signal is trated as a periodic one

            'sp1': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'per': periodization -- like periodic-padding but gives the
            smallest possible number of decomposition coefficients.
        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            If axes is given nscales is discarded.

        Examples
        --------
        >>> import odl, pywt
        >>> wbasis = pywt.Wavelet('db1')
        >>> discr_domain = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> op = WaveletTransform(discr_domain, nscales=1,
        ...                               wbasis=wbasis, pad_mode='per')
        >>> op.is_biorthogonal
        True
        """
        self.nscales = int(nscales)
        self.pad_mode = str(pad_mode).lower()

        if isinstance(wbasis, pywt.Wavelet):
            self.wbasis = wbasis
        else:
            self.wbasis = pywt.Wavelet(wbasis)

        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance.'
                            ''.format(domain))

        if domain.exponent != 2.0:
            raise ValueError('`domain` Lp exponent is {} instead of 2.0.'
                             ''.format(domain.exponent))

        max_level = pywt.dwt_max_level(domain.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales > max_level:
            raise ValueError('cannot use more than {} scaling levels, '
                             'got {}'.format(max_level, self.nscales))

        self._axes = axes
        if axes is not None:
            if len(axes) != domain.ndim:
                raise ValueError('The length of the axes has to be equal to '
                                 'dimension. Axes len is {} and dimension '
                                 'is {}.'.format(len(axes), domain.ndim))
            else:
                self.size_list = coeff_size_list_axes(
                    domain.shape, self.wbasis, self.pad_mode, self._axes)
                ran_size = np.prod(self.size_list[0])

                if domain.ndim == 1:
                    ran_size += sum(np.prod(shape) for shape in
                                    self.size_list[1:-1])
                elif domain.ndim == 2:
                    ran_size += sum(3 * np.prod(shape) for shape in
                                    self.size_list[1:-1])
                elif domain.ndim == 3:
                    ran_size += sum(7 * np.prod(shape) for shape in
                                    self.size_list[1:-1])
                else:
                    raise NotImplementedError('ndim {} not 1, 2 or 3'
                                              ''.format(len(domain.ndim)))
        else:
            self.size_list = coeff_size_list(domain.shape, self.nscales,
                                             self.wbasis, self.pad_mode)

            ran_size = np.prod(self.size_list[0])
            if domain.ndim == 1:
                ran_size += sum(np.prod(shape) for shape in
                                self.size_list[1:-1])
            elif domain.ndim == 2:
                ran_size += sum(3 * np.prod(shape) for shape in
                                self.size_list[1:-1])
            elif domain.ndim == 3:
                ran_size += sum(7 * np.prod(shape) for shape in
                                self.size_list[1:-1])
            else:
                raise NotImplementedError('ndim {} not 1, 2 or 3'
                                          ''.format(len(domain.ndim)))

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        range = domain.dspace_type(ran_size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete wavelet transform.

        Parameters
        ----------
        x : `domain` element

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if self._axes is None:
            if x.space.ndim == 1:
                coeff_list = pywt.wavedec(x, self.wbasis, self.pad_mode,
                                          self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if x.space.ndim == 2:
                coeff_list = pywt.wavedec2(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if x.space.ndim == 3:
                coeff_list = pywt.wavedecn(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)

                return self.range.element(coeff_arr)
        else:
            coeff_dict = pywt.dwtn(x, self.wbasis, self.pad_mode, self._axes)
            coeff_arr = pywt_dict_to_array(coeff_dict, self.size_list)
            return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """Adjoint wavelet transform.

        Returns
        -------
        adjoint : `WaveletTransformInverse`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.
        """
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """Inverse wavelet transform.

        Returns
        -------
        inverse : `WaveletTransformInverse`

        See Also
        --------
        adjoint
        """
        return WaveletTransformInverse(
            range=self.domain, nscales=self.nscales, wbasis=self.wbasis,
            mode=self.pad_mode, axes=self._axes)


class WaveletTransformInverse(Operator):

    """Discrete inverse wavelet trafo between discrete L2 spaces."""

    def __init__(self, range, nscales, wbasis, pad_mode, axes=None):
        """Initialize a new instance.

         Parameters
        ----------
        range : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        nscales : int
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .

        wbasis :  ``pywt.Wavelet``
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zpd': zero-padding -- signal is extended by adding zero samples

            'cpd': constant padding -- border values are replicated

            'sym': symmetric padding -- signal extension by mirroring samples

            'ppd': periodic padding -- signal is trated as a periodic one

            'sp1': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'per': periodization -- like periodic-padding but gives the
            smallest possible number of decomposition coefficients.
        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            If axes is given nscales is discarded.
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.pad_mode = str(pad_mode).lower()

        if not isinstance(range, DiscreteLp):
            raise TypeError('range {!r} is not a `DiscreteLp` instance'
                            ''.format(range))

        if range.exponent != 2.0:
            raise ValueError('range Lp exponent should be 2.0, got {}'
                             ''.format(range.exponent))

        max_level = pywt.dwt_max_level(range.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales > max_level:
            raise ValueError('cannot use more than {} scaling levels, '
                             'got {}'.format(max_level, self.nscales))

        self._axes = axes
        if axes is not None:
            if len(axes) != range.ndim:
                raise ValueError('The length of the axes has to be equal to '
                                 'dimension. Axes len is {} and dimension '
                                 'is {}.'.format(len(axes), range.ndim))
            else:
                self.size_list = coeff_size_list_axes(
                    range.shape, self.wbasis, self.pad_mode, self._axes)
                dom_size = np.prod(self.size_list[0])

                if range.ndim == 1:
                    dom_size += sum(np.prod(shape) for shape in
                                    self.size_list[1:-1])
                elif range.ndim == 2:
                    dom_size += sum(3 * np.prod(shape) for shape in
                                    self.size_list[1:-1])
                elif range.ndim == 3:
                    dom_size += sum(7 * np.prod(shape) for shape in
                                    self.size_list[1:-1])
                else:
                    raise NotImplementedError('ndim {} not 1, 2 or 3'
                                              ''.format(len(range.ndim)))
        else:
            self.size_list = coeff_size_list(range.shape, self.nscales,
                                             self.wbasis, self.pad_mode)

            dom_size = np.prod(self.size_list[0])
            if range.ndim == 1:
                dom_size += sum(np.prod(shape) for shape in
                                self.size_list[1:-1])
            elif range.ndim == 2:
                dom_size += sum(3 * np.prod(shape) for shape in
                                self.size_list[1:-1])
            elif range.ndim == 3:
                dom_size += sum(7 * np.prod(shape) for shape in
                                self.size_list[1:-1])
            else:
                raise NotImplementedError('ndim {} not 1, 2 or 3'
                                          ''.format(range.ndim))

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        domain = range.dspace_type(dom_size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 1D, 2D or 3D inverse wavelet transform.

        Parameters
        ----------
        coeff : `domain` element
            Wavelet coefficients supplied to the wavelet reconstruction.

        Returns
        -------
        arr : `numpy.ndarray`
            Result of the wavelet reconstruction.

        """
        if self._axes is None:
            if len(self.range.shape) == 1:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 2:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec2(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 3:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverecn(coeff_list, self.wbasis, self.pad_mode)
                return x

        else:
            coeff_dict = array_to_pywt_dict(coeff, self.size_list)
            x = pywt.idwtn(coeff_dict, self.wbasis, self.pad_mode, self._axes)
            return x

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `WaveletTransform`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.

        See Also
        --------
        inverse
        """
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """Inverse of this operator.

        Returns
        -------
        inverse : `WaveletTransform`

        See Also
        --------
        adjoint
        """
        return WaveletTransform(domain=self.range, nscales=self.nscales,
                                wbasis=self.wbasis, pad_mode=self.pad_mode,
                                axes=self._axes)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
