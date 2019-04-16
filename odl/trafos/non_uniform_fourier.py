# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discretized non-uniform Fourier transform on L^p spaces."""

from __future__ import division

import numpy as np

from odl.discr import DiscreteLp
from odl.operator import Operator
from odl.space import cn
from odl.space.base_tensors import TensorSpace
from odl.trafos.backends.pynfft_bindings import (
    PYNFFT_AVAILABLE, normalize_samples)
from odl.util import complex_dtype

if PYNFFT_AVAILABLE:
    import pynfft


class NonUniformFourierTransformBase(Operator):

    """Base class for non-uniform Fourier Transform and its adjoint."""

    def __init__(self, domain, samples, range=None, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : DiscreteLp
            Uniformly discretized space where the Fourier transform takes
            its inputs.
        samples : array-like
            Array of points at which the non-uniform FFT should be evaluated.
            It must have shape ``(M, d)``, where ``M`` is the number of
            samples and ``d`` the dimension of ``domain``. The magnitude
            of the samples may not exceed ``pi / s``, where ``s`` is
            ``domain.cell_sides``.
        range : `TensorSpace`, optional
            Range of the non-uniform FFT operator. It must be a
            one-dimensional space with shape ``(M,)``. By default, it is
            chosen as ``cn(M)`` with the same arithmetic precision as
            ``domain.dtype``.
        skip_normalization : bool, optional
            Whether the samples normalization step should be skipped.
            If ``True``, all ``samples`` are expected to lie between -0.5
            (inclusive) and +0.5 (exclusive).
        nfft : pynfft.nfft.NFFT, optional
            Instance of class implementing this transform. By default, a
            new one is created.

            .. note::
                - The ``nfft.f`` and ``nfft.f_hat`` attributes will not be
                  used.
                - The ``nfft.precompute()`` method will be called in any case
                  during the first evaluation.

        kwargs
            Additional keyword arguments passed on to the ``NFFT`` class
            constructor, except for its ``N`` and ``M`` parameters. Not used
            if ``nfft`` is given.

        Notes
        -----
        - The ``pynfft`` backend library currently only supports double
          precision arithmetic, thus all computations are carried out with
          double precision irrespective of the ``dtype`` of ``domain`` and
          ``range``.
        """
        # Input checking and transformation
        if not isinstance(domain, DiscreteLp):
            raise TypeError(
                '`domain` must be a `DiscreteLp`, got {!r}'
                ''.format(domain)
            )
        if not domain.is_uniform:
            raise ValueError('`domain` is not uniformly discretized')

        samples = np.array(samples, copy=True, ndmin=2)
        if samples.dtype.kind != 'f':
            raise ValueError(
                '`samples` must be of real dtype, got array with `dtype={}`'
                ''.format(samples.dtype)
            )
        if samples.ndim != 2:
            raise ValueError(
                '`samples` must have 2 dimensions, got array with `ndim={}`'
                ''.format(samples.ndim)
            )

        M = samples.shape[0]
        if range is None:
            range = cn(M, dtype=complex_dtype(domain.dtype))

        if not isinstance(range, TensorSpace):
            raise TypeError(
                '`range` must be a `TensorSpace`, got {!r}'.format(range)
            )
        if range.shape != (M,):
            raise ValueError(
                '`range` must have shape {}, but `range.shape == {}`'
                ''.format((M,), range.shape)
            )
        if range.dtype.kind != 'c':
            raise ValueError(
                '`range` must have a complex `dtype`, but `range.dtype == {}`'
                ''.format(range.dtype)
            )

        skip_normalization = bool(kwargs.pop('skip_normalization', False))
        nfft = kwargs.pop('nfft', None)
        if nfft is None:
            nfft = pynfft.NFFT(N=domain.shape, M=M, **kwargs)

        if not isinstance(nfft, pynfft.NFFT):
            raise TypeError(
                '`nfft` must be a `pynfft.NFFT` instance, got {!r}'
                ''.format(nfft)
            )
        if nfft.N != domain.shape or nfft.M != samples.shape[0]:
            raise ValueError(
                '`nfft` attributes `N` and `M` ({} and {}) inconsistent '
                'with `domain.shape` and `len(samples) ({} and{})'
                ''.format(nfft.N, nfft.M, domain.shape, samples.shape[0])
            )

        # Init
        super(NonUniformFourierTransformBase, self).__init__(
            domain, range, linear=True,
        )

        self.__samples = samples
        self._skip_normalization = skip_normalization
        self._has_run = False
        self.__nfft = nfft

    @property
    def samples(self):
        """Samples at which this NFFT is evaluated."""
        return self.__samples

    @property
    def nfft(self):
        """Instance of the class implementing this NFFT."""
        return self.__nfft


class NonUniformFourierTransform(NonUniformFourierTransformBase):

    r"""Forward non-uniform Fourier Transform.

    The non-uniform FFT computes

    .. math::
        \widehat{f}(\xi_k)
        = \sum_{0 \leq j \leq N-1} f_j\,
        \mathrm{e}^{-\mathrm{i} s j \cdot \xi_k},\quad
        k=0, \dots, M-1.

    for function values :math:`f_j` on a (multi-dimensional) uniform grid
    and arbitrary frequencies :math:`\xi_k` within the cube
    :math:`-\pi\, s^{-1} \leq \xi < \pi\, s^{-1}`, where :math:`s` is the
    spacing of the grid on which the values :math:`f_j` are given.
    """

    def __init__(self, domain, samples, range=None, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : DiscreteLp
            Uniformly discretized space where the Fourier transform takes
            its inputs.
        samples : array-like
            Array of points at which the non-uniform FFT should be evaluated.
            It must have shape ``(M, d)``, where ``M`` is the number of
            samples and ``d`` the dimension of ``domain``. The magnitude
            of the samples may not exceed ``pi / s``, where ``s`` is
            ``domain.cell_sides``.
        range : `TensorSpace`, optional
            Range of the non-uniform FFT operator. It must be a
            one-dimensional space with shape ``(M,)``. By default, it is
            chosen as ``cn(M)`` with the same arithmetic precision as
            ``domain.dtype``.
        skip_normalization : bool, optional
            Whether the samples normalization step should be skipped.
            If ``True``, all ``samples`` are expected to lie between -0.5
            (inclusive) and +0.5 (exclusive).
        kwargs
            Additional keyword arguments passed on to the ``NFFT`` class
            constructor, except for its ``N`` and ``M`` parameters.

        Notes
        -----
        - The ``pynfft`` backend library currently only supports double
          precision arithmetic, thus all computations are carried out with
          double precision irrespective of the ``dtype`` of ``domain`` and
          ``range``.
        """
        super(NonUniformFourierTransform, self).__init__(
            domain, samples, range, **kwargs
        )

    def _call(self, x):
        """Return ``self(x)``."""
        if not self._has_run:
            if not self._skip_normalization:
                normalize_samples(
                    self.samples, self.domain.cell_sides, out=self.samples
                )
            self.nfft.precompute()
            self._has_run = True

        self.nfft.f_hat = np.asarray(x)
        out = self.nfft.trafo()
        # TODO(kohr-h): normalize to match uniform FT
        return out

    @property
    def adjoint(self):
        r"""Adjoint operator.

        The adjoint is given by

        .. math::
            g_j =

            = \sum_{k=0}^{M-1} \widehat{f}(\xi_k)\,
            \mathrm{e}^{\mathrm{i} s j \cdot \xi_k},\quad
            0 \leq j < N - 1.
        """
        return NonUniformFourierTransformAdjoint(
            domain=self.range,
            samples=self.samples,
            range=self.domain,
            nfft=self.nfft,
            skip_normalization=self._has_run,
        )


class NonUniformFourierTransformAdjoint(NonUniformFourierTransformBase):
    """Adjoint of Non uniform Fast Fourier Transform.
    """
    def __init__(
        self, space, samples, skip_normalization=False):
        """Initialize a new instance.

        Parameters
        ----------
        space : DiscreteLp
            The uniform space in which the data lies
        samples : aray-like
            List of the fourier space positions where the coefficients are
            computed.
        skip_normalization : bool, optional
            Whether the normalization step should be skipped
        """
        if not isinstance(space, DiscreteLp) or not space.is_uniform:
            raise ValueError("`space` should be a uniform `DiscreteLp`")
        super(NonUniformFourierTransformAdjoint, self).__init__(
            space=space,
            samples=samples,
            domain=cn(len(samples)),
            range=space,
            skip_normalization=skip_normalization,
        )

    @property
    def adjoint(self):
        """Adjoint of the adjoint, i.e., the forward transform."""
        return NonUniformFourierTransform(
            domain=self.range,
            samples=self.samples,
            range=self.domain,
            nfft=self.nfft,
            skip_normalization=self._has_run,
        )

    def _call(self, x):
        """Compute the adjoint non uniform FFT.

        Parameters
        ----------
        x : `numpy.ndarray`
            The data whose non uniform FFT adjoint you want to compute

        Returns
        -------
        out_normalized : `numpy.ndarray`
            Result of the adjoint transform
        """
        if not self.__has_run:
            if not self._skip_normalization:
                normalize_samples(
                    self.samples, self.domain.cell_sides, out=self.samples
                )
            self.nfft.precompute()
            self._has_run = True

        self.nfft.f = np.asarray(x)
        out = self.nfft.adjoint()
        # TODO(kohr-h): normalize to match uniform FT
        return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYNFFT_AVAILABLE)
