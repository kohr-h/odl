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

"""Phase contrast reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
from odl.discr.lp_discr import DiscreteLp
from odl.operator.operator import Operator


class IntensityOperator(Operator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must be a power space
            of ``range``.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.

        Notes
        -----
        This operator maps a real vector field :math:`f = (f_1, \dots, f_d)`
        to its pointwise intensity

            :math:`\mathcal{I}(f) = \\lvert f\\rvert^2 :
            x \mapsto \sum_{j=1}^d f_i(x)^2`.

        """
        if domain is None and range is None:
            raise ValueError('either domain or range must be specified.')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('range {!r} is not a DiscreteLp instance.'
                                ''.format(range))
            domain = range.complex_space

        if range is None:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('domain {!r} is not a `DiscreteLp` '
                                'instance.'.format(domain))
            range = domain.real_space

        super().__init__(domain, range, linear=False)

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        out[:] = x.real
        out *= out
        out += x.imag * x.imag

    def derivative(self, f):
        """Return the derivative operator in ``f``.

        Parameters
        ----------
        f : domain element
            Point at which the derivative is taken

        Returns
        -------
        deriv : `Operator`
            Derivative operator at the specified point

        Notes
        -----
        The derivative of the intensity operator is given by

            :math:`\partial \mathcal{I}(f_1, f_2)(h_1, h_2) =
            2 (f_1 h_1 + f_2 h_2)`.

        Its adjoint maps a function :math:`g` to the product space
        element

            :math:`\\left[\partial\mathcal{I}(f_1, f_2)\\right]^*(g) =
            2 (f_1 g, f_2 g)`.
        """
        op = self

        class IntensOpDeriv(Operator):
            def __init__(self):
                super().__init__(op.domain, op.range, linear=True)

            def _call(self, h):
                return 2 * (f.real * h.real + f.imag * h.imag)

            @property
            def adjoint(self):
                class IntensOpDerivAdj(Operator):
                    def __init__(self):
                        super().__init__(op.range, op.domain, linear=True)

                    def _call(self, g, out):
                        out.real[:] = 2 * g * f.real
                        out.imag[:] = 2 * g * f.imag

                return IntensOpDerivAdj()

        return IntensOpDeriv()


def propagation_kernel_ft(x, **kwargs):
    """Fresnel propagation kernel.

    Notes
    -----
    The kernel is defined as

    .. math::

        k(\\xi) = -\\frac{\kappa}{2}
        \exp\\left(\\frac{i d}{2\kappa} \|\\xi\|^2 \\right),

    where :math:`\kappa` is the wave number of the incoming wave and
    :math:`d` the propagation distance.
    """
    wavenum = float(kwargs.pop('wavenum', 1.0))
    prop_dist = float(kwargs.pop('prop_dist', 1.0))
    scaled = [np.sqrt(prop_dist / (2 * wavenum)) * xi for xi in x[1:]]
    kernel = sum(sxi ** 2 for sxi in scaled)
    kernel += wavenum * prop_dist
    result = np.exp(1j * kernel)
    result *= -wavenum / 2
    return result


# %% Example: Real and imaginary part, single distance

wavenum = 10000.0
prop_dist = 0.1

# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-0.1] * 3, max_pt=[0.1] * 3,
                               shape=[300] * 3, dtype='complex64')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, np.pi, 60)
# Detector: uniformly sampled, n = (558, 558), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-0.16] * 2, [0.16] * 2,
                                           [300] * 2)

geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
ft = odl.trafos.FourierTransform(ray_trafo.range, axes=[1, 2], impl='pyfftw')
prop_kernel = ft.range.element(propagation_kernel_ft, wavenum=wavenum,
                               prop_dist=prop_dist)

prop_op = ft.inverse * prop_kernel * ft

plane_wave = prop_op.range.element(
    np.exp(1j * wavenum * prop_dist) * prop_op.range.one())

intens_op = IntensityOperator(prop_op.range)

single_dist_phase_op = intens_op * (prop_op * ray_trafo + plane_wave)

phantom = (1e-5 * odl.phantom.shepp_logan(reco_space, modified=True) +
           1e-5j * odl.phantom.shepp_logan(reco_space, modified=True))
data = single_dist_phase_op(phantom)

reco = ray_trafo.domain.zero()
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
odl.solvers.conjugate_gradient_normal(single_dist_phase_op, reco, data,
                                      niter=10, callback=callback)


lin_at_one = single_dist_phase_op.derivative(single_dist_phase_op.domain.one())
backprop = lin_at_one.adjoint(data)
