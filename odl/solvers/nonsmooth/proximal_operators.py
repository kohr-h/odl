# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Factory functions for creating proximal operators.

Functions with ``convex_conj`` mean the proximal of the convex conjugate and
are provided for convenience.

For more details see :ref:`proximal_operators` and references therein. For
more details on proximal operators including how to evaluate the proximal
operator of a variety of functions see `[PB2014]
<https://web.stanford.edu/~boyd/papers/prox_algs.html>`_.

References
----------
[PB2014] Parikh, N, and Boyd, S. *Proximal Algorithms*.
Foundations and Trends in Optimization, 1 (2014), pp 127-239.
"""

#TODO: LinearSpace -> TensorSpace

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.operator import (
    ConstantOperator, DiagonalOperator, IdentityOperator, MultiplyOperator,
    Operator, PointwiseNorm)
from odl.set.space import LinearSpaceElement
from odl.space import ProductSpace
from odl.util import (
    signature_string_parts, repr_string, npy_printoptions, REPR_PRECISION,
    method_repr_string, array_str)

__all__ = ('proximal_separable_sum', 'proximal_convex_conj',
           'proximal_translation', 'proximal_arg_scaling',
           'proximal_quadratic_perturbation', 'proximal_composition',
           'proximal_const_func', 'proximal_indicator_box',
           'proximal_l1', 'proximal_indicator_linf_unit_ball',
           'proximal_l2', 'proximal_indicator_l2_unit_ball',
           'proximal_l1_l2', 'proximal_indicator_linf_l2_unit_ball',
           'proximal_kl', 'proximal_convex_conj_kl',
           'proximal_convex_conj_kl_cross_entropy', 'proximal_huber')


def proximal_separable_sum(*factory_funcs):
    r"""Return the proximal factory for the separable sum of functionals.

    Parameters
    ----------
    factory_func1, ..., factory_funcN : callable
        Proximal operator factories, one for each of the functionals in
        the separable sum.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the separable sum of functionals.

    Notes
    -----
    That two functionals :math:`F` and :math:`G` are separable across variables
    means that :math:`F((x, y)) = F(x)` and :math:`G((x, y)) = G(y)`. In
    this case, the proximal operator of the sum is given by

    .. math::
        \mathrm{prox}_{\sigma (F(x) + G(y))}(x, y) =
        (\mathrm{prox}_{\sigma F}(x), \mathrm{prox}_{\sigma G}(y)).
    """

    def separable_sum_prox_factory(sigma):
        """Proximal factory for a separable sum of functionals ``F_i``.

        Parameters
        ----------
        sigma : positive float or sequence of positive floats
            Step size parameter(s). If a sequence, the length must match
            the length of the ``factory_list``. Furthermore, each of the
            sequence entries can be sequences or arrays, depending on
            what the used proximal factories support.

        Returns
        -------
        diag_op : `DiagonalOperator`
            The operator ``(prox[sigma_1](F_i), ..., prox[sigma_n](F_n))``.
        """
        if np.isscalar(sigma):
            sigma = [sigma] * len(factory_funcs)

        return DiagonalOperator(
            *[factory(sigma_i)
              for sigma_i, factory in zip(sigma, factory_funcs)])

    return separable_sum_prox_factory


def proximal_convex_conj(prox_factory):
    r"""Return the proximal factory for the convex conjugate of a functional.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the convex conjugate.

    Notes
    -----
    The Moreau identity states that for any convex function :math:`F` with
    convex conjugate :math:`F^*`, the proximals satisfy

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) + \sigma \,
        \mathrm{prox}_{F / \sigma}(x / \sigma) = x

    where :math:`\sigma` is a scalar step size. Using this, the proximal of
    the convex conjugate is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) =
        x - \sigma \, \mathrm{prox}_{F / \sigma}(x / \sigma)

    Note that since :math:`(F^*)^* = F`, this can be used to get the proximal
    of the original function from the proximal of the convex conjugate.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    def convex_conj_prox_factory(sigma):
        """Proximal factory for the convex conjugate of a functional ``F``.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * F^*``.
        """
        # Get the underlying space. At the same time, check if the given
        # prox_factory accepts `sigma` of the given type.
        space = prox_factory(sigma).domain

        mult_right = MultiplyOperator(1 / sigma, domain=space, range=space)
        mult_left = MultiplyOperator(sigma, domain=space, range=space)
        result = (IdentityOperator(space) -
                  mult_left * prox_factory(1 / sigma) * mult_right)
        return result

    return convex_conj_prox_factory


def proximal_translation(prox_factory, y):
    r"""Return the proximal factory for a translated functional.

    The returned `proximal factory` is associated with the translated
    functional ::

        x --> F(x - y)

    given the proximal factory of the original functional ``F``.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    y : Element in domain of the functional by which should be translated.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the translated functional.

    Notes
    -----
    Given a functional :math:`F`, this is calculated according to the rule

    .. math::
        \mathrm{prox}_{\sigma F(\cdot - y)}(x) =
        y + \mathrm{prox}_{\sigma F}(x - y)

    where :math:`y` is the translation, and :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """

    def translation_prox_factory(sigma):
        """Proximal factory for the translated functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``s * F( . - y)`` where ``s`` is the
            step size.
        """
        return (ConstantOperator(y) + prox_factory(sigma) *
                (IdentityOperator(y.space) - ConstantOperator(y)))

    return translation_prox_factory


def proximal_arg_scaling(prox_factory, scaling):
    r"""Return the proximal factory for a right-scaled functional.

    The returned `proximal factory` is associated with the functional whose
    argument is scaled by a factor, ::

        x --> F(a * x)

    given the proximal factory of the original functional ``F``.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    scaling : float or array-like
        Scaling parameter. Can be a scalar, a pointwise positive space
        element or a sequence of positive floats if the provided
        ``prox_factory`` support such types as step size parameters.

        .. note::
            - A scalar 0 is valid, but arrays may not contain zeros since
              they lead to division by 0.
            - Complex factors with nonzero imaginary parts are not supported
              yet. For such scalars, an exception will be raised.
            - For arrays, these conditions are not checked for efficiency
              reasons.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the right-scaled functional.

    Notes
    -----
    Given a functional :math:`F` and a scaling factor :math:`\alpha`,
    the proximal calculated here is

    .. math::
        \mathrm{prox}_{\sigma F(\alpha \, \cdot)}(x) =
        \frac{1}{\alpha}
        \mathrm{prox}_{\sigma \alpha^2 F(\cdot) }(\alpha x),

    where :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    # TODO: Implement the correct proximal for arrays with zero entries.
    # This proximal maps the components where the factor is zero to
    # themselves.

    # TODO(kohr-h): Implement the full complex version of this?
    if np.isscalar(scaling):
        # We run these checks only for scalars, since they can potentially
        # be computationally expensive for arrays.
        if scaling == 0:
            # Special case
            return proximal_const_func(prox_factory(1.0).domain)
        elif scaling.imag != 0:
            raise NotImplementedError('complex scaling not supported.')
        else:
            scaling = float(scaling.real)
    else:
        scaling = np.asarray(scaling)

    def arg_scaling_prox_factory(sigma):
        """Proximal factory for the right-scaled functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that, and if the product
            ``sigma * scaling`` makes sense.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * F( . * a)``.
        """
        scaling_square = scaling * scaling
        prox = prox_factory(sigma * scaling_square)
        space = prox.domain
        mult_inner = MultiplyOperator(scaling, domain=space, range=space)
        mult_outer = MultiplyOperator(1 / scaling, domain=space, range=space)
        return mult_outer * prox * mult_inner

    return arg_scaling_prox_factory


def proximal_quadratic_perturbation(prox_factory, a, u=None):
    r"""Return the proximal factory for a quadratically perturbed functional.

    The returned `proximal factory` is associated with the functional ::

        x --> F(x) + <x, a * x + u>

    given the proximal factory of the original functional ``F``, where
    ``a`` is a scalar and ``u`` a vector.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    a : non-negative float
        Coefficient of the quadratic term.
    u : array-like, optional
        Element of the functional domain that defines the linear term.
        The default ``None`` means zero.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the perturbed functional.

    Notes
    -----
    Given a functional :math:`F`, this proximal is calculated according to
    the rule

    .. math::
        \mathrm{prox}_{\sigma \left(F( \cdot ) + a \| \cdot \|^2 +
        <u, \cdot >\right)}(x) =
        c \, \mathrm{prox}_{\sigma F( \cdot \, c)}
        \big((x - \sigma u)\cdot c\big),

    where :math:`c` is the constant

    .. math::
        c = \frac{1}{\sqrt{2 \sigma a + 1}},

    :math:`a` is the scaling parameter belonging to the quadratic term,
    :math:`u` is the space element defining the linear functional, and
    :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    a = float(a)
    if a < 0:
        raise ValueError('scaling parameter muts be non-negative, got {}'
                         ''.format(a))

    if u is not None and not isinstance(u, LinearSpaceElement):
        raise TypeError('`u` must be `None` or a `LinearSpaceElement` '
                        'instance, got {!r}.'.format(u))

    def quadratic_perturbation_prox_factory(sigma):
        """Proximal factory for the right-scaled functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``x --> sigma * (F(x) + <x, a * x + u>)``.
        """
        if np.isscalar(sigma):
            sigma = float(sigma)
        else:
            sigma = np.asarray(sigma)

        const = 1.0 / np.sqrt(sigma * 2.0 * a + 1)
        prox = proximal_arg_scaling(prox_factory, const)(sigma)
        if a != 0:
            space = prox.domain
            mult_op = MultiplyOperator(const, domain=space)

        if u is None and a == 0:
            return prox
        elif u is None and a != 0:
            return mult_op * prox * mult_op
        elif u is not None and a == 0:
            return prox - sigma * u
        else:
            return mult_op * prox * (mult_op - (sigma * const) * u)

    return quadratic_perturbation_prox_factory


def proximal_composition(prox_factory, operator, mu):
    r"""Return the proximal factory for a functional composed with an operator.

    The returned `proximal factory` is associated with the functional ::

        x --> F(L x)

    given the proximal factory of the original functional ``F``, where
    ``L`` is an operator.

    .. note::
        The explicit formula for the proximal used by this function only
        holds for operators :math:`L` that satisfy

        .. math::
            L^* L = \mu\, I_X,

        with the identity operator :math:`I_X` on the domain of :math:`L`
        and a positive constant :math:`\mu`.

        This property is not checked; it is up to the user to ensure that
        passed-in operators are valid in this sense.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    operator : `Operator`
        The operator to be composed with the functional.
    mu : float
        Scalar such that ``(operator.adjoint * operator)(x) = mu * x``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the composed functional.

    Notes
    -----
    Given a linear operator :math:`L` with :math:`L^*L x = \mu\, x`, and a
    convex functional :math:`F`, the following identity holds:

    .. math::
        \mathrm{prox}_{\sigma F \circ L}(x) = \frac{1}{\mu}
        L^* \left( \mathrm{prox}_{\mu \sigma F}(Lx) \right)

    There is no simple formula for more general operators.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    mu = float(mu)

    def proximal_composition_factory(sigma):
        """Proximal factory for the composed functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``x --> prox[sigma * F * L](x)``.
        """
        prox_sig_mu = prox_factory(sigma * mu)
        return (1 / mu) * operator.adjoint * prox_sig_mu * operator

    return proximal_composition_factory


def proximal_const_func(space):
    r"""Return the proximal factory for a constant functional.

    The returned `proximal factory` is associated with the functional ::

        x --> const

    It always returns the `IdentityOperator` on the space of ``x``.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the constant functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of a constant functional.
        It always returns the identity operator, independently of its
        input parameter.
    """

    def identity_factory(sigma):
        """Proximal factory for the identity functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter (unused but kept to maintain a uniform
            interface).

        Returns
        -------
        proximal : `IdentityOperator`
            The proximal operator of a constant functional.
        """
        return IdentityOperator(space)

    return identity_factory


def proximal_indicator_box(space, lower=None, upper=None):
    r"""Return the proximal factory for a box indicator functional.

    The box indicator function assigns the value ``+inf`` to all points
    outside the box, and ``0`` to points inside. Its proximal operator
    is the projection onto that box.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.
    lower : float or ``space`` `element-like`, optional
        The (pointwise) lower bound. The default ``None`` means no lower
        bound, i.e., ``-inf``.
    upper : float or ``space`` `element-like`, optional
        The (pointwise) upper bound. The default ``None`` means no upper
        bound, i.e., ``+inf``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of a box indicator functional.
        It always returns the box projection operator, independently of its
        input parameter.

    Notes
    -----
    The box indicator with lower bound :math:`a` and upper bound :math:`b`
    (can be scalars, vectors or functions) is
    defined as

    .. math::
        \iota_{[a,b]}(x) =
        \begin{cases}
            0      & \text{if } a \leq x \leq b \text{ everywhere}, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the box:

    .. math::
         \mathrm{prox}_{\sigma \iota_{[a,b]}}(x) =
         \begin{cases}
         a & \text{where } x < a, \\
         x & \text{where } a \leq x \leq b, \\
         b & \text{where } x > b.
         \end{cases}
    """
    if lower is not None:
        if np.isscalar(lower):
            lower = float(lower)
        else:
            lower = space.element(lower)
    if upper is not None:
        if np.isscalar(upper):
            upper = float(upper)
        else:
            upper = space.element(upper)

    if np.isscalar(lower) and np.isscalar(upper) and lower > upper:
        raise ValueError('`lower` may not be larger than `upper`, but '
                         '{} > {}'.format(lower, upper))

    class ProximalIndicatorBox(Operator):

        """Proximal operator for a box indicator function."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorBox, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = sigma

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``."""
            if lower is not None and upper is None:
                x.ufuncs.maximum(lower, out=out)
            elif lower is None and upper is not None:
                x.ufuncs.minimum(upper, out=out)
            elif lower is not None and upper is not None:
                x.ufuncs.maximum(lower, out=out)
                out.ufuncs.minimum(upper, out=out)
            else:
                out.assign(x)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> lower = 0
            >>> upper = space.one()
            >>> indicator = odl.solvers.IndicatorBox(space, lower, upper)
            >>> indicator.proximal(2)
            IndicatorBox(
                rn(2), lower=0.0, upper=rn(2).element([ 1.,  1.])
            ).proximal(1.0)
            """
            posargs = [space]
            optargs = [('lower', lower, None),
                       ('upper', upper, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('IndicatorBox', inner_parts)
            return method_repr_string(caller_repr, 'proximal', ['1.0'])

    return ProximalIndicatorBox


def proximal_l1(space):
    r"""Return the proximal factory for the L1 norm functional.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the L1 norm functional.

    Notes
    -----
    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma \|\cdot\|_1` is given by

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_1}(x) =
        \max\big\{|x| - \sigma,\, 0\big\}\ \mathrm{sign}(x),

    where all operations are to be read pointwise.

    For vector-valued :math:`\mathbf{x}`, the (non-isotropic) proximal
    operator is the component-wise scalar proximal:

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_1}(\mathbf{x}) = \left(
            \mathrm{prox}_{\sigma F}(x_1), \dots,
            \mathrm{prox}_{\sigma F}(x_d)
            \right).

    See Also
    --------
    proximal_convex_conj_l1
    proximal_l1_l2 : isotropic variant of the group L1 norm proximal
    """

    class ProximalL1(Operator):

        """Proximal operator of the L1 norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter. A scalar defines a global step size,
                and arrays follow the broadcasting rules.
            """
            super(ProximalL1, self).__init__(
                domain=space, range=space, linear=False)

            if isinstance(space, ProductSpace) and not space.is_power_space:
                dtype = float
            else:
                dtype = space.dtype

            self.sigma = np.asarray(sigma, dtype=dtype)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Assign here in case `x` and `out` are aliased
            sign_x = x.ufuncs.sign()

            x.ufuncs.absolute(out=out)
            out -= self.sigma
            out.ufuncs.maximum(0, out=out)
            out *= sign_x

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l1norm = odl.solvers.L1Norm(space)
            >>> l1norm.proximal(2)
            L1Norm(rn(2)).proximal(2.0)
            """
            posargs = [space]
            inner_parts = signature_string_parts(posargs, [])
            caller_repr = repr_string('L1Norm', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalL1


def proximal_indicator_linf_unit_ball(space):
    r"""Return the proximal factory for the L^inf unit ball indicator.

    The L^inf unit ball indicator function assigns the value ``+inf`` to all
    points outside the unit ball with respect to the inf-norm, and ``0`` to
    points inside. Its proximal operator is the projection onto that ball.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the ball indicator functional.
        It always returns the unit ball projection operator, independently
        of its input parameter.

    Notes
    -----
    The :math:`L^\infty` unit ball indicator is defined as

    .. math::
        \iota_{B_\infty}(x) =
        \begin{cases}
            0      & \text{if } \|x\|_\infty \leq 1, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the ball:

    .. math::
        \mathrm{prox}_{\sigma \iota_{B_\infty}}(x) =
        \mathrm{sign}(x)\, \min\{|x|,\, 1\},

    where all operations are to be understood pointwise.

    For vector-valued functions, since the :math:`\infty`-norm is separable
    across components, the proximal is given as

    .. math::
        \mathrm{prox}_{\sigma \iota_{B_\infty}}(\mathbf{x}) = \left(
            \mathrm{prox}_{\sigma \iota_{B_\infty}}(x_1), \dots,
            \mathrm{prox}_{\sigma \iota_{B_\infty}}(x_d)
            \right).

    See Also
    --------
    proximal_l1 : proximal of the convex conjugate
    proximal_convex_conj_l1_l2 :
        proximal of the isotropic variant for vector-valued functions
    """

    class ProximalIndicatorLinfUnitBall(Operator):

        """Proximal operator for the L^inf unit ball indicator."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorLinfUnitBall, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Take abs first due to possible aliasing of `x` and `out`
            abs_x = x.ufuncs.absolute()
            abs_x.ufuncs.minimum(1, out=abs_x)
            x.ufuncs.sign(out=out)
            out *= abs_x
            return out

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l1norm = odl.solvers.L1Norm(space)
            >>> l1norm.convex_conj.proximal(2)
            IndicatorLpUnitBall(rn(2), exponent='inf').proximal(1.0)
            """
            posargs = [space]
            optargs = [('exponent', float('inf'), None)]
            inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('IndicatorLpUnitBall', inner_parts)
            return method_repr_string(caller_repr, 'proximal', ['1.0'])

    return ProximalIndicatorLinfUnitBall


def proximal_l2(space):
    r"""Return the proximal factory for the L2 norm functional.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the L2 norm functional.

    Notes
    -----
    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma \|\cdot\|_2` is given by

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_2}(y) =
        \max\left\{1 - \frac{\sigma}{\|y\|_2},\ 0\right\}\ y.

    See Also
    --------
    proximal_l2_squared : proximal for squared norm/distance
    proximal_indicator_l2_unit_ball : proximal of the convex conjugate
    """

    class ProximalL2(Operator):

        """Proximal operator of the L2 norm."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalL2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Implement ``self(x, out)``."""
            dtype = getattr(self.domain, 'dtype', float)
            eps = np.finfo(dtype).resolution * 10

            x_norm = x.norm() * (1 + eps)
            if x_norm == 0:
                out.set_zero()
            else:
                out.lincomb(1 - self.sigma / x_norm, x)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l2norm = odl.solvers.L2Norm(space)
            >>> l2norm.proximal(2)
            L2Norm(rn(2)).proximal(2.0)
            """
            posargs = [space]
            inner_parts = signature_string_parts(posargs, [])
            caller_repr = repr_string('L2Norm', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalL2


def proximal_indicator_l2_unit_ball(space):
    r"""Return the proximal factory for the L2 unit ball indicator functional.

    The L2 unit ball indicator function assigns the value ``+inf`` to all
    points outside the unit ball, and ``0`` to points inside. Its proximal
    operator is the projection onto that ball.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the ball indicator functional.
        It always returns the unit ball projection operator, independently
        of its input parameter.

    Notes
    -----
    The :math:`L^2` unit ball indicator is defined as

    .. math::
        \iota_{B_2}(x) =
        \begin{cases}
            0      & \text{if } \|x\|_2 \leq 1, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the ball:

    .. math::
         \mathrm{prox}_{\sigma \iota_{B_2}}(x) =
         \begin{cases}
         \frac{x}{\|x\|_2} & \text{if } \|x\|_2 > 1, \\
         \ x & \text{otherwise.}
         \end{cases}

    See Also
    --------
    proximal_l2
    proximal_convex_conj_l2_squared
    """

    class ProximalIndicatorL2UnitBall(Operator):

        """Proximal operator for the L2 unit ball indicator."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorL2UnitBall, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Implement ``self(x, out)``."""
            dtype = getattr(self.domain, 'dtype', float)
            eps = np.finfo(dtype).resolution * 10

            x_norm = x.norm() * (1 + eps)
            if x_norm > 1:
                out.lincomb(1 / x_norm, x)
            else:
                out.assign(x)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l2norm = odl.solvers.L2Norm(space)
            >>> l2norm.convex_conj.proximal(2)
            IndicatorLpUnitBall(rn(2), exponent=2.0).proximal(1.0)
            """
            posargs = [space]
            optargs = [('exponent', 2.0, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('IndicatorLpUnitBall', inner_parts)
            return method_repr_string(caller_repr, 'proximal', ['1.0'])

    return ProximalIndicatorL2UnitBall


def proximal_l1_l2(space):
    r"""Return the proximal factory for the L1-L2 norm functional.

    The returned proximal is intended for the group L1 norm with inner
    L2 norm. For vector-valued functions, that norm is the isotropic
    variant of the vectorial L1 norm.

    Parameters
    ----------
    space : `ProductSpace`
        Domain of the functional, must be a power space, i.e.,
        ``space.is_power_space is True``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the L1-L2 norm functional.

    Notes
    -----
    For a step size :math:`\sigma`, the proximal operator of the
    :math:`L^{1,2}` norm is given by

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_{1,2}}(\mathbf{x}) =
        \max\left\{1 - \frac{\sigma}{|\mathbf{x}|_2},\ 0\right\}\ \mathbf{x},

    where the Euclidean norm is taken per point along the components of
    :math:`\mathbf{x}`,

    .. math::
        |\mathbf{x}|_2(t) = \bigg(\sum_{i=1}^d x_i(t)^2\bigg)^{1/2},

    and the maximum and multiplication operations are also to be understood
    pointwise.

    See Also
    --------
    proximal_l1 : Scalar or non-isotropic vectorial variant
    """

    class ProximalL1L2(Operator):

        """Proximal operator of the group-L1-L2 norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalL1L2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            if x is out:
                # Handle aliased `x` and `out` (original `x` needed later)
                tmp = x.copy()
            else:
                tmp = x

            # We write the operator as
            # x - x / max(|x|_2 / sig, 1)
            denom = PointwiseNorm(self.domain, exponent=2)(tmp)
            denom /= self.sigma
            denom.ufuncs.maximum(1, out=denom)

            # out <- x / max(|x|_2 / sig, 1)
            for out_i, tmp_i in zip(out, tmp):
                tmp_i.divide(denom, out=out_i)

            # out <- x - out
            # Need original `x` here!
            out.lincomb(1, x, -1, out)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2) ** 3
            >>> l12norm = odl.solvers.GroupL1Norm(space)
            >>> l12norm.proximal(4)
            GroupL1Norm(ProductSpace(rn(2), 3), exponent=2.0).proximal(4.0)
            """
            posargs = [space]
            optargs = [('exponent', 2.0, None)]
            inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('GroupL1Norm', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalL1L2


def proximal_indicator_linf_l2_unit_ball(space):
    r"""Return the proximal factory for the L^inf-L2 unit ball indicator.

    The L^inf-L2 unit ball indicator function assigns the value ``+inf``
    if the L^inf-L2-norm of the input is greater than 1, and 0 otherwise.
    The above condition is true if at any point, the 2-norm of the vector
    at that point is larger than 1. See ``Notes`` for details.

    The proximal of this indicator is the projection onto that unit ball.

    Parameters
    ----------
    space : `ProductSpace`
        Domain of the functional, must be a power space, i.e.,
        ``space.is_power_space is True``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the ball indicator functional.
        It always returns the unit ball projection operator, independently
        of its input parameter.

    Notes
    -----
    The :math:`L^{\infty, 2}` unit ball indicator is defined as

    .. math::
        \iota_{B_{\infty, 2}}(\mathbf{x}) =
        \begin{cases}
            0      & \text{if } |\mathbf{x}|_2 \leq 1 \text{ everywhere}, \\
            \infty & \text{otherwise},
        \end{cases}

    where the Euclidean norm is taken per point along the components of
    :math:`\mathbf{x}`,

    .. math::
        |\mathbf{x}|_2(t) = \bigg(\sum_{i=1}^d x_i(t)^2\bigg)^{1/2}.

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the ball:

    .. math::
        \mathrm{prox}_{\sigma \iota_{B_{\infty, 2}}}(x) =
        \min\{|\mathbf{x}|_2,\, 1\}\, \frac{\mathbf{x}}{|\mathbf{x}|_2},

    where all operations are to be understood pointwise.

    See Also
    --------
    proximal_convex_conj_l1 : Scalar or non-isotropic vectorial variant
    """

    class ProximalIndicatorLinfL2UnitBall(Operator):

        """Proximal operator for the L^inf-L2 unit ball indicator."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalIndicatorLinfL2UnitBall, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # x / max(1, |x|_2)

            denom = PointwiseNorm(self.domain, exponent=2)(x)
            denom.ufuncs.maximum(1, out=denom)

            # Pointwise division (no aliasing issue here)
            for out_i, x_i in zip(out, x):
                x_i.divide(denom, out=out_i)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2) ** 3
            >>> l12norm = odl.solvers.GroupL1Norm(space)
            >>> l12norm.convex_conj.proximal(4)
            IndicatorGroupLinfUnitBall(
                ProductSpace(rn(2), 3), exponent=2.0
            ).proximal(1.0)
            """
            posargs = [space]
            optargs = [('exponent', 2.0, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('IndicatorGroupLinfUnitBall',
                                      inner_parts)
            return method_repr_string(caller_repr, 'proximal', ['1.0'])

    return ProximalIndicatorLinfL2UnitBall


def proximal_kl(space, g=None):
    r"""Return the proximal factory for the KL divergence functional.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.
    g : ``space`` element, optional
        Data term (or prior), must be positive everywhere. For the default
        ``None``, ``space.one()`` is taken.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the KL divergence.

    Notes
    -----
    Since the KL functional is differentiable, its proximal is obtained by
    solving

    .. math::
        \nabla \mathrm{KL}(y) + \frac{y - x}{\sigma}
        = 1 - \frac{g}{y} + \frac{y - x}{\sigma}
        = 0

    for :math:`y`. This results in

    .. math::
        \mathrm{prox}_{\sigma\mathrm{KL}}(x)
        = \frac{x - \sigma + \sqrt{(x - \sigma)^2 + 4\sigma g}}{2}.

    See Also
    --------
    proximal_kl_convex_conj
    """
    if g is not None:
        g = space.element(g)

    class ProximalKL(Operator):

        """Proximal operator of the KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalKL, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # (x - sig + sqrt((x - sig)^2 + 4*sig*g)) / 2

            # out = (x - sig)^2
            if x is out:
                # Handle aliased `x` and `out` (need original `x` later on)
                x = x.copy()
            else:
                out.assign(x)
            out -= self.sigma
            out.ufuncs.square(out=out)

            # out = ... + 4*sigma*g
            if g is None:
                out += 4 * self.sigma
            else:
                out.lincomb(1, out, 4 * self.sigma, g)

            # out = x + sqrt(...) - sig
            out.ufuncs.sqrt(out=out)
            out.lincomb(1, x, 1, out)
            out -= self.sigma

            # out = 1/2 * ...
            out /= 2

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> g = 2 * space.one()
            >>> kl = odl.solvers.KullbackLeibler(space, prior=g)
            >>> kl.proximal(2)
            KullbackLeibler(
                rn(2), prior=rn(2).element([ 2.,  2.])
            ).proximal(2.0)
            """
            posargs = [space]
            optargs = [('prior', g, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('KullbackLeibler',
                                      inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalKL


def proximal_convex_conj_kl(space, g=None):
    r"""Return the proximal factory for the KL divergence convex conjugate.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.
    g : ``space`` element, optional
        Data term (or prior), must be positive everywhere. For the default
        ``None``, ``space.one()`` is taken.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of convex conjugate of the KL
        divergence.

    Notes
    -----
    The convex conjugate of the KL divergence is (for :math:`L^p`-like
    spaces)

    .. math::
        \mathrm{KL}^*(x) = \int \big[-g(t) \ln\big(1 - x(t)\big)\big]\,
        \mathrm{d}t,

    with value :math:`+\infty` if :math:`x(t) \geq 1` anywhere. Its proximal
    operator can be obtained by solving

    .. math::
        \nabla \mathrm{KL^*}(y) + \frac{y - x}{\sigma}
        = \frac{g}{1 - y} + \frac{y - x}{\sigma}
        = 0

    for :math:`y`. This results in

    .. math::
        \mathrm{prox}_{\sigma\mathrm{KL^*}}(x)
        = \frac{x + 1 - \sqrt{(x - 1)^2 + 4\sigma g}}{2}.

    See Also
    --------
    proximal_kl
    proximal_convex_conj_kl_cross_entropy :
        Proximal for functional with switched roles of ``x`` and ``g``
    """
    if g is not None:
        g = space.element(g)

    class ProximalKLConvexConj(Operator):

        """Proximal operator of the convex conjugate of the KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalKLConvexConj, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # (x + 1 - sqrt((x - 1)^2 + 4*sig*g)) / 2

            # out = (x - 1)^2
            if x is out:
                # Handle aliased `x` and `out` (need original `x` later on)
                x = x.copy()
            else:
                out.assign(x)
            out -= 1
            out.ufuncs.square(out=out)

            # out = ... + 4*sigma*g
            if g is None:
                out += 4 * self.sigma
            else:
                out.lincomb(1, out, 4 * self.sigma, g)

            # out = x - sqrt(...) + 1
            out.ufuncs.sqrt(out=out)
            out.lincomb(1, x, -1, out)
            out += 1

            # out = 1/2 * ...
            out /= 2

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> g = 2 * space.one()
            >>> kl = odl.solvers.KullbackLeibler(space, prior=g)
            >>> kl.convex_conj.proximal(2)
            KullbackLeiblerConvexConj(
                rn(2), prior=rn(2).element([ 2.,  2.])
            ).proximal(2.0)
            """
            posargs = [space]
            optargs = [('prior', g, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('KullbackLeiblerConvexConj',
                                      inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalKLConvexConj


def proximal_convex_conj_kl_cross_entropy(space, g=None):
    r"""Return the proximal factory for the KL cross entropy convex conjugate.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.
    g : ``space`` element, optional
        Data term (or prior), must be positive everywhere. For the default
        ``None``, ``space.one()`` is taken.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the convex conjugate of the KL
        divergence.

    Notes
    -----
    The convex conjugate of the KL cross entropy is (for :math:`L^p`-like
    spaces)

    .. math::
        \widetilde{\mathrm{KL}}^*(x) = \int g(t)\, \mathrm{e}^{x(t)- 1}\,
        \mathrm{d}t.

    Its proximal operator is given by

    .. math::
        \mathrm{prox}_{\sigma \widetilde{\mathrm{KL}}^*}(x) =
        x - W(\sigma\, g\, \mathrm{e}^{x}),

    where :math:`W` is the `Lambert W function
    <https://en.wikipedia.org/wiki/Lambert_W_function>`_.

    See Also
    --------
    proximal_kl
    proximal_convex_conj_kl :
        Proximal for functional with switched roles of ``x`` and ``g``
    """
    if g is not None:
        g = space.element(g)

    class ProximalKLCrossEntropyConvexConj(Operator):

        """Proximal operator of the KL cross entropy convex conjugate."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            self.sigma = float(sigma)
            super(ProximalKLCrossEntropyConvexConj, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Lazy import to improve `import odl` time
            import scipy.special

            tmp = np.exp(x)
            tmp *= self.sigma
            if g is not None:
                tmp *= g

            # Different branches of lambertw are not an issue since
            # its input is positive.
            # The returned array is complex, so we usually have to cast
            # to real.
            lambw = scipy.special.lambertw(tmp)
            if not np.issubsctype(self.domain.dtype, np.complexfloating):
                lambw = lambw.real

            lambw = x.space.element(lambw)
            out.lincomb(1, x, -1, lambw)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> g = 2 * space.one()
            >>> kl = odl.solvers.KullbackLeiblerCrossEntropy(space, prior=g)
            >>> kl.convex_conj.proximal(2)
            KullbackLeiblerCrossEntropyConvexConj(
                rn(2), prior=rn(2).element([ 2.,  2.])
            ).proximal(2.0)
            """
            posargs = [space]
            optargs = [('prior', g, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('KullbackLeiblerCrossEntropyConvexConj',
                                      inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalKLCrossEntropyConvexConj


def proximal_huber(space, gamma):
    r"""Return the proximal factory for the Huber norm.

    Parameters
    ----------
    space : `TensorSpace`
        Domain of the functional.
    gamma : float
        The smoothing parameter of the Huber norm functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the Huber norm.

    See Also
    --------
    odl.solvers.Huber : the Huber norm functional

    Notes
    -----
    The proximal operator of the Huber norm :math:`H_\gamma` is (for
    :math:`L^p`-like spaces)

    .. math::
        \mathrm{prox}_{\sigma H_\gamma}(\mathbf{x}) =
        \begin{cases}
            \frac{\gamma}{\gamma + \sigma}\, \mathbf{x} &
            \text{where } |\mathbf{x}|_2 \leq \gamma + \sigma, \\
            \mathbf{x} - \sigma\, \mathrm{sign}(\mathbf{x}) &
            \text{elsewhere,}
        \end{cases}

    with :math:`|\cdot|_2` being the pointwise Euclidean norm along the
    vector components, and all operations being understood pointwise.
    """

    gamma = float(gamma)

    class ProximalHuber(Operator):

        """Proximal operator of the Huber norm."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            self.sigma = float(sigma)
            super(ProximalHuber, self).__init__(domain=space, range=space,
                                                linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            if isinstance(self.domain, ProductSpace):
                norm = PointwiseNorm(self.domain, 2)(x)
            else:
                norm = x.ufuncs.absolute()

            idx = norm.ufuncs.less_equal(gamma + self.sigma)
            out[idx] = gamma / (gamma + self.sigma) * x[idx]

            idx.ufuncs.logical_not(out=idx)
            sign_x = x.ufuncs.sign()
            out[idx] = x[idx] - self.sigma * sign_x[idx]

            return out

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> huber = odl.solvers.Huber(space, gamma=2)
            >>> huber.proximal(4)
            Huber(rn(2), gamma=2.0).proximal(4.0)
            """
            posargs = [space]
            optargs = [('gamma', gamma, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            caller_repr = repr_string('Huber', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(caller_repr, 'proximal', [prox_arg_str])

    return ProximalHuber


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
