# coding=utf-8

# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Default operators defined on any (reasonable) space."""

from __future__ import absolute_import, division, print_function

from copy import copy

import numpy as np

from odl.operator.operator import Operator
from odl.set import ComplexNumbers, RealNumbers
from odl.set.sets import Field
from odl.set.space import LinearSpace
from odl.space.pspace import ProductSpace

__all__ = (
    'ScalingOperator',
    'IdentityOperator',
    'LinCombOperator',
    'MultiplyOperator',
    'PowerOperator',
    'InnerProductOperator',
    'NormOperator',
    'DistOperator',
    'ConstantOperator',
    'ZeroOperator',
    'RealPart',
    'ImagPart',
    'ComplexEmbedding',
    'ComplexModulus',
    'ComplexModulusSquared',
)


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar.

    Implements::

        ScalingOperator(s)(x) == s * x
    """

    def __init__(self, domain, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which this operator acts.
        scalar : ``domain.field`` element
            Fixed scaling factor of this operator.

        Examples
        --------
        >>> A = op.ScalingOperator(odl.rn(3), 2.0)
        >>> A([1, 2, 3])
        array([ 2.,  4.,  6.])
        """
        if not isinstance(domain, (LinearSpace, Field)):
            raise TypeError('`domain` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(domain))

        super(ScalingOperator, self).__init__(domain, domain, linear=True)
        self.__scalar = domain.field.element(scalar)

    @property
    def scalar(self):
        """Fixed scaling factor of this operator."""
        return self.__scalar

    def _call(self, x, out=None):
        """Scale ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.scalar * x
        else:
            self.domain.lincomb(self.scalar, x, 0, out, out)
        return out

    @property
    def inverse(self):
        """Return the inverse operator.

        Examples
        --------
        >>> A = op.ScalingOperator(odl.rn(3), 2.0)
        >>> x = A.domain.element([1, 2, 3])
        >>> Ainv = A.inverse
        >>> Ainv(A(x)) == x
        array([ True,  True,  True], dtype=bool)
        >>> A(Ainv(x)) == x
        array([ True,  True,  True], dtype=bool)
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('scaling operator not invertible for '
                                    'scalar==0')
        return ScalingOperator(self.domain, 1.0 / self.scalar)

    @property
    def adjoint(self):
        """Adjoint, given as scaling with the conjugate of the scalar.

        Examples
        --------
        In the real case, the adjoint is the same as the operator:

        >>> A = op.ScalingOperator(odl.rn(3), 2)
        >>> A([1, 2, 3])
        array([ 2.,  4.,  6.])
        >>> A.adjoint([1, 2, 3])
        array([ 2.,  4.,  6.])

        In the complex case, the scalar is conjugated:

        >>> B = op.ScalingOperator(odl.cn(3), 1 + 1j)
        >>> z = B.domain.element([1, 1j, 1 - 1j])
        >>> B.adjoint(z)
        array([ 1.-1.j,  1.+1.j,  0.-2.j])
        >>> z * (1 - 1j)
        array([ 1.-1.j,  1.+1.j,  0.-2.j])

        Returns
        -------
        adjoint : `ScalingOperator`
            ``self`` if `scalar` is real, else `scalar` is conjugated.
        """
        if complex(self.scalar).imag == 0.0:
            return self
        else:
            return ScalingOperator(self.domain, self.scalar.conjugate())

    def norm(self, estimate=False, **kwargs):
        """Return the operator norm of this operator.

        Parameters
        ----------
        estimate, kwargs : bool
            Ignored. Present to conform with base-class interface.

        Returns
        -------
        norm : float
            The operator norm, absolute value of `scalar`.

        Examples
        --------
        >>> A = op.ScalingOperator(odl.rn(3), 3.0)
        >>> A.norm(estimate=True)
        3.0
        """
        # TODO(kohr-h): this is not always true
        return np.abs(self.scalar)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.scalar)

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self.scalar)


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself.

    Implements::

        IdentityOperator()(x) == x
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        """
        super(IdentityOperator, self).__init__(space, 1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return "I"


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

    Implements::

        LinCombOperator(a, b)([x, y]) == a * x + b * y
    """

    def __init__(self, space, a, b):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace`
            Space of elements on which the operator acts. Must be a power
            space.
        a, b : ``space[0].field`` elements
            Scalars to multiply ``x[0]`` and ``x[1]`` with, respectively.

        Examples
        --------
        >>> X = odl.rn(3)
        >>> A = op.LinCombOperator(X * X, 1.0, 1.0)
        >>> A([[1, 2, 3],
        ...    [1, 2, 3]])
        array([ 2.,  4.,  6.])
        """
        if not isinstance(space, ProductSpace) or not space.is_power_space:
            raise TypeError(
                '`space` must be a power space, got {!r}'.format(space)
            )
        super(LinCombOperator, self).__init__(space, space[0], linear=True)
        self.a = self.range.field.element(a)
        self.b = self.range.field.element(b)

    def _call(self, x, out=None):
        """Linearly combine ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.range.element()
        self.range.lincomb(self.a, x[0], self.b, x[1], out)
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.range, self.a, self.b)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}*x + {}*y".format(self.a, self.b)


class MultiplyOperator(Operator):

    """Operator multiplying by a fixed space or field element.

    Implements::

        MultiplyOperator(y)(x) == x * y

    Here, ``y`` is a `LinearSpaceElement` or `Field` element and
    ``x`` is a `LinearSpaceElement`.
    Hence, this operator can be defined either on a `LinearSpace` or on
    a `Field`. In the first case it is the pointwise multiplication,
    in the second the scalar multiplication.
    """

    def __init__(self, domain, multiplicand, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set to which the operator can be applied.
        multiplicand : `LinearSpaceElement` or scalar
            Value to multiply by.
        range : `LinearSpace` or `Field`, optional
            Set to which the operator maps.

            Default: ``domain``.

        Examples
        --------
        Multiply by vector:

        >>> A = op.MultiplyOperator(odl.rn(3), [1, 2, 3])
        >>> A([2, 3, 4])
        array([  2.,   6.,  12.])

        Multiply by scalar:

        >>> X = odl.rn(3)
        >>> B = op.MultiplyOperator(X.field, [1, 2, 3], range=X)
        >>> B(3)
        array([ 3.,  6.,  9.])
        """
        if range is None:
            range = domain

        super(MultiplyOperator, self).__init__(domain, range, linear=True)

        self.__multiplicand = np.array(multiplicand)
        self.__domain_is_field = isinstance(domain, Field)
        self.__range_is_field = isinstance(range, Field)

    @property
    def multiplicand(self):
        """Value to multiply by."""
        return self.__multiplicand

    def _call(self, x, out=None):
        """Multiply ``x`` and write to ``out`` if given."""
        if out is None:
            return x * self.multiplicand
        elif not self.__range_is_field:
            if self.__domain_is_field:
                self.range.lincomb(self.multiplicand, x, out=out)
            else:
                np.multiply(self.multiplicand, x, out=out)
        else:
            raise ValueError('can only use `out` with `LinearSpace` range')

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `InnerProductOperator` or `MultiplyOperator`
            If the domain of this operator is the scalar field of a
            `LinearSpace` the adjoint is the inner product with ``y``,
            else it is the multiplication with ``y``.

        Examples
        --------
        Multiply by a space element:

        >>> A = op.MultiplyOperator(odl.rn(3), [1, 2, 3])
        >>> A.adjoint([2, 3, 4])
        array([  2.,   6.,  12.])

        Multiply scalars with a fixed vector:

        >>> X = odl.rn(3)
        >>> B = op.MultiplyOperator(X.field, [1, 2, 3], range=X)
        >>> B.adjoint([2, 3, 4])
        20.0

        Multiply vectors with a fixed scalar:

        >>> C = op.MultiplyOperator(odl.rn(3), 3.0)
        >>> C.adjoint([2, 3, 4])
        array([  6.,   9.,  12.])

        Multiplication operator with complex space:

        >>> C = op.MultiplyOperator(odl.cn(3), [1, 1j, 1-1j])
        >>> C.adjoint.multiplicand
        array([ 1.-0.j,  0.-1.j,  1.+1.j])
        """
        if self.__domain_is_field:
            if isinstance(self.domain, RealNumbers):
                return InnerProductOperator(self.range, self.multiplicand)
            elif isinstance(self.domain, ComplexNumbers):
                return InnerProductOperator(
                    self.range, self.multiplicand.conjugate()
                )
            else:
                raise NotImplementedError(
                    'adjoint not implemented for domain{!r}'
                    ''.format(self.domain)
                )
        elif self.domain.is_complex:
            return MultiplyOperator(
                domain=self.range, range=self.domain,
                multiplicand=np.conj(self.multiplicand),
            )
        else:
            return MultiplyOperator(
                domain=self.range, range=self.domain,
                multiplicand=self.multiplicand,
            )

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.multiplicand)

    def __str__(self):
        """Return ``str(self)``."""
        return "x * {}".format(self.y)


class PowerOperator(Operator):

    """Operator taking a fixed power of a space or field element.

    Implements::

        PowerOperator(p)(x) == x ** p

    Here, ``x`` is a `LinearSpaceElement` or `Field` element and ``p`` is
    a number. Hence, this operator can be defined either on a
    `LinearSpace` or on a `Field`.
    """

    def __init__(self, domain, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which the operator can be applied.
        exponent : float
            Exponent parameter of the power function applied to an element.

        Examples
        --------
        Use with vectors

        >>> A = op.PowerOperator(odl.rn(3), exponent=2)
        >>> A([1, 2, 3])
        array([ 1.,  4.,  9.])

        or scalars

        >>> A = op.PowerOperator(odl.RealNumbers(), exponent=2)
        >>> A(2.0)
        4.0
        """
        super(PowerOperator, self).__init__(
            domain, domain, linear=(exponent == 1))
        self.__exponent = float(exponent)
        self.__domain_is_field = isinstance(domain, Field)

    @property
    def exponent(self):
        """Power of the input element to take."""
        return self.__exponent

    def _call(self, x, out=None):
        """Take the power of ``x`` and write to ``out`` if given."""
        if out is None:
            return x ** self.exponent
        elif self.__domain_is_field:
            raise ValueError('cannot use `out` with field')
        else:
            out[:] = x
            out **= self.exponent

    def derivative(self, point):
        """Derivative of this operator.

            ``PowerOperator(p).derivative(y)(x) == p * y ** (p - 1) * x``

        Parameters
        ----------
        point : `domain` element
            The point in which to take the derivative

        Returns
        -------
        derivative : `Operator`
            The derivative in ``point``

        Examples
        --------
        Use on vector spaces:

        >>> A = op.PowerOperator(odl.rn(3), exponent=2)
        >>> dA = A.derivative(A.domain.element([1, 2, 3]))
        >>> dA([1, 1, 1])
        array([ 2.,  4.,  6.])

        Use with scalars:

        >>> A = op.PowerOperator(odl.RealNumbers(), exponent=2)
        >>> dA = A.derivative(2.0)
        >>> dA(2.0)
        8.0
        """
        return self.exponent * MultiplyOperator(
            self.domain, point ** (self.exponent - 1), range=self.range
        )

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.exponent)

    def __str__(self):
        """Return ``str(self)``."""
        return "x ** {}".format(self.exponent)


class InnerProductOperator(Operator):
    """Operator taking the inner product with a fixed space element.

    Implements::

        InnerProductOperator(y)(x) <==> y.inner(x)

    This is only applicable in inner product spaces.

    See Also
    --------
    DistOperator : Distance to a fixed space element.
    NormOperator : Vector space norm as operator.
    """

    def __init__(self, space, vector):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space in which the inner product is taken.
        vector
            Element with which the inner product is taken.

        Examples
        --------
        >>> A = op.InnerProductOperator(odl.rn(3), [1, 2, 3])
        >>> A([2, 3, 4])
        20.0
        """
        super(InnerProductOperator, self).__init__(
            space, space.field, linear=True
        )
        self.__vector = space.element(vector)

    @property
    def vector(self):
        """Element to take the inner product with."""
        return self.__vector

    def _call(self, x):
        """Return the inner product with ``x``."""
        return self.domain.inner(self.vector, x)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `MultiplyOperator`
            The operator of multiplication with `vector`.

        Examples
        --------
        >>> A = op.InnerProductOperator(odl.rn(3), [1, 2, 3])
        >>> A.adjoint(2.0)
        array([ 2.,  4.,  6.])
        """
        return MultiplyOperator(self.range, self.vector, range=self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}.T'.format(self.vector)


class NormOperator(Operator):

    """Vector space norm as an operator.

    Implements::

        NormOperator()(x) <==> x.norm()

    This is only applicable in normed spaces, i.e., spaces implementing
    a ``norm`` method.

    See Also
    --------
    InnerProductOperator : Inner product with a fixed space element.
    DistOperator : Distance to a fixed space element.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space to take the norm in.

        Examples
        --------
        >>> A = op.NormOperator(odl.rn(2))
        >>> A([3, 4])
        5.0
        """
        super(NormOperator, self).__init__(space, RealNumbers(), linear=False)

    def _call(self, x):
        """Return the norm of ``x``."""
        return self.domain.norm(x)

    def derivative(self, point):
        r"""Derivative of this operator in ``point``.

            ``NormOperator().derivative(y)(x) == (y / y.norm()).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        point : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point.norm() == 0``, in which case the derivative is not well
            defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::
            (D \|\cdot\|)(y)(x) = \langle y / \|y\|, x \rangle

        Examples
        --------
        >>> A = op.NormOperator(odl.rn(3))
        >>> dA = A.derivative([1, 0, 0])
        >>> dA([1, 0, 0])
        1.0
        """
        point = self.domain.element(point)
        pt_norm = self.domain.norm(point)
        if pt_norm == 0:
            raise ValueError('not differentiable in 0')

        return InnerProductOperator(self.domain, point / pt_norm)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({})'.format(self.__class__.__name__, self.domain)


class DistOperator(Operator):

    """Operator taking the distance to a fixed space element.

    Implements::

        DistOperator(y)(x) == y.dist(x)

    This is only applicable in metric spaces, i.e., spaces implementing
    a ``dist`` method.

    See Also
    --------
    InnerProductOperator : Inner product with fixed space element.
    NormOperator : Vector space norm as an operator.
    """

    def __init__(self, space, vector):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space in which to take the norm.
        vector
            Point to which to calculate the distance.

        Examples
        --------
        >>> A = op.DistOperator(odl.rn(2), [1, 1])
        >>> A([4, 5])
        5.0
        """
        super(DistOperator, self).__init__(space, RealNumbers(), linear=False)
        self.__vector = space.element(vector)

    @property
    def vector(self):
        """Element to which to take the distance."""
        return self.__vector

    def _call(self, x):
        """Return the distance from ``self.vector`` to ``x``."""
        return self.domain.dist(self.vector, x)

    def derivative(self, point):
        r"""The derivative operator.

            ``DistOperator(y).derivative(z)(x) ==
            ((y - z) / y.dist(z)).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        x : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point == self.vector``, in which case the derivative is not
            well defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::
            (D d(\cdot, y))(z)(x) = \langle (y-z) / d(y, z), x \rangle

        Examples
        --------
        >>> A = op.DistOperator(odl.rn(2), [1, 1])
        >>> dA = A.derivative([2, 1])
        >>> dA([1, 0])
        1.0
        """
        point = self.domain.element(point)
        # TODO(kohr-h): this is not correct if dist(x, y) != ||x - y||
        diff = point - self.vector
        dist = self.domain.dist(self.vector, point)
        if dist == 0:
            raise ValueError('not differentiable at the reference vector {!r}'
                             ''.format(self.vector))

        return InnerProductOperator(self.domain, diff / dist)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({})'.format(self.__class__.__name__, self.vector)


class ConstantOperator(Operator):

    """Operator that always returns the same value.

    Implements::

        ConstantOperator(y)(x) == y
    """

    def __init__(self, range, constant, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        range : `LinearSpace`
            Range of the operator, space in which the returned element lies.
        constant : ``range`` `element-like`
            The constant space element to be returned.
        domain : `LinearSpace`, optional
            Domain of the operator.

            Default: ``range``.

        Examples
        --------
        >>> A = op.ConstantOperator(odl.rn(3), [1, 2, 3])
        >>> A([2, 3, 4])
        array([ 1.,  2.,  3.])
        """
        if domain is None:
            domain = range

        super(ConstantOperator, self).__init__(domain, range, linear=False)
        self.__constant = range.element(constant)

    @property
    def constant(self):
        """Constant space element returned by this operator."""
        return self.__constant

    def _call(self, x, out=None):
        """Return the constant vector or assign it to ``out``."""
        if out is None:
            return self.range.element(copy(self.constant))
        else:
            out[:] = self.constant

    def derivative(self, point):
        """Derivative of this operator, always zero.

        Returns
        -------
        derivative : `ZeroOperator`

        Examples
        --------
        >>> A = op.ConstantOperator(odl.rn(3), [1, 2, 3])
        >>> dA = A.derivative([1, 1, 1])
        >>> dA([2, 2, 2])
        array([ 0.,  0.,  0.])
        """
        return ZeroOperator(domain=self.domain, range=self.range)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.constant)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.constant)


class ZeroOperator(Operator):

    """Operator mapping each element to the zero element.

    Implements::

        ZeroOperator(space)(x) == space.zero()
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the operator.
        range : `LinearSpace`, optional
            Range of the operator. Default: ``domain``

        Examples
        --------
        >>> A = op.ZeroOperator(odl.rn(3))
        >>> A([1, 2, 3])
        array([ 0.,  0.,  0.])

        Also works with domain != range:

        >>> A = op.ZeroOperator(odl.rn(3), range=odl.cn(4))
        >>> A([1, 2, 3])
        array([ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])
        """
        if range is None:
            range = domain

        super(ZeroOperator, self).__init__(domain, range, linear=True)

    def _call(self, x, out):
        """Return the zero vector or assign it to ``out``."""
        out[:] = 0  # TODO: use assign()
        return out

    @property
    def adjoint(self):
        """Adjoint of the operator.

        If ``self.domain == self.range`` the zero operator is self-adjoint,
        otherwise it is the `ZeroOperator` from `range` to `domain`.
        """
        return ZeroOperator(domain=self.range, range=self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '0'


class RealPart(Operator):

    """Operator that extracts the real part of a vector.

    Implements::

        RealPart(x) == x.real
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the real part should be taken, needs to implement
            ``space.real_space``.

        Examples
        --------
        Take the real part of complex vector:

        >>> A = op.RealPart(odl.cn(3))
        >>> A([1 + 2j, 2, 3 - 1j])
        array([ 1.,  2.,  3.])

        The operator is the identity on real spaces:

        >>> A = op.RealPart(odl.rn(3))
        >>> A([1, 2, 3])
        array([ 1.,  2.,  3.])

        The operator also works on other `TensorSpace` spaces such as
        `DiscreteLp` spaces:

        >>> X = odl.uniform_discr(0, 1, 3, dtype=complex)
        >>> A = op.RealPart(X)
        >>> A([1 + 2j, 2, 3 - 1j])
        array([ 1.,  2.,  3.])
        """
        real_space = space.real_space
        self.space_is_real = (space == real_space)
        # NOTE: `linear=True` is chosen for efficiency reasons (shortcut
        # in `derivative`). The choice makes sense if the operator is
        # interpreted as R^2 -> R.
        super(RealPart, self).__init__(space, real_space, linear=True)

    def _call(self, x, out):
        """Return ``self(x)``."""
        out[:] = x.real

    def derivative(self, x):
        r"""Return the derivative operator in the "C = R^2" sense.

        The returned operator (``self``) is the derivative of the
        operator variant where the complex domain is reinterpreted as
        a product of two real spaces.

        Parameters
        ----------
        x : `domain` element
            Point in which to take the derivative, has no effect.
        """
        return self

    @property
    def inverse(self):
        """Return the (pseudo-)inverse.

        Examples
        --------
        The inverse is its own inverse if its domain is real:

        >>> A = op.RealPart(odl.rn(3))
        >>> A.inverse(A([1, 2, 3]))
        array([ 1.,  2.,  3.])

        This is not a true inverse, only a pseudoinverse, the complex part
        will by necessity be lost.

        >>> A = op.RealPart(odl.cn(3))
        >>> A.inverse(A([1 + 2j, 2, 3 - 1j]))
        array([ 1.+0.j,  2.+0.j,  3.+0.j])
        """
        if self.space_is_real:
            return self
        else:
            return ComplexEmbedding(self.domain, scalar=1)

    @property
    def adjoint(self):
        r"""Return the (left) adjoint.

        Notes
        -----
        Due to technicalities of operators from a complex space into a real
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle AA^*x, y \rangle = \langle A^*x, A^*y \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for real spaces:

        >>> A = op.RealPart(odl.rn(3))
        >>> X, Y = A.domain, A.range
        >>> x = X.element([1, 2, 3])
        >>> y = Y.element([3, 2, 1])
        >>> Y.inner(A(x), y) == X.inner(x, A.adjoint(y))
        True

        If the domain is complex, it only satisfies the weaker definition:

        >>> A = op.RealPart(odl.cn(3))
        >>> X, Y = A.domain, A.range
        >>> y = Y.element([3, 2, 1])
        >>> z = Y.element([4, 3, 2])
        >>> AAtyz = Y.inner(A(A.adjoint(y)), z)
        >>> AtyAtz = X.inner(A.adjoint(y), A.adjoint(z))
        >>> AAtyz == AtyAtz
        True
        """
        if self.space_is_real:
            return self
        else:
            return ComplexEmbedding(self.domain, scalar=1)


class ImagPart(Operator):

    """Operator that extracts the imaginary part of a vector.

    Implements::

        ImagPart(x) == x.imag
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the imaginary part should be taken, needs to
            implement ``space.real_space``.

        Examples
        --------
        Take the imaginary part of complex vector:

        >>> A = op.ImagPart(odl.cn(3))
        >>> A([1 + 2j, 2, 3 - 1j])
        array([ 2.,  0., -1.])

        The operator is the zero operator on real spaces:

        >>> A = op.ImagPart(odl.rn(3))
        >>> A([1, 2, 3])
        array([ 0.,  0.,  0.])
        """
        real_space = space.real_space
        self.space_is_real = (space == real_space)
        linear = True
        super(ImagPart, self).__init__(space, real_space, linear=linear)

    def _call(self, x, out):
        """Return ``self(x)``."""
        out[:] = x.imag

    def derivative(self, x):
        r"""Return the derivative operator in the "C = R^2" sense.

        The returned operator (``self``) is the derivative of the
        operator variant where the complex domain is reinterpreted as
        a product of two real spaces.

        Parameters
        ----------
        x : `domain` element
            Point in which to take the derivative, has no effect.
        """
        return self

    @property
    def inverse(self):
        """Return the pseudoinverse.

        Examples
        --------
        The inverse is the zero operator if the domain is real:

        >>> A = op.ImagPart(odl.rn(3))
        >>> A.inverse(A([1, 2, 3]))
        array([ 0.,  0.,  0.])

        This is not a true inverse, only a pseudoinverse, the real part
        will by necessity be lost.

        >>> A = op.ImagPart(odl.cn(3))
        >>> A.inverse(A([1 + 2j, 2, 3 - 1j]))
        array([ 0.+2.j,  0.+0.j, -0.-1.j])
        """
        if self.space_is_real:
            return ZeroOperator(self.domain)
        else:
            return ComplexEmbedding(self.domain, scalar=1j)

    @property
    def adjoint(self):
        r"""Return the (left) adjoint.

        Notes
        -----
        Due to technicalities of operators from a complex space into a real
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle AA^*x, y \rangle = \langle A^*x, A^*y \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for real spaces:

        >>> A = op.ImagPart(odl.rn(3))
        >>> X, Y = A.domain, A.range
        >>> x = X.element([1, 2, 3])
        >>> y = Y.element([3, 2, 1])
        >>> Y.inner(A(x), y) == X.inner(x, A.adjoint(y))
        True

        If the domain is complex, it only satisfies the weaker definition:

        >>> A = op.ImagPart(odl.cn(3))
        >>> X, Y = A.domain, A.range
        >>> y = Y.element([1, 2, 3])
        >>> z = Y.element([3, 2, 1])
        >>> AAtyz = Y.inner(A(A.adjoint(y)), z)
        >>> AtyAtz = X.inner(A.adjoint(y), A.adjoint(z))
        >>> AAtyz == AtyAtz
        True
        """
        if self.space_is_real:
            return ZeroOperator(self.domain)
        else:
            return ComplexEmbedding(self.domain, scalar=1j)


class ComplexEmbedding(Operator):

    """Operator that embeds a vector into a complex space.

    Implements::

        ComplexEmbedding(space)(x) <==> space.complex_space.element(x)
    """

    def __init__(self, space, scalar=1.0):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space that should be embedded into its complex counterpart.
            It must implement `TensorSpace.complex_space`.
        scalar : ``space.complex_space.field`` element, optional
            Scalar to be multiplied with incoming vectors in order
            to get the complex vector.

        Examples
        --------
        Embed real vector into complex space:

        >>> A = op.ComplexEmbedding(odl.rn(3))
        >>> A([1, 2, 3])
        array([ 1.+0.j,  2.+0.j,  3.+0.j])

        Embed real vector as imaginary part into complex space:

        >>> A = op.ComplexEmbedding(odl.rn(3), scalar=1j)
        >>> A([1, 2, 3])
        array([ 0.+1.j,  0.+2.j,  0.+3.j])

        On complex spaces the operator is the same as simple multiplication by
        scalar:

        >>> A = op.ComplexEmbedding(odl.cn(3), scalar=1 + 2j)
        >>> A([1 + 1j, 2 + 2j, 3 + 3j])
        array([-1.+3.j, -2.+6.j, -3.+9.j])
        """
        complex_space = space.complex_space
        self.scalar = complex_space.field.element(scalar)
        super(ComplexEmbedding, self).__init__(
            space, complex_space, linear=True)

    def _call(self, x, out):
        """Return ``self(x)``."""
        # TODO(kohr-h): Optimize cases 0 and 1
        if self.domain.is_real:
            # Real domain, multiply separately
            out.real = self.scalar.real * x
            out.imag = self.scalar.imag * x
        else:
            # Complex domain
            self.domain.lincomb(self.scalar, x, out=out)

    @property
    def inverse(self):
        """Return the (left) inverse.

        If the domain is a real space, this is not a true inverse,
        only a (left) inverse.

        Examples
        --------
        >>> A = op.ComplexEmbedding(odl.rn(3), scalar=1)
        >>> A.inverse(A([1, 2, 4]))
        array([ 1.,  2.,  4.])
        """
        if self.domain.is_real:
            # Real domain
            # Optimizations for simple cases.
            if self.scalar.real == self.scalar:
                return (1 / self.scalar.real) * RealPart(self.range)
            elif 1j * self.scalar.imag == self.scalar:
                return (1 / self.scalar.imag) * ImagPart(self.range)
            else:
                # General case
                inv_scalar = (1 / self.scalar).conjugate()
                return ((inv_scalar.real) * RealPart(self.range) +
                        (inv_scalar.imag) * ImagPart(self.range))
        else:
            # Complex domain
            return ComplexEmbedding(self.range, self.scalar.conjugate())

    @property
    def adjoint(self):
        r"""Return the (right) adjoint.

        Notes
        -----
        Due to technicalities of operators from a real space into a complex
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle A^*Ax, y \rangle = \langle Ax, Ay \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for complex spaces

        >>> A = op.ComplexEmbedding(odl.cn(3), scalar=1j)
        >>> X, Y = A.domain, A.range
        >>> x = X.element([1 + 1j, 2 + 2j, 3 + 3j])
        >>> y = Y.element([3 + 1j, 2 + 2j, 3 + 1j])
        >>> Axy = Y.inner(A(x), y)
        >>> xAty = X.inner(x, A.adjoint(y))
        >>> Axy == xAty
        True

        For real domains, it only satisfies the (right) adjoint equation

        >>> A = op.ComplexEmbedding(odl.rn(3), scalar=1j)
        >>> X, Y = A.domain, A.range
        >>> u = X.element([1, 2, 3])
        >>> v = X.element([3, 2, 3])
        >>> AtAuv = X.inner(A.adjoint(A(u)), v)
        >>> AuAv = Y.inner(A(u), A(v))
        >>> AtAuv == AuAv
        True
        """
        if self.domain.is_real:
            # Real domain
            # Optimizations for simple cases.
            if self.scalar.real == self.scalar:
                return self.scalar.real * RealPart(self.range)
            elif 1j * self.scalar.imag == self.scalar:
                return self.scalar.imag * ImagPart(self.range)
            else:
                # General case
                return (self.scalar.real * RealPart(self.range) +
                        self.scalar.imag * ImagPart(self.range))
        else:
            # Complex domain
            return ComplexEmbedding(self.range, self.scalar.conjugate())


class ComplexModulus(Operator):

    """Operator that computes the modulus (absolute value) of a vector."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the modulus should be taken, needs to implement
            ``space.real_space``.

        Examples
        --------
        Take the modulus of a complex vector:

        >>> A = op.ComplexModulus(odl.cn(2))
        >>> A([3 + 4j, 2])
        array([ 5.,  2.])

        The operator is the absolute value on real spaces:

        >>> A = op.ComplexModulus(odl.rn(2))
        >>> A([1, -2])
        array([ 1.,  2.])

        The operator also works on other `TensorSpace`'s such as
        `DiscreteLp`:

        >>> X = odl.uniform_discr(0, 1, 2, dtype=complex)
        >>> A = op.ComplexModulus(X)
        >>> A([3 + 4j, 2])
        array([ 5.,  2.])
        """
        real_space = space.real_space
        super(ComplexModulus, self).__init__(space, real_space, linear=False)

    def _call(self, x):
        """Return ``self(x)``."""
        # TODO(kohr-h): generalize
        return np.sqrt(x.real ** 2 + x.imag ** 2)

    def derivative(self, x):
        r"""Return the derivative operator in the "C = R^2" sense.

        The returned operator (``self``) is the derivative of the
        operator variant where the complex domain is reinterpreted as
        a product of two real spaces::

            M'(x) = y --> ((Re(x) * Re(y) + Im(x) * Im(y)) /
                           sqrt(Re(x)**2 + Re(y) ** 2))

        Parameters
        ----------
        x : `domain` element
            Point in which to take the derivative.

        Examples
        --------
        >>> A = op.ComplexModulus(odl.cn(2))
        >>> A([3 + 4j, 2])
        array([ 5.,  2.])
        >>> dA = A.derivative([3 + 4j, 2])
        >>> dA.domain
        cn(2)
        >>> dA.range
        rn(2)
        >>> dA([2 + 1j, 4j])  # [(3*2 + 4*1) / 5, (2*0 + 0*4) / 2]
        array([ 2.,  0.])

        Notes
        -----
        The derivative of the complex modulus

        .. math::
            &M: X(\mathbb{C}) \to X(\mathbb{R}), \\
            &M(x) = \sqrt{\Re(x)^2 + \Im(x)^2},

        with :math:`X(\mathbb{F}) = \mathbb{F}^n` or
        :math:`L^2(\Omega, \mathbb{F})`, is given as

        .. math::
            &M'(x): X(\mathbb{C}) \to X(\mathbb{R}), \\
            &M'(x)(y) = \frac{\Re(x)\,\Re(y) + \Im(x)\,\Im(y)}{M(x)}.

        It is linear when identifying :math:`\mathbb{C}` with
        :math:`\mathbb{R}^2`, but not complex-linear.
        """
        op = self
        x = self.domain.element(x)

        class ComplexModulusDerivative(Operator):

            """Derivative of the complex modulus operator."""

            def _call(self, y, out):
                """Return ``self(y)``."""
                out[:] = x.real * y.real
                out += x.imag * y.imag
                out /= op(x)
                return out

            @property
            def adjoint(self):
                r"""Adjoint in the "C = R^2" sense.

                Examples
                --------
                Adjoint of the derivative:

                >>> A = op.ComplexModulus(odl.cn(2))
                >>> A([3 + 4j, 2])
                array([ 5.,  2.])
                >>> dA = A.derivative([3 + 4j, 2])
                >>> dAt = dA.adjoint
                >>> dAt.domain
                rn(2)
                >>> dAt.range
                cn(2)
                >>> dAt([5, 5])  # [5*(3 + 4j)/5, 5*2/2]
                array([ 3.+4.j,  5.+0.j])

                Adjointness only holds in the weaker sense that inner products
                are the same when testing with vectors from the real space, but
                not when testing complex vectors:

                >>> X, Y = dA.domain, dA.range
                >>> y1 = Y.element([5, 5])
                >>> y2 = Y.element([1, 2])
                >>> X.inner(dAt(y1), dAt(y2))  # <M^* y1, M^* y2>
                (15+0j)
                >>> Y.inner(dA(dAt(y1)), y2)  # <M M^* y1, y2>
                15.0
                >>> x1 = X.element([6 + 3j, 2j])
                >>> x2 = X.element([5, 10 + 4j])
                >>> Y.inner(dA(x1), dA(x2))  # <M x1, M x2>
                18.0
                >>> X.inner(dAt(dA(x1)), x2)  # <M^* M x1, x2>
                (18+24j)

                Notes
                -----
                The complex modulus derivative is given by

                .. math::
                    &M'(x): X(\mathbb{C}) \to X(\mathbb{R}), \\
                    &M'(x)(y) = \frac{\Re(x)\,\Re(y) + \Im(x)\,\Im(y)}{M(x)}.

                Thus, its adjoint can (formally) be identified as

                .. math::
                    &M'(x)^*: X(\mathbb{R}) \to X(\mathbb{C}), \\
                    &M'(x)^*(u) = \frac{(\Re(x)\,u,\ \Im(x)\,u}{M(x)}.

                The operator :math:`A = M'(x)` has the weak adjointness
                property

                .. math::
                    \langle A^* y_1,\ A^* y_2 \rangle_{X(\mathbb{C})} =
                    \langle AA^* y_1,\ y_2 \rangle_{X(\mathbb{R})},

                but in general,

                .. math::
                    \langle A x,\ y \rangle_{X(\mathbb{R})} \neq
                    \langle x,\ A^* y \rangle_{X(\mathbb{C})},

                in particular

                .. math::
                    \langle A x_1,\ A x_2 \rangle_{X(\mathbb{R})} \neq
                    \langle A^*A x_1,\ x_2 \rangle_{X(\mathbb{C})}.
                """
                deriv = self

                class ComplexModulusDerivativeAdjoint(Operator):

                    def _call(self, u, out):
                        """Implement ``self(u, out)``."""
                        out[:] = x
                        tmp = u / op(x)
                        out.real *= tmp
                        out.imag *= tmp
                        return out

                    @property
                    def adjoint(self):
                        """Adjoint in the "C = R^2" sense."""
                        return deriv

                return ComplexModulusDerivativeAdjoint(
                    deriv.range, deriv.domain, linear=deriv.domain.is_real)

        return ComplexModulusDerivative(op.domain, op.range,
                                        linear=op.domain.is_real)


class ComplexModulusSquared(Operator):

    """Operator that computes the squared complex modulus (absolute value)."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the modulus should be taken, needs to implement
            ``space.real_space``.

        Examples
        --------
        Take the squared modulus of a complex vector:

        >>> A = op.ComplexModulusSquared(odl.cn(2))
        >>> A([3 + 4j, 2])
        array([ 25.,   4.])

        On a real space, this is the same as squaring:

        >>> A = op.ComplexModulusSquared(odl.rn(2))
        >>> A([1, -2])
        array([ 1.,  4.])

        The operator also works on other `TensorSpace`'s such as
        `DiscreteLp`:

        >>> X = odl.uniform_discr(0, 1, 2, dtype=complex)
        >>> A = op.ComplexModulusSquared(X)
        >>> A([3 + 4j, 2])
        array([ 25.,   4.])
        """
        real_space = space.real_space
        super(ComplexModulusSquared, self).__init__(
            space, real_space, linear=False)

    def _call(self, x):
        """Return ``self(x)``."""
        return x.real ** 2 + x.imag ** 2

    def derivative(self, x):
        r"""Return the derivative operator in the "C = R^2" sense.

        The returned operator (``self``) is the derivative of the
        operator variant where the complex domain is reinterpreted as
        a product of two real spaces.

        Parameters
        ----------
        x : `domain` element
            Point in which to take the derivative.

        Examples
        --------
        >>> A = op.ComplexModulusSquared(odl.cn(2))
        >>> A([3 + 4j, 2])
        array([ 25.,   4.])
        >>> dA = A.derivative([3 + 4j, 2])
        >>> dA.domain
        cn(2)
        >>> dA.range
        rn(2)
        >>> dA([2 + 1j, 4j])  # [(3*2 + 4*1) / 5, (2*0 + 0*4) / 2]
        array([ 10.,   0.])

        Notes
        -----
        The derivative of the squared complex modulus

        .. math::
            &S: X(\mathbb{C}) \to X(\mathbb{R}), \\
            &S(x) = \Re(x)^2 + \Im(x)^2,

        with :math:`X(\mathbb{F}) = \mathbb{F}^n` or
        :math:`L^2(\Omega, \mathbb{F})`, is given as

        .. math::
            &S'(x): X(\mathbb{C}) \to X(\mathbb{R}), \\
            &S'(x)(y) = \Re(x)\,\Re(y) + \Im(x)\,\Im(y).

        It is linear when identifying :math:`\mathbb{C}` with
        :math:`\mathbb{R}^2`, but not complex-linear.
        """
        op = self
        x = self.domain.element(x)

        # TODO(kohr-h): Move all operators to top level

        class ComplexModulusSquaredDerivative(Operator):

            """Derivative of the squared complex modulus operator."""

            def _call(self, y, out):
                """Return ``self(y)``."""
                np.multiply(x.real, y.real, out=out)
                out += x.imag * y.imag
                return out

            @property
            def adjoint(self):
                r"""Adjoint in the "C = R^2" sense.

                Adjoint of the derivative:

                Examples
                --------
                >>> A = op.ComplexModulusSquared(odl.cn(2))
                >>> dA = A.derivative([3 + 4j, 2])
                >>> dAt = dA.adjoint
                >>> dAt.domain
                rn(2)
                >>> dAt.range
                cn(2)
                >>> dAt([2, 1])  # [2*(3 + 4j), 1*2]
                array([ 6.+8.j,  2.+0.j])

                Adjointness only holds in the weaker sense that inner products
                are the same when testing with vectors from the real space, but
                not when testing complex vectors:

                >>> X, Y = dA.domain, dA.range
                >>> y1 = Y.element([1, 1])
                >>> y2 = Y.element([1, -1])
                >>> X.inner(dAt(y1), dAt(y2))  # <M^* y1, M^* y2>
                (21+0j)
                >>> Y.inner(dA(dAt(y1)), y2)  # <M M^* y1, y2>
                21.0
                >>> x1 = X.element([1j, 1j])
                >>> x2 = X.element([1 + 1j, 1j])
                >>> Y.inner(dA(x1), dA(x2))  # <M x1, M x2>
                28.0
                >>> X.inner(dAt(dA(x1)), x2)  # <M^* M x1, x2>
                (28+4j)

                Notes
                -----
                The squared complex modulus derivative is given by

                .. math::
                    &S'(x): X(\mathbb{C}) \to X(\mathbb{R}), \\
                    &S'(x)(y) = \Re(x)\,\Re(y) + \Im(x)\,\Im(y).

                Thus, its adjoint can (formally) be identified as

                .. math::
                    &S'(x)^*: X(\mathbb{R}) \to X(\mathbb{C}), \\
                    &S'(x)^*(u) = (\Re(x)\,u,\ \Im(x)\,u).

                The operator :math:`A = S'(x)` has the weak adjointness
                property

                .. math::
                    \langle A^* y_1,\ A^* y_2 \rangle_{X(\mathbb{C})} =
                    \langle AA^* y_1,\ y_2 \rangle_{X(\mathbb{R})},

                but in general,

                .. math::
                    \langle A x,\ y \rangle_{X(\mathbb{R})} \neq
                    \langle x,\ A^* y \rangle_{X(\mathbb{C})},

                in particular

                .. math::
                    \langle A x_1,\ A x_2 \rangle_{X(\mathbb{R})} \neq
                    \langle A^*A x_1,\ x_2 \rangle_{X(\mathbb{C})}.
                """
                deriv = self

                class ComplexModulusSquaredDerivAdj(Operator):

                    def _call(self, u, out):
                        """Implement ``self(u, out)``."""
                        out[:] = x
                        out.real *= u
                        out.imag *= u
                        return out

                    @property
                    def adjoint(self):
                        """Adjoint in the "C = R^2" sense."""
                        return deriv

                return ComplexModulusSquaredDerivAdj(
                    deriv.range, deriv.domain, linear=deriv.domain.is_real)

        return ComplexModulusSquaredDerivative(op.domain, op.range,
                                               linear=op.domain.is_real)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
