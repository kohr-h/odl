# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cartesian products of `LinearSpace` instances."""

from __future__ import print_function, division, absolute_import
from numbers import Integral
import numpy as np

from odl.set.space import LinearSpace
from odl.util import is_real_dtype, signature_string, indent
from odl.util.utility import protocol


__all__ = ()


class ProductSpace(LinearSpace):

    """Cartesian product of `LinearSpace`'s.

    A product space is the Cartesian product ``X_1 x ... x X_n`` of
    linear spaces ``X_i``. It is itself a linear space, where the linear
    combination is defined component-wise. Inner product, norm and
    distance can also be defined in natural ways from the corresponding
    functions in the individual components.
    """

    def __init__(self, *spaces, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        space1,...,spaceN : `LinearSpace` or int
            The individual spaces ("factors / parts") in the product
            space. Can also be given as ``space, n`` with ``n`` integer,
            in which case the power space ``space ** n`` is created.
        exponent : non-zero float or ``float('inf')``, optional
            Order of norm and distance, i.e. ::

                norm(x) <=> np.linalg.norm(x, ord=exponent)
                dist(x, y) <=> np.linalg.norm(x - y, ord=exponent)

            Values ``0 <= exponent < 1`` are currently unsupported
            due to numerical instability. See ``Notes`` for further
            information about the interpretation of the values.

            Default: 2.0

        field : `Field`, optional
            Scalar field of the resulting space.
            Default: ``spaces[0].field``

        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            ``None`` : no weighting (default)

            `array-like` : weigh each component with one entry from the
            array. The array must be one-dimensional and have the same
            length as the number of spaces.

            float : same weighting factor in each component

        Examples
        --------
        Product of two rn spaces

        >>> r2x3 = ProductSpace(odl.rn(2), odl.rn(3))

        Powerspace of rn space

        >>> r2x2x2 = ProductSpace(odl.rn(2), 3)

        Notes
        -----
        Inner product, norm and distance are evaluated by collecting
        the result of the corresponding operation in the individual
        components and reducing the resulting vector to a single number.
        The ``exponent`` parameter influences only this last part,
        not the computations in the individual components. We give the
        exact definitions in the following:

        Let :math:`\mathcal{X} = \mathcal{X}_1 \\times \dots \\times
        \mathcal{X}_d` be a product space, and
        :math:`\langle \cdot, \cdot\\rangle_i`,
        :math:`\lVert \cdot \\rVert_i`, :math:`d_i(\cdot, \cdot)` be
        inner products, norms and distances in the respective
        component spaces.

        **Inner product:**

        .. math::
            \langle x, y \\rangle = \\sum_{i=1}^d \langle x_i, y_i \\rangle_i

        **Norm:**

        - :math:`p < \infty`:

        .. math::
            \lVert x\\rVert =
            \left( \sum_{i=1}^d \lVert x_i \\rVert_i^p \\right)^{1/p}

        - :math:`p = \infty`:

        .. math::
            \lVert x\\rVert = \max_i \lVert x_i \\rVert_i

        **Distance:**

        - :math:`p < \infty`:

        .. math::
            d(x, y) = \left( \sum_{i=1}^d d_i(x_i, y_i)^p \\right)^{1/p}

        - :math:`p = \infty`:

        .. math::
            d(x, y) = \max_i d_i(x_i, y_i)

        To implement own versions of these functions, you can use
        the following snippet to gather the vector of norms (analogously
        for inner products and distances)::

            norms = np.fromiter(
                (xi.norm() for xi in x),
                dtype=np.float64, count=len(x))

        See Also
        --------
        ProductSpaceArrayWeighting
        ProductSpaceConstWeighting
        """
        field = kwargs.pop('field', None)
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = float(kwargs.pop('exponent', 2.0))
        if kwargs:
            raise TypeError('got unexpected keyword arguments: {}'
                            ''.format(kwargs))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weighting)) < 3:
            raise ValueError('invalid combination of options weighting, '
                             'dist, norm and inner')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             'inner, norm or dist')

        # Make a power space if the second argument is an integer.
        # For the case that the integer is 0, we already set the field here.
        if len(spaces) == 2 and isinstance(spaces[1], Integral):
            field = spaces[0].field
            spaces = [spaces[0]] * spaces[1]

        # Validate the space arguments
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            raise TypeError(
                'all arguments must be `LinearSpace` instances, or the '
                'first argument must be `LinearSpace` and the second '
                'integer; got {!r}'.format(spaces))
        if not all(spc.field == spaces[0].field for spc in spaces):
            raise ValueError('all spaces must have the same field')

        # Assign spaces and field
        self.__spaces = tuple(spaces)

        # Cache for efficiency
        self.__is_power_space = all(spc == self.spaces[0]
                                    for spc in self.spaces[1:])

        # Assing or infer field
        if field is None:
            if len(self) == 0:
                raise ValueError('no spaces provided, cannot deduce field')
            else:
                field = self.spaces[0].field

        super(ProductSpace, self).__init__(field)

        self.__exponent = float(kwargs.pop('exponent', 2.0))

        # Assign weighting
        if weighting is None:
            weighting = 1.0

        # FIXME: use _weight_type
        if np.isscalar(weighting):
            self.__weighting = float(weighting)
            self.__weighting_type = 'const'
        else:
            self.__weighting = np.asarray(weighting)
            self.__weighting_type = 'array'

    def __len__(self):
        """Return ``len(self)``.

        This length is the number of spaces at the top level only,
        and is equal to ``self.shape[0]``.
        """
        return len(self.spaces)

    @property
    def nbytes(self):
        """Total number of bytes in memory used by an element of this space."""
        return sum(spc.nbytes for spc in self.spaces)

    @property
    def shape(self):
        """Total spaces per axis, computed recursively.

        The recursion ends at the fist level that does not have a shape.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> pspace = odl.ProductSpace(r2, r3)
        >>> pspace.shape
        (2,)
        >>> pspace2 = odl.ProductSpace(pspace, 3)
        >>> pspace2.shape
        (3, 2)

        If the space is a "pure" product space, shape recurses all the way
        into the components:

        >>> r2_2 = odl.ProductSpace(r2, 3)
        >>> r2_2.shape
        (3, 2)
        """
        if len(self) == 0:
            return ()
        elif self.is_power_space:
            try:
                sub_shape = self[0].shape
            except AttributeError:
                sub_shape = ()
        else:
            sub_shape = ()

        return (len(self),) + sub_shape

    @property
    def size(self):
        """Total number of involved spaces, computed recursively.

        The recursion ends at the fist level that does not comprise
        a *power* space, i.e., which is not made of equal spaces.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> pspace = odl.ProductSpace(r2, r3)
        >>> pspace.size
        2
        >>> pspace2 = odl.ProductSpace(pspace, 3)
        >>> pspace2.size
        6
        """
        return (0 if self.shape == () else
                int(np.prod(self.shape, dtype='int64')))

    @property
    def spaces(self):
        """A tuple containing all spaces."""
        return self.__spaces

    @property
    def is_power_space(self):
        """``True`` if all member spaces are equal."""
        return self.__is_power_space

    @property
    def exponent(self):
        """Exponent of the product space norm/dist, ``None`` for custom."""
        return self.__exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def weighting_type(self):
        """This space's type of weighting."""
        return self.__weighting_type

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (self.weighting_type == 'const' and self.weighting == 1.0)

    @property
    def dtype(self):
        """The data type of this space.

        This is only well defined if all subspaces have the same dtype.

        Raises
        ------
        AttributeError
            If any of the subspaces does not implement `dtype` or if the dtype
            of the subspaces does not match.
        """
        dtypes = [space.dtype for space in self.spaces]

        if all(dtype == dtypes[0] for dtype in dtypes):
            return dtypes[0]
        else:
            raise AttributeError("`dtype`'s of subspaces not equal")

    @property
    def is_real(self):
        """True if this is a space of real valued vectors."""
        return all(spc.is_real for spc in self.spaces)

    @property
    def is_complex(self):
        """True if this is a space of complex valued vectors."""
        return all(spc.is_complex for spc in self.spaces)

    @property
    def real_space(self):
        """Variant of this space with real dtype."""
        return ProductSpace(*[space.real_space for space in self.spaces])

    @property
    def complex_space(self):
        """Variant of this space with complex dtype."""
        return ProductSpace(*[space.complex_space for space in self.spaces])

    def astype(self, dtype):
        """Return a copy of this space with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newspace : `ProductSpace`
            Version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('`None` is not a valid data type')

        dtype = np.dtype(dtype)
        current_dtype = getattr(self, 'dtype', object)

        if dtype == current_dtype:
            return self
        else:
            return ProductSpace(*[space.astype(dtype)
                                  for space in self.spaces])

    def element(self, inp=None, cast=True):
        """Create an element in the product space.

        Parameters
        ----------
        inp : optional
            If ``inp`` is ``None``, a new element is created from
            scratch by allocation in the spaces. If ``inp`` is
            already an element of this space, it is re-wrapped.
            Otherwise, a new element is created from the
            components by calling the ``element()`` methods
            in the component spaces.
        cast : bool, optional
            If ``True``, casting is allowed. Otherwise, a ``TypeError``
            is raised for input that is not a sequence of elements of
            the spaces that make up this product space.

        Returns
        -------
        element : `ProductSpaceElement`
            The new element

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> vec_2, vec_3 = r2.element(), r3.element()
        >>> r2x3 = odl.ProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.element()
        >>> vec_2x3[0] in r2x3[0]
        True
        >>> vec_2x3[1] in r2x3[1]
        True

        Create an element of the product space

        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> prod = odl.ProductSpace(r2, r3)
        >>> x2 = r2.element([1, 2])
        >>> x3 = r3.element([1, 2, 3])
        >>> x = prod.element([x2, x3])
        >>> x
        array([array([ 1.,  2.]), array([ 1.,  2.,  3.])], dtype=object)
        """
        # If data is given as keyword arg, prefer it over arg list
        if inp is None:
            inp = [space.element() for space in self.spaces]

        elif inp in self:
            return inp

        if len(inp) != len(self):
            raise ValueError('length of `inp` {} does not match length of '
                             'space {}'.format(len(inp), len(self)))

        if cast:
            # Delegate constructors
            parts = [space.element(arg)
                     for arg, space in zip(inp, self.spaces)]
        else:
            raise TypeError('input {!r} not a sequence of elements of the '
                            'component spaces'.format(inp))

        # If we are a power space, use the space dtype for the array
        if self.is_power_space:
            return np.array(parts, dtype=self.dtype)

        # Otherwise, we use an object array. Note that it must be created in
        # advance, since otherwise NumPy may still try to loop over the
        # inputs. See https://github.com/numpy/numpy/issues/12479
        # TODO(kohr-h): remove when above issue is resolved
        ret = np.empty(len(parts), dtype=object)
        for i, p in enumerate(parts):
            ret[i] = p
        return ret

    def __contains__(self, other):
        if not isinstance(other, np.ndarray):
            return False
        return all(p in spc for p, spc in zip(other, self.spaces))

    def _lincomb(self, a, x, b, y, out):
        """Linear combination ``out = a*x + b*y``."""
        for space, xp, yp, outp in zip(self.spaces, x.parts, y.parts,
                                       out.parts):
            space._lincomb(a, xp, b, yp, outp)

    def _inner(self, x1, x2):
        """Inner product of two elements."""
        return weighted_inner(x1, x2, self.weighting, self.spaces)

    def _norm(self, x):
        """Norm of an element."""
        return weighted_norm(x, self.weighting, self.exponent, self.spaces)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `ProductSpace` instance, has
            the same length and the same factors. ``False`` otherwise.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> rn, rm = odl.rn(2), odl.rn(3)
        >>> r2x3, rnxm = odl.ProductSpace(r2, r3), odl.ProductSpace(rn, rm)
        >>> r2x3 == rnxm
        True
        >>> r3x2 = odl.ProductSpace(r3, r2)
        >>> r2x3 == r3x2
        False
        >>> r5 = odl.ProductSpace(*[odl.rn(1)]*5)
        >>> r2x3 == r5
        False
        >>> r5 = odl.rn(5)
        >>> r2x3 == r5
        False
        """
        if other is self:
            return True

        return (
            isinstance(other, ProductSpace) and
            len(self) == len(other) and
            all(x == y for x, y in zip(self.spaces, other.spaces))
        )

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Examples
        --------
        Integers are used to pick components, slices to pick ranges:

        >>> r2, r3, r4 = odl.rn(2), odl.rn(3), odl.rn(4)
        >>> pspace = odl.ProductSpace(r2, r3, r4)
        >>> pspace[1]
        rn(3)
        >>> pspace[1:]
        ProductSpace(rn(3), rn(4))

        With lists, arbitrary components can be stacked together:

        >>> pspace[[0, 2, 1, 2]]
        ProductSpace(rn(2), rn(4), rn(3), rn(4))

        Tuples, i.e. multi-indices, will recursively index higher-order
        product spaces. However, remaining indices cannot be passed
        down to component spaces that are not product spaces:

        >>> pspace2 = odl.ProductSpace(pspace, 3)  # 2nd order product space
        >>> pspace2
        ProductSpace(ProductSpace(rn(2), rn(3), rn(4)), 3)
        >>> pspace2[0]
        ProductSpace(rn(2), rn(3), rn(4))
        >>> pspace2[1, 0]
        rn(2)
        >>> pspace2[:-1, 0]
        ProductSpace(rn(2), 2)
        """
        if isinstance(indices, Integral):
            return self.spaces[indices]

        elif isinstance(indices, slice):
            return ProductSpace(*self.spaces[indices], field=self.field)

        elif isinstance(indices, tuple):
            # Use tuple indexing for recursive product spaces, i.e.,
            # pspace[0, 0] == pspace[0][0]
            if not indices:
                return self
            idx = indices[0]
            if isinstance(idx, Integral):
                # Single integer in tuple, picking that space and passing
                # through the rest of the tuple. If the picked space
                # is not a product space and there are still indices left,
                # raise an error.
                space = self.spaces[idx]
                rest_indcs = indices[1:]
                if not rest_indcs:
                    return space
                elif isinstance(space, ProductSpace):
                    return space[rest_indcs]
                else:
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
            elif isinstance(idx, slice):
                # Doing the same as with single integer with all spaces
                # in the slice, but wrapping the result into a ProductSpace.
                spaces = self.spaces[idx]
                rest_indcs = indices[1:]
                if len(spaces) == 0 and rest_indcs:
                    # Need to catch this situation since the code further
                    # down doesn't trigger an error
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
                if not rest_indcs:
                    return ProductSpace(*spaces)
                elif all(isinstance(space, ProductSpace) for space in spaces):
                    return ProductSpace(
                        *(space[rest_indcs] for space in spaces),
                        field=self.field)
                else:
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
            else:
                raise TypeError('index tuple can only contain'
                                'integers or slices')

        elif isinstance(indices, list):
            return ProductSpace(*[self.spaces[i] for i in indices],
                                field=self.field)

        else:
            raise TypeError('`indices` must be integer, slice, tuple or '
                            'list, got {!r}'.format(indices))

    # TODO: fix
    def show(sel, x, title=None, indices=None, **kwargs):
        """Display the parts of this product space element graphically.

        Parameters
        ----------
        title : string, optional
            Title of the figures

        indices : int, slice, tuple or list, optional
            Display parts of ``self`` in the way described in the following.

            A single list of integers selects the corresponding parts
            of this vector.

            For other tuples or lists, the first entry indexes the parts of
            this vector, and the remaining entries (if any) are used to
            slice into the parts. Handling those remaining indices is
            up to the ``show`` methods of the parts to be displayed.

            The types of the first entry trigger the following behaviors:

                - ``int``: take the part corresponding to this index
                - ``slice``: take a subset of the parts
                - ``None``: equivalent to ``slice(None)``, i.e., everything

            Typical use cases are displaying of selected parts, which can
            be achieved with a list, e.g., ``indices=[0, 2]`` for parts
            0 and 2, and plotting of all parts sliced in a certain way,
            e.g., ``indices=[None, 20, None]`` for showing all parts
            sliced with indices ``[20, None]``.

            A single ``int``, ``slice``, ``list`` or ``None`` object
            indexes the parts only, i.e., is treated roughly as
            ``(indices, Ellipsis)``. In particular, for ``None``, all
            parts are shown with default slicing.

        in_figs : sequence of `matplotlib.figure.Figure`, optional
            Update these figures instead of creating new ones. Typically
            the return value of an earlier call to ``show`` is used
            for this parameter.

        kwargs
            Additional arguments passed on to the ``show`` methods of
            the parts.

        Returns
        -------
        figs : tuple of `matplotlib.figure.Figure`
            The resulting figures. In an interactive shell, they are
            automatically displayed.

        See Also
        --------
        odl.discr.lp_discr.DiscreteLpElement.show :
            Display of a discretized function
        odl.space.base_tensors.Tensor.show :
            Display of sequence type data
        odl.util.graphics.show_discrete_data :
            Underlying implementation
        """
        if title is None:
            title = 'ProductSpaceElement'

        if indices is None:
            if len(self) < 5:
                indices = list(range(len(self)))
            else:
                indices = list(np.linspace(0, len(self) - 1, 4, dtype=int))
        else:
            if (isinstance(indices, tuple) or
                    (isinstance(indices, list) and
                     not all(isinstance(idx, Integral) for idx in indices))):
                # Tuples or lists containing non-integers index by axis.
                # We use the first index for the current pspace and pass
                # on the rest.
                indices, kwargs['indices'] = indices[0], indices[1:]

            # Support `indices=[None, 0, None]` like syntax (`indices` is
            # the first entry as of now in that case)
            if indices is None:
                indices = slice(None)

            if isinstance(indices, slice):
                indices = list(range(*indices.indices(len(self))))
            elif isinstance(indices, Integral):
                indices = [indices]
            else:
                # Use `indices` as-is
                pass

        in_figs = kwargs.pop('fig', None)
        in_figs = [None] * len(indices) if in_figs is None else in_figs

        figs = []
        parts = self[indices]
        if len(parts) == 0:
            return ()
        elif len(parts) == 1:
            # Don't extend the title if there is only one plot
            fig = parts[0].show(title=title, fig=in_figs[0], **kwargs)
            figs.append(fig)
        else:
            # Extend titles by indexed part to make them distinguishable
            for i, part, fig in zip(indices, parts, in_figs):
                fig = part.show(title='{}. Part {}'.format(title, i), fig=fig,
                                **kwargs)
                figs.append(fig)

        return tuple(figs)

    def __str__(self):
        """Return ``str(self)``."""
        if len(self) == 0:
            return '{}'
        elif self.is_power_space:
            return '({}) ** {}'.format(self.spaces[0], len(self))
        else:
            return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        """Return ``repr(self)``."""
        edgeitems = np.get_printoptions()['edgeitems']
        if len(self) == 0:
            posargs = []
            posmod = ''
            optargs = [('field', self.field, None)]
            oneline = True
        elif self.is_power_space:
            posargs = [self.spaces[0], len(self)]
            posmod = '!r'
            optargs = []
            oneline = True
        elif self.size <= 2 * edgeitems:
            posargs = self.spaces
            posmod = '!r'
            optargs = []
            argstr = ', '.join(repr(s) for s in self.spaces)
            oneline = (len(argstr) <= 40 and '\n' not in argstr)
        else:
            posargs = (self.spaces[:edgeitems] +
                       ('...',) +
                       self.spaces[-edgeitems:])
            posmod = ['!r'] * edgeitems + ['!s'] + ['!r'] * edgeitems
            optargs = []
            oneline = False

        if oneline:
            inner_str = signature_string(posargs, optargs, sep=', ',
                                         mod=[posmod, '!r'])
            return '{}({})'.format(self.__class__.__name__, inner_str)
        else:
            inner_str = signature_string(posargs, optargs, sep=',\n',
                                         mod=[posmod, '!r'])
            return '{}(\n{}\n)'.format(self.__class__.__name__,
                                       indent(inner_str))


def _weight_type(w):
    """Dispatch function for the type of weight."""
    if np.isscalar(w) or (isinstance(w, np.ndarray) and w.size == 1):
        return 'const'
    elif isinstance(w, np.ndarray) and w.ndim == 1:
        return 'array'
    else:
        return 'unknown'


_dispatcher = lambda x1, x2, weights, subspaces: _weight_type(weights)


@protocol(dispatcher=_dispatcher)
def weighted_inner(x1, x2, weights, subspaces):
    r"""Weighted inner product on product spaces.

    Parameters
    ----------
    x1, x2
        Space elements whose inner product should be calculated.
    weights : numpy.ndarray or scalar
        One-dimensional array or scalar constant. All weights should be
        positive, but this is not checked.
    subspaces : sequence of `LinearSpace`
        Spaces in which the parts of ``x1`` and ``x2`` live.

    Returns
    -------
    inner : scalar
        The weighted inner product of ``x1`` and ``x2``.

    Notes
    -----
    - If :math:`w` is an array, the weighted inner product is defined as

      .. math::
          \langle x, y \rangle_w = \langle w \odot x, y \rangle

      with component-wise multiplication :math:`w \odot x`.

    - If :math:`w` is a scalar constant, the weighted inner product is given
      as

      .. math::
          \langle x, y \rangle_w = w\, \langle x, y \rangle.
    """


@weighted_inner.register('array')
def _array_weighted_inner(x1, x2, weights, subspaces):
    """Inner product on a product space, weighted by an array."""
    inners = np.array(
        [spc.inner(x1i, x2i) for x1i, x2i, spc in zip(x1, x2, subspaces)]
    )
    inner = np.dot(inners, weights)
    if is_real_dtype(inners.dtype):
        return float(inner)
    else:
        return complex(inner)


@weighted_inner.register('const')
def _const_weighted_inner(x1, x2, weight, subspaces):
    """Inner product on a product space, weighted by a constant."""
    inners = np.array(
        [spc.inner(x1i, x2i) for x1i, x2i, spc in zip(x1, x2, subspaces)]
    )
    inner = weight * np.sum(inners)
    if is_real_dtype(inners.dtype):
        return float(inner)
    else:
        return complex(inner)


_dispatcher = lambda x, weights, exponent, subspaces: _weight_type(weights)


@protocol(dispatcher=_dispatcher)
def weighted_norm(x, weights, exponent, subspaces):
    r"""Weighted norm on product spaces.

    Parameters
    ----------
    x
        Element whose norm is calculated.
    weights : numpy.ndarray or scalar
        One-dimensional array or scalar constant. All weights should be
        positive, but this is not checked.
    exponent : float
        Exponent of the norm.
    subspaces : sequence of `LinearSpace`
        Spaces in which the parts of ``x`` live.

    Returns
    -------
    norm : float
        The weighted norm of ``x``.

    Notes
    -----
    - If :math:`w` is an array, the weighted norm is defined as

      .. math::
          \|x\|_{w,p} = \|w^{1/p} \odot x\|_p,\quad & p < \infty,

          \|x\|_{w,\infty} = \|w \odot x\|_\infty,\quad & p = \infty,

      with component-wise multiplication :math:`w \odot x`.

    - If :math:`w` is a scalar constant, the weighted norm is given as

      .. math::
          \|x\|_{w,p} = w^{1/p} \|x\|_p,\quad & p < \infty,

          \|x\|_{w,\infty} = w \|x\|_\infty,\quad & p = \infty.
    """


@weighted_norm.register('array')
def _array_weighted_norm(x, weights, exponent, subspaces):
    """Norm on a product space, weighted by an array."""
    if exponent == 2.0:
        norm_squared = _array_weighted_inner(x, x, weights, subspaces).real
        return float(np.sqrt(norm_squared))

    norms = np.array(
        [spc.norm(xi) for xi, spc in zip(x, subspaces)],
        dtype=float
    )
    if exponent in (1.0, float('inf')):
        norms *= weights
    else:
        norms *= weights ** (1.0 / exponent)

    return float(np.linalg.norm(norms, ord=exponent))


@weighted_norm.register('const')
def _const_weighted_norm(x, weight, exponent, subspaces):
    r"""Norm on a product space, weighted by a constant."""
    if exponent == 2.0:
        norm_squared = _const_weighted_inner(x, x, weight, subspaces).real
        return float(np.sqrt(norm_squared))

    norms = np.array(
        [spc.norm(xi) for xi, spc in zip(x, subspaces)],
        dtype=float
    )
    norm = np.linalg.norm(norms, ord=exponent)
    if exponent in (1.0, float('inf')):
        norm *= weight
    else:
        norm *= weight ** (1.0 / exponent)

    return float(norm)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
