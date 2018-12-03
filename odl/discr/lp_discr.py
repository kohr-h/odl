# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Lebesgue L^p type discretizations of function spaces."""

from __future__ import print_function, division, absolute_import
from numbers import Integral
import numpy as np

from odl.discr.discretization import (
    DiscretizedSpace, tspace_type)
from odl.discr.discr_mappings import (
    PointCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)
from odl.discr.partition import (
    RectPartition, uniform_partition_fromintv, uniform_partition)
from odl.set import RealNumbers, ComplexNumbers, IntervalProd
from odl.space.fspace import FunctionSpace
from odl.space.pspace import ProductSpace
from odl.space.entry_points import tensor_space_impl
from odl.space.weighting import ConstWeighting
from odl.util import (
    apply_on_boundary, is_real_dtype, is_complex_floating_dtype, is_string,
    is_floating_dtype, is_numeric_dtype,
    dtype_str, array_str, signature_string, indent, npy_printoptions,
    normalized_scalar_param_list, safe_int_conv, normalized_nodes_on_bdry)

__all__ = ('DiscreteLp',
           'uniform_discr_frompartition', 'uniform_discr_fromspace',
           'uniform_discr_fromintv', 'uniform_discr',
           'uniform_discr_fromdiscr', 'discr_sequence_space')

_SUPPORTED_INTERP = ('nearest', 'linear')


class DiscreteLp(DiscretizedSpace):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, fspace, partition, tspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The continuous space to be discretized.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain``.
        tspace : `TensorSpace`
            Space of elements used for data storage. It must have the
            same `TensorSpace.field` as ``fspace`` and the same
            `TensorSpace.shape` as ``partition``.
        interp : str or sequence of str, optional
            Interpolation type to be used for discretization.
            A sequence is interpreted as interpolation scheme per axis.
            Possible values:

            - ``'nearest'`` : use nearest-neighbor interpolation.
            - ``'linear'`` : use linear interpolation.

        axis_labels : sequence of str, optional
            Names of the axes to use for plotting etc.
            Default:

            - 1D: ``['$x$']``
            - 2D: ``['$x$', '$y$']``
            - 3D: ``['$x$', '$y$', '$z$']``
            - nD: ``['$x_1$', '$x_2$', ..., '$x_n$']``

            Note: The ``$`` signs ensure rendering as LaTeX.
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('{!r} is not a FunctionSpace instance'
                            ''.format(fspace))
        if not isinstance(fspace.domain, IntervalProd):
            raise TypeError('function space domain {!r} is not an '
                            'IntervalProd instance'.format(fspace.domain))
        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a RectPartition '
                            'instance'.format(partition))
        if not fspace.domain.contains_set(partition.set):
            raise ValueError('`partition` {} is not a subset of the function '
                             'domain {}'.format(partition, fspace.domain))
        if fspace.scalar_out_dtype != tspace.dtype:
            raise ValueError('`fspace.scalar_out_dtype` does not match '
                             '`tspace.dtype`: {} != {}'
                             ''.format(fspace.scalar_out_dtype, tspace.dtype))

        self.__partition = partition

        # Handle interp
        if is_string(interp):
            interp, interp_in = str(interp).lower(), interp
            if interp not in _SUPPORTED_INTERP:
                raise ValueError('`interp` {!r} not understood'
                                 ''.format(interp_in))
            # Ensure that there is 1 entry for ndim == 0
            self.__interp_byaxis = (interp,) * max(partition.ndim, 1)
        else:
            # Got sequence of strings
            if len(interp) != partition.ndim:
                raise ValueError('expected {} (ndim) entries in `interp`, '
                                 'got {}'.format(partition.ndim, len(interp)))

            self.__interp_byaxis = tuple(str(s).lower() for s in interp)
            if any(s not in _SUPPORTED_INTERP for s in self.interp_byaxis):
                raise ValueError('`interp` sequence {} contains illegal '
                                 'values'.format(interp))

        # Assign sampling and interpolation operators
        sampling = PointCollocation(fspace, self.partition, tspace)
        if all(s == 'nearest' for s in self.interp_byaxis):
            interpol = NearestInterpolation(fspace, self.partition, tspace)
        elif all(s == 'linear' for s in self.interp_byaxis):
            interpol = LinearInterpolation(fspace, self.partition, tspace)
        else:
            interpol = PerAxisInterpolation(
                fspace, self.partition, tspace, self.interp_byaxis)

        super(DiscreteLp, self).__init__(fspace, tspace, sampling, interpol)

        # Set axis labels
        axis_labels = kwargs.pop('axis_labels', None)
        if axis_labels is None:
            if self.ndim <= 3:
                self.__axis_labels = ('$x$', '$y$', '$z$')[:self.ndim]
            else:
                self.__axis_labels = tuple('$x_{}$'.format(axis)
                                           for axis in range(self.ndim))
        else:
            self.__axis_labels = tuple(str(label) for label in axis_labels)

        if kwargs:
            raise ValueError('got unexpected keyword arguments {}'
                             ''.format(kwargs))

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        if self.ndim == 0:
            return 'nearest'
        elif all(interp == self.interp_byaxis[0]
                 for interp in self.interp_byaxis):
            return self.interp_byaxis[0]
        else:
            return self.interp_byaxis

    @property
    def interp_byaxis(self):
        """Interpolation by axis type of this discretization."""
        return self.__interp_byaxis

    @property
    def axis_labels(self):
        """Labels for axes when displaying space elements."""
        return self.__axis_labels

    @property
    def partition(self):
        """`RectPartition` of the function domain."""
        return self.__partition

    @property
    def exponent(self):
        """Exponent of this space, the ``p`` in ``L^p``."""
        return self.tspace.exponent

    @property
    def min_pt(self):
        """Vector of minimal coordinates of the function domain."""
        return self.partition.min_pt

    @property
    def max_pt(self):
        """Vector of maximal coordinates of the function domain."""
        return self.partition.max_pt

    @property
    def is_uniform_byaxis(self):
        """Boolean tuple showing uniformity of ``self.partition`` per axis."""
        return self.partition.is_uniform_byaxis

    @property
    def is_uniform(self):
        """``True`` if `partition` is uniform."""
        return self.partition.is_uniform

    @property
    def grid(self):
        """Sampling grid of the discretization mappings."""
        return self.partition.grid

    @property
    def shape(self):
        """Shape of the underlying partition."""
        return self.partition.shape

    @property
    def ndim(self):
        """Number of dimensions (= number of axes)."""
        return self.partition.ndim

    @property
    def size(self):
        """Total number of underlying partition cells."""
        return self.partition.size

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.partition.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying *uniform* partition."""
        return self.partition.cell_volume

    @property
    def meshgrid(self):
        """All sampling points in the partition as a sparse meshgrid."""
        return self.partition.meshgrid

    def points(self, order='C'):
        """All sampling points in the partition.

        Parameters
        ----------
        order : {'C', 'F'}
            Axis ordering in the resulting point array.

        Returns
        -------
        points : `numpy.ndarray`
            The shape of the array is ``size x ndim``, i.e. the points
            are stored as rows.
        """
        return self.partition.points(order)

    @property
    def default_order(self):
        """Default storage order for new elements in this space.

        This is equal to the default order of `tspace`.
        """
        return self.tspace.default_order

    @property
    def tangent_bundle(self):
        """The tangent bundle associated with `domain` using `partition`.

        The tangent bundle of a space ``X`` of functions ``R^d --> F`` can be
        interpreted as the space of vector-valued functions ``R^d --> F^d``.
        This space can be identified with the power space ``X^d`` as used
        in this implementation.
        """
        if self.ndim == 0:
            return ProductSpace(field=self.field)
        else:
            return ProductSpace(self, self.ndim)

    @property
    def is_uniformly_weighted(self):
        """``True`` if the weighting is the same for all space points."""
        try:
            is_uniformly_weighted = self.__is_uniformly_weighted
        except AttributeError:
            bdry_fracs = self.partition.boundary_cell_fractions
            is_uniformly_weighted = (
                np.allclose(bdry_fracs, 1.0) or
                self.exponent == float('inf') or
                not getattr(self.tspace, 'is_weighted', False))

            self.__is_uniformly_weighted = is_uniformly_weighted

        return is_uniformly_weighted

    def element(self, inp=None, order=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input used to initialize the new element. The following options
            are available:

            - ``None``: an empty element is created with no guarantee of
              its state (memory allocation only). The new element will
              use ``order`` as storage order if provided, otherwise
              `default_order`.

            - array-like: an element wrapping a `tensor` is created,
              where a copy is avoided whenever possible. This usually
              requires correct `shape`, `dtype` and `impl` if applicable,
              and if ``order`` is provided, also contiguousness in that
              ordering. See the ``element`` method of `tspace` for more
              information.

              If any of these conditions is not met, a copy is made.

            - callable: a new element is created by sampling the function
              using the `sampling` operator.

        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.
        vectorized : bool, optional
            If ``True``, assume that a provided callable ``inp`` supports
            vectorized evaluation. Otherwise, wrap it in a vectorizer.
            Default: ``True``.
        kwargs :
            Additional arguments passed on to `sampling` when called
            on ``inp``, in the form ``sampling(inp, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Returns
        -------
        element : `DiscreteLpElement`
            The discretized element, calculated as ``sampling(inp)`` or
            ``tspace.element(inp)``, tried in this order.

        Examples
        --------
        Elements can be created from array-like objects that represent
        an already discretized function:

        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element([1, 2, 3, 4])
        uniform_discr(-1.0, 1.0, 4).element([ 1.,  2.,  3.,  4.])
        >>> vector = odl.rn(4).element([0, 1, 2, 3])
        >>> space.element(vector)
        uniform_discr(-1.0, 1.0, 4).element([ 0.,  1.,  2.,  3.])

        On the other hand, non-discretized objects like Python functions
        can be discretized "on the fly":

        >>> space.element(lambda x: x * 2)
        uniform_discr(-1.0, 1.0, 4).element([-1.5, -0.5, 0.5, 1.5])

        This works also with parameterized functions, however only
        through keyword arguments (not positional arguments with
        defaults):

        >>> def f(x, c=0.0):
        ...     return np.maximum(x, c)
        ...
        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element(f, c=0.5)
        uniform_discr(-1.0, 1.0, 4).element([ 0.5 ,  0.5 ,  0.5 ,  0.75])

        See Also
        --------
        sampling : create a discrete element from a non-discretized one
        """
        if inp is None:
            return self.tspace.element(order=order)
        elif inp in self and order is None:
            return inp
        elif inp in self.tspace and order is None:
            return self.element_type(self, inp)
        elif callable(inp):
            vectorized = kwargs.pop('vectorized', True)
            # fspace element -> discretize
            inp_elem = self.fspace.element(inp, vectorized=vectorized)
            sampled = self.sampling(inp_elem, **kwargs)
            return self.tspace.element(sampled, order=order)
        else:
            # Sequence-type input
            return self.tspace.element(inp, order=order)

    def _astype(self, dtype):
        """Internal helper for ``astype``."""
        fspace = self.fspace.astype(dtype)
        tspace = self.tspace.astype(dtype)
        return type(self)(
            fspace, self.partition, tspace, interp=self.interp,
            axis_labels=self.axis_labels)

    # Overrides for space functions depending on partition
    #
    # The inherited methods by default use a weighting by a constant
    # (the grid cell size). In dimensions where the partitioned set contains
    # only a fraction of the outermost cells (e.g. if the outermost grid
    # points lie at the boundary), the corresponding contribuitons to
    # discretized integrals need to be scaled by that fraction.
    def _inner(self, x, y):
        """Return ``self.inner(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=1.0)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super(DiscreteLp, self)._inner(self.element(x_arr), y)
        else:
            return super(DiscreteLp, self)._inner(x, y)

    def _norm(self, x):
        """Return ``self.norm(x)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super(DiscreteLp, self)._norm(self.element(x_arr))
        else:
            return super(DiscreteLp, self)._norm(x)

    def _dist(self, x, y):
        """Return ``self.dist(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            arrs = [apply_on_boundary(vec, func=func_list, only_once=False)
                    for vec in (x, y)]

            return super(DiscreteLp, self)._dist(
                self.element(arrs[0]), self.element(arrs[1]))
        else:
            return super(DiscreteLp, self)._dist(x, y)

    # TODO: add byaxis_out when discretized tensor-valued functions are
    # available

    @property
    def byaxis_in(self):
        """Object to index along input (domain) dimensions.

        Examples
        --------
        Indexing with integers or slices:

        >>> space = odl.uniform_discr([0, 0, 0], [1, 2, 3], (5, 10, 15))
        >>> space.byaxis_in[0]
        uniform_discr(0.0, 1.0, 5)
        >>> space.byaxis_in[1]
        uniform_discr(0.0, 2.0, 10)
        >>> space.byaxis_in[1:]
        uniform_discr([ 0.,  0.], [ 2.,  3.], (10, 15))

        Lists can be used to stack spaces arbitrarily:

        >>> space.byaxis_in[[2, 1, 2]]
        uniform_discr([ 0.,  0.,  0.], [ 3.,  2.,  3.], (15, 10, 15))
        """
        space = self

        class DiscreteLpByaxisIn(object):

            """Helper class for indexing by domain axes."""

            def __getitem__(self, indices):
                """Return ``self[indices]``.

                Parameters
                ----------
                indices : index expression
                    Object used to index the space domain.

                Returns
                -------
                space : `DiscreteLp`
                    The resulting space with indexed domain and otherwise
                    same properties (except possibly weighting).
                """
                fspace = space.fspace.byaxis_in[indices]
                part = space.partition.byaxis[indices]

                if isinstance(space.weighting, ConstWeighting):
                    # Need to manually construct `tspace` since it doesn't
                    # know where its weighting factor comes from
                    try:
                        iter(indices)
                    except TypeError:
                        newshape = space.shape[indices]
                    else:
                        newshape = tuple(space.shape[int(i)] for i in indices)

                    weighting = part.cell_volume
                    tspace = type(space.tspace)(
                        newshape, space.dtype,
                        exponent=space.exponent, weighting=weighting)
                else:
                    # Other weighting schemes are handled correctly by
                    # the tensor space
                    tspace = space.tspace.byaxis[indices]

                try:
                    iter(indices)
                except TypeError:
                    interp = space.interp_byaxis[indices]
                    labels = space.axis_labels[indices]
                else:
                    interp = tuple(space.interp_byaxis[int(i)]
                                   for i in indices)
                    labels = tuple(space.axis_labels[int(i)]
                                   for i in indices)

                return DiscreteLp(fspace, part, tspace, interp,
                                  axis_labels=labels)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis_in'

        return DiscreteLpByaxisIn()

    def __repr__(self):
        """Return ``repr(self)``."""
        # Clunky check if the factory repr can be used
        if (uniform_partition_fromintv(
                self.fspace.domain, self.shape,
                nodes_on_bdry=False) == self.partition):
            use_uniform = True
            nodes_on_bdry = False
        elif (uniform_partition_fromintv(
                self.fspace.domain, self.shape,
                nodes_on_bdry=True) == self.partition):
            use_uniform = True
            nodes_on_bdry = True
        else:
            use_uniform = False

        if use_uniform:
            ctor = 'uniform_discr'
            if self.ndim == 1:
                posargs = [self.min_pt[0], self.max_pt[0], self.shape[0]]
                posmod = [':.4', ':.4', '']
            else:
                posargs = [self.min_pt, self.max_pt, self.shape]
                posmod = [array_str, array_str, '']

            default_dtype_s = dtype_str(
                self.tspace.default_dtype(RealNumbers()))

            dtype_s = dtype_str(self.dtype)
            optargs = [('interp', self.interp, 'nearest'),
                       ('impl', self.impl, 'numpy'),
                       ('nodes_on_bdry', nodes_on_bdry, False),
                       ('dtype', dtype_s, default_dtype_s)]

            # Add weighting stuff if not equal to default
            if (self.exponent == float('inf') or
                    self.ndim == 0 or
                    not is_floating_dtype(self.dtype)):
                # In these cases, weighting constant 1 is the default
                if (not isinstance(self.weighting, ConstWeighting) or
                        not np.isclose(self.weighting.const, 1.0)):
                    optargs.append(('weighting', self.weighting.const, None))
            else:
                if (not isinstance(self.weighting, ConstWeighting) or
                        not np.isclose(self.weighting.const,
                                       self.cell_volume)):
                    optargs.append(('weighting', self.weighting.const, None))

            optmod = [''] * len(optargs)
            if self.dtype in (float, complex, int, bool):
                optmod[3] = '!s'

            with npy_printoptions(precision=4):
                inner_str = signature_string(posargs, optargs,
                                             mod=[posmod, optmod])
            return '{}({})'.format(ctor, inner_str)

        else:
            ctor = self.__class__.__name__
            posargs = [self.fspace, self.partition, self.tspace]
            optargs = [('interp', self.interp, 'nearest')]

            with npy_printoptions(precision=4):
                inner_str = signature_string(posargs, optargs,
                                             sep=[',\n', ', ', ',\n'],
                                             mod=['!r', '!s'])

            return '{}({})'.format(ctor, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


# TODO: replace `self`
def show(self, title=None, method='', coords=None, indices=None,
         force_show=False, fig=None, **kwargs):
    """Display the function graphically.

    Parameters
    ----------
    title : string, optional
        Set the title of the figure

    method : string, optional
        1d methods:

        - ``'plot'`` : graph plot (default for 1d data)
        - ``'scatter'`` : scattered 2d points (2nd axis <-> value)

        2d methods:

        - ``'imshow'`` : image plot with coloring according to value,
          including a colorbar (default for 2d data).
        - ``'scatter'`` : cloud of scattered 3d points
          (3rd axis <-> value)

    coords : `array-like`, optional
        Display a slice of the array instead of the full array.
        The values are shown accordinging to the given values,
        where ``None`` means all values along that dimension. For
        example, ``[None, None, 0.5]`` shows all values in the first
        two dimensions, with the third coordinate equal to 0.5.
        If a sequence is provided, it specifies the minimum and maximum
        point to be shown, i.e. ``[None, [0, 1]]`` shows all of the
        first axis and values between 0 and 1 in the second.
        This option is mutually exclusive with ``indices``.

    indices : int, slice, Ellipsis or sequence, optional
        Display a slice of the array instead of the full array.
        If a sequence is given, the i-th entry indexes the i-th axis,
        with the following behavior for the different types of entries:

            - ``int``: take entries with this index along axis ``i``,
              removing this axis from the result
            - ``slice``: take a subset along axis ``i``, keeping it
              intact
            - ``None``: equivalent to ``slice(None)``
            - ``Ellipsis`` (``...``): equivalent to the number of
              ``None`` entries required to fill up the sequence to
              correct length.

        The typical use case is to show a slice for a fixed index in
        a specific axis, which can be done most easily by setting, e.g.,
        ``indices=[None, 50, None]`` to take the 2d slice parallel to
        the x-z coordinate plane at index ``y = 50``.

        A single ``int`` or ``slice`` object indexes the first
        axis, i.e., is treated as ``(int_or_slice, Ellipsis)``.
        For the default ``None``, the array is kepty as-is for data
        that has at most 2 dimensions. For higher-dimensional
        data, the 2d slice in the first two axes at the middle
        position along the remaining axes is shown
        (semantically ``[:, :, shape[2:] // 2]``).
        This option is mutually exclusive with ``coords``.

    force_show : bool, optional
        Whether the plot should be forced to be shown now or deferred until
        later. Note that some backends always displays the plot, regardless
        of this value.

    fig : `matplotlib.figure.Figure`, optional
        The figure to show in. Expected to be of same "style", as
        the figure given by this function. The most common use case
        is that ``fig`` is the return value of an earlier call to
        this function.

    kwargs : {'figsize', 'saveto', 'clim', ...}, optional
        Extra keyword arguments passed on to the display method.
        See the Matplotlib functions for documentation of extra
        options.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The resulting figure. It is also shown to the user.

    See Also
    --------
    odl.util.graphics.show_discrete_data : Underlying implementation
    """
    from odl.util.graphics import show_discrete_data

    if 'interp' not in kwargs and not isinstance(self.space.interp, tuple):
        # Otherwise different interp per axis
        kwargs['interp'] = self.space.interp

    if self.ndim == 0:
        raise ValueError('nothing to show for 0-dimensional vector')

    if coords is not None:
        if indices is not None:
            raise ValueError('cannot provide both coords and indices')

        partition = self.space.partition
        shape = self.shape
        indices = []
        for axis, (n, coord) in enumerate(zip(shape, coords)):
            try:
                coord_minp, coord_maxp = coord
            except TypeError:
                coord_minp = coord_maxp = coord

            subpart = partition.byaxis[axis]

            # Validate input
            if coord_minp is not None:
                coord_minp = subpart.set.element(coord_minp)
            if coord_maxp is not None:
                coord_maxp = subpart.set.element(coord_maxp)

            if len(subpart) == 0:  # trivial cases
                indices.append(0)
            elif coord_minp is not None and coord_minp == coord_maxp:
                indices.append(subpart.index(coord_minp))
            else:
                if coord_minp is None:
                    min_ind = 0
                else:
                    min_ind = np.floor(subpart.index(coord_minp,
                                                     floating=True))

                if coord_maxp is None:
                    max_ind = len(subpart)
                else:
                    max_ind = np.ceil(subpart.index(coord_maxp,
                                                    floating=True))

                indices.append(slice(int(min_ind), int(max_ind)))

    # Default to showing x-y slice "in the middle"
    if indices is None and self.ndim >= 3:
        indices = ((slice(None),) * 2 +
                   tuple(n // 2 for n in self.space.shape[2:]))

    # Normalize indices
    if isinstance(indices, (Integral, slice)):
        indices = (indices,)
    elif indices is None or indices == Ellipsis:
        indices = (slice(None),) * self.ndim

    # Single index or slice indexes the first axis, rest untouched
    if len(indices) == 1:
        indices = tuple(indices) + (Ellipsis,)

    # Convert `Ellipsis` objects
    if indices.count(Ellipsis) > 1:
        raise ValueError('cannot use more than 1 `Ellipsis` (`...`)')
    elif Ellipsis in indices:
        # Replace Ellipsis with the correct number of `slice(None)`
        pos = indices.index(Ellipsis)
        indices = (tuple(indices[:pos]) +
                   (slice(None),) * (self.ndim - len(indices) + 1) +
                   tuple(indices[pos + 1:]))

    # Now indices should be exactly of length `ndim`
    if len(indices) < self.ndim:
        raise ValueError('too few axes ({} < {})'.format(len(indices),
                                                         self.ndim))
    if len(indices) > self.ndim:
        raise ValueError('too many axes ({} > {})'.format(len(indices),
                                                          self.ndim))

    # Map `None` to `slice(None)` in indices for syntax like `coords`
    indices = tuple(slice(None) if idx is None else idx
                    for idx in indices)

    squeezed_axes = [axis for axis in range(self.ndim)
                     if not isinstance(indices[axis], Integral)]
    axis_labels = [self.space.axis_labels[axis] for axis in squeezed_axes]

    # Squeeze grid and values according to the index expression
    part = self.space.partition[indices].squeeze()
    values = self.asarray()[indices].squeeze()

    return show_discrete_data(values, part, title=title, method=method,
                              force_show=force_show, fig=fig,
                              axis_labels=axis_labels, **kwargs)


def uniform_discr_frompartition(partition, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    partition : `RectPartition`
        Uniform partition to be used for discretization.
        It defines the domain and the functions and the grid for
        discretization.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space.

    Examples
    --------
    >>> part = odl.uniform_partition(0, 1, 10)
    >>> uniform_discr_frompartition(part)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_fromspace : uniform Lp discretization from an existing
        function space
    odl.discr.partition.uniform_partition :
        partition of the function domain
    """
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` {!r} is not a `RectPartition` instance'
                        ''.format(partition))
    if not partition.is_uniform:
        raise ValueError('`partition` is not uniform')

    if dtype is not None:
        dtype = np.dtype(dtype)

    fspace = FunctionSpace(partition.set, out_dtype=dtype)
    ds_type = tspace_type(fspace, impl, dtype)

    if dtype is None:
        dtype = ds_type.default_dtype()

    weighting = kwargs.pop('weighting', None)
    exponent = kwargs.pop('exponent', 2.0)
    if weighting is None and is_numeric_dtype(dtype):
        if exponent == float('inf') or partition.ndim == 0:
            weighting = 1.0
        else:
            weighting = partition.cell_volume

    tspace = ds_type(partition.shape, dtype, exponent=exponent,
                     weighting=weighting)
    return DiscreteLp(fspace, partition, tspace, **kwargs)


def uniform_discr_fromspace(fspace, shape, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    fspace : `FunctionSpace`
        Continuous function space. Its domain must be an `IntervalProd`.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> intv = odl.IntervalProd(0, 1)
    >>> space = odl.FunctionSpace(intv)
    >>> uniform_discr_fromspace(space, 10)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromintv : uniform discretization from an existing
        interval product
    odl.discr.partition.uniform_partition :
        partition of the function domain
    """
    if not isinstance(fspace, FunctionSpace):
        raise TypeError('`fspace` {!r} is not a `FunctionSpace` instance'
                        ''.format(fspace))
    if not isinstance(fspace.domain, IntervalProd):
        raise TypeError('domain {!r} of the function space is not an '
                        '`IntervalProd` instance'.format(fspace.domain))

    # Set data type. If given, check consistency with fspace's field and
    # out_dtype. If not given, take the latter.
    if dtype is None:
        dtype = fspace.out_dtype
    else:
        dtype, dtype_in = np.dtype(dtype), dtype
        if not np.can_cast(fspace.scalar_out_dtype, dtype, casting='safe'):
            raise ValueError('cannot safely cast from output data {} type of '
                             'the function space to given data type {}'
                             ''.format(fspace.out, dtype_in))

    if fspace.field == RealNumbers() and not is_real_dtype(dtype):
        raise ValueError('cannot discretize real space {} with '
                         'non-real data type {}'
                         ''.format(fspace, dtype))
    elif (fspace.field == ComplexNumbers() and
          not is_complex_floating_dtype(dtype)):
        raise ValueError('cannot discretize complex space {} with '
                         'non-complex-floating data type {}'
                         ''.format(fspace, dtype))

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    partition = uniform_partition_fromintv(fspace.domain, shape,
                                           nodes_on_bdry)

    return uniform_discr_frompartition(partition, dtype, impl, **kwargs)


def uniform_discr_fromintv(intv_prod, shape, dtype=None, impl='numpy',
                           **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Function domain of the uniformly discretized space.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : str, optional
        Implementation of the data storage arrays.
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> intv = IntervalProd(0, 1)
    >>> uniform_discr_fromintv(intv, 10)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    """
    if dtype is None:
        dtype = tensor_space_impl(str(impl).lower()).default_dtype()

    fspace = FunctionSpace(intv_prod, out_dtype=dtype)
    return uniform_discr_fromspace(fspace, shape, dtype, impl, **kwargs)


def uniform_discr(min_pt, max_pt, shape, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    min_pt, max_pt : float or sequence of floats
        Minimum/maximum corners of the desired function domain.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays.

    Other Parameters
    ----------------
    exponent : positive float, optional
        The parameter :math:`p` in :math:`L^p`. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

        - ``'nearest'`` : use nearest-neighbor interpolation.
        - ``'linear'`` : use linear interpolation.

    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``len(shape)``.
        A single boolean is interpreted as a global choice for all
        boundaries.
        Default: ``False``.
    weighting : optional
        Use weighted inner product, norm, and dist. The following
        types are supported as ``weighting``:

        - ``None``: Use the cell volume as weighting constant (default).
        - ``float``: Weighting by a constant.
        - array-like: Point-wise weighting by an array.
        - `Weighting`: Use weighting class as-is. Compatibility
          with this space's elements is not checked during init.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    Create real space:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10))
    >>> space
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10))
    >>> space.cell_sides
    array([ 0.1,  0.1])
    >>> space.dtype
    dtype('float64')
    >>> space.is_real
    True

    Create complex space by giving a dtype:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10), dtype=complex)
    >>> space
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10), dtype=complex)
    >>> space.is_complex
    True
    >>> space.real_space  # Get real counterpart
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10))

    See Also
    --------
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    uniform_discr_fromintv : uniform discretization from an existing
        interval product
    """
    intv_prod = IntervalProd(min_pt, max_pt)
    return uniform_discr_fromintv(intv_prod, shape, dtype, impl, **kwargs)


def discr_sequence_space(shape, dtype=None, impl='numpy', **kwargs):
    """Return an object mimicing the sequence space ``l^p(R^d)``.

    The returned object is a `DiscreteLp` on the domain ``[0, shape - 1]``,
    using a uniform grid with stride 1.

    Parameters
    ----------
    shape : int or sequence of ints
        Number of element entries per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays.
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.
        Note that ``nodes_on_bdry`` cannot be given.

    Returns
    -------
    seqspc : `DiscreteLp`
        Sequence-space-like discrete Lp.

    Examples
    --------
    >>> seq_spc = discr_sequence_space((3, 3))
    >>> seq_spc.one().norm() == 3.0
    True
    >>> seq_spc = discr_sequence_space((3, 3), exponent=1)
    >>> seq_spc.one().norm() == 9.0
    True
    """
    shape = np.atleast_1d(shape)
    return uniform_discr([0] * len(shape), shape - 1, shape, dtype, impl,
                         nodes_on_bdry=True, **kwargs)


def uniform_discr_fromdiscr(discr, min_pt=None, max_pt=None,
                            shape=None, cell_sides=None, **kwargs):
    """Return a discretization based on an existing one.

    The parameters that are explicitly given are used to create the
    new discretization, and the missing information is taken from
    the template space. See Notes for the exact functionality.

    Parameters
    ----------
    discr : `DiscreteLp`
        Uniformly discretized space used as a template.
    min_pt, max_pt: float or sequence of floats, optional
        Desired minimum/maximum corners of the new space domain.
    shape : int or sequence of ints, optional
        Desired number of samples per axis of the new space.
    cell_sides : float or sequence of floats, optional
        Desired cell side lenghts of the new space's partition.
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``discr.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

        Default: ``False``.

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

    Notes
    -----
    The parameters ``min_pt``, ``max_pt``, ``shape`` and
    ``cell_sides`` can be combined in the following ways (applies in
    each axis individually):

    **0 arguments:**
        Return a copy of ``discr``

    **1 argument:**
        [min,max]_pt -> keep sampling but translate domain so it
        starts/ends at ``[min,max]_pt``

        shape/cell_sides -> keep domain but change sampling.
        See `uniform_partition` for restrictions.

    **2 arguments:**
        min_pt + max_pt -> translate and resample with the same
        number of samples

        [min,max]_pt + shape/cell_sides -> translate and resample

        shape + cell_sides -> error due to ambiguity (keep
        ``min_pt`` or ``max_pt``?)

    **3+ arguments:**
        The underlying partition is uniquely determined by the new
        parameters. See `uniform_partition`.

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    odl.discr.partition.uniform_partition :
        underlying domain partitioning scheme

    Examples
    --------
    >>> discr = odl.uniform_discr([0, 0], [1, 2], (10, 5))
    >>> discr.cell_sides
    array([ 0.1,  0.4])

    If no additional argument is given, a copy of ``discr`` is
    returned:

    >>> odl.uniform_discr_fromdiscr(discr) == discr
    True
    >>> odl.uniform_discr_fromdiscr(discr) is discr
    False

    Giving ``min_pt`` or ``max_pt`` results in a
    translation, while for the other two options, the domain
    is kept but re-partitioned:

    >>> odl.uniform_discr_fromdiscr(discr, min_pt=[1, 1])
    uniform_discr([ 1.,  1.], [ 2.,  3.], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, max_pt=[0, 0])
    uniform_discr([-1., -2.], [ 0.,  0.], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, cell_sides=[1, 1])
    uniform_discr([ 0.,  0.], [ 1.,  2.], (1, 2))
    >>> odl.uniform_discr_fromdiscr(discr, shape=[5, 5])
    uniform_discr([ 0.,  0.], [ 1.,  2.], (5, 5))
    >>> odl.uniform_discr_fromdiscr(discr, shape=[5, 5]).cell_sides
    array([ 0.2,  0.4])

    The cases with 2 or more additional arguments and the syntax
    for specifying quantities per axis is illustrated in the following:

    # axis 0: translate to match max_pt = 3
    # axis 1: recompute max_pt using the original shape with the
    # new min_pt and cell_sides
    >>> new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=[None, 1],
    ...                                         max_pt=[3, None],
    ...                                         cell_sides=[None, 0.25])
    >>> new_discr
    uniform_discr([ 2.,  1.], [ 3.  ,  2.25], (10, 5))
    >>> new_discr.cell_sides
    array([ 0.1 ,  0.25])

    # axis 0: recompute min_pt from old cell_sides and new
    # max_pt and shape
    # axis 1: use new min_pt, shape and cell_sides only
    >>> new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=[None, 1],
    ...                                         max_pt=[3, None],
    ...                                         shape=[5, 5],
    ...                                         cell_sides=[None, 0.25])
    >>> new_discr
    uniform_discr([ 2.5,  1. ], [ 3.  ,  2.25], (5, 5))
    >>> new_discr.cell_sides
    array([ 0.1 ,  0.25])
    """
    if not isinstance(discr, DiscreteLp):
        raise TypeError('`discr` {!r} is not a DiscreteLp instance'
                        ''.format(discr))
    if not discr.is_uniform:
        raise ValueError('`discr` {} is not uniformly discretized'
                         ''.format(discr))

    # Normalize partition parameters
    min_pt = normalized_scalar_param_list(min_pt, discr.ndim,
                                          param_conv=float, keep_none=True)
    max_pt = normalized_scalar_param_list(max_pt, discr.ndim,
                                          param_conv=float, keep_none=True)
    shape = normalized_scalar_param_list(shape, discr.ndim,
                                         param_conv=safe_int_conv,
                                         keep_none=True)
    cell_sides = normalized_scalar_param_list(cell_sides, discr.ndim,
                                              param_conv=float, keep_none=True)

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    nodes_on_bdry = normalized_nodes_on_bdry(nodes_on_bdry, discr.ndim)

    new_min_pt = []
    new_max_pt = []
    new_shape = []
    new_csides = []
    for i, (xmin, xmax, n, s, old_xmin, old_xmax, old_n, old_s) in enumerate(
            zip(min_pt, max_pt, shape, cell_sides,
                discr.min_pt, discr.max_pt, discr.shape,
                discr.cell_sides)):
        num_params = sum(p is not None for p in (xmin, xmax, n, s))

        if num_params == 0:
            new_params = [old_xmin, old_xmax, old_n, None]

        elif num_params == 1:
            if xmin is not None:
                new_params = [xmin, old_xmax + (xmin - old_xmin), old_n, None]
            elif xmax is not None:
                new_params = [old_xmin + (xmax - old_xmax), xmax, old_n, None]
            elif n is not None:
                new_params = [old_xmin, old_xmax, n, None]
            else:
                new_params = [old_xmin, old_xmax, None, s]

        elif num_params == 2:
            if xmin is not None and xmax is not None:
                new_params = [xmin, xmax, old_n, None]
            elif xmin is not None and n is not None:
                new_params = [xmin, None, n, old_s]
            elif xmin is not None and s is not None:
                new_params = [xmin, None, old_n, s]
            elif xmax is not None and n is not None:
                new_params = [None, xmax, n, old_s]
            elif xmax is not None and s is not None:
                new_params = [None, xmax, old_n, s]
            else:
                raise ValueError('in axis {}: cannot use `shape` and '
                                 '`cell_size` only due to ambiguous values '
                                 'for `min_pt` and `max_pt`.'.format(i))

        else:
            new_params = [xmin, xmax, n, s]

        new_min_pt.append(new_params[0])
        new_max_pt.append(new_params[1])
        new_shape.append(new_params[2])
        new_csides.append(new_params[3])

    new_part = uniform_partition(min_pt=new_min_pt, max_pt=new_max_pt,
                                 shape=new_shape,
                                 cell_sides=new_csides,
                                 nodes_on_bdry=nodes_on_bdry)

    return uniform_discr_frompartition(new_part, exponent=discr.exponent,
                                       interp=discr.interp, impl=discr.impl,
                                       **kwargs)


def _scaling_func_list(bdry_fracs, exponent):
    """Return a list of lists of scaling functions for the boundary."""
    def scaling(factor):
        def scaling_func(x):
            return x * factor
        return scaling_func

    func_list = []
    for frac_l, frac_r in bdry_fracs:
        func_list_entry = []
        if np.isclose(frac_l, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_l ** (1 / exponent)))

        if np.isclose(frac_r, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_r ** (1 / exponent)))

        func_list.append(func_list_entry)
    return func_list


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
