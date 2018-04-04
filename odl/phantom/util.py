# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for phantom generation."""

import numpy as np

from odl.util import many_dot, many_matvec, many_matmul


def closest_points_ellipsoid_plane(ell_center, ell_vectors, ell_halfaxes,
                                   plane_point, plane_normal):
    r"""Return the closest points of an ellipsoid and a hyperplane.

    Parameters
    ----------
    ell_center : array-like
        Center point of the ``d``-dimensional ellipsoid, of shape ``(d,)``.
    ell_vectors : array-like
        Symmetry axes of the ellipsoid, arranged as rows of a matrix with
        shape ``(d, d)``. The vectors should be mutually orthogonal.
    ell_halfaxes : array-like
        Lengths of the ellipsoid half-axes. Must have shape ``(d,)``, and
        all entries must be positive.
    plane_point : array-like
        An arbitrary point in the plane, of shape ``(d,)``. To use multiple
        planes at once, an array of shape ``(N, d)`` can be given.
    plane_normal : array-like
        Normal vector of the plane, must be of shape ``(d,)`` and not
        equal to the zero vector. For multiple planes, the shape should
        be ``(N, d)``, the same as for ``plane_point``.

    Returns
    -------
    ell_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the ellipsoid that is closest to the hyperplane.
    plane_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the hyperplane that is closest to the ellipsoid.

    Notes
    -----
    A solid ellipsoid is defined as

    .. math::
        E = \Big\{
            c + \sum_{j=1}^d x_j\, u_j\ \Big|\ \|e^{-1} \odot x \| \leq 1
            \Big\}

    with center :math:`c`, orthogonal unit vectors :math:`u_1, \dots, u_d`
    defining the symmetry axes, and a vector :math:`e > 0` of half-axis
    lengths. The operation ":math:`e^{-1} \odot`" denotes elementwise
    multiplication with the reciprocal entries of :math:`e`.
    We write ellipsoid points in short as

    .. math::
        z = c + Ux,\quad
        U =
        \begin{pmatrix}
            u_1 & \cdots & u_d
        \end{pmatrix}

    A hyperplane is defined as

    .. math::
        H = \big\{
            x\ \big|\ (x - p)^{\mathrm{T}} n = 0
        \big\}

    with a point :math:`p` and a normal vector :math:`n` of unit length.

    The distance between :math:`E` and :math:`H` is given as

    .. math::
        \mathrm{d}(E, H) = \max\Big\{
            \big| (p - c)^{\mathrm{T}} n \big| - \|e \odot \alpha\|,
            \ 0
        \Big\},

    where the vector :math:`\alpha` is defined as projection

    .. math::
        \alpha = U^{\mathrm{T}} n,

    of the unit vectors onto the normal vector of the plane.

    If :math:`\mathrm{d}(E, H) > 0`, the closest point in the ellipsoid
    is :math:`z = c + Ux` with coordinates

    .. math::
        x = -\frac{e^2 \odot \alpha}{\|e \odot \alpha\|}.

    (A second solution with positive sign yields the farthest point.)

    The corresponding closest point in the hyperplane is the projection

    .. math::
        y = z - \big(n^{\mathrm{T}}(z - p)\big) n.

    The function returns the points :math:`z` and :math:`y` in this case.

    If ellipsoid and hyperplane have a non-empty intersection, the above
    defined point :math:`y` is a common point, comprising both return
    values of the function in this case.
    """
    c = np.asarray(ell_center, dtype=float)
    U = np.asarray(ell_vectors, dtype=float).T
    e = np.asarray(ell_halfaxes, dtype=float)
    p = np.asarray(plane_point, dtype=float)
    n = np.asarray(plane_normal, dtype=float)

    # --- Check inputs --- #

    squeeze = False
    if c.ndim != 1:
        raise ValueError('`ell_center` must be 1-dimensional, but '
                         '`ell_center.ndim == {}`'.format(c.ndim))
    d = c.size

    if U.shape != (d, d):
        raise ValueError('`ell_vectors` must have shape `(d, d)`, but '
                         '`d == {}` and `ell_vectors.shape == {}`'
                         ''.format(d, U.shape))
    if e.shape != (d,):
        raise ValueError('`ell_halfaxes` must have shape `(d,)`, but '
                         '`d == {}` and `ell_halfaxes.shape == {}`'
                         ''.format(d, e.shape))
    if p.ndim == 2 and p.shape[1] == d:
        N = p.shape[0]
        if n.shape != (N, d):
            raise ValueError('`plane_normal` must have shape `(N, d)`, but '
                             '`(N, d) == {}` and `plane_normal.shape == {}`'
                             ''.format((N, d), n.shape))
    elif p.shape == (d,):
        if n.shape != (d,):
            raise ValueError('`plane_normal` must have shape `(d,)`, but '
                             '`d == {}` and `plane_normal.shape == {}`'
                             ''.format(d, n.shape))
        p = p[None, :]
        n = n[None, :]
        squeeze = True
    else:
        raise ValueError('`plane_point` must have shape `(d,)` or `(N, d)`, '
                         'but `d == {}` and `plane_point.shape == {}`'
                         ''.format(d, p.shape))

    if not np.all(e > 0):
        raise ValueError('`ell_halfaxes` must be all positive, got '
                         '{}'.format(e))
    if np.linalg.norm(n) == 0:
        raise ValueError('`plane_normal` may not be the zero vector')

    # --- Re-normalize normal vectors --- #

    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)

    # Add extra axes to match the "many" type other arrays
    c = c[None, :]
    U = U[None, :]
    e = e[None, :]

    UT = np.transpose(U, axes=(0, 2, 1))

    # --- Compute the points --- #

    alpha = many_matvec(UT, n)
    dist = np.abs(many_dot(p - c, n)) - np.linalg.norm(e * alpha, axis=1)
    dist = dist.ravel()
    isect = (dist <= 0)

    # TODO: check if this is always the closest or if it depends
    x = -e ** 2 * alpha / np.linalg.norm(e * alpha, axis=1, keepdims=True)
    z = c + many_matvec(U, x)
    y = z - n * many_dot(n, z - p)

    z[isect] = y[isect]
    if squeeze:
        z = z.reshape((d,))
        y = y.reshape((d,))
    return z, y


def closest_points_ellipsoids(ell1_center, ell1_vectors, ell1_halfaxes,
                              ell2_center, ell2_vectors, ell2_halfaxes,
                              angle_tol=1e-4, max_iter=50):
    r"""Return the closest points of two ellipsoids.

    Parameters
    ----------
    ell1_center : array-like
        Center point of the first ``d``-dimensional ellipsoid, of shape
        ``(d,)``.
    ell1_vectors : array-like
        Symmetry axes of the first ellipsoid, arranged as rows of a matrix
        with shape ``(d, d)``. The vectors should be mutually orthogonal.
    ell1_halfaxes : array-like
        Lengths of the half-axes of the first ellipsoid. Must have shape
        ``(d,)``, and all entries must be positive.
    ell2_center : array-like
        Center point of the second ellipsoid, of shape ``(d,)``. To use
        multiple ellipsoids at once, an array of shape ``(N, d)`` can be
        given.
    ell2_vectors : array-like
        Symmetry axes of the second ellipsoid, arranged as rows of a matrix
        with shape ``(d, d)`` or ``(N, d, d)`` with the same ``N`` as for
        ``ell_center``. Each set of vectors, i.e., each ``(d, d)`` matrix,
        should be mutually orthogonal.
    ell2_halfaxes : array-like
        Lengths of the half-axes of the second ellipsoid. Must have shape
        ``(d,)`` or ``(N, d)`` with ``N`` as for ``ell2_center``, and all
        entries must be positive.
    angle_tol : positive float, optional
        Tolerance for angles that determines convergence in the iteration.
        See Notes for details.
    max_iter : positive int, optional
        Maximum number of iterations that should be run.

    Returns
    -------
    ell1_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the first ellipsoid that is closest to the second ellipsoid.
    ell2_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the second ellipsoid that is closest to the first ellipsoid.

    Notes
    -----
    A solid ellipsoid is defined as

    .. math::
        E = \Big\{
            c + \sum_{j=1}^d x_j\, u_j\ \Big|\ \|e^{-1} \odot x \| \leq 1
            \Big\}

    with center :math:`c`, orthogonal unit vectors :math:`u_1, \dots, u_d`
    defining the symmetry axes, and a vector :math:`e > 0` of half-axis
    lengths. The operation ":math:`e^{-1} \odot`" denotes elementwise
    multiplication with the reciprocal entries of :math:`e`.
    We write ellipsoid points in short as

    .. math::
        z = c + Ux,\quad
        U =
        \begin{pmatrix}
            u_1 & \cdots & u_d
        \end{pmatrix}

    An alternative notation is to write an ellipsis as the sublevel set of
    a quadratic function,

    .. math::
        E = \{z\,|\, q(z) \leq 0\}, \quad
        q(z) = \frac{1}{2} z^{\mathrm{T}} A z + b^{\mathrm{T}} z + \beta,

    with a symmetric and positive definite matrix :math:`A`.

    The former representation can be converted into the latter by choosing

    .. math::
        A &= U E^{-2} U^{\mathrm{T}}, \\
        b &= -A c, \\
        \beta & = \frac{1}{2} c^{\mathrm{T}} A c - \frac{1}{2},

    with :math:`E = \mathrm{diag}(e)`.

    To find the two points :math:`z_1 \in E_1,\ z_2 \in E_2` that are
    closest to each other, an iterative algorithm from
    `[LH2002] <https://doi.org/10.1137/S1052623401396510>`_ is applied,
    consisting of the following steps:

    - Start: Choose :math:`c_1, c_2` as the centers of the ellipsoids.
      Select a threshold :math:`\epsilon > 0`.
    - Step :math:`k \to k + 1` -- we write, e.g., :math:`x` for the old
      quantity and :math:`\bar x` for the new one.

      1. Find :math:`t_1, t_2 \in [0, 1]` such that
         :math:`\gamma(t) = c_1 + t (c_2 - c_1)` intersects the boundaries
         of :math:`E_1, E_2` at those parameter values.

         They can be calculated as

         .. math::
             t_1 &= \frac{1}{
                 \| e_1^{-1} \odot U_1^{\mathrm{T}}(c_2 - c_1) \|
             }, \\
             t_2 &= 1 - \frac{1}{
                 \| e_2^{-1} \odot U_2^{\mathrm{T}}(c_2 - c_1) \|
             }.

      2. If :math:`t_2 \leq t_1`, the distance is 0. Return the common point
         :math:`\gamma(t_1)` twice.

         Else, set :math:`\bar z_1 = \gamma(t_1),\ \bar z_2 = \gamma(t_2)`.

      3. Compute the angles

         .. math::
             \theta_1 &= \theta(\bar z_2 - \bar z_1, A_1 \bar z_1 + b_1), \\
             \theta_2 &= \theta(\bar z_1 - \bar z_2, A_2 \bar z_2 + b_2), \\
             \theta(u, v) &= \mathrm{arccos}\bigg(
                 \frac{u^{\mathrm{T}}v}{\|u\|\,\|v\|}
             \bigg) \in [0, \pi],

         between the line connecting the probe points and the tangent planes
         to the ellipsoids at those points.

         If :math:`\theta_1 \leq \epsilon` and :math:`\theta_2 \leq \epsilon`,
         terminate -- this signals convergence (:math:`\epsilon` corresponds
         to the ``angle_tol`` parameter).

      4. Compute the new centers

         .. math::
             \bar c_1 &= \bar z_1 - \gamma_1(A_1 \bar z_1 + b_1), \\
             \bar c_2 &= \bar z_2 - \gamma_2(A_2 \bar z_2 + b_2),

         with :math:`\gamma_1 = \|A_1\|^{-1},\ \gamma_2 = \|A_2\|^{-1}`,
         using the 1- or :math:`\infty`-matrix norm.

    This iteration produces points :math:`z_1^k,\ z_2^k` that converge to
    the mutually closest points, which are returned by this function.

    According to `[LH2002] <https://doi.org/10.1137/S1052623401396510>`_,
    convergence generally fast, except in the cases when the ellipsoids are
    far apart, or if an ellipsoid has a large elongation.

    References
    ----------
    [LH2002] Lin, A and Han, S-P.
    *On the Distance between Two Ellipsoids*.
    SIAM Journal on Optimization, 13-1 (2002), pp. 298–308.
    """
    c1 = np.asarray(ell1_center, dtype=float)
    U1T = np.asarray(ell1_vectors, dtype=float)
    e1 = np.asarray(ell1_halfaxes, dtype=float)
    c2 = np.asarray(ell2_center, dtype=float)
    U2T = np.asarray(ell2_vectors, dtype=float)
    e2 = np.asarray(ell2_halfaxes, dtype=float)
    eps = float(angle_tol)

    # --- Check inputs --- #

    squeeze = False
    if c1.ndim != 1:
        raise ValueError('`ell1_center` must be 1-dimensional, but '
                         '`ell1_center.ndim == {}`'.format(c1.ndim))
    d = c1.size

    if U1T.shape != (d, d):
        raise ValueError('`ell1_vectors` must have shape `(d, d)`, but '
                         '`d == {}` and `ell1_vectors.shape == {}`'
                         ''.format(d, U1T.shape))
    if e1.shape != (d,):
        raise ValueError('`ell1_halfaxes` must have shape `(d,)`, but '
                         '`d == {}` and `ell1_halfaxes.shape == {}`'
                         ''.format(d, e1.shape))

    if c2.ndim == 2 and c2.shape[1] == d:
        N = c2.shape[0]
        if U2T.shape != (N, d, d):
            raise ValueError('`ell2_vectors` must have shape `(N, d, d)`, but '
                             '`(N, d, d) == {}` and `ell2_vectors.shape == {}`'
                             ''.format((N, d, d), U2T.shape))
        if e2.shape != (N, d):
            raise ValueError('`ell2_halfaxes` must have shape `(N, d)`, but '
                             '`(N, d) == {}` and `ell2_halfaxes.shape == {}`'
                             ''.format((N, d), e2.shape))
    elif c2.shape == (d,):
        if U2T.shape != (d, d):
            raise ValueError('`ell2_vectors` must have shape `(d, d)`, but '
                             '`(d, d) == {}` and `ell2_vectors.shape == {}`'
                             ''.format((d, d), U2T.shape))
        if e2.shape != (d,):
            raise ValueError('`ell2_halfaxes` must have shape `(d,)`, but '
                             '`d == {}` and `ell2_halfaxes.shape == {}`'
                             ''.format(d, c2.shape))
        c2 = c2[None, :]
        U2T = U2T[None, :, :]
        e2 = e2[None, :]
        N = 1
        squeeze = True
    else:
        raise ValueError('`ell2_center` must have shape `(d,)` or `(N, d)`, '
                         'but `d == {}` and `ell2_center.shape == {}`'
                         ''.format(d, c2.shape))

    if not np.all(e1 > 0):
        raise ValueError('`ell1_halfaxes` must be all positive, got '
                         '{}'.format(e1))
    if not np.all(e2 > 0):
        raise ValueError('`ell2_halfaxes` must be all positive, got '
                         '{}'.format(e2))

    if eps <= 0:
        raise ValueError('`angle_tol` must be positive, got {}'.format(eps))

    # --- Blow up `1` arrays to size of the `2` arrays

    U1T = np.repeat(U1T[None, :, :], N, axis=0)
    e1 = np.repeat(e1[None, :], N, axis=0)
    c1 = np.repeat(c1[None, :], N, axis=0)

    # --- Re-normalize and transpose the U matrices --- #

    U1T = U1T / np.linalg.norm(U1T, axis=2, keepdims=True)
    U1 = np.transpose(U1T, (0, 2, 1))
    U2T = U2T / np.linalg.norm(U2T, axis=2, keepdims=True)
    U2 = np.transpose(U2T, (0, 2, 1))

    # --- Compute level set representation parts --- #

    A1 = many_matmul(U1 / e1[:, None, :] ** 2, U1T)
    b1 = -many_matvec(A1, c1)
    A2 = many_matmul(U2 / e2[:, None, :] ** 2, U2T)
    b2 = -many_matvec(A2, c2)

    # Reciprocal of the spectral radii of `A1` and `A2`
    gamma1 = np.min(e1, axis=1) ** 2
    gamma2 = np.min(e2, axis=1) ** 2

    # --- Define helpers --- #

    def angle(u, v):
        """Return the angle between u and v (many vectors)."""
        norm_u = np.linalg.norm(u, axis=1)
        norm_v = np.linalg.norm(v, axis=1)
        nonzero = (norm_u > 1e-10) & (norm_v > 1e-10)
        ang = np.zeros(u.shape[0])
        inner = np.zeros_like(ang)
        inner[nonzero] = (
            many_dot(u[nonzero], v[nonzero]) /
            (norm_u[nonzero] * norm_v[nonzero])
        ).ravel()
        ang[nonzero] = np.arccos(np.clip(inner[nonzero], -1, 1))
        return ang

    def find_t_param(u, v):
        r"""Helper to find a t value for the iteration.

        This function solves the quadratic equation

        .. math::
            \|v\|^2\, t^2 + 2 u^{\mathrm{T}}v\, t + (\|u\|^2 - 1) = 0

        for :math:`t`. The two possible solutions are

        .. math::
            t_\pm = \frac{
                \pm \sqrt{(u^{\mathrm{T}}v)^2 - \|v\|^2(\|u\|^2 - 1)} -
                u^{\mathrm{T}}v
                }{
                    \|v\|^2
                }.

        We prefer a solution in :math:`[0, 1]` if it exists, since it's
        the parameter where the segment between two ellipsoid centers
        intersect the ellipsoid boundary.
        """
        u_norm2 = np.linalg.norm(u, axis=1) ** 2
        v_norm2 = np.linalg.norm(v, axis=1) ** 2
        u_dot_v = many_dot(u, v)
        sqrt = np.sqrt(np.maximum(u_dot_v ** 2 - v_norm2 * (u_norm2 - 1), 0))
        tplus = (sqrt - u_dot_v) / v_norm2
        tminus = (-sqrt - u_dot_v) / v_norm2

        # One solution is always > 1, the other one is either in [0, 1]
        # or negative. We choose the one in [0, 1] if possible, otherwise
        # `NaN`
        t = np.full_like(tplus, fill_value=float('nan'))
        plus = (tplus >= 0) & (tplus <= 1)
        minus = (tminus >= 0) & (tminus <= 1)
        t[plus] = tplus[plus]
        t[minus] = tminus[minus]
        return t

    # --- Perform the iteration --- #

    isect = np.zeros(c2.shape[0], dtype=bool)  # which ones intersect
    cent1, cent2 = c1, c2  # for backup
    closest_ell1 = z1 = np.empty_like(c1)
    closest_ell2 = z2 = np.empty_like(c2)

    for i in range(max_iter):
        # Compute the parameter values for which the segment connecting the
        # current centers intersects the ellipsoids
        u1 = many_matvec(U1T, c1 - cent1) / e1
        v1 = many_matvec(U1T, c2 - c1) / e1
        t1 = find_t_param(u1, v1)
        u2 = many_matvec(U2T, c1 - cent2) / e2
        v2 = many_matvec(U2T, c2 - c1) / e2
        t2 = find_t_param(u2, v2)

        # Where one of the segments from `c1` to `c2` does not intersect both
        # ellipsoid boundaries, we have intersection
        invalid = np.isnan(t1) | np.isnan(t2)
        isect[invalid] = True
        # The same if the "foreign" boundary comes before the own boundary
        isect[~invalid][t2[~invalid] <= t1[~invalid]] = True
        # Choose point from ellipsoid 1 for intersecting ones
        closest_ell2[isect] = closest_ell1[isect]
        if np.all(isect):
            break

        # Compute new boundary points
        z1[~isect] = (c1 + t1[:, None] * (c2 - c1))[~isect]
        z2[~isect] = (c1 + t2[:, None] * (c2 - c1))[~isect]

        # Compute angles with surface normals to test convergence
        normal1 = many_matvec(A1, z1) + b1
        normal2 = many_matvec(A2, z2) + b2
        theta1 = angle(z2 - z1, normal1)
        theta2 = angle(z1 - z2, normal2)

        converged1 = (theta1 <= eps)
        converged2 = (theta2 <= eps)
        if np.all(converged1 & converged2):
            break

        # Compute new centers
        c1[~isect & ~converged1] = (
            z1 - gamma1[:, None] * normal1
        )[~isect & ~converged1]
        c2[~isect & ~converged2] = (
            z2 - gamma2[:, None] * normal2
        )[~isect & ~converged2]

    # Set non-intersecting parts of the final vectors
    closest_ell1[~isect] = z1[~isect]
    closest_ell2[~isect] = z2[~isect]

    if squeeze:
        closest_ell1 = closest_ell1.squeeze(axis=0)
        closest_ell2 = closest_ell2.squeeze(axis=0)

    return closest_ell1, closest_ell2


def closest_point_ellipsoids_bbox(ell1_center, ell1_vectors, ell1_halfaxes,
                                  ell2_center, ell2_vectors, ell2_halfaxes):
    r"""Return the closest point in an ellipsoid's bbox to another ellipsoid.

    This function is mainly intended to yield good start values for
    `closest_points_ellipsoids`.

    Parameters
    ----------
    ell1_center : array-like
        Center point of the first ``d``-dimensional ellipsoid, of shape
        ``(d,)``.
    ell1_vectors : array-like
        Symmetry axes of the first ellipsoid, arranged as rows of a matrix
        with shape ``(d, d)``. The vectors should be mutually orthogonal.
    ell1_halfaxes : array-like
        Lengths of the half-axes of the first ellipsoid. Must have shape
        ``(d,)``, and all entries must be positive.
    ell2_center : array-like
        Center point of the second ellipsoid, of shape ``(d,)``. To use
        multiple ellipsoids at once, an array of shape ``(N, d)`` can be
        given.
    ell2_vectors : array-like
        Symmetry axes of the second ellipsoid, arranged as rows of a matrix
        with shape ``(d, d)`` or ``(N, d, d)`` with the same ``N`` as for
        ``ell_center``. Each set of vectors, i.e., each ``(d, d)`` matrix,
        should be mutually orthogonal.
    ell2_halfaxes : array-like
        Lengths of the half-axes of the second ellipsoid. Must have shape
        ``(d,)`` or ``(N, d)`` with ``N`` as for ``ell2_center``, and all
        entries must be positive.

    Returns
    -------
    ell2_bbox_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the second ellipsoid's bounding box that is closest to the
        first ellipsoid.

    Notes
    -----
    A solid ellipsoid is defined as

    .. math::
        E = \Big\{
            c + \sum_{j=1}^d x_j\, u_j\ \Big|\ \|e^{-1} \odot x \| \leq 1
            \Big\}

    with center :math:`c`, orthogonal unit vectors :math:`u_1, \dots, u_d`
    defining the symmetry axes, and a vector :math:`e > 0` of half-axis
    lengths. The operation ":math:`e^{-1} \odot`" denotes elementwise
    multiplication with the reciprocal entries of :math:`e`.

    The bounding box of :math:`E` is defined by :math:`2d` hyperplanes

    .. math::
        H_j^\pm &= \big\{x\ \big|
            \ \big( x - p_j^\pm \big)^{\mathrm{T}}n_j = 0\}, \\
        n_j &= u_j,\quad p_j^\pm = c \pm e_j u_j.

    These hyperplanes are the tangent planes at the intersections of the
    ellipsoid boundary with the symmetry axes.

    A point :math:`x \in H_j^\pm` is part of the bounding box if

    .. math::
        (x - p_j^\pm)^{\mathrm{T}} u_i \in [-e_i, e_i] \quad
        \text{for all } i \neq j.

    Thus, the procedure to determine the closest points between an ellipsoid
    :math:`E_1` and the bounding box of a second ellipsoid :math:`E`
    (with vector notation as above for :math:`E`) are as follows:

    For each of the :math:`2d` hyperplanes :math:`H_j^\pm`:

    - Compute the closest points :math:`z_j^\pm, y` between :math:`E_1` and
      :math:`H_j^\pm` using `closest_points_ellipsoid_plane`.
    - Transform :math:`y` into local coordinates of :math:`H_j^\pm` via
      :math:`x = U^{\mathrm{T}}(y - p_j^\pm)`.
    - Clip :math:`x` to :math:`[-e, e]`.
    - Set :math:`y_j^\pm = c + Ux`.

    The pair of closest points is the :math:`(z_j^\pm, y_j^\pm)` pair
    with minimum distance.
    """
    c1 = np.asarray(ell1_center, dtype=float)
    c2 = np.asarray(ell2_center, dtype=float)
    U2T = np.asarray(ell2_vectors, dtype=float)
    e2 = np.asarray(ell2_halfaxes, dtype=float)

    # --- Check inputs --- #

    squeeze = False
    if c1.ndim != 1:
        raise ValueError('`ell1_center` must be 1-dimensional, but '
                         '`ell1_center.ndim == {}`'.format(c1.ndim))
    d = c1.size

    if c2.ndim == 2 and c2.shape[1] == d:
        N = c2.shape[0]
        if U2T.shape != (N, d, d):
            raise ValueError('`ell2_vectors` must have shape `(N, d, d)`, but '
                             '`(N, d, d) == {}` and `ell2_vectors.shape == {}`'
                             ''.format((N, d, d), U2T.shape))
        if e2.shape != (N, d):
            raise ValueError('`ell2_halfaxes` must have shape `(N, d)`, but '
                             '`(N, d) == {}` and `ell2_halfaxes.shape == {}`'
                             ''.format((N, d), e2.shape))
    elif c2.shape == (d,):
        if U2T.shape != (d, d):
            raise ValueError('`ell2_vectors` must have shape `(d, d)`, but '
                             '`(d, d) == {}` and `ell2_vectors.shape == {}`'
                             ''.format((d, d), U2T.shape))
        if e2.shape != (d,):
            raise ValueError('`ell2_halfaxes` must have shape `(d,)`, but '
                             '`d == {}` and `ell2_halfaxes.shape == {}`'
                             ''.format(d, c2.shape))
        c2 = c2[None, :]
        U2T = U2T[None, :, :]
        e2 = e2[None, :]
        squeeze = True
    else:
        raise ValueError('`ell2_center` must have shape `(d,)` or `(N, d)`, '
                         'but `d == {}` and `ell2_center.shape == {}`'
                         ''.format(d, c2.shape))

    if not np.all(e2 > 0):
        raise ValueError('`ell2_halfaxes` must be all positive, got '
                         '{}'.format(e2))

    # --- Re-normalize and transpose U2 --- #

    U2T = U2T / np.linalg.norm(U2T, axis=2, keepdims=True)
    U2 = np.transpose(U2T, (0, 2, 1))

    # --- Find the closest points --- #

    zmin = np.empty_like(c2)
    ymin = np.empty_like(c2)
    dmin = np.full(c2.shape[0], fill_value=float('inf'))
    for j in range(d):
        for sign in (-1, 1):
            u = U2[:, :, j]
            e = e2[:, j]

            # Point in plane
            p = c2 + sign * e[:, None] * u

            # Get closest points and clip coordinates of `y`
            z, y = closest_points_ellipsoid_plane(
                ell1_center, ell1_vectors, ell1_halfaxes, p, u)
            x = many_matvec(U2T, y - p)
            x = np.clip(x, -e2, e2)
            y = p + many_matvec(U2, x)

            # Take `y` and `z` if distance is smaller
            dist = np.linalg.norm(z - y, axis=1)
            update = (dist < dmin)
            dmin[update] = dist[update]
            zmin[update, :] = z[update, :]
            ymin[update, :] = y[update, :]

    if squeeze:
        zmin = zmin.squeeze(axis=0)
        ymin = ymin.squeeze(axis=0)

    return zmin, ymin
