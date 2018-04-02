# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for phantom generation."""

import numpy as np

from odl.util import many_matvec, many_matmul


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
    dist = (np.abs(np.tensordot(p - c, n, axes=[1, 1])) -
            np.linalg.norm(e * alpha, axis=1))
    dist = dist.ravel()
    isect = (dist <= 0)

    # TODO: check if this is always the closest or if it depends
    x = -e ** 2 * alpha / np.linalg.norm(e * alpha, axis=1, keepdims=True)
    z = c + many_matvec(U, x)
    y = z - n * np.tensordot(n, z - p, axes=[1, 1])

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

    # --- Re-normalize and transpose the U matrices --- #

    U1T = U1T / np.linalg.norm(U1T, axis=1, keepdims=True)
    U1T = U1T[None, :, :]
    U1 = np.transpose(U1T, (0, 2, 1))
    U2T = U2T / np.linalg.norm(U2T, axis=2, keepdims=True)
    U2 = np.transpose(U2T, (0, 2, 1))

    # Add empty axes for the vectors to match the "many" type arrays of
    # ellipsoid 2
    c1 = c1[None, :]
    e1 = e1[None, :]

    # --- Compute level set representation parts --- #

    A1 = many_matmul(U1 / e1[:, None, :] ** 2, U1T)
    b1 = -many_matvec(A1, c1)
    A2 = many_matmul(U2 / e2[:, None, :] ** 2, U2T)
    b2 = -many_matvec(A2, c2)

    gamma1 = 1 / np.linalg.norm(A1, float('inf'), axis=(1, 2))
    gamma2 = 1 / np.linalg.norm(A2, float('inf'), axis=(1, 2))

    # --- Define helpers --- #

    def angle(u, v):
        """Return the angle between u and v (many vectors)."""
        norm_u = np.linalg.norm(u, axis=1)
        norm_v = np.linalg.norm(v, axis=1)
        nonzero = (norm_u > 1e-10) & (norm_v > 1e-10)
        ang = np.zeros(u.shape[0])
        inner = np.zeros_like(ang)
        inner[nonzero] = (
            np.tensordot(u[nonzero], v[nonzero], axes=[1, 1]) /
            (norm_u[nonzero] * norm_v[nonzero])
        ).ravel()
        ang[nonzero] = np.arccos(np.clip(inner, -1, 1))
        return ang

    def find_t_in_0_1(u, v):
        r"""Helper to find a t value for the iteration.

        This function solves the quadratic equation

        .. math::
            \|v\|^2\, t^2 + 2 u^{\mathrm{T}}v\, t + (\|u\|^2 - 1) = 0

        for :math:`t \in [0, 1]`. Due to the way the vectors :math:`u`
        and :math:`v` are defined, such a solution always exists.

        The two possible solutions are

        .. math::
            t_\pm = \frac{
                \pm \sqrt{(u^{\mathrm{T}}v)^2 - \|v\|^2(\|u\|^2 - 1)} -
                u^{\mathrm{T}}v
                }{
                    \|v\|^2
                }.
        """
        u_norm2 = np.linalg.norm(u, axis=1) ** 2
        v_norm2 = np.linalg.norm(v, axis=1) ** 2
        u_dot_v = np.tensordot(u, v, axes=[1, 1]).ravel()
        sqrt = np.sqrt(u_dot_v ** 2 - v_norm2 * (u_norm2 - 1))
        tplus = (sqrt - u_dot_v) / v_norm2
        tminus = (-sqrt - u_dot_v) / v_norm2
        # Make sure that not finding a root in [0, 1] breaks subsequent code
        t = np.full_like(tplus, fill_value=float('nan'))
        plus = (tplus >= 0) & (tplus <= 1)
        minus = (tminus >= 0) & (tminus <= 1)
        t[plus] = tplus[plus]
        t[minus] = tminus[minus]
        return t

    # --- Perform the iteration --- #

    isect = np.zeros(c2.shape[0], dtype=bool)  # which ones intersect
    converged = np.zeros(c2.shape[0], dtype=bool)  # done iterating
    cent1, cent2 = c1, c2  # for backup

    for _ in range(max_iter):
        # Compute the parameter values for which the segment connecting the
        # current centers intersects the ellipsoids
        u1 = many_matvec(U1T, c1 - cent1) / e1
        v1 = many_matvec(U1T, c2 - c1) / e1
        t1 = find_t_in_0_1(u1, v1)
        u2 = many_matvec(U2T, c1 - cent2) / e2
        v2 = many_matvec(U2T, c2 - c1) / e2
        t2 = find_t_in_0_1(u2, v2)

        isect[t2 <= t1] = True
        if np.all(isect):
            break

        # New intersection points with the ellipsoid boundaries
        z1 = c1 + t1 * (c2 - c1)
        z2 = c1 + t2 * (c2 - c1)

        # Compute angles with surface normals to test convergence
        normal1 = many_matvec(A1, z1) + b1
        normal2 = many_matvec(A2, z2) + b2
        theta1 = angle(z2 - z1, normal1)
        theta2 = angle(z1 - z2, normal2)

        converged = (theta1 <= eps) & (theta2 <= eps)
        if np.all(converged):
            break

        # Compute new centers
        c1 = z1 - gamma1 * normal1
        c2 = z2 - gamma2 * normal2

    # Choose point from ellipsoid 1 for intersecting ones
    z2[isect] = z1[isect]

    if squeeze:
        z1 = z1.squeeze(axis=0)
        z2 = z2.squeeze(axis=0)

    return z1, z2


def closest_points_ellipsoids_bbox(ell1_center, ell1_vectors, ell1_halfaxes,
                                   ell2_center, ell2_vectors, ell2_halfaxes):
    """Return the closest points of ellipsoids using a bounding box for one.

    This function is a "quick and dirty" variant of
    `closest_points_ellipsoids` that computes the distance of the first
    elliposoid to the bounding box of the second ellipsoid, making use of
    `closes_points_ellipsoid_plane`. It can be used either as a means to
    compute good start values for `closest_points_ellipsoids`, or even as
    replacement in cases where convergence can be expected to be slow.

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
    ell1_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
        Point in the first ellipsoid that is closest to the second
        ellipsoid's bounding box.
    ell2_closest_point : `numpy.ndarray`, shape ``(d,)`` or ``(N, d)``
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
        n_j &= u_j,\ p_j^\pm = c \pm e_j u_j.

    These hyperplanes are the tangent planes at the intersections of the
    ellipsoid boundary with the symmetry axes.

    A point :math:`x \in H_j^\pm` is part of the bounding box if

    .. math::
        (x - p_j^\pm)^{\mathrm{T}} u_i \in [-e_i, e_i] \quad
        \text{for all } i \neq j.
    """
