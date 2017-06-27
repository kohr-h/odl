# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the Ray transform."""

from __future__ import division
import numpy as np
from pkg_resources import parse_version
import pytest

import odl
from odl.tomo.backends import ASTRA_VERSION
from odl.tomo.util.testutils import (skip_if_no_astra, skip_if_no_astra_cuda,
                                     skip_if_no_skimage)
from odl.util.testutils import almost_equal, all_almost_equal, simple_fixture


# --- pytest fixtures --- #


impl_params = [skip_if_no_astra('astra_cpu'),
               skip_if_no_astra_cuda('astra_cuda'),
               skip_if_no_skimage('skimage')]
impl = simple_fixture('impl', impl_params, fmt=" {name} = '{value.args[1]}' ")

geometry_params = ['par2d', 'par3d', 'cone2d', 'cone3d', 'helical']
geometry_ids = [' geometry = {} '.format(p) for p in geometry_params]


@pytest.fixture(scope='module', ids=geometry_ids, params=geometry_params)
def geometry(request):
    geom = request.param
    m = 100
    n_angles = 100

    if geom == 'par2d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'par3d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -30], [30, 30], (m, m))
        return odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone2d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.FanFlatGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
    elif geom == 'cone3d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-60, -60], [60, 60], (m, m))
        return odl.tomo.CircularConeFlatGeometry(apart, dpart, src_radius=200,
                                                 det_radius=100)
    elif geom == 'helical':
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], (m, m))
        return odl.tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                                src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')


# Find the valid projectors
projectors = [skip_if_no_astra('par2d astra_cpu uniform'),
              skip_if_no_astra('par2d astra_cpu nonuniform'),
              skip_if_no_astra('par2d astra_cpu random'),
              skip_if_no_astra('cone2d astra_cpu uniform'),
              skip_if_no_astra('cone2d astra_cpu nonuniform'),
              skip_if_no_astra('cone2d astra_cpu random'),
              skip_if_no_astra_cuda('par2d astra_cuda uniform'),
              skip_if_no_astra_cuda('par2d astra_cuda half_uniform'),
              skip_if_no_astra_cuda('par2d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('par2d astra_cuda random'),
              skip_if_no_astra_cuda('cone2d astra_cuda uniform'),
              skip_if_no_astra_cuda('cone2d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('cone2d astra_cuda random'),
              skip_if_no_astra_cuda('par3d astra_cuda uniform'),
              skip_if_no_astra_cuda('par3d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('par3d astra_cuda random'),
              skip_if_no_astra_cuda('cone3d astra_cuda uniform'),
              skip_if_no_astra_cuda('cone3d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('cone3d astra_cuda random'),
              skip_if_no_astra_cuda('helical astra_cuda uniform'),
              skip_if_no_skimage('par2d skimage uniform'),
              skip_if_no_skimage('par2d skimage half_uniform')]


projector_ids = [' geom={}, impl={}, angles={} '
                 ''.format(*p.args[1].split()) for p in projectors]


@pytest.fixture(scope='module', params=projectors, ids=projector_ids)
def projector(request):
    n = 100
    m = 100
    n_angles = 100
    dtype = 'float32'

    geom, impl, angle = request.param.split()

    if angle == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angle == 'half_uniform':
        apart = odl.uniform_partition(0, np.pi, n_angles)
    elif angle == 'random':
        # Linearly spaced with random noise
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt, max_pt, n_angles)
        points += np.random.rand(n_angles) * (max_pt - min_pt) / (5 * n_angles)
        apart = odl.nonuniform_partition(points)
    elif angle == 'nonuniform':
        # Angles spaced quadratically
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt ** 0.5, max_pt ** 0.5, n_angles) ** 2
        apart = odl.nonuniform_partition(points)
    else:
        raise ValueError('angle not valid')

    if geom == 'par2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.tomo.Parallel2dGeometry(apart, dpart)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'par3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [m] * 2)
        geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.tomo.FanFlatGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition([-60] * 2, [60] * 2, [m] * 2)
        geom = odl.tomo.CircularConeFlatGeometry(apart, dpart, src_radius=200,
                                                 det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'helical':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [n] * 3, dtype=dtype)
        # Geometry, overwriting angle partition
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [m] * 2)
        geom = odl.tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                                src_radius=200, det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)
    else:
        raise ValueError('geom not valid')


@pytest.fixture(scope='module',
                params=[True, False],
                ids=[' in-place ', ' out-of-place '])
def in_place(request):
    return request.param


# --- RayTransform tests --- #


def test_projector(projector, in_place):
    """Test Ray transform forward projection."""

    # TODO: this needs to be improved
    # Accept 10% errors
    places = 1

    # Create Shepp-Logan phantom
    vol = projector.domain.one()

    # Calculate projection
    if in_place:
        proj = projector.range.zero()
        projector(vol, out=proj)
    else:
        proj = projector(vol)

    # We expect maximum value to be along diagonal
    expected_max = projector.domain.partition.extent[0] * np.sqrt(2)
    assert almost_equal(proj.ufuncs.max(), expected_max, places=places)


def test_adjoint(projector):
    """Test Ray transform backward projection."""
    # Relative tolerance, still rather high due to imperfectly matched
    # adjoint in the cone beam case
    if (parse_version(ASTRA_VERSION) < parse_version('1.8rc1') and
            isinstance(projector.geometry, odl.tomo.HelicalConeFlatGeometry)):
        rtol = 0.1
    else:
        rtol = 0.05

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    backproj = projector.adjoint(proj)

    # Verified the identity <Ax, Ax> = <A^* A x, x>
    result_AxAx = proj.inner(proj)
    result_xAtAx = backproj.inner(vol)
    assert result_AxAx == pytest.approx(result_xAtAx, rel=rtol)


def test_adjoint_of_adjoint(projector):
    """Test Ray transform adjoint of adjoint."""

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    proj_adj_adj = projector.adjoint.adjoint(vol)

    # Verify A(x) == (A^*)^*(x)
    assert all_almost_equal(proj, proj_adj_adj)

    # Calculate adjoints
    proj_adj = projector.adjoint(proj)
    proj_adj_adj_adj = projector.adjoint.adjoint.adjoint(proj)

    # Verify A^*(y) == ((A^*)^*)^*(x)
    assert all_almost_equal(proj_adj, proj_adj_adj_adj)


def test_angles(projector):
    """Test Ray transform angle conventions."""

    # Smoothed line/hyperplane with offset
    vol = projector.domain.element(
        lambda x: np.exp(-(2 * x[0] - 10 + x[1]) ** 2))

    # Create projection
    result = projector(vol).asarray()

    # Find the angle where the projection has a maximum (along the line).
    axes = 1 if projector.domain.ndim == 2 else (1, 2)
    ind_angle = np.argmax(np.max(result, axis=axes))
    # Restrict to [0, 2 * pi) for helical
    maximum_angle = np.fmod(projector.geometry.angles[ind_angle], 2 * np.pi)

    # Verify correct maximum angle. The line is defined by the equation
    # x1 = 10 - 2 * x0, i.e. the slope -2. Thus the angle arctan(1/2) should
    # give the maximum projection values.
    expected = np.arctan2(1, 2)
    assert np.fmod(maximum_angle, np.pi) == pytest.approx(expected, abs=0.1)

    # Find the pixel where the projection has a maximum at that angle
    axes = () if projector.domain.ndim == 2 else 1
    ind_pixel = np.argmax(np.max(result[ind_angle], axis=axes))
    max_pixel = projector.geometry.det_partition[ind_pixel, ...].mid_pt[0]

    # The line is at distance 2 * sqrt(5) from the origin, which translates
    # to the same distance from the detector midpoint, with positive sign
    # if the angle is smaller than pi and negative sign otherwise.
    expected = 2 * np.sqrt(5) if maximum_angle < np.pi else -2 * np.sqrt(5)
    # This is a bit hard to check strictly, so we mostly test for the
    # correct side
    assert max_pixel == pytest.approx(expected, abs=abs(max_pixel))


def test_complex(impl):
    """Test transform of complex input for parallel 2d geometry."""
    space_c = odl.uniform_discr([-1, -1], [1, 1], (10, 10), dtype='complex64')
    space_r = space_c.real_space
    geom = odl.tomo.parallel_beam_geometry(space_c)
    ray_trafo_c = odl.tomo.RayTransform(space_c, geom, impl=impl)
    ray_trafo_r = odl.tomo.RayTransform(space_r, geom, impl=impl)
    vol = odl.phantom.shepp_logan(space_c)
    vol.imag = odl.phantom.cuboid(space_r)

    data = ray_trafo_c(vol)
    true_data_re = ray_trafo_r(vol.real)
    true_data_im = ray_trafo_r(vol.imag)

    assert all_almost_equal(data.real, true_data_re)
    assert all_almost_equal(data.imag, true_data_im)


def test_anisotropic_voxels(geometry):
    """Test projection and backprojection with anisotropic voxels."""
    ndim = geometry.ndim
    shape = [10] * (ndim - 1) + [5]
    space = odl.uniform_discr([-1] * ndim, [1] * ndim, shape=shape,
                              dtype='float32')

    # If no implementation is available, skip
    if ndim == 2 and not odl.tomo.ASTRA_AVAILABLE:
        pytest.skip(msg='ASTRA not available, skipping 2d test')
    elif ndim == 3 and not odl.tomo.ASTRA_CUDA_AVAILABLE:
        pytest.skip(msg='ASTRA_CUDA not available, skipping 3d test')

    ray_trafo = odl.tomo.RayTransform(space, geometry)
    vol_one = ray_trafo.domain.one()
    data_one = ray_trafo.range.one()

    if ndim == 2:
        # Should raise
        with pytest.raises(NotImplementedError):
            ray_trafo(vol_one)
        with pytest.raises(NotImplementedError):
            ray_trafo.adjoint(data_one)
    elif ndim == 3:
        # Just check that this doesn't crash and computes something nonzero
        data = ray_trafo(vol_one)
        backproj = ray_trafo.adjoint(data_one)
        assert data.norm() > 0
        assert backproj.norm() > 0
    else:
        assert False


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
