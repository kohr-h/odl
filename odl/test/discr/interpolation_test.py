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

"""Unit tests for ``odl.discr.interoplation``."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.interpolation import remap_points, indices_weights
from odl.util.testutils import (
    all_almost_equal, all_equal, almost_equal,
    simple_fixture)


# --- pytest fixtures --- #


pad_mode = simple_fixture(
    name='pad_mode',
    params=['constant', 'periodic', 'symmetric', 'order0', 'order1'])
ndim = simple_fixture(name='ndim', params=[1, 2])
points_type = simple_fixture(name='points_type', params=['array', 'meshgrid'])


@pytest.fixture(scope='module')
def remap_setup(pad_mode, ndim, points_type):
    """Fixture for original and remapped points, partition and pad mode."""
    if ndim == 1:
        min_pt = 1.0
        max_pt = 2.0
        shape = (5,)
        partition = odl.uniform_partition(min_pt, max_pt, shape)

        # Generate points in the order of periodic repetitions of the
        # interval [min_pt, max_pt], from left to right.
        # This is relevant for periodic and symmetric padding.
        points = ([-1.4] +                      # 3rd period left
                  [-1.0, -0.6, -0.01] +         # 2nd
                  [0.0, 0.1, 0.35, 0.8, 1.0] +  # 1st
                  [1.01, 1.35, 1.8, 1.99] +     # main period
                  [2.0, 2.4, 2.75, 2.95] +      # 1st right
                  [3.0, 3.6, 3.99] +            # 2nd
                  [4.7])                        # 3rd

        points = np.array(points)
        outside_left = (np.arange(0, 9),)
        inside = (np.arange(9, 13),)
        outside_right = (np.arange(13, 21),)

        x0 = float(min_pt)
        xn = float(max_pt)
        gx0 = float(partition.grid.min_pt)
        gxn = float(partition.grid.max_pt)

        if pad_mode == 'constant':
            # Everything <= x0 is mapped to x0, everything >= xn to xn
            remapped = ([x0] +                       # 3rd period left
                        [x0] * 3 +                   # 2nd
                        [x0] * 5 +                   # 1st
                        [1.01, 1.35, 1.8, 1.99] +    # main period
                        [xn] * 4 +                   # 1st right
                        [xn] * 3 +                   # 2nd
                        [xn])                        # 3rd
        elif pad_mode == 'periodic':
            # Take values modulo 1 with offset 1
            remapped = ([1.6] +                       # 3rd period left
                        [1.0, 1.4, 1.99] +            # 2nd
                        [1.0, 1.1, 1.35, 1.8, 1.0] +  # 1st
                        [1.01, 1.35, 1.8, 1.99] +     # main period
                        [1.0, 1.4, 1.75, 1.95] +      # 1st right
                        [1.0, 1.6, 1.99] +            # 2nd
                        [1.7])                        # 3rd
        elif pad_mode == 'symmetric':
            # Take values modulo 2 with offset 1 and reflect at x=2
            remapped = ([1.4] +                       # 3rd period left
                        [1.0, 1.4, 1.99] +            # 2nd
                        [2.0, 1.9, 1.65, 1.2, 1.0] +  # 1st
                        [1.01, 1.35, 1.8, 1.99] +     # main period
                        [2.0, 1.6, 1.25, 1.05] +      # 1st right
                        [1.0, 1.6, 1.99] +            # 2nd
                        [1.3])                        # 3rd
        elif pad_mode in ('order0', 'order1'):
            # Everything <= x0 is mapped to gx0, everything >= xn to gxn
            remapped = ([gx0] +                       # 3rd period left
                        [gx0] * 3 +                   # 2nd
                        [gx0] * 5 +                   # 1st
                        [1.01, 1.35, 1.8, 1.99] +     # main period
                        [gxn] * 4 +                   # 1st right
                        [gxn] * 3 +                   # 2nd
                        [gxn])                        # 3rd
        else:
            assert False

        remapped = np.array(remapped)
        if points_type == 'array':
            return (points, partition, pad_mode, remapped, outside_left,
                    outside_right, inside)
        elif points_type == 'meshgrid':
            return ((points,), partition, pad_mode, (remapped,), outside_left,
                    outside_right, inside)
        else:
            assert False

    elif ndim == 2:
        min_pt = [0.0, 1.0]
        max_pt = [1.0, 4.0]
        shape = (4, 3)
        partition = odl.uniform_partition(min_pt, max_pt, shape)

        x0, y0 = min_pt
        xn, yn = max_pt
        gx0, gy0 = partition.grid.min_pt
        gxn, gyn = partition.grid.max_pt

        if points_type == 'array':
            points = np.array(
                [[-2.1, -1.0, -0.4, 0.0, 0.2, 0.6, 1.0, 1.3, 2.0, 2.7],
                 [-3.4, -2.0, 0.4, 1.0, 2.9, 4.0, 4.4, 5.0, 6.9, 8.2]])
            outside_left = (np.arange(0, 4), np.arange(0, 4))
            inside = (np.arange(4, 6), np.arange(4, 5))
            outside_right = (np.arange(6, 10), np.arange(5, 10))

            if pad_mode == 'constant':
                remapped = np.array(
                    [[x0, x0, x0, 0.0, 0.2, 0.6, 1.0, xn, xn, xn],
                     [y0, y0, y0, 1.0, 2.9, 4.0, yn, yn, yn, yn]])
            elif pad_mode == 'periodic':
                remapped = np.array(
                    [[0.9, 0.0, 0.6, 0.0, 0.2, 0.6, 0.0, 0.3, 0.0, 0.7],
                     [2.6, 1.0, 3.4, 1.0, 2.9, 1.0, 1.4, 2.0, 3.9, 2.2]])
            elif pad_mode == 'symmetric':
                remapped = np.array(
                    [[0.1, 1.0, 0.4, 0.0, 0.2, 0.6, 1.0, 0.7, 0.0, 0.7],
                     [2.6, 4.0, 1.6, 1.0, 2.9, 4.0, 3.6, 3.0, 1.1, 2.2]])
            elif pad_mode in ('order0', 'order1'):
                remapped = np.array(
                    [[gx0, gx0, gx0, gx0, 0.2, 0.6, gxn, gxn, gxn, gxn],
                     [gy0, gy0, gy0, gy0, 2.9, gyn, gyn, gyn, gyn, gyn]])
            else:
                assert False

        elif points_type == 'meshgrid':
            points = sparse_meshgrid(
                np.array([-0.6, 0.0, 0.7, 1.0, 2.8]),
                np.array([-4.2, 0.5, 1.7, 4.0, 5.8]))
            outside_left = (np.arange(0, 2), np.arange(0, 2))
            inside = (np.arange(2, 3), np.arange(2, 3))
            outside_right = (np.arange(3, 5), np.arange(3, 5))

            if pad_mode == 'constant':
                remapped = (np.array([x0, 0.0, 0.7, 1.0, xn]),
                            np.array([y0, y0, 1.7, 4.0, yn]))
            elif pad_mode == 'periodic':
                remapped = (np.array([0.4, 0.0, 0.7, 0.0, 0.8]),
                            np.array([1.8, 3.5, 1.7, 1.0, 2.8]))
            elif pad_mode == 'symmetric':
                remapped = (np.array([0.6, 0.0, 0.7, 1.0, 0.8]),
                            np.array([1.8, 1.5, 1.7, 4.0, 2.2]))
            elif pad_mode in ('order0', 'order1'):
                remapped = (np.array([gx0, gx0, 0.7, gxn, gxn]),
                            np.array([gy0, gy0, 1.7, gyn, gyn]))
            else:
                assert False
        else:
            assert False  # points_type
        return (points, partition, pad_mode, remapped, outside_left,
                outside_right, inside)
    else:
        assert False  # ndim


def test_remap_points(remap_setup):
    """Check remapped points and indices against truth from fixture."""

    pts, part, pad_mode, true_pts, true_il, true_ir, true_iin = remap_setup
    new_pts, il, ir, iin = remap_points(pts, part, pad_mode)

    if isinstance(pts, tuple):
        # Meshgrid, compare each array
        for p, tp in zip(new_pts, true_pts):
            assert np.allclose(p, tp)
    else:
        # Array, compare at once
        assert np.allclose(new_pts, true_pts)

    for i in range(part.ndim):
        assert np.array_equal(il[i], true_il[i])
        assert np.array_equal(ir[i], true_ir[i])
        assert np.array_equal(iin[i], true_iin[i])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
