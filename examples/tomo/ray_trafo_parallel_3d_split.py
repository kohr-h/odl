"""Example using the ray transform with 3d parallel beam geometry."""

import numpy as np
import odl
from odl.discr import DiscreteLp
from odl.operator import Operator
from odl.util import writable_array

space_3d = odl.uniform_discr(
    [-20, -20, -20], [20, 20, 20], (200, 256, 200), dtype='float32'
)
num_slcs = space_3d.shape[1]
space_2d = odl.uniform_discr(
    [-20, -20], [20, 20], (200, 200), dtype='float32'
)

angles = np.linspace(0, np.pi, 180, endpoint=False)
apart = odl.nonuniform_partition(angles)
dpart_1d = odl.uniform_partition(-30, 30, 512)
geom_2d = odl.tomo.Parallel2dGeometry(apart, dpart_1d)
ray_trafo_2d = odl.tomo.RayTransform(space_2d, geom_2d)

x = odl.phantom.shepp_logan(space_3d, modified=True)
pspace_op = odl.DiagonalOperator(ray_trafo_2d, num_slcs)


class SplitOp(Operator):
    def __init__(self, space_3d, space_2d):
        assert isinstance(space_3d, DiscreteLp)
        assert isinstance(space_2d, DiscreteLp)
        assert space_3d.ndim == 3
        assert space_2d.ndim == 2
        assert space_2d.shape == (space_3d.shape[0], space_3d.shape[2])
        nsplit = space_3d.shape[1]
        range = odl.ProductSpace(space_2d, nsplit)
        super().__init__(domain=space_3d, range=range, linear=True)

    def _call(self, x, out):
        for i in range(len(out)):
            out[i] = x.asarray()[:, i, :]

    @property
    def adjoint(self):
        return (
            (1 / self.domain.cell_sides[1])
            * MergeOp(self.range[0], self.domain)
        )


class MergeOp(Operator):
    def __init__(self, space_2d, space_3d):
        assert isinstance(space_2d, DiscreteLp)
        assert isinstance(space_3d, DiscreteLp)
        assert space_2d.ndim == 2
        assert space_3d.ndim == 3
        assert space_2d.shape == (space_3d.shape[0], space_3d.shape[2])
        nsplit = space_3d.shape[1]
        domain = odl.ProductSpace(space_2d, nsplit)
        super().__init__(domain=domain, range=space_3d, linear=True)

    def _call(self, px, out):
        with writable_array(out) as out_arr:
            for i in range(len(px)):
                out_arr[:, i, :] = px[i]

    @property
    def adjoint(self):
        return self.range.cell_sides[1] * SplitOp(self.range, self.domain[0])


dom_split_op = SplitOp(space_3d, space_2d)
proj_op = pspace_op * dom_split_op

data = proj_op(x)
backproj = proj_op.adjoint(data)

x.show(coords=[None, None, 0], title='Phantom, Middle Z Slice')
backproj.show(coords=[None, None, 0], title='Back-projection, Middle Z Slice')
data[128].show(title='Sinogram, Middle Slice', force_show=True)
backproj.show(coords=[None, 0, None])
