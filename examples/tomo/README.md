# Tomography examples

These examples demonstrate the capability of ODL to perform tomographic projections, back-projections and filtered back-projection reconstruction in the various supported geometries. They also serve as copy-and-paste templates for the basic setup of more complex applications in tomography.

## List of examples

Example | Purpose | Complexity
------- | ------- | ----------
[`anisotropic_voxels.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/anisotropic_voxels.py) | Tomographic projection with non-cube voxels | middle
[`filtered_backprojection_cone_2d_partial_scan.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_cone_2d_partial_scan.py) | (Inexact) FBP reconstruction in 2D fan beam geometry with partial scan (less than 360°) | middle
[`filtered_backprojection_cone_2d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_cone_2d.py) | (Inexact) FBP reconstruction in 2D fan beam geometry | middle
[`filtered_backprojection_cone_3d_partial_scan.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_cone_3d_partial_scan.py) | (Inexact) FBP reconstruction in 3D circular cone beam geometry with partial scan (less than 360°) | middle
[`filtered_backprojection_cone_3d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_cone_3d.py) | (Inexact) FBP reconstruction in 3D circular cone beam geometry | middle
[`filtered_backprojection_helical_3d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_helical_3d.py) | (Inexact) FBP reconstruction in 3D helical cone beam geometry | middle
[`filtered_backprojection_parallel_2d_complex.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_parallel_2d_complex.py) | FBP reconstruction in 2D parallel beam geometry with complex-valued data | middle
[`filtered_backprojection_parallel_2d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_parallel_2d.py) | FBP reconstruction in 2D parallel beam geometry | middle
[`filtered_backprojection_parallel_3d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/filtered_backprojection_parallel_3d.py) | FBP reconstruction in 3D parallel beam single-axis geometry | middle
[`ray_trafo_circular_cone.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_circular_cone.py) | Projection and back-projection in 3D circular cone beam geometry | middle
[`ray_trafo_fanflat.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_fanflat.py) | Projection and back-projection in 2D fan beam geometry | middle
[`ray_trafo_helical.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_helical.py) | Projection and back-projection in 3D helical cone beam geometry | middle
[`ray_trafo_parallel_2d_complex.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_parallel_2d_complex.py) | Projection and back-projection in 2D parallel beam geometry with complex-valued data | middle
[`ray_trafo_parallel_2d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_parallel_2d.py) | Projection and back-projection in 2D parallel beam geometry | middle
[`ray_trafo_parallel_3d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/ray_trafo_parallel_3d.py) | Projection and back-projection in 3D parallel beam single-axis geometry | middle
[`skimage_ray_trafo_parallel_2d.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/skimage_ray_trafo_parallel_2d.py) | Projection and back-projection in 2D parallel beam geometry using the `scikit-image` back-end | middle
[`stir_project.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/stir_project.py) | Projection and back-projection in 3D PET geometry using the `stir` back-end | middle
[`stir_reconstruct.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/stir_reconstruct.py) | Iterative reconstruction in 3D PET geometry using the `stir` back-end | middle
