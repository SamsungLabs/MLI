__all__ = ['RayDescriptor']

from typing import Optional, NamedTuple, Tuple

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.utils.base import product


class RayDescriptor(NamedTuple):
    direction: torch.Tensor
    displacement_to_normal: torch.Tensor
    inner_product: Optional[torch.Tensor] = None
    plucker: Optional[torch.Tensor] = None


def calculate_ray_descriptors(source_camera: CameraMultiple,
                              reference_camera: CameraMultiple,
                              source_pixel_coords: torch.Tensor,
                              intermediate_dims: Optional[Tuple[int, ...]] = None,
                              ) -> RayDescriptor:
    """

    Args:
        source_camera:  B x 1 x n_source x KRT
        source_pixel_coords: B x 1 x n_source x n_steps x n_rays x UV
        intermediate_dims: tuple of int, that have product equal to n_rays

    Returns:

    """
    n_rays = source_pixel_coords.shape[-2]
    if intermediate_dims is not None:
        assert product(intermediate_dims) == n_rays
    else:
        intermediate_dims = (n_rays,)

    mpi_camera_broadcasted = CameraMultiple.broadcast_cameras(reference_camera, source_camera)

    # B x 1 x n_source x n_steps * n_rays x XYZ
    rays_dir_source = source_camera.pixel_to_another_cam_ray_direction(
        source_pixel_coords.reshape(*source_camera.cameras_shape, -1, 2),
        mpi_camera_broadcasted,
    )

    # rays_dir_source = source_camera.pixel_to_world_ray_direction(
    #     source_pixel_coords.reshape(*source_camera.cameras_shape, -1, 2)
    # )

    # B x 1 x n_source x n_steps x n_rays x XYZ
    rays_dir_source = rays_dir_source.reshape(*source_camera.cameras_shape, -1, n_rays, 3)

    # B x n_source x n_steps x n_rays x XYZ
    mpi_cam_dir = reference_camera.cam_view_direction() \
        .view(-1, 1, 1, 1, 1, 3) \
        .expand(*source_pixel_coords.shape[:-1], -1)

    displacement_to_normal = F.normalize(rays_dir_source - mpi_cam_dir, dim=-1)
    output_shape = (tuple(displacement_to_normal.shape[:-2])
                    + tuple(intermediate_dims)
                    + (-1,)
                    )
    displacement_to_normal = displacement_to_normal.reshape(output_shape)

    inner_product = torch.sum(rays_dir_source * mpi_cam_dir, dim=-1, keepdim=True)
    inner_product = inner_product.reshape(output_shape)

    ray_origin = mpi_camera_broadcasted.world_to_cam(source_camera.world_position.unsqueeze(-2)).unsqueeze(-2)
    plucker = torch.cross(ray_origin.expand_as(rays_dir_source),
                          rays_dir_source,
                          dim=-1).reshape(output_shape)

    direction = rays_dir_source.reshape(output_shape)
    return RayDescriptor(
        direction=direction,
        displacement_to_normal=displacement_to_normal,
        inner_product=inner_product,
        plucker=plucker,
    )
