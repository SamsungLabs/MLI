__all__ = ['RaySampler']

from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.utils.base import product
from lib.networks.generators.gen_parts.surfaces.surfaces_mpi import SurfacesMPI
from lib.networks.generators.gen_parts.surfaces.surfaces_msi import SurfacesMSI
from lib.networks.generators.gen_parts.surfaces.ray_descriptors import RayDescriptor, \
    calculate_ray_descriptors


def sample_rays(n_steps: int,
                rays_origin: torch.Tensor,
                rays_dir: torch.Tensor,
                max_distance: Optional[float],
                min_distance: Optional[float] = 0.0,
                depth_mode: str = None,
                random_sampling: bool = True):
    """
    Sample points along rays.

    Args:
        n_steps: num of points on ray
        rays_origin: B x ... x XYZ rays origin points
        rays_dir: B x ... x XYZ rays directions
        max_distance: end sampling position
        min_distance: start sampling position
        depth_mode: could be  'disparity'
        random_sampling: enable random sampling

    Returns:
        rays_pts: B x ... x n_steps x XYZ

    """

    assert 0 < min_distance < max_distance and max_distance > 0

    # B x ...
    near_depth = min_distance * torch.ones_like(rays_dir[..., 0])
    far_depth = max_distance * torch.ones_like(rays_dir[..., 0])

    if depth_mode == 'disparity':
        start = 1. / near_depth
        step = (1. / far_depth - start) / (n_steps - 1)
        # B x ... x n_steps
        inv_z_vals = torch.stack([start + i * step for i in range(n_steps)], dim=-1)
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (n_steps - 1)
        # B x ... x n_steps
        z_vals = torch.stack([start + i * step for i in range(n_steps)], dim=-1)

    if random_sampling:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        # B x ... x n_steps
        z_vals = lower + (upper - lower) * t_rand

    # B x ... x 1 x 3
    rays_dir = rays_dir.unsqueeze(-2)
    rays_origin = rays_origin.unsqueeze(-2)

    # B x ... x n_steps x 3
    rays_pts = z_vals.unsqueeze(-1) * rays_dir + rays_origin

    return rays_pts


def rays_surfaces_intersect(rays_origin: torch.Tensor,
                            rays_dir: torch.Tensor,
                            surfaces: Union[SurfacesMPI, SurfacesMSI],
                            surface_idx: int = None,
                            ):
    """
    Found intersections between rays and mpi planes for mpi_camera

    Args:
        rays_origin: B x 1 x 1 x n_rays x XYZ rays origin points
        rays_dir: B x 1 x 1 x n_rays x XYZ rays directions
        surfaces: surfaces

    Returns:
        rays_pts: B x 1 x 1 x n_rays x n_steps x XYZ

    """

    # B x 1 x 1 x n_surfaces x n_rays x XYZ
    intersection, _, _ = surfaces.find_intersection(rays_dir, rays_origin, surface_idx)
    intersection = intersection.permute(0, 1, 2, 4, 3, 5)

    return intersection


class RaySampler:
    """
    Ray sampler

    Sample rays for camera
    """

    def __init__(self,
                 n_steps: Optional[int] = None,
                 min_distance: Optional[float] = 1.,
                 max_distance: Optional[float] = 100.,
                 mode: str = 'disparity',
                 random_sampling=True,
                 surfaces_type: str = 'mpi',
                 ):
        self.reference_camera: Optional[CameraMultiple] = None
        self.n_steps = n_steps
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mode = mode
        self.random_sampling = random_sampling
        self.surfaces = None
        self.surfaces_type = surfaces_type
        self.mpi_camera = None

        self.update_surfaces_parameters()

    def update_surfaces_parameters(self,
                                   n_steps: Optional[int] = None,
                                   min_distance: Optional[float] = None,
                                   max_distance: Optional[float] = None,
                                   mode: Optional[str] = None,
                                   random_sampling: Optional[str] = None,
                                   surfaces_type: Optional[str] = None,
                                   ):

        if n_steps is not None:
            self.n_steps = n_steps
        if min_distance is not None:
            self.min_distance = min_distance
        if max_distance is not None:
            self.max_distance = max_distance
        if mode is not None:
            self.mode = mode
        if random_sampling:
            self.random_sampling = random_sampling
        if surfaces_type is not None:
            self.surfaces_type = surfaces_type
        if self.surfaces_type == 'mpi':
            surfaces_class = SurfacesMPI
        elif self.surfaces_type == 'msi':
            surfaces_class = SurfacesMSI
        else:
            raise ValueError(f'Unknown surfaces_type: {self.surfaces_type}')
        self.surfaces = surfaces_class(n_surfaces=self.n_steps,
                                       min_distance=self.min_distance,
                                       max_distance=self.max_distance,
                                       mode=self.mode,
                                       multi_reference_cams=False)

    def set_position(self,
                     camera: CameraMultiple,
                     mpi_camera: CameraMultiple = None,
                     ) -> None:
        self.reference_camera = camera  # B x 1 x 1 x KRT
        self.mpi_camera = mpi_camera
        self.surfaces.set_position(self.mpi_camera)

    def set_surfaces_depths(self,
                            depths
                            ) -> None:
        self.surfaces.set_surfaces_depths(depths)
        self.n_steps = depths.shape[0]
        self.min_distance = torch.min(depths)
        self.max_distance = torch.max(depths)

    def project_on(self,
                   source_features: torch.Tensor,
                   source_camera: CameraMultiple,
                   reference_pixel_coords: torch.Tensor,
                   relative_intrinsics: bool = False,
                   return_source_vectors_displacement: Union[str, Sequence[str], None] = None,
                   surface_idx: int = None,
                   return_displacement_namedtuple: bool = False,
                   depth_tensor: torch.Tensor = None,
                   ):
        """
        Projects features on points sampled from rays which runs from reference camera pixels.

        Args:
            source_features: B x n_source x n_features x H_feat x W_feat
                or B x n_source x n_steps x n_features x H_feat x W_feat
            source_camera: B x 1 x n_source x KRT
            reference_pixel_coords: B x ... x UV
            relative_intrinsics: bool

        Returns:
            sampled_rays: B x n_source x n_steps x n_features x ...
            mask: B x 1 x n_source x n_steps x 1 x ...
            source_vectors_displacement:  B x 1 x n_source x n_steps x ... x XYZ

        """
        batch_size, *intermediate_dims, uv = reference_pixel_coords.shape
        # B x ... x UV -> B x n_rays x UV
        reference_pixel_coords = reference_pixel_coords.reshape(batch_size, -1, uv)
        n_rays = reference_pixel_coords.shape[-2]
        reference_pixel_coords = reference_pixel_coords[:, None, None]

        # B x 1 x 1 x n_rays x XYZ
        rays_dir = self.reference_camera.pixel_to_world_ray_direction(reference_pixel_coords)
        world_positions = self.reference_camera.world_position.view(-1, 1, 1, 1, 3).expand_as(rays_dir)

        # B x 1 x 1 x n_steps x n_rays x XYZ
        if depth_tensor is not None:
            depth_tensor = depth_tensor.reshape(*depth_tensor.shape[:-2], n_rays)
            if surface_idx is None:
                rays_points = self.reference_camera.pixel_to_world(
                    reference_pixel_coords.unsqueeze(-3).expand(-1, -1, -1, self.n_steps, -1, -1),
                    depth=depth_tensor[:, None, None, :, :, None],
                )
            else:
                rays_points = self.reference_camera.pixel_to_world(
                    reference_pixel_coords.unsqueeze(-3).expand(-1, -1, -1, 1, -1, -1),
                    depth=depth_tensor[:, None, None, [surface_idx], :, None],
                )
        elif self.surfaces is not None:
            rays_points = rays_surfaces_intersect(rays_origin=world_positions,
                                                  rays_dir=rays_dir,
                                                  surfaces=self.surfaces,
                                                  surface_idx=surface_idx)
            rays_points = rays_points.permute(0, 1, 2, 4, 3, 5)
        else:
            rays_points = sample_rays(n_steps=self.n_steps,
                                      rays_origin=world_positions,
                                      rays_dir=rays_dir,
                                      max_distance=self.max_distance,
                                      min_distance=self.min_distance,
                                      depth_mode=self.mode,
                                      random_sampling=self.random_sampling)
            rays_points = rays_points.permute(0, 1, 2, 4, 3, 5)

        # B x 1 x n_source x n_steps x n_rays x XYZ
        rays_points = rays_points.expand(*source_camera.cameras_shape, -1, -1, -1)

        # B x 1 x n_source x n_steps x n_rays x UV
        source_pixel_coords = source_camera.world_to_pixel(rays_points)

        source_vectors_descriptors = None
        if return_source_vectors_displacement == 'angle':
            assert False, "Not implemented. " \
                          "Need to implement ray_direction from reference camera pixel grid in mpi camera system"
        elif return_source_vectors_displacement is not None:
            source_vectors_descriptors = calculate_ray_descriptors(source_camera=source_camera,
                                                                   source_pixel_coords=source_pixel_coords,
                                                                   reference_camera=self.mpi_camera,
                                                                   intermediate_dims=intermediate_dims
                                                                   )
            if not return_displacement_namedtuple:
                source_vectors_descriptors = source_vectors_descriptors.displacement_to_normal

        if relative_intrinsics:
            source_relative_coords = source_pixel_coords * 2 - 1
        else:
            h_feat, w_feat = source_features.shape[-2:]
            normalizer = -1 + torch.tensor([w_feat, h_feat], dtype=torch.float, device=source_features.device)
            source_relative_coords = source_pixel_coords / normalizer * 2 - 1

        # B x n_source x n_steps x n_features x H_feat x W_feat
        if surface_idx is not None:
            n_steps = 1
        else:
            n_steps = self.n_steps

        if source_features.ndim != 6:
            source_features = source_features.unsqueeze(2).expand(-1, -1, n_steps, -1, -1, -1)
        elif source_features.shape[2] == 1:
            source_features = source_features.expand(-1, -1, n_steps, -1, -1, -1)
        elif surface_idx is not None:
            source_features = source_features[:, :, [surface_idx]]
        else:
            assert source_features.shape[2] == n_steps, \
                f'Support only {n_steps} layers for projection, ' \
                f'but got {source_features.shape[2]} as input'

        source_relative_coords = source_relative_coords.contiguous().view(-1, 1, *source_relative_coords.shape[-2:])
        # B*n_source*n_steps x n_features x 1 x n_rays
        sampled_rays = F.grid_sample(source_features.contiguous().view(-1, *source_features.shape[-3:]),
                                     source_relative_coords)

        mask = torch.all(torch.abs(source_relative_coords) <= 1, dim=-1, keepdim=True)

        # B x n_source x n_steps x n_features x n_rays
        sampled_rays = sampled_rays.contiguous().view(*source_features.shape[:2], n_steps, -1, *intermediate_dims)
        mask = mask.contiguous().view(*source_features.shape[:2], n_steps, 1, *intermediate_dims)

        return sampled_rays, mask, source_vectors_descriptors

    def look_at(self,
                reference_features,
                reference_cameras,
                novel_camera,
                novel_pixel_coords,
                relative_intrinsics
                ):
        self.set_position(reference_cameras, mpi_camera=reference_cameras)

        features_reproject, timestamps, frustum_mask = self.surfaces.look_at(
            reference_features=reference_features,
            novel_camera=novel_camera,
            novel_pixel_coords=novel_pixel_coords,
            relative_intrinsics=relative_intrinsics
        )

        return features_reproject, timestamps, frustum_mask