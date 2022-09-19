__all__ = ['SurfacesMPI']

from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.utils.base import product
from lib.networks.generators.gen_parts.surfaces.surface_base import SurfacesBase
from lib.networks.generators.gen_parts.surfaces.ray_descriptors import RayDescriptor, \
    calculate_ray_descriptors
from lib.utils.base import get_grid


class SurfacesMPI(SurfacesBase):
    """Multi-Plane Images"""

    def __init__(self,
                 n_surfaces: Optional[int] = None,
                 min_distance: Optional[float] = 1.,
                 max_distance: Optional[float] = 100.,
                 mode: str = 'disparity',
                 multi_reference_cams=False,
                 reference_pixel_resolution=None,
                 ):
        self.reference_camera: Optional[CameraMultiple] = None
        self.n_surfaces = n_surfaces
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mode = mode
        self.multi_reference_cams = multi_reference_cams
        self.mpi_depths = self._build_surfaces(n_surfaces=self.n_surfaces,
                                               min_distance=self.min_distance,
                                               max_distance=self.max_distance,
                                               mode=self.mode,
                                               )
        self.reference_pixel_resolution = reference_pixel_resolution

    def depths(self, normalize: bool = False) -> torch.Tensor:
        if not normalize:
            return self.mpi_depths
        else:
            return (self.mpi_depths - self.min_distance) / (self.max_distance - self.min_distance)

    def disparities(self, normalize: bool = False) -> torch.Tensor:
        if not normalize:
            return self.mpi_depths.reciprocal()
        else:
            disp = self.mpi_depths.reciprocal()
            return (disp - 1 / self.max_distance) / (1 / self.min_distance - 1 / self.max_distance)

    def set_position(self,
                     camera: CameraMultiple,
                     n_surfaces: Optional[int] = None,
                     min_distance: Optional[float] = None,
                     max_distance: Optional[float] = None,
                     mode: Optional[str] = None,
                     ) -> None:
        self.reference_camera = camera  # B x n_ref x 1 x KRT

        if n_surfaces is not None:
            self.n_surfaces = n_surfaces
        if min_distance is not None:
            self.min_distance = min_distance
        if max_distance is not None:
            self.max_distance = max_distance
        if mode is not None:
            self.mode = mode

        self.mpi_depths = self._build_surfaces(
            n_surfaces=self.n_surfaces,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            mode=self.mode,
            device=camera.device,
        )

    def set_surfaces_depths(self,
                            depths: torch.Tensor
                            ):
        """
        depth: 1D torch.Tensor length of N
        """
        self.mpi_depths = depths

    @property
    def n_intersections(self):
        return self.n_surfaces

    def find_intersection(self,
                          velocity: torch.Tensor,
                          start_point: torch.Tensor,
                          surface_idx: int = None,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            velocity: B x n_ref x n_novel x ... x XYZ
            start_point: B x n_ref x n_novel x XYZ or B x n_ref x n_novel x ... x XYZ
            surface_idx: surface idx if need intersection with particular surface

        Returns:
            intersection: B x n_ref x n_novel x n_surfaces x ... x XYZ
            time: B x n_ref x n_novel x n_surfaces x ...
            mask: B x n_ref x n_novel x n_surfaces x ...
        """
        hidden_dim = velocity.ndim - 4
        # B x n_ref x 1 x XYZ x ...
        planes_normal = self.reference_camera.world_view_direction()[(...,) + (None,) * hidden_dim]

        if start_point.ndim == 4:
            start_point = start_point[(...,) + (None,) * hidden_dim]  # B x n_ref x n_novel x XYZ x ...
        else:
            assert start_point.ndim == velocity.ndim
            perm = [0, 1, 2, 3 + hidden_dim] + list(range(3, 3 + hidden_dim))
            start_point = start_point.permute(perm)
        # B x n_ref x n_novel x  XYZ x ...
        translation = self.reference_camera.world_position[(...,) + (None,) * hidden_dim] - start_point
        scalar_product = torch.sum(planes_normal * translation, dim=3, keepdim=True)  # B x n_ref x n_novel x 1 x ...
        # B x n_ref x n_novel x n_surfaces x ...
        if surface_idx is None:
            numerator = scalar_product + self.mpi_depths[(...,) + (None,) * hidden_dim]
        else:
            numerator = scalar_product + self.mpi_depths[[surface_idx]][(...,) + (None,) * hidden_dim]

        perm = [0, 1, 2] + list(range(4, 4 + hidden_dim)) + [3]
        denominator = torch.sum(velocity * planes_normal.permute(perm),
                                dim=-1)  # B x n_ref x n_novel x ...

        time = numerator / (denominator.unsqueeze(3) + 1e-7)  # B x n_ref x n_novel x n_surfaces x ...
        intersection = start_point.permute(perm).unsqueeze(3) + time.unsqueeze(-1) * velocity.unsqueeze(3)
        mask = torch.ones(*time.shape,
                          dtype=torch.bool, device=time.device)

        return intersection, time, mask

    def project_on(self,
                   source_features: torch.Tensor,
                   source_camera: CameraMultiple,
                   reference_pixel_coords: torch.Tensor = None,
                   relative_intrinsics: bool = False,
                   ) -> torch.Tensor:
        """
        Project features on the MPI surfaces

        Args:
            source_features: B x n_ref x n_source x n_features x H_feat x W_feat
                or B x n_ref x n_source x n_intersections x n_features x H_feat x W_feat
            source_camera: B x n_ref x n_source x KRT
            reference_pixel_coords: B x n_ref x 1 x H_ref x W_ref x UV

        Returns:
            psv: B x n_ref x n_source x n_surfaces x n_features x H_ref x W_ref,
                plane-sweep-volume of the reference camera
        """
        if self.multi_reference_cams is False:
            source_features = source_features.unsqueeze(1)
            if reference_pixel_coords is not None:
                reference_pixel_coords = reference_pixel_coords.unsqueeze(1)

        if reference_pixel_coords is None:
            batch_size, n_ref, n_source, n_intersections, _, _, _ = source_features.shape
            reference_pixel_coords = get_grid(batch_size=1,
                                              height=self.reference_pixel_resolution[0],
                                              width=self.reference_pixel_resolution[1],
                                              relative=True,
                                              values_range='sigmoid',
                                              align_corners=True,
                                              device=source_features.device,
                                              )
            reference_pixel_coords = reference_pixel_coords[:, -1, -1, -1, ].expand(batch_size, n_ref, n_source,
                                                                                         -1, -1, -1, -1)

        h_ref, w_ref = reference_pixel_coords.shape[-3:-1]

        # reshape to B x n_ref x 1 x n_surfaces x H_ref x W_ref x UV
        reference_pixel_coords = reference_pixel_coords.unsqueeze(3).expand(
            -1, -1, -1, self.n_intersections, -1, -1, -1)
        depths = self.mpi_depths.view(-1, 1, 1, 1).expand_as(reference_pixel_coords[..., :1])

        # B x n_ref x 1 x n_surfaces x H_ref x W_ref x XYZ
        world_coords = self.reference_camera.pixel_to_world(reference_pixel_coords, depths)

        # B x n_ref x n_source x n_surfaces x H_ref x W_ref x XYZ
        world_coords = world_coords.expand(*source_camera.cameras_shape, -1, -1, -1, -1)

        # B x n_ref x n_source x n_surfaces x H_ref x W_ref x UV
        source_pixel_coords = source_camera.world_to_pixel(world_coords)

        if relative_intrinsics:
            source_relative_coords = source_pixel_coords * 2 - 1
        else:
            h_feat, w_feat = source_features.shape[-2:]
            normalizer = -1 + torch.tensor([w_feat, h_feat], dtype=torch.float, device=source_features.device)
            source_relative_coords = source_pixel_coords / normalizer * 2 - 1

        # source_features: B x n_ref x n_source x n_surfaces x n_features x H_feat x W_feat
        if source_features.ndim != 7:
            source_features = source_features.unsqueeze(3).expand(-1, -1, -1, self.n_intersections, -1, -1, -1)
        elif source_features.shape[3] == 1:
            source_features = source_features.expand(-1, -1, -1, self.n_intersections, -1, -1, -1)
        else:
            assert source_features.shape[3] == self.n_intersections, \
                f'Support only {self.n_intersections} layers for projection, ' \
                f'but got {source_features.shape[3]} as input'

        # B*n_ref*n_source*n_surfaces x n_features x H_ref x W_ref
        psv = F.grid_sample(source_features.contiguous().view(-1, *source_features.shape[-3:]),
                            source_relative_coords.contiguous().view(-1, *source_relative_coords.shape[-3:]),
                            )

        # B x n_ref x n_source x n_surfaces x n_features x H_ref x W_ref
        psv = psv.contiguous().view(*source_camera.cameras_shape, self.n_intersections, -1, h_ref, w_ref)

        if self.multi_reference_cams is False:
            psv = psv.squeeze(1)

        return psv

    def look_at(self,
                reference_features: torch.Tensor,
                novel_camera: CameraMultiple,
                novel_pixel_coords: torch.Tensor,
                relative_intrinsics: bool = False,
                return_novel_vectors_descriptors: Union[str, Sequence[str], None] = None,
                return_descriptors_namedtuple: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Render the surfaces for the given cameras.

        Args:
            reference_features: B x n_ref x n_surfaces x n_features x H_ref x W_ref
            novel_pixel_coords: B x n_ref x n_novel x H_novel x W_novel x UV
            novel_camera: B x n_ref x n_novel x KRT
            relative_intrinsics: relative_intrinsics flag
            return_novel_vectors_displacement: return diff between novel vectors and reference direction for each voxel

        Returns:
            rendered: B x n_ref x n_novel x n_surfaces x n_features x H_novel x W_novel
            time: B x n_ref x n_novel x n_surfaces x H_novel x W_novel
            inside_reference_frustum_mask: B x n_ref x n_novel x 1 x H_novel x W_novel
                                        masks with ones inside frustum reference cameras
            novel_vectors_displacement: B x n_ref x n_novel x XYZ x H_novel x W_novel (non normalized!)
        """
        # B x n_ref x n_novel x H_novel x W_novel x XYZ

        if self.multi_reference_cams is False:
            reference_features = reference_features.unsqueeze(1)

        # B x n_ref x n_novel x H_novel x W_novel x XYZ
        rays = novel_camera.pixel_to_world_ray_direction(novel_pixel_coords)
        rendered, time, inside_reference_frustum_mask = self.look_at_rays(
            reference_features=reference_features,
            rays=rays,
            start_points=novel_camera.world_position,
            relative_intrinsics=relative_intrinsics,
        )

        if return_novel_vectors_descriptors is not None:
            novel_vectors_descriptors = calculate_ray_descriptors(source_camera=novel_camera,
                                                                  source_pixel_coords=novel_pixel_coords,
                                                                  reference_camera=self.reference_camera
                                                                  )
            # if not return_novel_vectors_descriptors:
            #     novel_vectors_descriptors = novel_vectors_descriptors.displacement_to_normal

        if self.multi_reference_cams is False:
            rendered = rendered.squeeze(1)
            time = time.squeeze(1)
            inside_reference_frustum_mask = inside_reference_frustum_mask.squeeze(1)

            # if return_novel_vectors_descriptors is not None:
            #     if not return_descriptors_namedtuple:
            #         novel_vectors_descriptors = novel_vectors_descriptors.squeeze(1)

        if return_novel_vectors_descriptors is not None:
            return rendered, time, inside_reference_frustum_mask, novel_vectors_descriptors
        else:
            return rendered, time, inside_reference_frustum_mask

    def look_at_rays(self,
                     reference_features: torch.Tensor,
                     rays: torch.Tensor,
                     start_points: torch.Tensor,
                     relative_intrinsics: bool = False,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Render the surfaces for the given cameras.

        Args:
            reference_features: B x n_ref x n_surfaces x n_features x H_ref x W_ref
            rays: B x n_ref x n_novel x ... x XYZ
            start_points: B x n_ref x n_novel x XYZ or B x n_ref x n_novel x ... x XYZ
            relative_intrinsics: bool

        Returns:
            rendered: B x n_ref x n_novel x n_surfaces x n_features x ...
            time: B x n_ref x n_novel x n_surfaces x ...
            inside_reference_frustum_mask: B x n_ref x n_novel x 1 x ...
                masks with ones inside frustum reference cameras
        """
        # B x n_ref x n_novel x n_surfaces x ... x XYZ
        world_intersection, time, _ = self.find_intersection(rays, start_points)
        batch_size, n_ref, n_novel, _, *hidden_dims, _ = world_intersection.shape

        # a trick is needed as CameraMultiple does not support broadcasting
        # B x n_ref x 1 x n_novel*n_surfaces x ... x XYZ
        world_intersection = world_intersection.contiguous().view(batch_size, n_ref, 1, -1,
                                                                  *world_intersection.shape[4:]
                                                                  )

        # B x n_ref x 1 x n_novel*n_surfaces x ... x UV
        reference_pixel_coords = self.reference_camera.world_to_pixel(world_intersection)

        # B x n_ref x n_novel x n_surfaces x ... x UV
        reference_pixel_coords = reference_pixel_coords.contiguous().view(batch_size, n_ref, n_novel, -1,
                                                                          *reference_pixel_coords.shape[4:]
                                                                          )
        if relative_intrinsics:
            reference_relative_coords = reference_pixel_coords * 2 - 1
        else:
            h_ref, w_ref = reference_features.shape[-2:]
            normalizer = -1 + torch.tensor([w_ref, h_ref], dtype=torch.float, device=rays.device)
            reference_relative_coords = reference_pixel_coords / normalizer * 2 - 1

        inside_reference_frustum_mask = torch.all(reference_relative_coords.ge(-1.) *
                                                  reference_relative_coords.le(1.),
                                                  dim=-1).all(dim=3, keepdim=True)

        # B x n_ref x n_novel x n_surfaces x n_features x H_ref x W_ref
        reference_features = reference_features.unsqueeze(2).expand(batch_size, n_ref, n_novel,
                                                                    *reference_features.shape[2:])

        rendered = F.grid_sample(
            reference_features.contiguous().view(-1, *reference_features.shape[-3:]),
            reference_relative_coords.contiguous().view(batch_size * n_ref * n_novel * self.n_intersections, -1, 1, 2)
        )
        rendered = rendered.contiguous().view(batch_size, n_ref, n_novel, self.n_intersections, -1, *hidden_dims)

        return rendered, time, inside_reference_frustum_mask
