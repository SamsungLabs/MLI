__all__ = ['SurfacesMSI']

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from .surface_base import SurfacesBase


class SurfacesMSI(SurfacesBase):
    """
    Multi-Sphere Images.
    3D world is represented as images on the surfaces of concentric spheres.
    The center of the spheres coincides with the reference camera position.
    """

    def __init__(self,
                 n_surfaces: int,
                 min_distance: float,
                 max_distance: float = 100.,
                 mode: str = 'disparity',
                 multi_reference_cams: bool = False,
                 ):
        self.reference_camera: Optional[CameraMultiple] = None
        self.n_surfaces = n_surfaces
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mode = mode
        self.radii = self._build_surfaces(n_surfaces=self.n_surfaces,
                                          min_distance=self.min_distance,
                                          max_distance=self.max_distance,
                                          mode=self.mode,
                                          )
        self.multi_reference_cams = multi_reference_cams

    def depths(self, normalize: bool = False) -> torch.Tensor:
        if not normalize:
            return self.radii
        else:
            return self.radii.sub(self.min_distance).div(self.max_distance - self.min_distance)

    def disparities(self, normalize: bool = False) -> torch.Tensor:
        if not normalize:
            return self.radii.reciprocal()
        else:
            disp = self.radii.reciprocal()
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

        self.radii = self._build_surfaces(
            n_surfaces=self.n_surfaces,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            mode=self.mode,
            device=camera.device,
        )

    @property
    def n_intersections(self) -> int:
        return self.n_surfaces

    def find_intersection(self,
                          velocity: torch.Tensor,
                          start_point: torch.Tensor,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Find the intersection of the ray with spheres.
        Warning: Only farthest time-positive intersection is returned.

        Method: Denote velocity vector as v, start point as C' and sphere center as C. Let the sphere radius equal R.
        Then one needs to solve the equation w.r.t. time variable s:
            |C' + sv - C|^2 = R^2.
        This leads to a typical quadratic equation, where T is a translation vector T = C - C':
            |v|^2 s^2 - 2 * <v, T> * s + (|T|^2 - R^2) = 0.
        The discriminant of such an equation equals
            D = <V, T>^2 - |v|^2 * (|T|^2 - R^2).
        As explained in the warning above, we take only the greater root into account, therefore,
            s = (<v, T> + sqrt(D)) / |v|^2.

        Args:
            velocity: B x n_ref x n_novel x ... x XYZ
            start_point: B x n_ref x n_novel x XYZ or B x n_ref x n_novel x ... x XYZ

        Returns:
            intersection: B x n_ref x n_novel x n_surfaces x ... x XYZ
            time: B x n_ref x n_novel x n_surfaces x ...
            mask: B x n_ref x n_novel x n_surfaces x ...
        """

        hidden_dim = velocity.ndim - 4
        batch_size, n_ref, n_novel, *spatial, xyz = velocity.shape
        spatial = tuple(spatial)

        # from B x n_ref x n_novel x ... x XYZ to B x n_ref x n_novel x XYZ x ...
        perm = [0, 1, 2, 3 + hidden_dim] + list(range(3, 3 + hidden_dim))

        if start_point.ndim == 4:
            start_point = start_point[(...,) + (None,) * hidden_dim]  # B x n_ref x n_novel x XYZ x ...
        else:
            assert start_point.ndim == velocity.ndim
            start_point = start_point.permute(perm)

        # B x n_ref x n_novel x  XYZ x ...
        translation = self.reference_camera.world_position[(...,) + (None,) * hidden_dim] - start_point
        velocity = velocity.permute(perm)

        scalar_product = torch.sum(velocity * translation, dim=3)  # B x n_ref x n_novel x ...
        velocity_squared_norm = velocity.pow(2).sum(3)  # B x n_ref x n_novel x ...
        translation_squared_norm = translation.pow(2).sum(3)  # B x n_ref x n_novel x ...
        discriminant = (
            scalar_product.pow(2).unsqueeze(3)
            - velocity_squared_norm.unsqueeze(3) * (translation_squared_norm.unsqueeze(3)
                                                    - self.radii.pow(2)[(...,) + (None,) * hidden_dim]
                                                    )
        )  # B x n_ref x n_novel x n_surfaces x ...
        intersections_exist = discriminant.ge(0.)
        discriminant = F.relu(discriminant)  # set fake zero values instead of negative values
        discriminant_sqrt = discriminant.sqrt()

        # B x n_ref x n_novel x n_surfaces x ...
        time = (scalar_product.unsqueeze(3) + discriminant_sqrt) / (velocity_squared_norm.unsqueeze(3) + 1e-7)
        time_positive = time.ge(0.)
        mask = intersections_exist * time_positive

        # B x n_ref x n_novel x n_surfaces x XYZ x ...
        intersection = start_point.unsqueeze(3) + time.unsqueeze(4) * velocity.unsqueeze(3)
        perm = [0, 1, 2, 3] + list(range(5, 5 + hidden_dim)) + [4]
        intersection = intersection.permute(perm)  # B x n_ref x n_novel x n_surfaces x ... x XYZ

        assert intersection.shape == (batch_size, n_ref, n_novel, self.n_intersections) + spatial + (xyz,)
        assert time.shape == (batch_size, n_ref, n_novel, self.n_intersections) + spatial
        assert mask.shape == (batch_size, n_ref, n_novel, self.n_intersections) + spatial

        return intersection, time, mask

    def project_on(self,
                   source_features: torch.Tensor,
                   source_camera: CameraMultiple,
                   reference_pixel_coords: torch.Tensor,
                   relative_intrinsics: bool = False,
                   ) -> torch.Tensor:
        """
        Project features on the MSI surfaces

        Args:
            source_features: B x n_ref x n_source x n_features x H_feat x W_feat
            source_camera: B x n_ref x n_source x KRT
            reference_pixel_coords: B x n_ref x 1 x H_ref x W_ref x UV
            relative_intrinsics:

        Returns:
            psv: B x_ref x n_source x n_surfaces x n_features x H_ref x W_ref
        """
        if not self.multi_reference_cams:
            source_features = source_features.unsqueeze(1)
            reference_pixel_coords = reference_pixel_coords.unsqueeze(1)

        h_ref, w_ref = reference_pixel_coords.shape[-3:-1]

        # B x n_ref x 1 x H_ref x W_ref x XYZ
        rays = self.reference_camera.pixel_to_world_ray_direction(reference_pixel_coords)

        # B x n_ref x 1 x n_surfaces x H_ref x W_ref x XYZ
        world_intersection, time, mask = self.find_intersection(rays, self.reference_camera.world_position)

        # B x n_ref x n_source x n_surfaces x H_ref x W_ref x XYZ
        world_intersection = world_intersection.expand(*source_camera.cameras_shape, -1, -1, -1, -1)

        # B x n_ref x n_source x n_surfaces x H_ref x W_ref x UV
        source_pixel_coords = source_camera.world_to_pixel(world_intersection)

        if relative_intrinsics:
            source_relative_coords = source_pixel_coords * 2 - 1
        else:
            h_feat, w_feat = source_features.shape[-2:]
            normalizer = -1 + torch.tensor([w_feat, h_feat], dtype=torch.float, device=source_features.device)
            source_relative_coords = source_pixel_coords / normalizer * 2 - 1

        # replace fake coords to such that would be ignored by grid_sample
        source_relative_coords = torch.where(mask.unsqueeze(-1).expand_as(source_relative_coords),
                                             source_relative_coords,
                                             torch.empty_like(source_relative_coords).fill_(1e10),
                                             )

        source_features = source_features.unsqueeze(-4).expand(-1, -1, -1, self.n_intersections, -1, -1, -1)

        psv = F.grid_sample(source_features.reshape(-1, *source_features.shape[-3:]),
                            source_relative_coords.reshape(-1, *source_relative_coords.shape[-3:]),
                            )

        # B x_ref x n_source x n_surfaces x n_features x H_ref x W_ref
        psv = psv.reshape(*source_camera.cameras_shape, self.n_intersections, -1, h_ref, w_ref)

        if not self.multi_reference_cams:
            psv = psv.squeeze(1)
        return psv

    def look_at(self,
                reference_features: torch.Tensor,
                novel_camera: CameraMultiple,
                novel_pixel_coords: torch.Tensor,
                relative_intrinsics: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Render the surfaces for the given cameras.

        Args:
            reference_features: B x n_ref x  n_surfaces x n_features x H_ref x W_ref
            novel_pixel_coords: B x n_ref x n_novel x H_novel x W_novel x UV
            novel_camera: B x n_ref x n_novel x KRT
            relative_intrinsics:

        Returns:
            rendered: B x n_ref x n_novel x n_surfaces x n_features x H_novel x W_novel
            time: B x n_ref x n_novel x n_surfaces x H_novel x W_novel
            mask: B x n_ref x n_novel x 1 x H_novel x W_novel
                masks with ones inside frustum reference cameras
        """
        if not self.multi_reference_cams:
            reference_features = reference_features.unsqueeze(1)
        assert reference_features.shape[-4] == self.n_intersections

        h_novel, w_novel = novel_pixel_coords.shape[-3:-1]

        # B x n_ref x n_novel x H_novel x W_novel x XYZ
        rays = novel_camera.pixel_to_world_ray_direction(novel_pixel_coords)

        # B x n_ref x n_novel x n_surfaces x H_novel x W_novel x XYZ
        world_intersection, time, mask = self.find_intersection(rays, novel_camera.world_position)

        # a trick is needed as CameraMultiple does not support broadcasting
        # B x n_ref x 1 x n_novel*n_surfaces x H_novel x W_novel x XYZ
        world_intersection = world_intersection.reshape(
            *world_intersection.shape[:2], 1, -1, *world_intersection.shape[-3:])

        # B x n_ref x 1 x n_novel*n_surfaces x H_novel x W_novel x UV
        reference_pixel_coords = self.reference_camera.world_to_pixel(world_intersection)

        # B x n_ref x n_novel x n_surfaces x H_novel x W_novel x UV
        reference_pixel_coords = reference_pixel_coords.reshape(
            *novel_camera.cameras_shape, -1, *reference_pixel_coords.shape[-3:])

        if relative_intrinsics:
            reference_relative_coords = reference_pixel_coords * 2 - 1
        else:
            h_ref, w_ref = reference_features.shape[-2:]
            normalizer = -1 + torch.tensor([w_ref, h_ref], dtype=torch.float, device=rays.device)
            reference_relative_coords = reference_pixel_coords / normalizer * 2 - 1

        # replace fake coords to such that would be ignored by grid_sample
        reference_relative_coords = torch.where(mask.unsqueeze(-1).expand_as(reference_relative_coords),
                                                reference_relative_coords,
                                                torch.empty_like(reference_relative_coords).fill_(1e10)
                                                )
        # B x n_ref x n_novel x n_surfaces x n_features x H_ref x W_ref
        reference_features = reference_features.unsqueeze(2).expand(*novel_camera.cameras_shape,
                                                                    *reference_features.shape[2:])

        rendered = F.grid_sample(reference_features.contiguous().view(-1, *reference_features.shape[-3:]),
                                 reference_relative_coords.contiguous().view(-1, *reference_relative_coords.shape[-3:])
                                 )
        rendered = rendered.reshape(*novel_camera.cameras_shape, self.n_intersections, -1, h_novel, w_novel)
        mask = reference_relative_coords.abs().le(1.).all(-1).all(dim=3, keepdim=True)

        if not self.multi_reference_cams:
            rendered = rendered.squeeze(1)
            time = time.squeeze(1)
            mask = mask.squeeze(1)
        return rendered, time, mask
