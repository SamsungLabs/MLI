__all__ = ['Frustums',
           'CameraFrustum',
           ]

from typing import Callable

import torch
import torch.nn.functional as F

from .camera_pinhole import CameraPinhole


class Frustums:
    def __init__(self,
                 frustums_edges_directions: torch.Tensor,
                 frustums_apex_points: torch.Tensor
                 ):
        r"""
        Class for work with frustums.
        Frustums edges vectors must be in counterclockwise order when planes
        in right-handed system and far plane z_n < z_f.

           \                    /
            \                  /
             \                /
              \              /
              2 +---------> 3
              ^             +
          \   |             |    /
           \  |             |   /
            \ |             |  /
             \+             v /
              1 <---------+ 4

        Args:
            frustums_edges_directions (torch.Tensor): M_frustums x 4 x 3, Vectors of ribs which produce frustum.
            frustums_apex_points (torch.Tensor): M_frustums x 3, Apex of the pyramid which contain frustum.
        """

        super().__init__()

        self.frustums_edges_vectors = frustums_edges_directions
        self.frustums_apex_points = frustums_apex_points

    def __len__(self):
        return int(self.frustums_edges_vectors.shape[0] / 4)

    @property
    def frustums_planes_normals(self) -> torch.Tensor:
        """
        Calculation of the normals to the planes that create a frustums.

        Returns:
            torch.Tensor: M_frustums * 4 x 3
        """

        return torch.cross(self.frustums_edges_vectors,
                           torch.roll(self.frustums_edges_vectors, dims=1, shifts=1), dim=2)

    def check_points(self,
                     points: torch.Tensor
                     ) -> torch.Tensor:
        """
        Return bool tensor N_points x M_frustums, 'True'' if point in frustum or 'False' if not.

        Args:
            points (Tensor): N x 3

        Returns:
            torch.Tensor: N x M_frustums
        """

        apex_points = torch.repeat_interleave(self.frustums_apex_points.unsqueeze(1), 4, dim=1)
        result = ((points.unsqueeze(1).unsqueeze(1) - apex_points) * self.frustums_planes_normals).sum(3)
        return (result >= 0).all(2)

    def check_points_batch(self,
                           points: torch.Tensor
                           ) -> torch.Tensor:
        """
        Return bool tensor N_points x M_frustums, 'True'' if point in frustum or 'False' if not.

        Args:
            points (Tensor): Bc x N x 3

        Returns:
            torch.Tensor: Bc x N
        """

        apex_points = self.frustums_apex_points.unsqueeze(1)
        result = ((points - apex_points).unsqueeze(2) * self.frustums_planes_normals.unsqueeze(1)).sum(3)
        return (result >= 0).all(2)


class CameraFrustum(CameraPinhole):
    def __init__(self, *args, relative_intrinsics=True, **kwargs):
        """
        Camera with view frustum.

        Args:
            *args:
            relative_intrinsics: if True, uses image sizes and relative intrinsics for computing real intrinsics.
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.frustums = None
        self.cam_frustums_intersection_grid = None
        self.relative_intrinsics = relative_intrinsics

    def build_frustums(self) -> Frustums:
        """
        Build view frustums for cameras.
        Frustum can be used to check whether a point is in the camera’s field of vision or not.

        Returns:
            Frustums: frustums
        """

        bound_points = torch.tensor([[[0, 0], [0, 1], [1, 1], [1, 0]]],
                                    dtype=torch.float32,
                                    device=self.extrinsics.device)

        if not self.relative_intrinsics:
            images_bound_points_coords = torch.mul(bound_points, self.images_sizes.unsqueeze(1))
        else:
            images_bound_points_coords = bound_points.repeat(self.__len__(), 1, 1, )
        cameras_world_position = self.world_position

        world_frustum_bound_vectors_direction = self.cam_to_world(
            self.pixel_to_cam(images_bound_points_coords, 1)) - cameras_world_position.unsqueeze(1)

        world_frustum_bound_vectors_direction = F.normalize(world_frustum_bound_vectors_direction, dim=-1)

        return Frustums(world_frustum_bound_vectors_direction, cameras_world_position)

    def check_points(self,
                     points: torch.Tensor,
                     rebuild_frustums=True
                     ) -> torch.Tensor:
        """
        Checking is  point into any cameras frustum or not.

        Args:
            points (torch.Tensor): D1 x ... x Dn x 3
            rebuild_frustums (bool): If 'True' rebuild cameras frustums before checking.
                Use this if cameras params changing during training.

        Returns:
            torch.Tensor: Bc x D1 x ... x Dn
        """

        if rebuild_frustums or self.frustums is None:
            self.frustums = self.build_frustums()

        *dims, space_dim = points.shape
        if len(dims) > 0:
            points = points.contiguous().view(-1, space_dim)

        return self.frustums.check_points(points).permute(1, 0).view(len(self), *dims)

    def check_points_batch(self,
                           points: torch.Tensor,
                           rebuild_frustums=True
                           ) -> torch.Tensor:
        """
        Checking is  point into any cameras frustum or not.

        Args:
            points (torch.Tensor): Bc x Bp x 3
            rebuild_frustums (bool): If 'True' rebuild cameras frustums before checking.
                Use this if cameras params changing during training.

        Returns:
            torch.Tensor: Bc x N
        """

        if rebuild_frustums or self.frustums is None:
            self.frustums = self.build_frustums()

        return self.frustums.check_points_batch(points)

    def set_grid_sample_checker(self,
                                edge_half_length: int = 1,
                                resolution: int = 16,
                                rebuild_frustums: bool = True
                                ):
        """
        For 3d points grid, computed amount of the cameras frustums which contain point.
        Used to get the approximate number of cameras that look at a point.

        Args:
            edge_half_length (int): space cube edge half length
            resolution (int): grid sample resolution
            rebuild_frustums (bool): If 'True' rebuild cameras frustums before compute grid.
                Use this if cameras params changing during training.
        """

        if rebuild_frustums or self.frustums is None:
            self.frustums = self.build_frustums()

        x = torch.linspace(-edge_half_length, edge_half_length, resolution)
        y = torch.linspace(-edge_half_length, edge_half_length, resolution)
        z = torch.linspace(-edge_half_length, edge_half_length, resolution)

        grid = torch.stack(torch.meshgrid(x, y, z), dim=3)
        self.cam_frustums_intersection_grid = self.check_points(grid.to(self.extrinsics.device)).sum(
            dim=0).float().unsqueeze(0).unsqueeze(0)

    def grid_sample_checker(self, points: torch.Tensor,
                            resolution: int = 16,
                            min_cams: int = 3,
                            rebuild_frustums: bool = True) -> torch.Tensor:
        """
        Checking whether 'min_cams' cameras are looking at this point ('True') or not ('False').
        Args:
            points (torch.Tensor): D1 x ... x Dn x 3
            resolution (int): grid sample resolution
            min_cams (int): minimum number of cameras which look at point
            rebuild_frustums: If 'True' rebuild cameras frustums before checking.
                Use this if cameras params changing during training.

        Returns:
            torch.Tensor:  D1 x ... x Dn
        """

        if self.cam_frustums_intersection_grid is None \
                or self.cam_frustums_intersection_grid.shape[-1] != resolution \
                or rebuild_frustums:
            self.set_grid_sample_checker(resolution=resolution, rebuild_frustums=rebuild_frustums)

        *dims, space_dim = points.shape
        if len(dims) > 0:
            points = points.contiguous().view(-1, space_dim)
        result = F.grid_sample(self.cam_frustums_intersection_grid,
                               points.to(self.cam_frustums_intersection_grid.device)[None, None, None, ...]).reshape(-1)

        return result.view(*dims) > min_cams

    def build_grid_sample_checker_function(self,
                                           resolution: int = 16,
                                           min_cams: int = 3,
                                           rebuild_frustums: bool = True) -> Callable:
        """
        Build valid function.
        Args:
            points (torch.Tensor): D1 x ... x Dn x 3
            resolution (int): grid sample resolution
            min_cams (int): minimum number of cameras which look at point
            rebuild_frustums: If 'True' rebuild cameras frustums before checking.
                Use this if cameras params changing during training.

        Returns:
            function: validation function
        """

        def valid_fn(x):
            return self.grid_sample_checker(x,
                                            resolution=resolution,
                                            min_cams=min_cams,
                                            rebuild_frustums=rebuild_frustums)

        return valid_fn

