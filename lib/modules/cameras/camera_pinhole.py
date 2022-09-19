__all__ = ['CameraPinhole']

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from lib.utils.coord_conversion import (coords_cam_to_film,
                                        coords_cam_to_pixel,
                                        coords_cam_to_world,
                                        coords_film_to_cam,
                                        coords_film_to_pixel,
                                        coords_pixel_to_cam,
                                        coords_pixel_to_film,
                                        coords_world_to_cam,
                                        get_cameras_world_positions,
                                        )


class CameraPinhole:
    """
    Special class for processing space transforms in some camera rig system.
    'Bc' - batch of cameras, num cameras in system.
    """

    def __init__(self,
                 extrinsics: Optional[torch.Tensor] = None,
                 intrinsics: Optional[torch.Tensor] = None,
                 images_sizes: Optional[torch.Tensor] = None
                 ):
        """
        Special class for processing space transforms in some camera rig system.
        'Bc' - batch of cameras, num cameras in system.

        Args:
            extrinsics (torch.Tensor): Bc x 3 x 4 or Bc x 4 x 4, cameras extrinsics matrices
            intrinsics (torch.Tensor): Bc x 3 x 3, cameras intrinsics matrices
            images_sizes (torch.Tensor): Bc x 2 or 1 x 2 or 2, camera image plane size in pixels,
                needed for compute camera frustums.
        """

        if extrinsics is not None:
            self.extrinsics = extrinsics
        if intrinsics is not None:
            self.intrinsics = intrinsics
        if images_sizes is not None:
            self.images_sizes = images_sizes

    @classmethod
    def from_cameras(cls, cameras: List['CameraPinhole']):
        """
        Init :class:`~Cameras` instance from another :class:`~Cameras` instances.

        Args:
            cameras: cameras

        Returns:
            CameraPinhole: class:`~Cameras`
        """
        extrinsics = [camera.extrinsics for camera in cameras]
        intrinsics = [camera.intrinsics for camera in cameras]
        images_sizes = [camera.images_sizes for camera in cameras]

        if any([e is None for e in extrinsics]):
            extrinsics = None
        else:
            extrinsics = torch.cat(extrinsics)

        if any([i is None for i in intrinsics]):
            intrinsics = None
        else:
            intrinsics = torch.cat(intrinsics)

        if any([s is None for s in images_sizes]):
            images_sizes = None
        else:
            images_sizes = torch.cat(images_sizes)

        return cls(extrinsics=extrinsics,
                   intrinsics=intrinsics,
                   images_sizes=images_sizes)

    def __len__(self):
        return self.extrinsics.shape[0]

    def repeat(self, times: int = 1):
        """
        Repeat cameras n times.

        Args:
            times: how much times you want repeat this.

        Returns:
            CameraPinhole: class:`~Cameras`
        """
        if getattr(self, 'extrinsics', None) is not None:
            self.extrinsics = self.extrinsics.repeat(times, 1, 1)
        if getattr(self, 'intrinsics', None) is not None:
            self.intrinsics = self.intrinsics.repeat(times, 1, 1)
        if getattr(self, 'images_sizes', None) is not None:
            self.images_sizes = self.images_sizes.repeat(times, 1, 1)

        return self

    @property
    def device(self):
        return self.extrinsics.device

    @staticmethod
    def cam_to_film(points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_cam_to_film`
        """
        n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_cam_to_film(points).contiguous().view(n_cams, *dims, 2)

    @staticmethod
    def film_to_cam(points: torch.Tensor,
                    depth: Union[torch.Tensor, float, int]
                    ) -> torch.Tensor:
        """
        See :func:`~coords_film_to_cam`
        """

        n_cams, *dims, space_dim = points.shape
        if isinstance(depth, torch.Tensor):
            if depth.dim() == points.dim():
                depth = depth.contiguous().view(n_cams, -1, 1)
            elif depth.dim() == points.dim() - 1:
                depth = depth.contiguous().view(n_cams, -1)

        points = points.contiguous().view(n_cams, -1, space_dim)

        return coords_film_to_cam(points, depth).contiguous().view(n_cams, *dims, 3)

    def film_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_film_to_pixel`
        """

        n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_film_to_pixel(points, self.intrinsics).view(n_cams, *dims, 2)

    def pixel_to_film(self, points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_pixel_to_film`
        """

        n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_pixel_to_film(points, self.intrinsics).view(n_cams, *dims, 2)

    def cam_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_cam_to_pixel`
        """

        n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_cam_to_pixel(points, self.intrinsics).view(n_cams, *dims, 2)

    def pixel_to_cam(self,
                     points: torch.Tensor,
                     depth: Union[torch.Tensor, float, int]
                     ) -> torch.Tensor:
        """
        See :func:`~coords_pixel_to_cam`
        """

        n_cams, *dims, space_dim = points.shape
        if isinstance(depth, torch.Tensor):
            if depth.dim() == points.dim():
                depth = depth.contiguous().view(n_cams, -1, 1)
            elif depth.dim() == points.dim() - 1:
                depth = depth.contiguous().view(n_cams, -1)
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_pixel_to_cam(points, depth, self.intrinsics).view(n_cams, *dims, 3)

    def cam_to_world(self, points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_cam_to_world`
        """

        n_cams, *dims, space_dim = points.shape
        if len(dims) == 0:
            points = points.unsqueeze(0)
            n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_cam_to_world(points, self.extrinsics).view(-1, *dims, 3)

    def world_to_cam(self, points: torch.Tensor) -> torch.Tensor:
        """
        See :func:`~coords_world_to_cam`
        """

        n_cams, *dims, space_dim = points.shape
        if len(dims) == 0:
            points = points.unsqueeze(0)
            n_cams, *dims, space_dim = points.shape
        points = points.contiguous().view(n_cams, -1, space_dim)
        return coords_world_to_cam(points, self.extrinsics).view(n_cams, *dims, 3)

    def world_to_depth(self, points: torch.Tensor) -> torch.Tensor:
        return self.world_to_cam(points)[..., -1:]

    def world_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        return self.cam_to_pixel(self.world_to_cam(points))

    def pixel_to_world(self, points, depth):
        return self.cam_to_world(self.pixel_to_cam(points, depth))

    @property
    def world_position(self) -> torch.Tensor:
        """
        Returning is cameras position in global coord.

        Returns:
            torch.Tensor: Bc x 3 camera positions in global coord
        """

        return get_cameras_world_positions(self.extrinsics).squeeze(1)

    def world_view_direction(self, axis: str = 'z') -> torch.Tensor:
        """
        Returning is cameras view vector in global coord, usual camera look at z axis.

        Args:
            axis: 'x' | 'y' | 'z' cameras view direction

        Returns:
            torch.Tensor: Bc x 3 vector in global coord
        """

        world_position = self.world_position
        cam_direction = self.cam_view_direction(axis).unsqueeze(-2)
        # the redundant expanding above is necessary due to inherited CameraMultiple behaviour
        return self.cam_to_world(cam_direction).squeeze(-2) - world_position

    def cam_view_direction(self, axis: str = 'z') -> torch.Tensor:
        """
        Returning is cameras view vector in camera coord, usual camera look at z axis.

        Args:
            axis: 'x' | 'y' | 'z' cameras view direction

        Returns:
            torch.Tensor: Bc x 3 vector in global coord
        """

        world_position = self.world_position

        if axis == 'x':
            direction = [1, 0, 0]
        elif axis == 'y':
            direction = [0, 1, 0]
        elif axis == 'z':
            direction = [0, 0, 1]
        else:
            raise ValueError(f'Unknown axis {axis}: only x, y, z are supported')
        cam_direction = torch.tensor(direction,
                                     dtype=torch.float32,
                                     device=self.extrinsics.device
                                     ).expand_as(world_position)
        return cam_direction

    def pixel_to_world_ray_direction(self,
                                     points: torch.Tensor,
                                     ) -> torch.Tensor:
        """
        Compute direction for rays that start at camera center and intersect the image plane at given pixel coordinates.

        Args:
            points: points

        Returns:
            torch.Tensor: ray directions
        """
        world_points_coords = self.pixel_to_world(points, 1.)
        world_cam_position = self.world_position.contiguous()

        ndim_with_points = world_points_coords.dim()
        ndim_without_points = world_cam_position.dim()
        shape = (world_cam_position.shape[:-1]
                 + (1,) * (ndim_with_points - ndim_without_points)
                 + world_cam_position.shape[-1:]
                 )
        world_cam_position = world_cam_position.view(*shape)

        ray_direction = world_points_coords - world_cam_position
        ray_direction = F.normalize(ray_direction, dim=-1)

        return ray_direction

    def pixel_to_another_cam_ray_direction(self,
                                           points: torch.Tensor,
                                           another_camera: 'CameraPinhole',
                                           ) -> torch.Tensor:
        """
        Compute direction for rays that start at camera center and intersect the image plane at given pixel coordinates.
        Output in camera system of another_camera.

        Args:
            points: points

        Returns:
            torch.Tensor: ray directions
        """
        world_points_coords = self.pixel_to_world(points, 1.)
        another_cam_points_coords = another_camera.world_to_cam(world_points_coords)
        another_cam_cam_positions = another_camera.world_to_cam(self.world_position.contiguous())

        ndim_with_points = another_cam_points_coords.dim()
        ndim_without_points = another_cam_cam_positions.dim()
        shape = (another_cam_cam_positions.shape[:-1]
                 + (1,) * (ndim_with_points - ndim_without_points)
                 + another_cam_cam_positions.shape[-1:]
                 )
        another_cam_cam_positions = another_cam_cam_positions.view(*shape)

        ray_direction = another_cam_points_coords - another_cam_cam_positions
        ray_direction = F.normalize(ray_direction, dim=-1)

        return ray_direction

    def rescale_cameras(self, scale: Union[float, Tuple[float, float]]):
        """
        It's equivalent to changing the physical pixel size for all cameras,
        Useful for down-sampling, for example:
        Camera image have 256x256 original resolution,
        for down-sampling to 128x128 you need change pixel size on 2 times per dimension.
        It's same as change scale to 0.5.

        Args:
            scale: coeff

        """
        attributes_are_none = True

        if getattr(self, 'intrinsics', None) is not None:
            if isinstance(scale, (int, float)):
                self.intrinsics = self.intrinsics * scale
            else:
                vectorized_scale = torch.tensor([scale[0], scale[1], 1],
                                                dtype=torch.float, device=self.device)
                self.intrinsics = self.intrinsics * vectorized_scale
            attributes_are_none = False

        if getattr(self, 'images_sizes', None) is not None:
            if isinstance(scale, (int, float)):
                self.images_sizes = (self.images_sizes * scale).int()
            else:
                vectorized_scale = torch.tensor([scale[0], scale[1]],
                                                dtype=torch.float, device=self.images_sizes.device)
                self.images_sizes = self.images_sizes * vectorized_scale
            attributes_are_none = False

        if attributes_are_none:
            raise TypeError('Can not rescale camera without intrinsic and image size.')
