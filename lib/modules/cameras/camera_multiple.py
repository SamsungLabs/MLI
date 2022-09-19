__all__ = ['CameraMultiple']

import math
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F

from .camera_pinhole import CameraPinhole
from ...utils.coord_conversion import coords_pixel_to_film


class CameraMultiple(CameraPinhole):
    """
    The instance of this class contains a 'tensor' of cameras: D1 x ... x Dn cameras instead of just B cameras in
    CameraPinhole class.
    Arguments of the methods should start with the same dimensions.
    """

    def __init__(self,
                 extrinsics: torch.Tensor,
                 intrinsics: torch.Tensor,
                 images_sizes: Union[Tuple, List] = None,
                 ):
        assert extrinsics.shape[:-2] == intrinsics.shape[:-2], \
            f'{extrinsics.shape} vs {intrinsics.shape}'
        # if images_sizes is not None:
        #     print(images_sizes.shape, extrinsics.shape)
        #     assert extrinsics.shape[:-2] == images_sizes.shape[:-1]

        super().__init__(
            extrinsics=extrinsics.contiguous().view(-1, *extrinsics.shape[-2:]),
            intrinsics=intrinsics.contiguous().view(-1, *intrinsics.shape[-2:]),
            images_sizes=images_sizes if images_sizes is None else torch.tensor(images_sizes)
                .expand(*extrinsics.shape[:-2], -1).contiguous().view(-1, 2),
        )
        self.cameras_shape = extrinsics.shape[:-2]
        self.cameras_numel = torch.tensor(self.cameras_shape).prod().item()
        self.cameras_ndim = len(self.cameras_shape)
        self.images_size = images_sizes

    def __len__(self):
        return self.cameras_shape[0]

    def __getitem__(self, key):
        """
        Camera Multiple tensor-like indexing
        Args:
            key: slice indexing
        Returns:
            Selection of Camera Multiple
        """
        selected_extrinsics = self._unflatten_tensor(self.extrinsics)[key]
        selected_intrinsics = self._unflatten_tensor(self.intrinsics)[key]
        image_sizes = None if not hasattr(self, 'images_sizes') else self._unflatten_tensor(self.images_sizes)[key]
        return CameraMultiple(selected_extrinsics, selected_intrinsics, image_sizes)

    def _flatten_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[:self.cameras_ndim] == self.cameras_shape, \
            f'Expected {self.cameras_shape} but got {tensor.shape[:self.cameras_ndim]}'
        return tensor.contiguous().view(-1, *tensor.shape[self.cameras_ndim:])

    def _unflatten_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[0] == self.cameras_numel, \
            f'Expected length {self.cameras_numel} but got {tensor.shape[0]}'
        return tensor.contiguous().view(*self.cameras_shape, *tensor.shape[1:])

    def get_extrinsics(self):
        return self.extrinsics.view(*self.cameras_shape, *self.extrinsics.shape[-2:])

    def get_intrinsics(self):
        return self.intrinsics.view(*self.cameras_shape, *self.intrinsics.shape[-2:])

    @classmethod
    def from_cameras(cls, cameras):
        raise NotImplementedError

    @classmethod
    def broadcast_cameras(cls, broadcasted_camera, source_camera):
        camera_extrinsics_broadcasted = broadcasted_camera.get_extrinsics() \
            .expand(*source_camera.cameras_shape, -1, -1)
        camera_intrinsics_broadcasted = broadcasted_camera.get_intrinsics() \
            .expand(*source_camera.cameras_shape, -1, -1)
        camera_broadcasted = CameraMultiple(extrinsics=camera_extrinsics_broadcasted,
                                            intrinsics=camera_intrinsics_broadcasted,
                                            )
        return camera_broadcasted

    def cam_to_film(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_film(points)
        return self._unflatten_tensor(out)

    def film_to_cam(self,
                    points: torch.Tensor,
                    depth: Union[torch.Tensor, float, int]
                    ) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            depth = self._flatten_tensor(depth)
        points = self._flatten_tensor(points)
        out = super().film_to_cam(points, depth)
        return self._unflatten_tensor(out)

    def film_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().film_to_pixel(points)
        return self._unflatten_tensor(out)

    def pixel_to_film(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().pixel_to_film(points)
        return self._unflatten_tensor(out)

    def cam_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_pixel(points)
        return self._unflatten_tensor(out)

    def pixel_to_cam(self,
                     points: torch.Tensor,
                     depth: Union[torch.Tensor, float, int]
                     ) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            depth = self._flatten_tensor(depth)
        points = self._flatten_tensor(points)
        out = super().pixel_to_cam(points, depth)
        return self._unflatten_tensor(out)

    def cam_to_world(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_world(points)
        return self._unflatten_tensor(out)

    def world_to_cam(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().world_to_cam(points)
        return self._unflatten_tensor(out)

    @property
    def world_position(self) -> torch.Tensor:
        out = super().world_position
        return self._unflatten_tensor(out)

    def world_view_direction_unflatten(self, axis: str = 'z') -> torch.Tensor:
        out = super().world_view_direction(axis)
        return self._unflatten_tensor(out)

    def pixel_to_another_cam_ray_direction(self,
                                           points: torch.Tensor,
                                           another_camera: 'CameraMultiple',
                                           ) -> torch.Tensor:
        """
        Compute direction for rays that start at camera center and intersect the image plane at given pixel coordinates.
        Output in camera system of another_camera.

        Args:
            points: points

        Returns:
            torch.Tensor: ray directions
        """
        another_cam_cam_positions = another_camera.world_to_cam(self.world_position.unsqueeze(-2)).squeeze(-2)

        world_points_coords = self.pixel_to_world(points, 1.)
        another_cam_points_coords = another_camera.world_to_cam(world_points_coords)

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

    def crop_center(self,
                    crop_size
                    ):
        """
        Central crop the  camera intrinsics

        Args:
            crop_size: [h, w]
        """
        height, width = self.images_size
        scaling = torch.tensor([width, height, 1.], device=self.intrinsics.device).view(1, 3, 1)
        absolute_intrinsics = self.intrinsics * scaling

        crop_height, crop_width = crop_size

        crop_x = math.floor((width - crop_width) / 2)
        crop_y = math.floor((height - crop_height) / 2)

        pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float, device=self.intrinsics.device).view(1, 1, -1)
        film_coords = coords_pixel_to_film(pixel_coords, absolute_intrinsics)[:, 0]
        new_principal_point = - film_coords * torch.diagonal(absolute_intrinsics[:, :-1, :-1], dim1=1, dim2=2)
        cropped_intrinsic = absolute_intrinsics.clone()
        cropped_intrinsic[:, :-1, -1] = new_principal_point

        self.intrinsics = cropped_intrinsic / scaling
        self.images_size = crop_size
