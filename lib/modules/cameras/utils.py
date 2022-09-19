import logging
import math
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch

from lib.utils.geometry import average_rotation_matrices
from lib.utils.coord_conversion import coords_pixel_to_film
from .camera_multiple import CameraMultiple
from .camera_pytorch3d import CameraPytorch3d

logger = logging.getLogger(__name__)


def convert_to_camera_pytorch3d(camera: CameraMultiple,
                                convert_intrinsics_to_relative: bool = False,
                                height: Optional[int] = None,
                                width: Optional[int] = None,
                                ) -> CameraPytorch3d:
    assert camera.cameras_shape[1:3] == (1, 1)
    scaling = 1
    if convert_intrinsics_to_relative:
        if height is not None and width is not None:
            scaling = torch.tensor([width, height, 1], dtype=torch.float, device=camera.device).view(-1, 1)
        else:
            logger.warning('Asked to convert intrinsics to relative, but did not provide resolution')

    return CameraPytorch3d(
        extrinsics=camera.extrinsics.view(*camera.cameras_shape, *camera.extrinsics.shape[-2:])[:, 0, 0],
        intrinsics=camera.intrinsics.view(*camera.cameras_shape, *camera.intrinsics.shape[-2:])[:, 0, 0] / scaling,
    )


def interpolate_extrinsics(start: torch.Tensor,
                           end: torch.Tensor,
                           timestamp: Union[float, Sequence[float], torch.Tensor],
                           ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """

    Args:
        start: ... x 3 x 4 or ... x 4 x 4
        end:   ... x 3 x 4 or ... x 4 x 4
        timestamp: examples:
            - 0.75
            - (0.25, 0.5, 0.75)
            - torch.tensor([0.25, 0.5, 0.75])
            - tensor of shape ... x n_steps
    """
    assert start.shape == end.shape
    batch_shape = start.shape[:-2]
    if isinstance(timestamp, Number):
        single_timestamp = True
        shared_timestamps = None
        timestamp = torch.tensor([timestamp], dtype=torch.float, device=start.device)
    else:
        single_timestamp = False
        if torch.is_tensor(timestamp):
            shared_timestamps = timestamp.ndim == 1
            timestamp = timestamp.to(start.device)
        else:
            timestamp = torch.tensor(timestamp, dtype=torch.float, device=start.device)
            shared_timestamps = True

    weights = torch.stack([1 - timestamp, timestamp], dim=-1)  # n_steps x 2
    weights = weights.expand(*batch_shape, -1, -1)  # ... x n_steps x 2
    # ... x n_steps x 1 x 3 x 4
    start = start[..., None, None, :, :].expand(*batch_shape, timestamp.shape[-1], 1, -1, -1)
    end = end[..., None, None, :, :].expand(*batch_shape, timestamp.shape[-1], 1, -1, -1)

    # ... x n_steps x 3 x 4
    interpolated_extrinsics = average_extrinsics(
        torch.cat([start, end], dim=-3),
        weights=weights,
        keepdim=False,
    )

    if single_timestamp:
        return interpolated_extrinsics.squeeze(-3)
    elif not shared_timestamps:
        return interpolated_extrinsics
    else:
        return interpolated_extrinsics.unbind(dim=-3)


def average_extrinsics(extrinsics: torch.Tensor,
                       weights: Optional[torch.Tensor] = None,
                       keepdim: bool = True) -> torch.Tensor:
    """

    Args:
        extrinsics: ... x N x 3 x 4 or ... x N x 4 x 4
        weights: ... x N
        keepdim:
    Returns:
        torch.Tensor: ... x 1 x 4 x 4 average_extrinsic
    """
    extrinsics_shape = extrinsics.shape[-2:]

    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, -1:]

    n_samples = extrinsics.shape[-3]
    if weights is None:
        weights = torch.empty(*extrinsics.shape[:-2], dtype=extrinsics.dtype, device=extrinsics.device) \
            .fill_(1. / n_samples)  # ... x N
    else:
        assert weights.shape[-1] == n_samples, \
            f'number of weights {weights.shape[-1]} does not equal to number of extrinsics {n_samples}'
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        weights = weights.to(extrinsics.device)

    # ... x 3 x 3
    avg_rotations = average_rotation_matrices(rotation, weights=weights, keepdim=False)

    # for our camera format x_w = R^T @ (x_c - t) ==> position of the camera equals (-R^T @ t)
    cameras_world_positions = -rotation.transpose(-1, -2) @ translation  # ... x N x 3 x 1
    avg_world_position = torch.sum(cameras_world_positions * weights[..., None, None], dim=-3)  # ... x 3 x 1

    # for our camera format x_w = R^T @ (x_c - t) ==> position of the camera equals (-R^T @ t) ==> t = -R @ position
    avg_extrinsics = torch.cat([avg_rotations, -avg_rotations @ avg_world_position], dim=-1)

    if extrinsics_shape == (4, 4):
        extra_row = torch.tensor([0, 0, 0, 1], dtype=torch.float,
                                 device=extrinsics.device).expand_as(extrinsics[..., :1, :])
        avg_extrinsics = torch.cat([avg_extrinsics, extra_row], dim=-2)

    return avg_extrinsics.unsqueeze(-3) if keepdim else avg_extrinsics


def get_median_extrinsic(extrinsics: torch.Tensor,
                         keepdim: bool = True) -> torch.Tensor:
    """

    Args:
        extrinsics: ... x N x 3 x 4 or ... x N x 4 x 4
        keepdim:
    Returns:
        torch.Tensor: ... x 1 x 4 x 4 average_extrinsic
    """
    extrinsics_shape = extrinsics.shape[-2:]

    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, -1:]

    n_samples = extrinsics.shape[-3]

    # for our camera format x_w = R^T @ (x_c - t) ==> position of the camera equals (-R^T @ t)
    cameras_world_positions = -rotation.transpose(-1, -2) @ translation  # ... x N x 3 x 1
    avg_world_position = torch.sum(cameras_world_positions, dim=-3)  # ... x 3 x 1
    dists = torch.sum(torch.pow(avg_world_position - cameras_world_positions, 2))
    idx = torch.argmin(dists).item()
    median_extrinsics = torch.cat([rotation[..., idx, :, :], translation[..., idx, :, :]], dim=-1)

    if extrinsics_shape == (4, 4):
        extra_row = torch.tensor([0, 0, 0, 1], dtype=torch.float,
                                 device=extrinsics.device).expand_as(extrinsics[..., :1, :])
        median_extrinsics = torch.cat([median_extrinsics, extra_row], dim=-2)

    return median_extrinsics.unsqueeze(-3) if keepdim else median_extrinsics


def interpolate_cameras_pytorch3d(start: CameraPytorch3d,
                                  end: CameraPytorch3d,
                                  timestamp: Union[float, Sequence[float], torch.Tensor],
                                  take_start_intrinsics: bool = True,
                                  ) -> Union[CameraPytorch3d, List[CameraPytorch3d]]:
    start_extrinsics = start.extrinsics
    end_extrinsics = end.extrinsics
    intrinsics = start.intrinsics if take_start_intrinsics else end.intrinsics
    middle_extrinsics = interpolate_extrinsics(start_extrinsics, end_extrinsics, timestamp)
    if torch.is_tensor(middle_extrinsics):
        return CameraPytorch3d(extrinsics=middle_extrinsics,
                               intrinsics=intrinsics)
    else:
        return [
            CameraPytorch3d(extrinsics=extr, intrinsics=intrinsics)
            for extr in middle_extrinsics
        ]


def sample_camera_with_pixel_offset(camera: CameraPytorch3d,
                                    offset: float = 0.2,
                                    scale_y: float = 0.5,
                                    ) -> CameraPytorch3d:
    extrinsics = camera.extrinsics.clone()
    device = extrinsics.device
    focal = torch.diagonal(camera.intrinsics[:, :2, :2], dim1=1, dim2=2)  # B x 2
    batch_size = focal.shape[0]
    focal = torch.cat([focal, torch.ones(batch_size, 1, device=device, dtype=torch.float)], dim=1)  # B x 3

    direction_x = torch.tensor([1.0, 0, 0.0], dtype=torch.float, device=device)
    direction_y = torch.tensor([0, 1.0, 0.0], dtype=torch.float, device=device)
    angle = torch.rand(batch_size, device=device) * 2 * math.pi
    translation = (
                          direction_x * torch.cos(angle).unsqueeze(-1)
                          + direction_y * torch.sin(angle).unsqueeze(-1) * scale_y
                  ) * offset / focal  # B x 3

    extrinsics[:, :3, 3] += translation
    return CameraPytorch3d(extrinsics=extrinsics, intrinsics=camera.intrinsics)


def relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([width, height, 1.]).view(-1, 1)
    return intrinsic * scaling


def absolute_intrinsic_to_relative(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([width, height, 1.]).view(-1, 1)
    return intrinsic / scaling


def crop_center_intrinsic(
        intrinsic: torch.Tensor,
        crop_size,
        image_size,
        relative_intrinsics = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Central crop the absolute or relative intrinsics

    Args:
        intrinsic: 3 x 3
        crop_size: [h, w]
        image_size: [h, w]
        relative_intrinsics: if True work with relative intrinsics format

    Returns:
        cropped_intrinsic: 3 x 3
    """
    height, width = image_size
    if relative_intrinsics:
        intrinsic = relative_intrinsic_to_absolute(height, width, intrinsic)

    crop_height, crop_width = crop_size

    crop_x = math.floor((width - crop_width) / 2)
    crop_y = math.floor((height - crop_height) / 2)

    pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float).view(1, 1, -1)
    film_coords = coords_pixel_to_film(pixel_coords, intrinsic.unsqueeze(0))[0, 0]
    new_principal_point = - film_coords * torch.diagonal(intrinsic[:-1, :-1], dim1=0, dim2=1)
    cropped_intrinsic = intrinsic.clone()
    cropped_intrinsic[:-1, -1] = new_principal_point

    if relative_intrinsics:
        cropped_intrinsic = absolute_intrinsic_to_relative(*crop_size, cropped_intrinsic)

    return cropped_intrinsic
