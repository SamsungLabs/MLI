from typing import Union

import numpy as np
import torch



def change_coordinate_system_orientation_metashape(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert extrinsics from metashape system to normal system.

    Normal system where X_w = R^T * (X_c - T)
    X_w - points coords in world system
    X_c - points coords in camera system
    R - rotation matrix
    T - translation matrix

    Extrinsic matrix:
            R         T
    r_11 r_12 r_13 | t_1
    r_21 r_22 r_23 | t_2
    r_31 r_32 r_33 | t_3

    Args:
        extrinsic: 4 x 4

    Returns:
        4 x 4
    """

    axis_reversion = np.eye(3)
    axis_reversion[-1][-1] = axis_reversion[-1][-1] * -1

    extrinsic[:3, :3] = axis_reversion.T @ extrinsic[:3, :3] @ axis_reversion
    extrinsic[:3, 3] = axis_reversion @ extrinsic[:3, 3]

    return extrinsic


def change_coordinate_system_orientation_pyrender(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert extrinsics from pyrender system to normal system.
    For details see :func:`~change_coordinate_system_orientation_metashape`

    Args:
        extrinsic: 4 x 4

    Returns:
        4 x 4
    """

    axis_reversion = np.eye(3)
    axis_reversion[1:, 1:] *= -1

    extrinsic[:3, :3] = axis_reversion @ extrinsic[:3, :3].T
    extrinsic[:3, 3] = - axis_reversion @ extrinsic[:3, 3]

    return extrinsic


def coords_cam_to_film(points) -> torch.Tensor:
    """
    Convert from camera coordinates to film coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 3 (batch of cameras x batch of pixels x 3)

    Returns:
        torch.Tensor: Bc x Bp x 2
    """

    assert points.dim() == 3
    denominator = torch.where(points[..., -1:] >= 0,
                              points[..., -1:].clamp(min=1e-6),
                              points[..., -1:].clamp(max=-1e-6),
                              )
    film_coords = points[..., :2] / denominator
    return film_coords


def coords_film_to_pixel(points: torch.Tensor,
                         intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert from film coordinates to pixel coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 2
        intrinsic (torch.Tensor): Bc x 3 x 3

    Returns:
        torch.Tensor: Bc x Bp x 2
    """

    assert points.dim() == intrinsic.dim() == 3
    coords_scaled = points * torch.diagonal(intrinsic[:, :-1, :-1], dim1=1, dim2=2).unsqueeze(1)
    coords_shifted = coords_scaled + intrinsic[:, :-1, -1:].permute(0, 2, 1)
    return coords_shifted


def coords_cam_to_pixel(points: torch.Tensor,
                        intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert from camera coordinates to pixel coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 2
        intrinsic (torch.Tensor): Bc x 3 x 3

    Returns:
        torch.Tensor: Bc x Bp x 2
    """

    film_coords = coords_cam_to_film(points)
    return coords_film_to_pixel(film_coords, intrinsic)


def coords_cam_to_world(points: torch.Tensor,
                        extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert from camera coordinates to world coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 3 or 1 x Bp x 3 or Bp x 3
        extrinsic (torch.Tensor): Bc x 4 x 4

    Returns:
        torch.Tensor: Bc x Bp x 3
    """

    if points.dim() == 2:
        points = points.unsqueeze(0)

    rotation, translation = extrinsic[:, :3, :3], extrinsic[:, :3, -1].unsqueeze(1)
    translated = points - translation
    return torch.einsum('cji,cpj->cpi', [rotation, translated])


def coords_world_to_cam(points: torch.Tensor,
                        extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert from world coordinates to camera coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 3 or 1 x Bp x 3 or Bp x 3 (broadcasting is applied for two latter cases)
        extrinsic (torch.Tensor): Bc x 4 x 4

    Returns:
        torch.Tensor: Bc x Bp x 3
    """

    rotation, translation = extrinsic[:, :3, :3], extrinsic[:, :3, -1].unsqueeze(1)
    if points.dim() == 3 and points.shape[0] == 1:
        points = points.squeeze(0)

    if points.dim() == 3:
        rotated = torch.einsum('cij,cpj->cpi', [rotation, points])
    else:
        rotated = torch.einsum('cij,pj->cpi', [rotation, points])
    return rotated + translation


def coords_pixel_to_film(points: torch.Tensor,
                         intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert from pixel coordinates to film coordinates.

    Args:
        points (torch.Tensor): Bc x Bp x 2
        intrinsic (torch.Tensor): Bc x 3 x 3

    Returns:
        torch.Tensor:  Bc x Bp x 2
    """

    film_coords = (points - intrinsic[:, :-1, -1:].permute(0, 2, 1)) / torch.diagonal(intrinsic[:, :-1, :-1],
                                                                                      dim1=1,
                                                                                      dim2=2).unsqueeze(1)
    return film_coords


def coords_film_to_cam(points: torch.Tensor,
                       depth: Union[torch.Tensor, float, int]) -> torch.Tensor:
    """
    Convert points from film coordinates to camera.
    Since this transformation takes points from 2D space to 3D,
    you need to specify the depth value for each point,
    this value is along the direction axis of the camera’s view.

    Args:
        points (torch.Tensor): Bc x Bp x 2
        depth (Union[torch.Tensor, float, int]): Bc x Bp x 1 or Bc x Bp or 1

    Returns:
        torch.Tensor: Bc x Bp x 3
    """

    if isinstance(depth, (int, float)):
        depth = torch.ones_like(points[..., -1:]) * depth
    elif depth.dim() == 2:
        depth = depth.unsqueeze(-1)
    film_homogenous = torch.cat([points, torch.ones_like(points[..., -1:])], dim=-1)

    return film_homogenous * depth


def coords_pixel_to_cam(points: torch.Tensor,
                        depth: Union[torch.Tensor, float, int],
                        intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert points from pixel coordinates to camera.
    Since this transformation takes points from 2D space to 3D,
    you need to specify the depth value for each point,
    this value is along the direction axis of the camera’s view.

    Args:
        points (torch.Tensor):  Bc x Bp x 2
        depth (Union[torch.Tensor, float, int]): Bc x Bp x 1 or Bc x Bp or 1
        intrinsic (torch.Tensor): Bc x 3 x 3

    Returns:
        torch.Tensor: Bc x Bp x 3
    """

    film_coords = coords_pixel_to_film(points, intrinsic)
    return coords_film_to_cam(film_coords, depth)


def get_cameras_world_positions(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Get cameras position in world coordinates based on cameras extrinsics.

    Args:
        extrinsics (torch.Tensor): Bc x 4 x 4

    Returns:
        torch.Tensor: Bc x 1 x 3 cameras positions in global coord
    """

    x = torch.zeros(extrinsics.shape[0], 1, 3, device=extrinsics.device)
    return coords_cam_to_world(x, extrinsics)
