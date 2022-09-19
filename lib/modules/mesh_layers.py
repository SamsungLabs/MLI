__all__ = ['build_layers_from_view']

from typing import Tuple

import numpy as np
import torch

from lib.modules.cameras import CameraPytorch3d
from lib.modules import grids_generators


def build_layers_from_view(features_tensor: torch.Tensor,
                           depth_tensor: torch.Tensor,
                           source_cameras: CameraPytorch3d,
                           align_corners: bool = True,
                           only_cloud: bool = False,
                           grid_generator: str = 'gen_quad_planes'
                           ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """
    Get features, depth and camera params for view, and construct mesh which contains several textured layers.
    Uses for construct mesh from view depth.
    Example:
        input:
            features_tensor = image
            depth_tensor = image depth
            source_cameras = image camera
        output:
            vertices, each vertex matched with pixel in image.
            faces, with topology as on the picture, each vertex is a pixel on image.
            features_tensor, features for each vertex

    Args:
        features_tensor: B x N x C x H x W
        depth_tensor: B x N x H x W or B x N x 1 x H x W
        source_cameras:
        align_corners: bool
        only_cloud: return only verts coords
        grid_generator: function for grid generating

    Returns:
        verts: B x N*H*W x XYZ,
            vertices coordinates in world coordinates
        faces: B x 2*N*(H-1)*(W-1) x 3,
            indices of vertices for each face, 2*(H-1)*(W-1) faces per layer
        verts_uvs: B x N*H*W x UV, UV coordinates of vertices
        verts_features: B x N*H*W x C
    """

    batch_size, num_layers, num_channels, height, width = features_tensor.shape
    assert depth_tensor.shape[1] == num_layers

    # (B, N, C, H, W) -> (B, N, C, H*W) -> (B, N, H*W, C) -> [(B, H*W, C), ...] -> (B, N*H*W, C)
    verts_features = features_tensor \
        .contiguous() \
        .view(batch_size, num_layers, num_channels, -1) \
        .transpose(-1, -2) \
        .unbind(1)
    verts_features = torch.cat(verts_features, dim=1)
    grid_generator = getattr(grids_generators, grid_generator)
    verts, faces, verts_uvs = grid_generator(height=height,
                                             width=width,
                                             n_planes=num_layers,
                                             align_corners=align_corners,
                                             device=features_tensor.device)
    verts = verts.repeat(batch_size, 1, 1)

    depth_tensor = depth_tensor.view(batch_size, 1, -1)
    verts = source_cameras.pixel_to_world(verts, depth_tensor.permute(0, 2, 1))

    if not only_cloud:
        faces = faces.repeat(batch_size, 1, 1)
        if verts_uvs is not None:
            verts_uvs = verts_uvs.repeat(batch_size, 1, 1)

    return verts, faces, verts_uvs, verts_features



