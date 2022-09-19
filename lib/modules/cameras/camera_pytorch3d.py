__all__ = ['CameraPytorch3d',
           ]

import logging

import torch

from . import CameraFrustum

logger = logging.getLogger(__name__)


class CameraPytorch3d(CameraFrustum):
    def __init__(self, *args, **kwargs):
        """
        This camera module support only cameras with relative intrinsics.

        Class for working with pytorch3d NDC space.
        This space used for rendering, pytorch3d rasterizers work in this space.

        https://github.com/facebookresearch/pytorch3d/blob/master/docs/notes/renderer_getting_started.md

        Infinity_value and infinity_eps are copied from synsin code.
        Args:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

    def cam_to_opengl(self, points: torch.Tensor) -> torch.Tensor:
        # TODO WTF?
        EPS = 1e-2
        mask = (points[:, :, 2:3].abs() < EPS).detach()
        points[:, :, 2:3][mask] = EPS

        film_coords = self.cam_to_film(points)
        pixel_coords = self.film_to_pixel(film_coords)
        openlg_coords = torch.cat((pixel_coords * 2 - 1, points[..., -1:]), 2)

        return openlg_coords

    def opengl_to_cam(self, openlg_coords: torch.Tensor) -> torch.Tensor:
        pixel_coords = (openlg_coords[..., :2] + 1) / 2
        film_coords = self.pixel_to_film(pixel_coords)
        cam_coords = self.film_to_cam(film_coords, openlg_coords[..., 2:3])

        return cam_coords

    def cam_to_pytorch3d(self, points: torch.Tensor) -> torch.Tensor:
        pytorch3d_coords = self.cam_to_opengl(points)
        pytorch3d_coords[:, :, 1] = - pytorch3d_coords[:, :, 1]
        pytorch3d_coords[:, :, 0] = - pytorch3d_coords[:, :, 0]

        return pytorch3d_coords

    def pytorch3d_to_cam(self, pytorch3d_coords: torch.Tensor) -> torch.Tensor:
        pytorch3d_coords[:, :, 1] = - pytorch3d_coords[:, :, 1]
        pytorch3d_coords[:, :, 0] = - pytorch3d_coords[:, :, 0]

        cam_coords = self.opengl_to_cam(pytorch3d_coords)
        return cam_coords
