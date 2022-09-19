__all__ = ['RasterizerFFrast',
           'Fragments']

from typing import Tuple, NamedTuple

import torch
from torch import nn
import nvdiffrast.torch as dr

class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor
    ffrast_rast_out: torch.Tensor

class RasterizerFFrast(nn.Module):
    def __init__(
            self,
            image_size: int = 256,
            scale: int = 100,
            faces_per_pixel: int = 8,
            custom_resolution: Tuple = None,
    ):
        """

        Args:
            image_size: Size in pixels of the output raster image for each mesh
                in the batch. Assumes square images.
            scale: Depth should be in the range (0,1) for rasterizer, so scale should be >= max_depth
        """
        super().__init__()

        self.image_size = image_size
        self.scale = scale
        if custom_resolution is None:
            self.img_resolution = [image_size, image_size]
        else:
            self.img_resolution = custom_resolution
        self.faces_per_pixel = faces_per_pixel
        self.glctx = dr.RasterizeGLContext()

    def forward(self,
                verts: torch.Tensor,
                faces: torch.Tensor,
                custom_resolution: Tuple[int, int]=None,
                ) -> dict:
        """

        Args:
            verts: B x Nv x 3
            faces: Nf x 3 (faces must be equal for all meshes in batch)

        Returns:
          pix_to_face: B x H x W X FACES_PER_PIXEL,
            giving the indices of the nearest faces at each pixel, sorted in ascending z-order.
          zbuf: B x H x W X FACES_PER_PIXEL,
            giving the NDC z-coordinates of the nearest faces at each pixel,
            sorted in ascending z-order. Pixels hit by fewer than faces_per_pixel are padded with -1.
          barycentric: B x H x W X FACES_PER_PIXEL x 3,
            giving the barycentric coordinates in NDC units of the nearest faces at each pixel,
            sorted in ascending z-order. Pixels hit by fewer than faces_per_pixel are padded with -1.
          pix_dists: B x H x W X FACES_PER_PIXEL,
            giving the signed Euclidean distance (in NDC units) in the x/y plane of each point closest to the pixel.
            Pixels hit with fewer than ``faces_per_pixel`` are padded with -1.
        """

        rast_layers = []
        if custom_resolution is not None:
            self.img_resolution = custom_resolution
        with dr.DepthPeeler(self.glctx,
                            verts,
                            faces.type(torch.int32),
                            resolution=self.img_resolution,
                            ) as peeler:
            for i in range(self.faces_per_pixel):
                rast_out, _ = peeler.rasterize_next_layer()
                tmp = rast_out[:, :, :, :].unsqueeze(-2)
                rast_layers.append(tmp)

        rast_out = torch.cat(rast_layers, dim=-2)

        rast_zbuf = rast_out[:, :, :, :, 2] * self.scale
        rast_pix_to_face = rast_out[:, :, :, :, 3]
        rast_u = rast_out[:, :, :, :, 0]
        rast_v = rast_out[:, :, :, :, 1]
        rast_w = 1 - rast_u - rast_v
        rast_bary_coords = torch.stack([rast_u, rast_v, rast_w], dim=-1)

        return {'pix_to_face': rast_pix_to_face,
                'zbuf': rast_zbuf,
                'bary_coords': rast_bary_coords,
                'dists': rast_zbuf,
                'ffrast_rast_out': rast_out}
