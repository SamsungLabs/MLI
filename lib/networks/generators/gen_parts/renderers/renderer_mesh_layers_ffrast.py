__all__ = ['RendererMeshLayers']

import logging
from typing import Dict, Tuple, Optional

import torch
from torch import nn

from lib.modules import Textures
from lib.modules.pytorch3d_structures.meshes import Meshes
from lib.networks.generators.gen_parts.rasterizers.rasterizer_ffrast import Fragments
from lib.modules.cameras import CameraPytorch3d

logger = logging.getLogger(__name__)


class RendererMeshLayers(nn.Module):
    """
    Rendering mesh on select views.
    """

    def __init__(self, rasterizer, shader, num_layers):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader
        self.num_layers = num_layers

    def _build_and_transform(self,
                             view_cameras: CameraPytorch3d,
                             verts: torch.Tensor,
                             faces: torch.Tensor,
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare mesh layers for opengl format and batchify for ffrast rasterizer.
        """
        verts_view = verts
        verts_view = view_cameras.world_to_cam(verts_view)
        verts_view = view_cameras.cam_to_opengl(verts_view)
        verts_view[:, :, 2] = verts_view[:, :, 2] / self.rasterizer.scale
        verts_view_cat = torch.cat([verts_view, torch.ones_like(verts_view[:, :, :1])], dim=-1)

        return verts_view_cat, faces, verts_view

    def _rasterize(self,
                   verts: torch.Tensor,
                   faces: torch.Tensor,
                   batch_size: int,
                   custom_resolution: Optional[Tuple[int, int]],
                   ):
        fragments_dict = self.rasterizer(verts, faces[0], custom_resolution)
        fragments_dict['pix_to_face'] = fragments_dict['pix_to_face'].long()
        fragments_dict['pix_to_face'] -= 1
        # fix for continous face indexing
        for i in range(batch_size):
            fragments_dict['pix_to_face'][i][fragments_dict['pix_to_face'][i] != -1] += i * int(faces.shape[1])
        fragments_dict['zbuf'][fragments_dict['pix_to_face'] == -1] = torch.ones_like(fragments_dict['zbuf'][fragments_dict['pix_to_face'] == -1]) * -1
        temp = torch.ones_like(fragments_dict['bary_coords'][fragments_dict['pix_to_face'] == -1]) * -1
        fragments_dict['bary_coords'][fragments_dict['pix_to_face'] == -1] = temp
        fragments_dict['pix_to_face'] = fragments_dict['pix_to_face'].contiguous()
        fragments_dict['zbuf'] = fragments_dict['zbuf'].contiguous()
        fragments_dict['bary_coords'] = fragments_dict['bary_coords'].contiguous()
        fragments_dict['ffrast_rast_out'] = fragments_dict['ffrast_rast_out'].contiguous()
        fragments = Fragments(**fragments_dict)

        return fragments

    def forward(self,
                view_cameras: CameraPytorch3d,
                verts: torch.Tensor,
                faces: torch.Tensor,
                custom_resolution: Optional[Tuple[int, int]] = None,
                **texture_kwargs
                ) -> dict:
        """
        Args:
            view_cameras: cameras for views which are rendered
            verts: mesh vertices
            faces: mesh faces
            texture_kwargs: kwargs for Texture building, see :~pytorch3d.structures.Textures: for details:
                - maps
                - faces_uvs
                - verts_uvs
                - verts_rgb

        Returns:
            images: B x C x H x W, result images (or features maps)
            opacity: B x 1 x H x W, pixel opacity (if it's zero, ray correspondent to pixel
                haven't intersections with mesh)
            texel_weights: B x N x H x W, weight for each intersection in the resulting color value of the pixel
            fragments: the output of rasterization.
        """
        batch_size = verts.shape[0]
        verts_view, faces_view, verts_view_nocat = self._build_and_transform(view_cameras, verts, faces)

        fragments = self._rasterize(verts_view, faces_view, batch_size,
                                    custom_resolution=custom_resolution)
        meshes = Meshes(verts=verts_view_nocat, faces=faces_view)
        meshes.textures = Textures(**texture_kwargs)
        shader_output: Dict[str, torch.Tensor] = self.shader(fragments, meshes)
        shader_output['images'] = shader_output['images'].permute(0, 3, 1, 2)
        shader_output['opacity'] = shader_output['opacity'].permute(0, 3, 1, 2)
        shader_output['texel_weights'] = shader_output['texel_weights'].squeeze(-1).permute(0, 3, 1, 2)
        shader_output['fragments'] = fragments

        return shader_output

    def rasterize(self,
                  view_cameras: CameraPytorch3d,
                  verts: torch.Tensor,
                  faces: torch.Tensor,
                  custom_resolution: Optional[Tuple[int, int]]=None,
                  ) -> Fragments:
        """
        Method for rasterizing mesh, without applying shaders.
        Args:
            view_cameras: cameras for views which are rendered
            verts: mesh vertices
            faces: mesh faces

        Returns:
            fragments: the output of rasterization.

        """

        batch_size = verts.shape[0]
        verts_view, faces_view, _ = self._build_and_transform(view_cameras, verts, faces)
        fragments = self._rasterize(verts_view, faces_view, batch_size, custom_resolution=custom_resolution)

        return fragments