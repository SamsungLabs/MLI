import abc
from typing import Dict, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn

import nvdiffrast.torch as dr

from lib.modules.pytorch3d_structures.meshes import Meshes
from lib.networks.generators.gen_parts.rasterizers.rasterizer_ffrast import Fragments
from lib.networks.blocks.convmlp import ConvMLP

# # TODO must be moved to modern pytorch3d.
# def interpolate_vertex_features(fragments: Fragments,
#                                 meshes: Meshes) -> torch.Tensor:
#     """
#     Detemine the color for each rasterized face. Interpolate the colors for
#     vertices which form the face using the barycentric coordinates.
#     Args:
#         meshes: A Meshes class representing a batch of meshes.
#         fragments: The outputs of rasterization.
#
#     Returns:
#         texels: B x H x W x N x C, There will be one C dimensional value for each element in fragments.pix_to_face
#     """
#     C = meshes.textures.verts_rgb_padded().shape[-1]
#     vertex_textures = meshes.textures.verts_rgb_padded().reshape(-1, C)  # (V, C)
#     vertex_textures = vertex_textures[meshes.verts_padded_to_packed_idx(), :]
#     faces_packed = meshes.faces_packed()
#     faces_textures = vertex_textures[faces_packed]  # (F, 3, C)
#     texels = interpolate_face_attributes(
#         fragments.pix_to_face, fragments.bary_coords, faces_textures
#     )
#     return texels

def interpolate_vertex_features(fragments: Fragments,
                                meshes: Meshes) -> torch.Tensor:
    """
    Detemine the color for each rasterized face. Interpolate the colors for
    vertices which form the face using the barycentric coordinates.
    Args:
        meshes: A Meshes class representing a batch of meshes.
        fragments: The outputs of rasterization.

    Returns:
        texels: B x H x W x N x C, There will be one C dimensional value for each element in fragments.pix_to_face
    """
    # [minibatch_size, height, width, num_attributes]

    C = meshes.textures.verts_rgb_padded().shape[-1]
    vertex_textures = meshes.textures.verts_rgb_padded().reshape(-1, C)  # (V, C)
    vertex_textures = vertex_textures[meshes.verts_padded_to_packed_idx(), :]
    faces_packed = meshes.faces_packed().type(torch.int32)
    _, H, W, N, _ = fragments.ffrast_rast_out.shape
    texels, _ = dr.interpolate(
        attr=vertex_textures,
        rast=fragments.ffrast_rast_out.reshape(1, H, W * N, 4),
        tri=faces_packed,
    )

    return texels.reshape(1, H, W, N, -1)

def features_compose_over_blend(texels: torch.Tensor,
                                black_background: bool = False,
                                ) -> Dict[str, torch.Tensor]:
    """
    MPI like alpha blending.

    Args:
        texels: B x H x W x N x C+1,  C-channels for features and one for opacity for each
            of the top N nearest (front-to-back) faces per pixel.
        black_background: treat last texel as opaque
    Returns:
        images: B x H X W X C, features are blended for each pixel.
        opacity: B x H x W x 1, Zero for pixels which haven't fragments on according ray, except one.
        texel_weights: B x H x W x N x 1
    """

    eps = 1e-6
    opacity = texels[..., [-1]]

    if not black_background:
        opacity[:, :, :, [-1]] = torch.ones_like(opacity[:, :, :, [-1]])

    transmittance = torch.cumprod(1 - opacity + eps, dim=3)
    total_opacity = 1 - transmittance[..., -1, :]
    transmittance = torch.cat([torch.ones_like(transmittance[..., [0], :]), transmittance[..., :-1, :]],
                              dim=-2)
    texel_weights = transmittance * opacity
    result_features = torch.sum(texels[..., :-1] * texel_weights, dim=3)

    return dict(
        images=result_features,
        opacity=total_opacity,
        texel_weights=texel_weights,
        texel_opacity=opacity,
    )


def features_simple_alpha_blend(texels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Simple alpha blending, result_feature = f_1 * a_1 + f_2 * a_2 + ... + f_n * a_n

    Args:
        texels: B x H x W x N x C+1,  C-channels for features and one for opacity for each
            of the top N nearest (front-to-back) faces per pixel.
    Returns:
        images: B x H X W X C, features are blended for each pixel.
        opacity: B x H x W x 1, 0 Zero for pixels which haven't fragments on according ray, except one.
        texel_weights: B x H x W x N x 1
    """

    opacity = texels[..., [-1]]
    result_features = torch.sum(texels[..., :-1] * opacity, dim=3)

    return dict(
        images=result_features,
        opacity=opacity.sum(dim=-2).clamp(0, 1),
        texel_weights=opacity,
    )


class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1.0, 1.0, 1.0)


def softmax_rgb_layered_blend(texels: torch.Tensor,
                              fragments: Fragments,
                              faces_per_layer: int,
                              num_layers: int,
                              znear: float = 1,
                              zfar: float = 100,
                              black_background: bool = False,
                              ) -> Dict[str, torch.Tensor]:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    blend_params = BlendParams()
    num_channels = texels.shape[-1]

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, num_channels), dtype=texels.dtype, device=texels.device)
    background = [1.0] * num_channels
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    layers = []
    for i in range(num_layers):
        faces_min = i * faces_per_layer
        faces_max = (i + 1) * faces_per_layer

        layer_mask = (faces_min <= fragments.pix_to_face) * (fragments.pix_to_face < faces_max)

        z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask * layer_mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma) * layer_mask

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1)[..., None] + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num[..., None] * texels).sum(dim=-2)
        weighted_background = delta * background
        pixel_colors[..., :num_channels] = (weighted_colors + weighted_background) / denom
        layers.append(pixel_colors.unsqueeze(3))

    return features_compose_over_blend(torch.cat(layers, 3), black_background=black_background)


class ShaderBase(nn.Module):
    @staticmethod
    @abc.abstractmethod
    def do_interpolation(fragments: Fragments, meshes: Meshes) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def do_blending(self, texels: torch.Tensor, fragments: Optional[Fragments] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self,
                fragments: Fragments,
                meshes: Meshes,
                ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            images: B x H X W X C, features are blended for each pixel.
            opacity: B x H x W x 1, 0 Zero for pixels which haven't fragments on according ray, except one.
            texel_weights: B x H x W x N x 1
        """

        texels = self.__class__.do_interpolation(fragments, meshes)
        mask = fragments.pix_to_face == -1
        if not getattr(self, 'black_background', True):
            color, opacity = texels[..., :-1], texels[..., -1:]
            layer_nums = torch.arange(fragments.pix_to_face.shape[-1],
                                      dtype=torch.long, device=fragments.pix_to_face.device,
                                      ).expand_as(fragments.pix_to_face)
            last_nonempty_idx = torch.where(mask,
                                            torch.zeros_like(mask, dtype=torch.long),
                                            layer_nums
                                            ).max(dim=-1, keepdim=True)[0]
            opacity = opacity.scatter(dim=-2, index=last_nonempty_idx.unsqueeze(-1), value=1.)
            texels = torch.cat([color, opacity], dim=-1)
        texels[mask] = texels[mask] * 0
        out = self.do_blending(texels, fragments)
        out['texels'] = texels
        return out


class SimpleAlphaShader(ShaderBase):
    """
    Simple alpha blending shader.
    Only feature blending without light.
    """
    do_interpolation = interpolate_vertex_features

    def do_blending(self, texels: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return features_simple_alpha_blend(texels)


class MPIShader(ShaderBase):
    """
    MPI like shader. Vertex features variant.
    Only feature blending without light.
    """

    do_interpolation = interpolate_vertex_features

    def __init__(self, black_background=False):
        """

        Args:
            black_background: treat last texel as opaque
        """
        super().__init__()
        self.black_background = black_background

    def do_blending(self, texels: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return features_compose_over_blend(texels, black_background=True)


class MPIMLPShader(ShaderBase):
    """
    MPI like shader. Vertex features variant.
    Only feature blending without light.
    """

    do_interpolation = interpolate_vertex_features

    def __init__(self,
                 dim_input,
                 faces_per_pixel,
                 mlp_dims,
                 black_background=False):
        """

        Args:
            black_background: treat last texel as opaque
        """
        super().__init__()
        self.black_background = black_background
        self.faces_per_pixel = faces_per_pixel
        self.dim_input = dim_input
        self.mlp_dims = mlp_dims
        self.mlp = ConvMLP(
                input_dim=dim_input * faces_per_pixel,
                dims=self.mlp_dims,
                output_dim=4 * faces_per_pixel,
                activation='elu',
            )

    def do_blending(self, texels: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
          texels: B x H x W x N x C+1,  C-channels for features and one for opacity for each
            of the top N nearest (front-to-back) faces per pixel.
        """
        b, h, w, n, c = texels.shape
        texels = self.mlp(texels.reshape(b, h, w, c * n).permute(0, 3, 1, 2))
        texels = texels.permute(0, 2, 3, 1).reshape(b, h, w, n, 4)
        texels[:, :, :, :, [-1]] = torch.sigmoid(texels[:, :, :, :, [-1]])
        texels[:, :, :, :, -1] = torch.tanh(texels[:, :, :, :, -1])

        return features_compose_over_blend(texels, black_background=True)


class SoftmaxShaderLayered(MPIShader):
    def __init__(self,
                 faces_per_layer: int,
                 num_layers: int,
                 znear: float = 1,
                 zfar: float = 100,
                 black_background: bool = False,
                 ):
        super().__init__()
        self.faces_per_layer = faces_per_layer
        self.num_layers = num_layers
        self.znear = znear
        self.zfar = zfar

        # an ugly work-around to prevent setting some opacities to one
        self._black_background = black_background
        self.black_background = True

    def do_blending(self, texels: torch.Tensor, fragments: Optional[Fragments] = None) -> Dict[str, torch.Tensor]:
        if fragments is None:
            return super().do_blending(texels)
        return softmax_rgb_layered_blend(texels=texels,
                                         fragments=fragments,
                                         faces_per_layer=self.faces_per_layer,
                                         num_layers=self.num_layers,
                                         znear=self.znear,
                                         zfar=self.zfar,
                                         black_background=self._black_background,
                                         )
