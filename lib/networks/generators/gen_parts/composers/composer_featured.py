__all__ = ['ComposerFeaturedSurface',
           'ComposerInpaintingStereoMagnification']

from typing import Optional, Tuple

import torch

from .composer_base import ComposerBase
from .composer_stereo_mag import ComposerStereoMagnification
from lib.networks.generators.gen_parts import renderers


class ComposerFeaturedSurface(ComposerBase):
    def __init__(self,
                 render_net: dict,
                 is_3d: bool = False,
                 ):
        super().__init__()
        architecture = render_net.pop('architecture')
        self.render_network = getattr(renderers, architecture)(**render_net)
        self.is_3d = is_3d

    def forward(self,
                features: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                inside_frustum_mask: Optional[torch.BoolTensor] = None,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Compose-over using render Unet.

        Args:
            features: B x n_ref x n_cams x n_intersections x C x H x W
            timestamps: B x n_ref x n_cams x n_intersections x H x W. If None, features are treated as they are already properly ordered
            inside_frustum_mask: B x n_ref x n_cams x 1 x H x W

        Returns:
            out: B x n_ref x n_cams x C-1 x H x W
            opacity: B x n_ref x n_cams x 1 x H x W,  tensor with zeros
        """
        if timestamps is not None:
            features = self.sort_intersections(features, timestamps)

        b, n_ref, n_cams, n_intersections, c, h, w = features.size()

        out = torch.tanh(self.render_network(self.preprocess_features(features)))

        return out.contiguous().view(b, n_ref, n_cams, -1, h, w), torch.zeros(b, n_ref, n_cams, 1, h, w)

    def preprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor in a proper way for renderer
        if 2d render is used: then we merge depth and feature level
        else we use it separately
        Args:
            features: B x n_ref x n_cams x n_intersections x C x H x W

        Returns:
            features: with different shapes
        """
        b, n_ref, n_cams = features.shape[:3]
        features = features.contiguous().view(-1, *features.shape[-4:])
        if self.is_3d:
            return features.transpose(1, 2)
        else:
            return features.contiguous().view(b * n_ref * n_cams, -1, *features.shape[-2:])


class ComposerInpaintingStereoMagnification(ComposerFeaturedSurface, ComposerStereoMagnification):
    def __init__(self,
                 render_net: dict,
                 is_3d: False,
                 black_background: bool = False,
                 ):
        super().__init__(render_net, is_3d=is_3d)
        self.black_background = black_background

    def forward(self,
                features: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                inside_frustum_mask: Optional[torch.BoolTensor] = None,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Compose-over from back to front. The last channel is treated as opacity.

        Args:
            features: B x n_ref x n_cams x n_intersections x C x H x W
            timestamps: B x n_ref x n_cams x n_intersections x H x W.
                If None, features are treated as they are already properly ordered
            inside_frustum_mask: B x n_ref x n_cams x 1 x H x W

        Returns:
            out: B x n_ref x n_cams x C-1 x H x W
            opacity: B x n_ref x n_cams x 1 x H x W
        """
        if timestamps is not None:
            features = self.sort_intersections(features, timestamps)
        color, opacity, transmittance, total_transmittance = self.precompute_features(features)

        out = torch.sum(color * opacity * transmittance, dim=-4)

        b, n_ref, n_cams, c, h, w = out.size()

        outside_mask = 1 - self.preprocess_features(inside_frustum_mask.unsqueeze(-3)).float()
        prerender_out = torch.cat([out.contiguous().view(-1, c, h, w), outside_mask], dim=1)
        out = torch.tanh(self.render_network(prerender_out)).contiguous().view(b, n_ref, n_cams, -1, h, w)
        return out, 1 - total_transmittance.squeeze(-4)
