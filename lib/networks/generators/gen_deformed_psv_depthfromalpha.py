__all__ = ['GeneratorDeformedPSVDepthFromAlpha',
           ]

import logging

import torch
from .gen_parts import SurfacesMPI, RendererMeshLayers
try:
    from .gen_parts import RasterizerFFrast
except:
    pass
from .gen_deformed_psv import GeneratorDeformedPSV

logger = logging.getLogger(__name__)


class GeneratorDeformedPSVDepthFromAlpha(GeneratorDeformedPSV):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def _init_modules(self,
                      depth_estimator,
                      psv_net,
                      composer,
                      surfaces,
                      rasterizer,
                      shader,
                      ):
        self.psv_net = psv_net
        self.composer = composer
        self.surfaces: SurfacesMPI = surfaces
        self.renderer = RendererMeshLayers(rasterizer, shader, self.params['num_layers'])

        self.decoder = None
        self.depth_predictor = depth_estimator

    def group_logits_to_depth(self,
                              depth_logits: torch.Tensor,
                              ) -> torch.Tensor:
        """
        Args:
            depth_logits: B x n_psv_layers x H x W
        Returns:
            depth: B x n_num_layers x H x W
        """
        assert depth_logits.ndim == 4
        batch_size, *_, height, width = depth_logits.shape

        planes_groups_alphas = torch.cat(torch.sigmoid(depth_logits).chunk(self.num_layers, dim=1),
                                         dim=0)  # B*n_groups x planes_per_group x H x W
        default_depths = self.surfaces.mpi_depths.view(-1, 1, 1).expand(batch_size, -1, height, width)
        default_depths = torch.cat(default_depths.chunk(self.num_layers, dim=1), dim=0)

        # B*n_groups x planes_per_group x 2 x H x W
        default_depths_with_alphas = torch.cat([default_depths.unsqueeze(2), planes_groups_alphas.unsqueeze(2)], dim=2)

        # B*n_groups x 1 x H x W
        group_depth = self.composer(default_depths_with_alphas[:, None, None], black_background=False)[0][:, 0, 0]
        layered_depth = torch.cat(group_depth.chunk(self.num_layers, dim=0), dim=1)  # B x n_groups x H x W
        return layered_depth

    def _postprocess_depth(self,
                           depth: torch.Tensor,
                           mode=None,
                           ) -> torch.Tensor:
        """
        Postprocess depth and return layers sorted in back-to-front manner (in average).
        This format better suits visualisation purposes.

        Args:
            depth: B x n_psv_layers x H x W

        Returns:
            out: B x num_layers x H x W
        """
        return self.group_logits_to_depth(depth)
