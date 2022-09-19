__all__ = ['GeneratorMPIBase']

from typing import Optional

import torch

from lib.utils.base import get_grid
from .gen_base import GeneratorBase


class GeneratorMPIBase(GeneratorBase):
    def __init__(self, params):
        super().__init__(params=params)
        self.grid = None
        self.multi_reference_cams = params.get('multi_reference_cams', False)

    def _get_grid(self,
                  features: Optional[torch.Tensor],
                  height: Optional[int] = None,
                  width: Optional[int] = None,
                  relative: bool = False,
                  values_range: str = 'tanh',
                  align_corners: bool = True,
                  device: Optional[torch.device] = None,
                  ) -> torch.Tensor:
        """
        Args:
            features: B x n_reference x n_cams x ... x H x W

        Returns:
            grid: B x n_reference x n_cams x H x W x UV
        """
        if self.multi_reference_cams is False and features is not None:
            features = features.unsqueeze(1)

        if features is None:
            batch_size = n_ref = n_cam = 1
        else:
            batch_size, n_ref, n_cam = features.shape[:3]
            height, width = features.shape[-2:]
            device = features.device

        if self.grid is None or tuple(self.grid.shape[-2:]) != (height, width):
            self.grid = get_grid(1, height, width,
                                 relative=relative, values_range=values_range, align_corners=align_corners,
                                 device=device)[:, None, None, ...]

        if self.multi_reference_cams is False:
            return self.grid.expand(batch_size, 1, n_cam, -1, -1, -1)[:, 0]
        else:
            return self.grid.expand(batch_size, n_ref, n_cam, -1, -1, -1)
