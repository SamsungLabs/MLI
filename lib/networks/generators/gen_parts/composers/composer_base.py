__all__ = ['ComposerBase']

from typing import Optional, Tuple

import torch
from torch import nn


class ComposerBase(nn.Module):
    @staticmethod
    def sort_intersections(features: torch.Tensor,
                           timestamps: torch.Tensor,
                           dim: int = 3,
                           ) -> torch.Tensor:
        """
        Sort features from back to front.

        Args:
            features: ... x n_intersections x C x H x W
            timestamps: ... x n_intersections x H x W
            dim: positive number if intersection axis

        Returns:
            sorted_features: ... x n_intersections x C x H x W
        """
        assert dim >= 0
        idx = torch.argsort(timestamps, dim=dim, descending=True)  # ... x n_intersections x H x W
        idx = idx.unsqueeze(dim+1).expand_as(features)
        sorted_features = torch.gather(features, dim=dim, index=idx)
        return sorted_features

    def forward(self,
                features: torch.Tensor,
                timestamps: Optional[torch.Tensor],
                ) -> Tuple[torch.Tensor, ...]:

        raise NotImplementedError
