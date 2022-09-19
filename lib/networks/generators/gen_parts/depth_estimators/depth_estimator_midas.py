__all__ = ['DepthEstimatorMidas']

import torch
import torch.nn as nn

from lib.modules import MidasNet


class DepthEstimatorMidas(nn.Module):
    def __init__(self, path: str):
        """
        Depth estimator which uses pretrained MIDAS net.
        You must use it with 'frozen' flag, or not:3

        Magic params of shift and scale  are founded empirically.

        Args:
            path: path to MIDAS weights
        """
        super().__init__()
        self.midas = MidasNet(path)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        Args:
            images : B x 3 x H x W normalized from -1 to 1.

        Returns:
            depth : B x 1 x H x W
        """
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
        img_midas = (images * 0.5 + 0.5) * std[None, :, None, None] + mean[None, :, None, None]

        return (4000 / (self.midas.forward(img_midas) + 1) + 1).unsqueeze(1)
