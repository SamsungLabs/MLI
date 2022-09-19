__all__ = ['FlowEstimatorRAFT']

import torch
import torch.nn as nn

from lib.modules import FlowRAFT


class FlowEstimatorRAFT(nn.Module):
    def __init__(self, path: str):
        """
        Flow estimator which uses pretrained RAFT net.
        You must use it with 'frozen' flag, or not:3

        Args:
            path: path to RAFT weights
        """
        super().__init__()
        self.model = FlowRAFT(path)

    def forward(self,
                image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        """

        Args:
            images : B x 3 x H x W normalized from -1 to 1.

        Returns:
            depth : B x 1 x H x W
        """

        return self.model.forward(image_a, image_b)
