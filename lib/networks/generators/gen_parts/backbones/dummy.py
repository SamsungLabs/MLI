"""
This is coded to the original code from the original repository.
https://github.com/facebookresearch/synsin/blob/master/models/networks/architectures.py
"""

__all__ = ['DummyFeatExtractor']

import torch
import torch.nn as nn

from lib.networks.blocks.base import ConvBlock


class DummyFeatExtractor(nn.Module):
    def __init__(
            self,
            half_size_output: bool = False,
            upsample_mode: str = 'bilinear',
    ):
        super().__init__()
        self.half_size_output = half_size_output

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.half_size_output:
            self.dw_input = nn.Upsample(scale_factor=0.5, mode=upsample_mode)

    def forward(self, x_input):
        if self.half_size_output:
            x_input = self.dw_input(x_input)
        return x_input
