"""
This is coded to the original code from the original repository.
https://github.com/facebookresearch/synsin/blob/master/models/networks/architectures.py
"""

__all__ = ['SuperPointsFeatExtractor']

import torch
import torch.nn as nn

from lib.networks.blocks.base import ConvBlock


class SuperPointsFeatExtractor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            norm: str = 'ln',
            activation: str = 'relu',
            weight_norm: str = 'none',
            concat_x_to_output: bool = True,
            half_size_output: bool = False,
            upsample_mode: str = 'bilinear',
            kernel_size: int = 3,
    ):
        super().__init__()
        self.concat_x_to_output = concat_x_to_output
        self.half_size_output = half_size_output

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.half_size_output:
            self.up2 = nn.Upsample(scale_factor=1, mode=upsample_mode)
            self.up3 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up4 = nn.Upsample(scale_factor=4, mode=upsample_mode)
            self.dw_input = nn.Upsample(scale_factor=0.5, mode=upsample_mode)
        else:
            self.up2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up3 = nn.Upsample(scale_factor=4, mode=upsample_mode)
            self.up4 = nn.Upsample(scale_factor=8, mode=upsample_mode)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # self.conv1a = ConvBlock(input_dim, c1, kernel_size=3, stride=1, padding=1,
        #                         norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv1a = ConvBlock(input_dim, c1, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv1b = ConvBlock(c1, c1, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv2a = ConvBlock(c1, c2, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv2b = ConvBlock(c2, c2, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv3a = ConvBlock(c2, c3, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv3b = ConvBlock(c3, c3, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv4a = ConvBlock(c3, c4, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv4b = ConvBlock(c4, c4, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)

        self.convDa = ConvBlock(c1 + c2 + c3 + c4, c5, kernel_size=3, stride=1, padding=1,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.convDb = ConvBlock(c5, output_dim, kernel_size=1, stride=1, padding=0,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)

    def forward(self, x_input):
        x = self.conv1a(x_input)
        x1 = self.conv1b(x)
        x = self.pool(x1)
        if self.half_size_output:
            x1 = x
        x = self.conv2a(x)
        x2 = self.conv2b(x)
        x = self.pool(x2)
        x = self.conv3a(x)
        x3 = self.conv3b(x)
        x = self.pool(x3)
        x = self.conv4a(x)
        x4 = self.conv4b(x)

        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = self.up4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        if not self.training:
            del x1, x2, x3, x4
            torch.cuda.empty_cache()

        cDa = self.convDa(x)
        descriptors = self.convDb(cDa)
        if self.concat_x_to_output:
            if self.half_size_output:
                x_input = self.dw_input(x_input)
            return torch.cat([descriptors, x_input], dim=1)
        return descriptors
