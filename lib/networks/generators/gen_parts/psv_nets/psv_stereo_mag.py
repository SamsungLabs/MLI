__all__ = ['PSVNetStereoMagnification',
           'VertexNet']

import torch
from torch import nn

from lib.networks.blocks import UNet
from lib.networks.blocks.base import ConvBlock, ConvTransposeBlock


class PSVNetStereoMagnification(nn.Module):
    def __init__(self, input_dim, output_dim, dim, activation='relu'):
        super().__init__()
        self.conv1_1 = ConvBlock(input_dim, dim, 3, stride=1, padding=1, norm='ln', activation=activation)
        self.conv1_2 = ConvBlock(dim, dim * 2, 3, stride=2, padding=1, norm='ln', activation=activation)

        self.conv2_1 = ConvBlock(dim * 2, dim * 2, 3, stride=1, padding=1, norm='ln', activation=activation)
        self.conv2_2 = ConvBlock(dim * 2, dim * 4, 3, stride=2, padding=1, norm='ln', activation=activation)

        self.conv3_1 = ConvBlock(dim * 4, dim * 4, 3, stride=1, padding=1, norm='ln', activation=activation)
        self.conv3_2 = ConvBlock(dim * 4, dim * 8, 3, stride=1, padding=1, norm='ln', activation=activation)
        self.conv3_3 = ConvBlock(dim * 8, dim * 8, 3, stride=2, padding=1, norm='ln', activation=activation)

        self.conv4_1 = ConvBlock(dim * 8, dim * 8, 3, stride=1, dilation=2, padding=2, norm='ln', activation=activation)
        self.conv4_2 = ConvBlock(dim * 8, dim * 8, 3, stride=1, dilation=2, padding=2, norm='ln', activation=activation)
        self.conv4_3 = ConvBlock(dim * 8, dim * 8, 3, stride=1, dilation=2, padding=2, norm='ln', activation=activation)

        self.upconv5_1 = ConvTransposeBlock(dim * 16, dim * 4, 4, stride=2, padding=1, norm='ln', activation=activation)
        self.upconv5_2 = ConvTransposeBlock(dim * 4, dim * 4, 3, stride=1, padding=1, norm='ln', activation=activation)
        self.upconv5_3 = ConvTransposeBlock(dim * 4, dim * 4, 3, stride=1, padding=1, norm='ln', activation=activation)

        self.upconv6_1 = ConvTransposeBlock(dim * 8, dim * 2, 4, stride=2, padding=1, norm='ln', activation=activation)
        self.upconv6_2 = ConvTransposeBlock(dim * 2, dim * 2, 3, stride=1, padding=1, norm='ln', activation=activation)

        self.upconv7_1 = ConvTransposeBlock(dim * 4, dim, 4, stride=2, padding=1, norm='ln', activation=activation)
        self.upconv7_2 = ConvTransposeBlock(dim, dim, 1, stride=1, padding=0, norm='ln', activation=activation)

        self.final_conv = ConvBlock(dim, output_dim, 1, 1, norm='none', activation='none')

    def forward(self, x):
        c11 = self.conv1_1(x)
        c12 = self.conv1_2(c11)
        c21 = self.conv2_1(c12)
        c22 = self.conv2_2(c21)
        c31 = self.conv3_1(c22)
        c32 = self.conv3_2(c31)
        c33 = self.conv3_3(c32)
        c41 = self.conv4_1(c33)
        c42 = self.conv4_2(c41)
        c43 = self.conv4_3(c42)

        skip1 = torch.cat([c43, c33], dim=1)
        c51 = self.upconv5_1(skip1)
        c52 = self.upconv5_2(c51)
        c53 = self.upconv5_3(c52)

        skip2 = torch.cat([c53, c22], dim=1)

        c61 = self.upconv6_1(skip2)
        c62 = self.upconv6_2(c61)

        skip3 = torch.cat([c62, c12], dim=1)

        c71 = self.upconv7_1(skip3)
        c72 = self.upconv7_2(c71)

        return self.final_conv(c72)


class VertexNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dim,
                 unet_levels=4,
                 norm='ln',
                 conv_dim=2,
                 padding=1,
                 activation='relu',
                 upsampling_type='bilinear',
                 use_res_blocks=False,
                 weight_norm='none'
                 ):
        """
        Similar with original paper PSVnet
        """
        super().__init__()

        self.model = UNet(input_dim=input_dim,
                          output_dim=output_dim,
                          inner_dim=dim,
                          conv_dim=conv_dim,
                          norm=norm,
                          activation=activation,
                          upsampling_type=upsampling_type,
                          num_downsamples=unet_levels,
                          use_res_blocks=use_res_blocks,
                          weight_norm=weight_norm
                          )

    def forward(self, images):
        return self.model(images)
