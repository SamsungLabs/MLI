__all__ = ['UNet',
           'DownBlock',
           'UpBlock']

import torch
from torch import nn
from torch.nn import functional as F

from lib.networks.blocks import ConvBlock, DoubleConvBlock
from lib.networks.blocks.base import get_pooling_layer, get_upsample_layer


class DownBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 conv_dim: int = 2,
                 pooling_type: str = 'max',
                 use_residual: bool = False,
                 weight_norm: str = 'none',
                 conv_mode='single'
                 ):
        '''
        Unet decoder downsample block. Rescale factor is 0.5.

        Args:
            input_dim: num input channels.
            output_dim: num output channels.
            norm: list of supported normalizations here :func:`~lib.networks.blocks.base.get_norm_layer`
            activation: list of supported normalizations here :func:`~lib.networks.blocks.base.get_activation`
            conv_dim: dimension of convolutions 3D/2D/1D
            pooling_type: pooling type
            use_residual: enable residual connection between input and output
        '''
        # TODO add rescale factor to this block
        super().__init__()
        layers = []
        self.pool = get_pooling_layer(conv_dim, pooling_type=pooling_type)

        if conv_mode == 'double':
            self.conv = DoubleConvBlock(input_dim,
                                        output_dim,
                                        norm=norm,
                                        activation=activation,
                                        conv_dim=conv_dim,
                                        use_residual=use_residual,
                                        weight_norm=weight_norm)
        else:
            self.conv = ConvBlock(input_dim,
                                  output_dim,
                                  norm=norm,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  activation=activation,
                                  conv_dim=conv_dim,
                                  weight_norm=weight_norm,
                                  )

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 upsampling_type: str = None,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 conv_dim: int = 2,
                 scale_factor: int = 2,
                 align_corners: bool = False,
                 use_residual: bool = False,
                 weight_norm: str = 'none',
                 conv_mode='single'
                 ):
        '''
        Unet encoder upconvolution block.

        Args:
            input_dim: num input channels.
            output_dim: num output channels.
            upsampling_type: list of supported normalizations here :func:`~lib.networks.blocks.base.get_upsample_layer`
            norm: list of supported normalizations here :func:`~lib.networks.blocks.base.get_norm_layer`
            activation: list of supported normalizations here :func:`~lib.networks.blocks.base.get_activation`
            conv_dim: dimension of convolutions 3D/2D/1D
            scale_factor: upsample scale
            align_corners: align corners
            use_residual: enable residual connection between input and output
            conv_mode: Use DoubleConv blocks or single conv
        '''
        # TODO add support of 1D/2D
        super().__init__()
        assert conv_dim in {1, 2, 3}, f'Unsupported conv_dim={conv_dim}'

        self.conv_dim = conv_dim
        self.up = get_upsample_layer(scale_factor=scale_factor,
                                     dim=conv_dim,
                                     mode=upsampling_type,
                                     align_corners=align_corners)

        if conv_mode == 'double':
            self.conv = DoubleConvBlock(input_dim,
                                        output_dim,
                                        norm=norm,
                                        activation=activation,
                                        conv_dim=conv_dim,
                                        use_residual=use_residual,
                                        weight_norm=weight_norm)
        else:
            self.conv = ConvBlock(input_dim,
                                  output_dim,
                                  norm=norm,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  activation=activation,
                                  conv_dim=conv_dim,
                                  weight_norm=weight_norm,
                                  )

    def forward(self, x1, residual):
        x1 = self.up(x1)
        padding = []
        for i in range(1, self.conv_dim + 1):
            dim_diff = residual.size()[-i] - x1.size()[-i]
            padding += [dim_diff // 2] * 2
        x1 = F.pad(x1, padding)
        x = torch.cat([residual, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_encoding_blocks: int,
            norm: str = 'ln',
            activation: str = 'relu',
            conv_dim: int = 2,
            pooling_type: str = 'max',
            use_res_blocks: bool = False,
            weight_norm: str = 'none',
            conv_mode: str = 'single'
    ):
        super().__init__()
        self.encoding_blocks = nn.ModuleList()
        self.input_conv = DoubleConvBlock(input_dim,
                                          inner_dim,
                                          conv_dim=conv_dim,
                                          norm='none',
                                          activation=activation,
                                          use_residual=use_res_blocks,
                                          weight_norm=weight_norm
                                          )

        inner_input_dim = inner_dim
        inner_output_dim = 2 * inner_dim
        for _ in range(num_encoding_blocks):
            encoding_block = DownBlock(
                inner_input_dim,
                inner_output_dim,
                norm=norm,
                activation=activation,
                conv_dim=conv_dim,
                pooling_type=pooling_type,
                use_residual=use_res_blocks,
                weight_norm=weight_norm,
                conv_mode=conv_mode
            )

            self.encoding_blocks.append(encoding_block)
            self.output_dim = inner_output_dim

            inner_input_dim = inner_output_dim
            inner_output_dim *= 2

    def forward(self, x):
        skip_connections = []
        x = self.input_conv(x)
        skip_connections.append(x)
        for encoding_block in self.encoding_blocks:
            skip_connections.append(x)
            x = encoding_block(x)
        return skip_connections, x


class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_decoding_blocks: int,
                 upsampling_type: str = None,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 conv_dim: int = 2,
                 use_res_blocks: bool = False,
                 weight_norm: str = 'none',
                 conv_mode: str = 'single'
                 ):
        super().__init__()

        self.decoding_blocks = nn.ModuleList()
        for _ in range(num_decoding_blocks):
            input_dim //= 2

            decoding_block = UpBlock(
                input_dim * (1 + 2),
                input_dim,
                upsampling_type,
                conv_dim=conv_dim,
                activation=activation,
                norm=norm,
                use_residual=use_res_blocks,
                weight_norm=weight_norm,
                conv_mode=conv_mode
            )
            self.decoding_blocks.append(decoding_block)

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for i, (skip_connection, decoding_block) in enumerate(zipped):
            x = decoding_block(x, skip_connection)
        return x


class UNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 inner_dim: int,
                 conv_dim: int = 2,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 upsampling_type: str = None,
                 num_downsamples: int = 4,
                 use_res_blocks: bool = False,
                 weight_norm: str = 'none',
                 conv_mode: str = 'single',
                 ):
        super().__init__()

        supported_dimensions = (2, 3)
        if conv_dim not in (2, 3):
            raise ValueError(f'Dimension: {conv_dim} is unsupported, only: {supported_dimensions} supported.')

        self.encoder = Encoder(input_dim,
                               inner_dim,
                               num_downsamples,
                               norm=norm,
                               activation=activation,
                               conv_dim=conv_dim,
                               use_res_blocks=use_res_blocks,
                               weight_norm=weight_norm,
                               conv_mode=conv_mode)

        self.decoder = Decoder(self.encoder.output_dim,
                               num_downsamples,
                               upsampling_type,
                               conv_dim=conv_dim,
                               norm=norm,
                               activation=activation,
                               use_res_blocks=use_res_blocks,
                               weight_norm=weight_norm,
                               conv_mode=conv_mode)

        self.final_conv = ConvBlock(inner_dim,
                                    output_dim,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    conv_dim=conv_dim,
                                    norm='none',
                                    activation='none',
                                    weight_norm='none')

    def forward(self, x):
        x = self.decoder(*self.encoder(x))
        out = self.final_conv(x)
        return out
