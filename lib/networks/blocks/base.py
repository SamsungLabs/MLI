__all__ = ['LinearBlock',
           'MLP',
           'ConvBlock',
           'ResBlock',
           'ResBlocks',
           'ConvTransposeBlock',
           'DoubleConvBlock',
           'SqueezeAndExcitationBlock',
           'Identity',
           'get_activation',
           ]

import math
from numbers import Number
from typing import Optional, Union

import torch
from torch import nn


from lib.networks.blocks.norm import LayerNormTotal, LayerNormChannels
from lib.networks.blocks.adaptive import AdaptiveInstanceNorm

UPSAMPLING_TYPES = {'nearest': 1, 'linear': 1, 'bilinear': 2, 'bicubic': 2, 'trilinear': 3}


def get_norm_layer(norm: str,
                   channels: int,
                   dim: int = 2,
                   ) -> Optional[nn.Module]:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    module = None

    if norm == 'bn':
        if dim == 1:
            module = nn.BatchNorm1d(channels)
        elif dim == 2:
            module = nn.BatchNorm2d(channels)
        elif dim == 3:
            module = nn.BatchNorm3d(channels)

    elif norm == 'in':
        if dim == 1:
            module = nn.InstanceNorm1d(channels)
        elif dim == 2:
            module = nn.InstanceNorm2d(channels)
        elif dim == 3:
            module = nn.InstanceNorm3d(channels)

    elif norm == 'ln':
        module = LayerNormTotal(channels)

    elif norm == 'ln-channels':
        module = LayerNormChannels(channels)

    elif norm == 'adain':
        module = AdaptiveInstanceNorm(channels)

    elif norm in {'none'}:
        module = None

    else:
        raise ValueError(f'Unsupported normalization: {norm}')

    return module


def get_padding_layer(pad_type: str,
                      padding: int,
                      dim: int = 2,
                      ) -> nn.Module:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    module = None

    if pad_type == 'reflect':
        if dim == 1:
            module = nn.ReflectionPad1d(padding)
        elif dim == 2:
            module = nn.ReflectionPad2d(padding)
        elif dim == 3:
            raise ValueError(f'Pytorch does not have ReflectionPad3d module')

    elif pad_type == 'replicate':
        if dim == 1:
            module = nn.ReplicationPad1d(padding)
        elif dim == 2:
            module = nn.ReplicationPad2d(padding)
        elif dim == 3:
            module = nn.ReplicationPad3d(padding)

    elif pad_type == 'zero':
        if dim == 1:
            module = nn.ConstantPad1d(padding, 0.)
        elif dim == 2:
            module = nn.ConstantPad2d(padding, 0.)
        elif dim == 3:
            module = nn.ConstantPad3d(padding, 0.)

    else:
        raise ValueError(f'Unsupported padding type {pad_type}')

    return module


def get_upsample_layer(scale_factor: int,
                       dim: int,
                       mode: str = None,
                       align_corners: bool = False) -> nn.Module:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    if mode is None:
        if dim == 1:
            mode = 'linear'
        elif dim == 2:
            mode = 'bilinear'
        elif dim == 3:
            mode = 'trilinear'

    if mode not in UPSAMPLING_TYPES.keys():
        raise ValueError(f'Upsampling type: {mode} is unsupported \
               but should be one of the following: {UPSAMPLING_TYPES}')
    if UPSAMPLING_TYPES[mode] != dim:
        raise ValueError(f'Upsampling type: {mode} is wrong \
                           for dim: {dim} ')

    module = nn.Upsample(scale_factor=scale_factor,
                         mode=mode,
                         align_corners=align_corners)
    return module


def set_weight_normalization(module: nn.Module,
                             norm: Optional[str]) -> nn.Module:
    if norm == 'sn':
        return nn.utils.spectral_norm(module)

    elif norm in {'none', None}:
        return module

    else:
        raise ValueError(f'Unsupported weight normalization: {norm}')


def get_pooling_layer(dimensions: int,
                      pooling_type: str,
                      kernel_size: int = 2,
                      ) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_pool = getattr(nn, class_name)
    return class_pool(kernel_size)


def get_activation(act: str,
                   dim: int = 1,
                   ) -> Optional[nn.Module]:
    if act == 'relu':
        module = nn.ReLU(inplace=True)
    elif act == 'lrelu':
        module = nn.LeakyReLU(0.2, inplace=True)
    elif act == 'prelu':
        module = nn.PReLU()
    elif act == 'elu':
        module = nn.ELU(inplace=True)
    elif act == 'tanh':
        module = nn.Tanh()
    elif act == 'sigmoid':
        module = nn.Sigmoid()
    elif act == 'log_sigmoid':
        module = nn.LogSigmoid()
    elif act == 'softplus':
        module = nn.Softplus()
    elif act == 'softmax':
        module = nn.Softmax(dim)
    elif act == 'log_softmax':
        module = nn.LogSoftmax(dim)
    elif act == 'none':
        module = None
    else:
        raise ValueError(f'Unsupported act: {act}')

    return module


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', use_bias=True, weight_norm='none'):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.fc = set_weight_normalization(self.fc, norm=weight_norm)

        self.norm = get_norm_layer(norm, output_dim, 1)
        self.activation = get_activation(activation, -1)

    def forward(self, x):
        out = self.fc(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_blocks, norm='none', activation='relu', flatten_batch_dims=True):
        super().__init__()
        self.flatten_batch_dims = flatten_batch_dims
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activation)]
        for i in range(num_blocks - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activation)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        if self.flatten_batch_dims:
            x = x.view(x.shape[0], -1)
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 dilation: int = 1,
                 pad_type: str = 'zero',
                 norm: str = 'none',
                 activation: str = 'relu',
                 use_bias: bool = True,
                 conv_dim: int = 2,
                 weight_norm: str = 'none'
                 ):
        super().__init__()
        self.pad = get_padding_layer(pad_type, padding, conv_dim)
        # TODO padding is redundant here
        assert conv_dim in {1, 2, 3}, f'Unsupported conv_dim={conv_dim}'
        if conv_dim == 1:
            conv = nn.Conv1d
        elif conv_dim == 2:
            conv = nn.Conv2d
        elif conv_dim == 3:
            conv = nn.Conv3d

        self.conv = conv(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=use_bias)
        self.conv = set_weight_normalization(self.conv, norm=weight_norm)

        self.norm_type = norm
        self.norm = get_norm_layer(norm, output_dim, conv_dim)

        self.activation = get_activation(activation, 1)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SqueezeAndExcitationBlock(nn.Module):
    """
    Based  on the paper:

    Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://dx.doi.org/10.1007/978-3-030-00928-1_48

    Spatial SE (general case) - average all the pixels for each channel and calculate per-channel gating.
    Channel SE (segmentation-like case) - collapse all the channels into the single one and calculate
        per-spatial position gating.
    """

    def __init__(self,
                 input_dim: int,
                 ratio: Union[Number, str, None] = None,
                 channel_se: bool = False,
                 conv_dim: int = 2,
                 ):
        super().__init__()
        if isinstance(ratio, Number):
            dim = math.ceil(input_dim / ratio)
            global_pooling = getattr(nn, f'AdaptiveAvgPool{conv_dim:d}d')(1)
            self.spatial_se = nn.Sequential(global_pooling,
                                            ConvBlock(input_dim, dim, 1, 1, 0, activation='relu', conv_dim=conv_dim),
                                            ConvBlock(dim, input_dim, 1, 1, 0, activation='sigmoid', conv_dim=conv_dim),
                                            )
        else:
            self.spatial_se = None

        if channel_se:
            self.channel_se = ConvBlock(input_dim, 1, 1, 1, 0, activation='sigmoid', conv_dim=conv_dim)
        else:
            self.channel_se = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channel_se is None and self.spatial_se is None:
            return tensor
        else:
            gate = 0.
            if self.channel_se is not None:
                gate = gate + self.channel_se(tensor)
            if self.spatial_se is not None:
                gate = gate + self.spatial_se(tensor)
            return tensor * gate


class ResBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 inner_dim: Optional[int] = None,
                 norm: str = 'in',
                 activation: str = 'relu',
                 pad_type: str = 'zero',
                 res_off: bool = False,
                 spatial_se_ratio: Union[float, str, None] = None,
                 channel_se: bool = False,
                 conv_dim: int = 2,
                 ):
        super().__init__()
        self.res_off = res_off
        if inner_dim is None:
            inner_dim = dim
        self.model = nn.ModuleList([
            ConvBlock(dim, inner_dim, 3, 1, 1,
                      norm=norm, activation=activation, pad_type=pad_type, conv_dim=conv_dim),
            ConvBlock(inner_dim, dim, 3, 1, 1,
                      norm=norm, activation='none', pad_type=pad_type, conv_dim=conv_dim),
        ])
        if not isinstance(spatial_se_ratio, Number) and not channel_se:
            self.se = None
        else:
            self.se = SqueezeAndExcitationBlock(input_dim=dim,
                                                ratio=spatial_se_ratio,
                                                channel_se=channel_se,
                                                conv_dim=conv_dim,
                                                )

    def forward(self, x):
        residual = x
        for layer in self.model:
            x = layer(x)
        if self.se is not None:
            x = self.se(x)
        if self.res_off:
            return x
        else:
            return x + residual


class ResBlocks(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 dim: int,
                 inner_dim: int = None,
                 output_dim: int = None,
                 norm: str = 'in',
                 activation: str = 'relu',
                 pad_type: str = 'zero',
                 conv_dim: int = 2,
                 spatial_se_ratio: Union[float, str, None] = None,
                 channel_se: bool = False,
                 ):
        super().__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    inner_dim=inner_dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type,
                                    conv_dim=conv_dim,
                                    spatial_se_ratio=spatial_se_ratio,
                                    channel_se=channel_se)]

        if output_dim is not None:
            self.model += [ConvBlock(dim,
                                     output_dim,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     conv_dim=conv_dim,
                                     norm='none',
                                     activation='none',
                                     weight_norm='none')]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 output_padding: int = 0,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 conv_dim: int = 2,
                 ):
        super().__init__()
        assert conv_dim in {1, 2, 3}, f'Unsupported conv_dim={conv_dim}'
        if conv_dim == 1:
            conv = nn.ConvTranspose1d
        elif conv_dim == 2:
            conv = nn.ConvTranspose2d
        elif conv_dim == 3:
            conv = nn.ConvTranspose3d

        self.conv = conv(input_dim, output_dim, kernel_size,
                         stride=stride,
                         padding=padding,
                         output_padding=output_padding,
                         )
        self.norm = get_norm_layer(norm, output_dim, conv_dim)
        self.activation = get_activation(activation, 1)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 inner_dim: int = None,
                 conv_dim: int = 2,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 pad_type: str = 'replicate',
                 output_activation: bool = True,
                 output_norm: bool = True,
                 use_residual: bool = False,
                 weight_norm: str = 'none'
                 ):
        """
        Generalized block with two convolutions.

        Args:
            input_dim: num input channels.
            output_dim: num output channels.
            inner_dim: num channels after first convolution.
            conv_dim: dimension of convolutions 3D/2D/1D
            norm: list of supported normalizations here :func:`~lib.networks.blocks.base.get_norm_layer`
            activation: list of supported normalizations here :func:`~lib.networks.blocks.base.get_activation`
            output_activation: If output activation is not needed, set False, else it is set same as activation.
            output_norm: If output normalization is not needed, set False, else it is set same as normalization.
            use_residual: enable residual connection between input and output
            weight_norm: type of weight normalization
        """
        super().__init__()
        self.output_activation = None
        self.use_residual = use_residual
        self.residual_conv = None

        if output_activation:
            self.output_activation = get_activation(activation)
        if output_norm:
            output_norm = norm
        else:
            output_norm = 'none'

        if inner_dim is None:
            inner_dim = output_dim
        if output_dim != input_dim:
            self.residual_conv = ConvBlock(input_dim, output_dim,
                                           kernel_size=1, stride=1, padding=0,
                                           norm='none', activation='none',
                                           conv_dim=conv_dim,
                                           weight_norm=weight_norm)

        self.double_conv = nn.Sequential(
            ConvBlock(input_dim, inner_dim,
                      kernel_size=3, stride=1, padding=1,
                      pad_type=pad_type, norm=norm, activation=activation,
                      conv_dim=conv_dim,
                      weight_norm=weight_norm),
            ConvBlock(inner_dim, output_dim,
                      kernel_size=3, stride=1, padding=1,
                      pad_type=pad_type, norm=output_norm, activation='none',
                      conv_dim=conv_dim,
                      weight_norm=weight_norm)
        )

    def forward(self, x):
        if self.use_residual:
            residual = x
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x = self.double_conv(x)
            output = x + residual
        else:
            output = self.double_conv(x)

        if self.output_activation is not None:
            return self.output_activation(output)
        else:
            return output
