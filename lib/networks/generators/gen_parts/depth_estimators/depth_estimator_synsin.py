__all__ = ['DepthEstimatorSynsin',
           'DepthEstimatorSynsinAdaIN',
           'SynSinUnet',
           'DepthEstimatorUnet',
           'DepthEstimatorSynsinLayered',
           'PSVNetStereoMagnification'
           ]

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import networks
from lib.networks.generators import gen_parts
from lib.networks.blocks.adaptive import get_num_adaptive_params, assign_adaptive_params
from lib.networks.blocks.base import ConvBlock, get_activation
from lib.networks.generators.gen_parts import PSVNetStereoMagnification

class SynSinUnet(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
            self,
            input_dim=3,
            dim=32,
            output_dim=3,
            levels=3,
            norm='bn',
            weight_norm='none',
    ):
        super().__init__()
        self.levels = levels

        self.conv1 = ConvBlock(input_dim=input_dim,
                               output_dim=dim,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               activation='lrelu',
                               norm='none',
                               weight_norm=weight_norm)
        self.conv2 = ConvBlock(input_dim=dim,
                               output_dim=dim * 2,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               activation='lrelu',
                               norm=norm,
                               weight_norm=weight_norm)
        self.conv3 = ConvBlock(input_dim=dim * 2,
                               output_dim=dim * 4,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               activation='lrelu',
                               norm=norm,
                               weight_norm=weight_norm)

        self.level_convs = nn.ModuleList([
            ConvBlock(input_dim=dim * 4,
                      output_dim=dim * 8,
                      kernel_size=4,
                      padding=1,
                      stride=2,
                      activation='lrelu',
                      norm=norm,
                      weight_norm=weight_norm)]
        )

        for i in range(levels):
            self.level_convs.append(ConvBlock(input_dim=dim * 8,
                                              output_dim=dim * 8,
                                              kernel_size=4,
                                              padding=1,
                                              stride=2,
                                              activation='lrelu',
                                              norm=norm,
                                              weight_norm=weight_norm)
                                    )

        self.conv_last = ConvBlock(input_dim=dim * 8,
                                   output_dim=dim * 8,
                                   kernel_size=4,
                                   padding=1,
                                   stride=2,
                                   activation='relu',
                                   norm='none',
                                   weight_norm='none')

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.level_dconv = nn.ModuleList([ConvBlock(input_dim=dim * 8,
                                                    output_dim=dim * 8,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=1,
                                                    activation='relu',
                                                    norm=norm,
                                                    weight_norm=weight_norm)])
        for i in range(levels):
            self.level_dconv.append(ConvBlock(input_dim=dim * 8 * 2,
                                              output_dim=dim * 8,
                                              kernel_size=3,
                                              padding=1,
                                              stride=1,
                                              activation='relu',
                                              norm=norm,
                                              weight_norm=weight_norm)
                                    )

        self.dconv4 = ConvBlock(input_dim=dim * 8 * 2,
                                output_dim=dim * 4, kernel_size=3,
                                padding=1,
                                stride=1,
                                activation='relu',
                                norm=norm,
                                weight_norm=weight_norm)
        self.dconv3 = ConvBlock(input_dim=dim * 4 * 2,
                                output_dim=dim * 2,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                activation='relu',
                                norm=norm,
                                weight_norm=weight_norm)
        self.dconv2 = ConvBlock(input_dim=dim * 2 * 2,
                                output_dim=dim, kernel_size=3,
                                padding=1,
                                stride=1,
                                activation='relu',
                                norm=norm)
        self.dconv1 = ConvBlock(input_dim=dim * 2,
                                output_dim=output_dim, kernel_size=3,
                                padding=1,
                                stride=1,
                                activation='none',
                                norm='none',
                                weight_norm='none')

    def forward(self, x, **kwargs):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(x)
        # state size is (num_filters) x 128 x 128
        e2 = self.conv2(e1)
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.conv3(e2)
        # state size is (num_filters x 4) x 32 x 32
        x = e3
        e_levels = []
        for i, conv_layer in enumerate(self.level_convs):
            x = conv_layer(x)
            e_levels.append(x)
        e_levels = e_levels[::-1]

        # No batch norm on output of Encoder
        x = self.conv_last(x)

        # Decoder
        # Deconvolution layers:
        for i, (conv_layer, e_level) in enumerate(zip(self.level_dconv, e_levels)):
            x = conv_layer(self.up(x))
            x = torch.cat((x, e_level), 1)

        d3_ = self.dconv4(self.up(x))
        # state size is (num_filters x 4) x 32 x 32
        d3 = torch.cat((d3_, e3), 1)
        d2_ = self.dconv3(self.up(d3))
        # state size is (num_filters x 2) x 64 x 64
        d2 = torch.cat((d2_, e2), 1)
        d1_ = self.dconv2(self.up(d2))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d1 = torch.cat((d1_, e1), 1)
        d0 = self.dconv1(self.up(d1))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return d0


class DepthEstimatorSynsin(nn.Module):
    def __init__(self,
                 input_dim=3,
                 dim=32,
                 output_dim=1,
                 unet_levels=3,
                 norm="bn",
                 max_depth=100,
                 min_depth=1,
                 weight_norm='none'
                 ):
        # TODO dockstring
        """
        Similar with original paper DepthEstimator.
        In original paper depth value range is sets manual by max_z/min_z.

        Args:
            input_dim: 
            dim:
            output_dim: 
            unet_levels: 
            norm:
            max_depth:
            min_depth:
            weight_norm:
        """
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth

        self.model = SynSinUnet(input_dim=input_dim,
                                dim=dim,
                                output_dim=output_dim,
                                levels=unet_levels,
                                norm=norm,
                                weight_norm=weight_norm)

    def forward(self, images, **kwargs):
        prediction = self.model(images, **kwargs)
        if self.max_depth != 'none' and self.min_depth != 'none':
            return torch.sigmoid(prediction) * (self.max_depth - self.min_depth) + self.min_depth
        else:
            return prediction


class DepthEstimatorSynsinAdaIN(DepthEstimatorSynsin):
    def __init__(self,
                 input_dim=3,
                 dim=32,
                 output_dim=1,
                 unet_levels=3,
                 max_depth=100,
                 min_depth=1,
                 weight_norm='none',
                 adain_net_params=None,
                 ):
        super().__init__(input_dim=input_dim,
                         dim=dim,
                         output_dim=output_dim,
                         unet_levels=unet_levels,
                         norm='adain',
                         max_depth=max_depth,
                         min_depth=min_depth,
                         weight_norm=weight_norm)

        architecture = adain_net_params.pop('architecture')
        num_adain_params = get_num_adaptive_params(self)
        adain_net_params['output_dim'] = num_adain_params
        self.adain_net = getattr(networks.blocks.base, architecture)(**adain_net_params)
        self.style_dim = adain_net_params['input_dim']
        self.pred_adain_params = 'adain'

    def forward(self, images, adain_input, **kwargs):
        if self.pred_adain_params:
            adain_params = self.adain_net(adain_input)
            assign_adaptive_params(adain_params, self)

        depth_tensor = super().forward(images, **kwargs)

        return depth_tensor


class DepthEstimatorUnet(nn.Module):
    def __init__(self,
                 type='SynSinUnet',
                 max_depth=100,
                 min_depth=1,
                 out_act='none',
                 params=None,
                 ):
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.out_act = out_act

        self.model = getattr(gen_parts, type)(**params)

    def forward(self, images, **kwargs):
        prediction = self.model(images)
        if self.max_depth != 'none' and self.min_depth != 'none':
            return torch.sigmoid(prediction) * (self.max_depth - self.min_depth) + self.min_depth
        else:
            if self.out_act != 'none':
                return get_activation(self.out_act, 1)(prediction)
            return prediction

class DepthEstimatorSynsinLayered(SynSinUnet):
    def __init__(self,
                 input_dim=3,
                 dim=32,
                 output_dim=1,
                 unet_levels=3,
                 norm='bn',
                 weight_norm='none',
                 min_depth=1,
                 max_depth=100,
                 ):
        super().__init__(input_dim=input_dim,
                         dim=dim,
                         output_dim=output_dim,
                         levels=unet_levels,
                         norm=norm,
                         weight_norm=weight_norm,
                         )
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.borders_regressor = nn.ModuleList([
            ConvBlock(input_dim=dim * 8,
                      output_dim=dim * 4,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      activation='lrelu',
                      norm=norm,
                      weight_norm=weight_norm),
            ConvBlock(input_dim=dim * 4,
                      output_dim=dim * 2,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      activation='lrelu',
                      norm=norm,
                      weight_norm=weight_norm),
            ConvBlock(input_dim=dim * 2,
                      output_dim=output_dim,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      activation='none',
                      norm='none',
                      weight_norm='none'),
        ])

    def forward(self, x, t=1, **kwargs):
        # Encoder
        # input is (nc) x 256 x 256
        e1 = self.conv1(x)
        # state size is (num_filters) x 128 x 128
        e2 = self.conv2(e1)
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.conv3(e2)
        # state size is (num_filters x 4) x 32 x 32
        x = e3
        e_levels = []
        for i, conv_layer in enumerate(self.level_convs):
            x = conv_layer(x)
            e_levels.append(x)
        e_levels = e_levels[::-1]

        # No batch norm on output of Encoder
        x = self.conv_last(x)

        # Regress segment_len for layers
        segment_len = x
        for layer in self.borders_regressor[:-1]:
            segment_len = layer(segment_len)
        segment_len = segment_len.mean(dim=[-1, -2], keepdim=True)
        segment_len = self.borders_regressor[-1](segment_len)  # B x output_dim x 1 x 1
        segment_len = F.softplus(segment_len)
        if self.min_depth != 'none' and self.max_depth != 'none':
            segment_len = segment_len * (self.max_depth - self.min_depth)
        low_bounds = segment_len.cumsum(dim=1)  # front-to-back
        low_bounds = torch.cat([torch.zeros_like(low_bounds[:, :1]),
                                low_bounds[:, :-1]
                                ], dim=1)
        if self.min_depth != 'none':
            low_bounds += self.min_depth

        # Decoder
        # Deconvolution layers:
        for i, (conv_layer, e_level) in enumerate(zip(self.level_dconv, e_levels)):
            x = conv_layer(self.up(x))
            x = torch.cat((x, e_level), 1)

        d3_ = self.dconv4(self.up(x))
        # state size is (num_filters x 4) x 32 x 32
        d3 = torch.cat((d3_, e3), 1)
        d2_ = self.dconv3(self.up(d3))
        # state size is (num_filters x 2) x 64 x 64
        d2 = torch.cat((d2_, e2), 1)
        d1_ = self.dconv2(self.up(d2))
        # state size is (num_filters) x 128 x 128
        d1 = torch.cat((d1_, e1), 1)
        d0 = self.dconv1(self.up(d1))
        # state size is (nc) x 256 x 256

        depth = torch.sigmoid(d0) / t * segment_len + low_bounds
        return depth.flip(1)  # back-to-front
