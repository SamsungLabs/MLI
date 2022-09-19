__all__ = ['RenderUNet2d',
           'RenderUNet3d',
           ]

from typing import Optional

import torch
import torch.nn as nn
from lib.networks.blocks import ConvBlock
from lib.networks.blocks.unet import Encoder, Decoder


class DotProductAttention(nn.Module):
    def __init__(self, num_features: Optional[int] = None):
        super().__init__()
        self.use_projection = False
        if num_features is not None:
            self.use_projection = True
            self.w_q = nn.Linear(num_features, num_features)
            self.w_k = nn.Linear(num_features, num_features)
            self.w_v = nn.Linear(num_features, num_features)

    def forward(self, query, key, value):
        if self.use_projection:
            query = self.w_q(query)
            key = self.w_k(key)
            value = self.w_v(value)

        out = torch.softmax(torch.bmm(key, query.transpose(1, 2)), dim=-1)
        return torch.bmm(out, value)


class DepthPool(nn.Module):
    def __init__(self, pool_type: str = 'mean',
                 num_features: Optional[int] = None):
        super().__init__()
        self.pool_type = pool_type

        if self.pool_type == 'attention':
            self.attn = DotProductAttention(num_features)

    def forward(self, features, dim=2):
        if self.pool_type == 'max':
            return features.max(dim, keepdim=True)[0]
        elif self.pool_type == 'mean':
            return features.mean(dim, keepdim=True)
        elif self.pool_type == 'attention':
            b, c, d, h, w = features.size()
            rebatched_features = features.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, d, c)
            attn_score = self.attn(rebatched_features, rebatched_features, rebatched_features)
            return attn_score.mean(1).contiguous().view(b, h, w, c, 1).permute(0, 3, 4, 1, 2)
        else:
            raise ValueError(f'Unsupported depth pooling {self.pool_type} type')


class RenderUNet2d(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 dim,
                 conv_dim=2,
                 norm='ln',
                 activation='relu',
                 upsampling_type='bilinear',
                 num_downsamples=4,
                 ):
        super().__init__()
        self.encoder = Encoder(input_dim,
                               dim,
                               num_downsamples,
                               norm=norm,
                               activation=activation,
                               conv_dim=conv_dim)

        self.decoder = Decoder(self.encoder.output_dim,
                               num_downsamples,
                               upsampling_type,
                               conv_dim=conv_dim,
                               norm=norm,
                               activation=activation)

        self.final_conv = ConvBlock(dim, output_dim, 1, 1, norm='none', activation='none')

    def forward(self, x):
        x = self.decoder(*self.encoder(x))
        out = self.final_conv(x)
        return out


class RenderUNet3d(RenderUNet2d):
    def __init__(self, input_dim,
                 output_dim,
                 dim=5,
                 norm='ln',
                 activation='relu',
                 upsampling_type='trilinear',
                 num_downsamples=3,
                 depth_pool_type='max',
                 ):
        super().__init__(input_dim, output_dim, dim,
                         norm=norm,
                         activation=activation,
                         conv_dim=3,
                         upsampling_type=upsampling_type,
                         num_downsamples=num_downsamples)

        self.pool_depth = DepthPool(depth_pool_type, num_features=dim)

        self.final_conv = ConvBlock(dim, output_dim, 1, 1, pad_type='replicate',
                                    norm='none', activation='none', conv_dim=2)

    def forward(self, x):
        x = self.decoder(*self.encoder(x))
        x = self.pool_depth(x, dim=2)
        out = self.final_conv(x.squeeze(2))
        return out
