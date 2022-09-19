"""Custom normalization layers"""

__all__ = ['LayerNormTotal',
           'LayerNormChannels',
           ]

import torch
from torch import nn


class LayerNormTotal(nn.Module):
    """LayerNorm that acts on all the axes except the batch axis"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNormTotal, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.rand(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.shape[0] == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.shape[0], -1).mean(1).view(*shape)
            std = x.view(x.shape[0], -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class LayerNormChannels(nn.LayerNorm):
    """LayerNorm that acts on the channels dimension only (i.e. for tensors of shape B x C x ...)"""

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        permutation = [0] + list(range(2, x.dim())) + [1]  # e.g., for images: [0, 2, 3, 1]
        x = super().forward(x.permute(*permutation).contiguous())
        permutation = [0, x.dim() - 1] + list(range(1, x.dim() - 1))  # e.g., for images: [0, 3, 1, 2]
        return x.permute(*permutation).contiguous()
