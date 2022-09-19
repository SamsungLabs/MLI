__all__ = ['ConvMLP',
           'ConvMLPResidual']

import torch
from torch import nn

from lib.networks.blocks import ConvBlock


class ConvMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 dims: list,
                 output_dim: int,
                 activation: str = 'relu',
                 weight_norm: str = 'none',
                 norm: str = 'none',
                 out_activation: str = 'none'
                 ):
        super().__init__()
        self.model = []

        input_dims = [input_dim] + dims
        output_dims = dims + [output_dim]

        for block_id in range(len(input_dims) - 1):
            self.model += [ConvBlock(
                input_dim=input_dims[block_id],
                output_dim=output_dims[block_id],
                norm=norm,
                activation=activation,
                weight_norm=weight_norm,
                kernel_size=1,
                stride=1,
            )]

        self.model += [ConvBlock(
            input_dim=input_dims[-1],
            output_dim=output_dims[-1],
            norm=norm,
            activation=out_activation,
            weight_norm=weight_norm,
            kernel_size=1,
            stride=1,
        )]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ConvMLPResidual(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dims_stage1: list,
                 dims_stage2: list,
                 norm: str = 'none',
                 activation: str = 'relu',
                 weight_norm: str = 'none',
                 ):
        super().__init__()

        self.stage_1 = ConvMLP(
            input_dim=input_dim,
            output_dim=dims_stage1[-1],
            dims=dims_stage1[:-1],
            norm=norm,
            activation=activation,
            weight_norm=weight_norm,
            out_activation=activation
        )

        self.stage_2 = ConvMLP(
            input_dim=dims_stage1[-1] + input_dim,
            output_dim=output_dim,
            dims=dims_stage2,
            norm=norm,
            activation=activation,
            weight_norm=weight_norm,
            out_activation='none'
        )

    def forward(self, x):
        y = self.stage_1(x)
        out = self.stage_2(torch.cat([y, x], dim=1))
        return out
