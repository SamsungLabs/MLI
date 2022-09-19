__all__ = ['RayAttention',
           'AnchoredRayAttention',
           'FNetAttention',
           'MultiLevelAnchoredRayAttention',
           ]

from typing import Optional

import torch
from torch import nn

from lib.networks.blocks.attention import MultiHeadAttention
from lib.networks.blocks.positional_encoders import PositionalEncoderSelfAttention
from lib.utils import fourier


class RayAttention(nn.Module):
    def __init__(self,
                 dim_input: int,
                 num_heads: int = 1,
                 dim_hidden: int = 4,
                 num_self_att_blocks: int = 1,
                 positional_encoding:bool = False,
                 ):
        super().__init__()

        self.num_self_att_blocks = num_self_att_blocks
        self.positional_encoding = positional_encoding
        if num_self_att_blocks == 1:
            self.ray_attention = MultiHeadAttention(
                num_heads=num_heads,
                dim_input_v=dim_input,
                dim_input_k=dim_input,
                dim_hidden_v=dim_hidden,
                dim_hidden_k=dim_hidden,
                dim_output=dim_input,
                raw_attention=False,
                residual=False,
            )
        else:
            ray_attention_blocks = []
            for i in range(self.num_self_att_blocks):
                ray_attention_blocks.append(MultiHeadAttention(
                    num_heads=num_heads,
                    dim_input_v=dim_input,
                    dim_input_k=dim_input,
                    dim_hidden_v=dim_hidden,
                    dim_hidden_k=dim_hidden,
                    dim_output=dim_input,
                    raw_attention=False,
                    residual=False,
                ))
                self.ray_attention = nn.ModuleList(ray_attention_blocks)

        self.ray_attention_encoder = PositionalEncoderSelfAttention(
            dim=dim_input,
        )

    def forward(self, x, positions):
        x = x.permute(0, 3, 4, 1, 2)
        x_input_shape = x.shape
        x = x.reshape(-1, *x.shape[-2:])

        if self.positional_encoding:
            x = self.ray_attention_encoder(x, position=positions)

        if self.num_self_att_blocks == 1:
            x, _ = self.ray_attention(x, x, x)
        else:
            for i in range(self.num_self_att_blocks):
                x, _ = self.ray_attention[i](x, x, x)

        x = x.reshape(*x_input_shape)
        return x.permute(0, 3, 4, 1, 2)


class AnchoredRayAttention(nn.Module):
    def __init__(self,
                 dim_input: int,
                 num_heads: int = 1,
                 dim_hidden: int = 4,
                 num_anchors: int = 4,
                 num_self_att_blocks: int = 1,
                 permute: bool = True,
                 ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_self_att_blocks = num_self_att_blocks
        if num_self_att_blocks == 1:
            self.ray_attention = MultiHeadAttention(
                num_heads=num_heads,
                dim_input_v=dim_input,
                dim_input_k=dim_input,
                dim_hidden_v=dim_hidden,
                dim_hidden_k=dim_hidden,
                dim_output=dim_input,
                raw_attention=False,
                residual=False,
            )
        else:
            ray_attention_blocks = []
            for i in range(self.num_self_att_blocks):
                ray_attention_blocks.append(MultiHeadAttention(
                    num_heads=num_heads,
                    dim_input_v=dim_input,
                    dim_input_k=dim_input,
                    dim_hidden_v=dim_hidden,
                    dim_hidden_k=dim_hidden,
                    dim_output=dim_input,
                    raw_attention=False,
                    residual=False,
                ))
                self.ray_attention = nn.ModuleList(ray_attention_blocks)

        self.permute = permute
        self.anchor = nn.Parameter(torch.randn([num_anchors, dim_input]))

        self.ray_attention_encoder = PositionalEncoderSelfAttention(
            dim=dim_input,
        )

    def forward(self, x, positions=None):
        if self.permute:
            x = x.permute(0, 3, 4, 1, 2)
        x_input_shape = x.shape
        x = x.reshape(-1, *x.shape[-2:])
        x = self.ray_attention_encoder(x, position=positions)

        if self.num_self_att_blocks == 1:
            x, _ = self.ray_attention(self.anchor.expand(x.shape[0], -1, -1), x, x)
        else:
            for i in range(self.num_self_att_blocks - 1):
                x, _ = self.ray_attention[i](x, x, x)
            x, _ = self.ray_attention[self.num_self_att_blocks - 1](self.anchor.expand(x.shape[0], -1, -1), x, x)

        x = x.reshape(*x_input_shape[:3], self.num_anchors, -1)
        if self.permute:
            x = x.permute(0, 3, 4, 1, 2)
        return x


class MultiLevelAnchoredRayAttention(nn.Module):
    def __init__(self,
                 dim_input: int,
                 num_levels: int,
                 num_heads: int = 1,
                 dim_hidden: int = 4,
                 num_anchors: int = 4,
                 num_self_att_blocks: int = 1,
                 ):
        super().__init__()
        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_levels
        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * num_levels
        if isinstance(num_anchors, int):
            num_anchors = [num_anchors] * num_levels
        if isinstance(num_self_att_blocks, int):
            num_self_att_blocks = [num_self_att_blocks] * num_levels

        self.model = []
        for i in range(num_levels):
            self.model.append(
                AnchoredRayAttention(
                    dim_input=dim_input,
                    num_heads=num_heads[i],
                    dim_hidden=dim_hidden[i],
                    num_anchors=num_anchors[i],
                    num_self_att_blocks=num_self_att_blocks[i],
                    permute=False
                )
            )

        self.model = nn.Sequential(*self.model)

    def forward(self, x, positions):
        x = x.permute(0, 3, 4, 1, 2)
        for module in self.model:
            x = module(x, positions)
            positions = None
        x = x.permute(0, 3, 4, 1, 2)
        return x


class FNetAttention(nn.Module):
    """
    Quasi-self-attention implemented with Fourier transform.

    Proposed by the paper `FNet: Mixing Tokens with Fourier Transforms` https://arxiv.org/abs/2105.03824
    The codebase is provided by
        https://github.com/rishikksh20/FNet-pytorch/blob/6288eca8af31a5c2a2f753c11eb0a56783e9483a/fnet.py
    """

    def __init__(self,
                 dim_input: int,
                 dim_hidden: Optional[int] = None,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.ray_attention_encoder = PositionalEncoderSelfAttention(dim=dim_input)
        self.ln1 = nn.LayerNorm(dim_input)
        self.ln2 = nn.LayerNorm(dim_input)
        if dim_hidden is None:
            dim_hidden = dim_input
        self.fc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x, positions):
        """

        Args:
            x: B x n_source x C x H x W
            positions: B x n_source

        Returns:
            out: B x n_source x C x H x W
        """
        x = x.permute(0, 3, 4, 1, 2)
        x_input_shape = x.shape
        x = x.reshape(-1, *x.shape[-2:])
        x = self.ray_attention_encoder(x, position=positions)

        y = self.ln1(x)
        y = fourier.fft(fourier.FourierOutput(y), signal_ndim=1, normalized=True)
        y = fourier.FourierOutput(y.real.transpose(1, 2), y.imag.transpose(1, 2))
        y = fourier.fft(y, signal_ndim=1, normalized=True).real.transpose(1, 2)
        x = x + y

        x = x + self.fc(self.ln2(x))

        x = x.reshape(*x_input_shape)
        return x.permute(0, 3, 4, 1, 2)
