__all__ = ['ScaledDotProductAttention',
           'MultiHeadAttention',
           'SimpleAttention',
           'MultiHeadSelfAttention',
           'MultiHeadSelfAttentionUniversal'
           ]

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.networks.blocks.positional_encoders import PositionalEncoderSelfAttention
from lib.utils.base import product

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self,
                 temperature: float,
                 ):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class SimpleAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 ):
        super().__init__()
        self.temperature=dim ** 0.5

    def forward(self, q, k, v, mask=None):
        """
         Args:
               q: B x N_Q x C
               k: B x N_V x C
               v: B x N_V x C
               mask: B x N_Q x N_V
           Returns:
               output: B x N_Q x C
               attn: B x N_Q x N_V
        """
        attn = torch.matmul(q / self.temperature, k.permute(0, 2, 1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,
                 num_heads: int,
                 dim_input_v: int,
                 dim_input_k: int,
                 dim_hidden_v: int,
                 dim_hidden_k: int,
                 dim_output: int,
                 residual: bool = True,
                 raw_attention: bool = False,
                 ):
        super().__init__()

        self.residual = residual
        self.num_heads = num_heads
        self.dim_hidden_k = dim_hidden_k
        self.dim_hidden_v = dim_hidden_v
        self.raw_attention = raw_attention

        self.w_qs = nn.Linear(dim_input_k, num_heads * dim_hidden_k, bias=False)
        self.w_ks = nn.Linear(dim_input_k, num_heads * dim_hidden_k, bias=False)
        self.w_vs = nn.Linear(dim_input_v, num_heads * dim_hidden_v, bias=False)
        self.fc = nn.Linear(num_heads * dim_hidden_v, dim_output, bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_hidden_k ** 0.5)

        self.layer_norm = nn.LayerNorm(dim_output, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        if self.residual:
            residual = v

        # Pass through the pre-attention projection: b x lq x (num_heads * dv)
        # Separate different heads: b x lq x num_heads x dv
        q = self.w_qs(q).view(batch_size, len_q, self.num_heads, self.dim_hidden_k)
        k = self.w_ks(k).view(batch_size, len_k, self.num_heads, self.dim_hidden_k)
        v = self.w_vs(v).view(batch_size, len_v, self.num_heads, self.dim_hidden_v)

        # Transpose for attention dot product: b x num_heads x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        v, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lv x num_heads x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lv x (num_heads * dv)
        v = v.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        if self.raw_attention:
            return v, attn

        v = self.fc(v)

        if self.residual:
            v += residual

        v = self.layer_norm(v)

        return v, attn


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self,
                 num_heads: int,
                 dim_input: int,
                 dim_hidden: Optional[int] = None,
                 dim_output: Optional[int] = None,
                 residual: bool = True,
                 ):
        if dim_hidden is None:
            dim_hidden = dim_input
        if dim_output is None:
            dim_output = dim_input
        super().__init__(num_heads=num_heads,
                         dim_input_v=dim_input,
                         dim_input_k=dim_input,
                         dim_hidden_v=dim_hidden,
                         dim_hidden_k=dim_hidden,
                         dim_output=dim_output,
                         residual=residual,
                         )

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask)


class MultiHeadSelfAttentionUniversal(nn.Module):
    def __init__(self,
                 dim_input: int,
                 num_heads: int = 1,
                 dim_hidden: Optional[int] = None,
                 dim_output: Optional[int] = None,
                 residual: bool = True,
                 n_times: int = 1,
                 pos_encoder_on: bool = False,
                 ):
        super().__init__()
        self.pos_encode = pos_encoder_on
        self.layers = nn.ModuleList([])
        for _ in range(n_times - 1):
            self.layers.append(MultiHeadSelfAttention(
                num_heads=num_heads,
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                dim_output=dim_input,  # keep input size till the final layer
                residual=residual,
            ))
        self.layers.append(MultiHeadSelfAttention(
            num_heads=num_heads,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,  # finally change the size
            residual=residual,
        ))


        if self.pos_encode:
            self.position_encoder = PositionalEncoderSelfAttention(
                dim=dim_input
            )

    def forward(self,
                x,
                positions=None,
                mask: Union[None, torch.Tensor, torch.BoolTensor] = None,
                att_dim: Union[int, Sequence[int]] = -2,
                feat_dim: int = -1,
                ):

        if self.pos_encode:
            assert positions is not None, 'Positions must be set!'
        else:
            assert positions is None, 'Positions received but positional encoder did not set, ' \
                                          'please set pos_encoder_on flag.'

        if isinstance(att_dim, (int, float)):
            att_dim = [att_dim]
        att_dim = [d if d >= 0 else x.ndim + d for d in att_dim]
        if feat_dim < 0:
            feat_dim = feat_dim + x.ndim
        assert feat_dim not in att_dim

        original_shape = x.shape
        indices = list(range(x.ndim))
        blind_dim = [i for i in indices if i != feat_dim and i not in att_dim]
        permutation = blind_dim + att_dim + [feat_dim]
        inv_permutation = [permutation.index(i) for i in indices]

        x = x.permute(permutation)
        x = x.reshape(product([original_shape[i] for i in blind_dim]),
                      product([original_shape[i] for i in att_dim]),
                      original_shape[feat_dim]
                      )

        attns = []
        x = self.position_encoder(x, position=positions)
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)

        output_shape = x.shape
        x = x.reshape(
            *(original_shape[i] for i in blind_dim),
            *(original_shape[i] for i in att_dim),
            output_shape[-1]
        )
        x = x.permute(inv_permutation)
        return x, attns