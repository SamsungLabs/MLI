__all__ = ['PositionalEncoderNeRF',
           'PositionalEncoderSelfAttention'
           ]

import math

import torch
from torch import nn


class PositionalEncoderNeRF(nn.Module):
    def __init__(self,
                 n_funcs: int = 0,
                 include_input: bool = False,
                 mul_by_pi: bool = True,
                 ):
        """
        Positional Encoder from NeRF paper.

        Args:
            n_funcs: number of multipliers to apply. If n_funcs == 0, encoder returns the input with an additional axis
            include_input: whether to include the original inputs in the encoder outputs
            mul_by_pi: whether to multiply by pi
        """
        super().__init__()
        if n_funcs == 0:
            self.multipliers = None
            self.out_features = 1
        elif n_funcs > 0:
            multipliers = (math.pi if mul_by_pi else 1) * torch.tensor([2 ** i for i in range(n_funcs)],
                                                                       dtype=torch.float)
            self.register_buffer('multipliers', multipliers)
            self.out_features = n_funcs * 2 + (1 if include_input else 0)
            self.include_input = include_input
        else:
            raise ValueError(f'Expected non-negative n_funcs, but obtained {n_funcs} instead.')

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        Args:
            tensor: D1 x ... x Dn

        Returns:
            output:
                D1 x ... Dn x 2*n_funcs if n_funcs > 0 and include_input == False,
                D1 x ... Dn x (2*n_funcs + 1) if n_funcs > 0 and include_input == True,
                D1 x ... x Dn x 1 if n_funcs == 0.
        """

        if self.multipliers is None:
            return tensor.unsqueeze(-1)
        else:
            embedding_input = tensor.unsqueeze(-1) * self.multipliers  # D1 x ... x Dn x n_funcs
            out = [torch.sin(embedding_input), torch.cos(embedding_input)]
            if self.include_input:
                out.append(tensor.unsqueeze(-1))
            return torch.cat(out, dim=-1)


class PositionalEncoderSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 n_samples: int = None,
                 base: float = 10_000,
                 ):
        super().__init__()

        self.n_samples = n_samples
        self.dim = dim

        self.presinusoid_table = torch.pow(
            base,
            -torch.tensor([j // 2 * 2 / self.dim for j in range(self.dim)])
        )  # (dim,)

    def forward(self, x, position=None):
        """

        Args:
            x: B x n_positions x dim
            position: B x n_positions

        Returns:

        """
        assert x.shape[-1] == self.dim, f'{x.shape[-1]} != {self.dim}'
        if position is not None:
            assert position.min().item() >= 0, f'{position.min().item()} < 0'
            assert position.max().item() <= 1, f'{position.max().item()} > 1'

        n_samples = self.n_samples
        if n_samples is None:
            n_samples = x.shape[1]

        if position is None:
            position = torch.linspace(0, 1, n_samples)[None, :].repeat(x.shape[0], 1)
            position = position.to(x.device)

        self.presinusoid_table = self.presinusoid_table.to(x.device)  # (dim,)
        table = position.unsqueeze(-1) * (n_samples - 1) * self.presinusoid_table  # B x n_positions x dim
        sinusoid_table = torch.empty_like(table)
        sinusoid_table[..., 0::2] = torch.sin(table[..., 0::2])
        sinusoid_table[..., 1::2] = torch.cos(table[..., 1::2])
        return x + sinusoid_table
