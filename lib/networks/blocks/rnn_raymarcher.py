__all__ = ['RNNRaymarcher']

import math

import torch
from torch import nn
import torch.nn.functional as F


class RNNRaymarcher(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int = 1,
                 rnn_type: str = 'gru',
                 activ: str = 'softplus',
                 distribution: str = None,
                 ):
        """
        :param input_size: number of input features
        :param hidden_size: hidden dim of recurrent cell
        :param output_size: number of output features
        :param rnn_type: type of rnn cell: rnn | gru | lstm
        :param activ: activation function for `none` distribution (see below): softplus | relu
        :param distribution: sampling distribution. Random samples are used during training, the expected value -
            during evaluation. Supported distributions:

            none: deterministic output
            normal: Normal distribution with predicted location and scale
            lognormal: Log-Normal distribution with predicted location and scale
            exp: Exponential distribution with predicted parameter
            exp-trunc: Truncated Exponential distribution with support [0, sqrt(3)] and predicted parameter
            exp-trunc-high: Truncated Exponential distribution with support [0, predicted_higher_bound] and predicted
                parameter
            exp-trunc-bounds: Truncated Exponential distribution with support
                [predicted_lower_bound, predicted_higher_bound] and predicted parameter
        """
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type == 'rnn':
            self.cell = nn.RNNCell(input_size, hidden_size)
        elif rnn_type == 'lstm':
            self.cell = nn.LSTMCell(input_size, hidden_size)
        elif rnn_type == 'gru':
            self.cell = nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError(f'Unknown rnn_type: {rnn_type}: only `rnn`, `lstm` and `gru` are supported')

        for name, param in self.cell.named_parameters():
            if name.startswith('weight'):
                nn.init.orthogonal_(param)
            elif name.startswith('bias'):
                nn.init.constant_(param, 1.)

        activ = activ.lower()
        if activ == 'softplus':
            self.activ = nn.Softplus()
        elif activ == 'relu':
            self.activ = nn.ReLU()
        else:
            raise ValueError(f'Unknown activ {activ}. Supported types are: `softplus`, `relu`.')

        self.distribution = distribution

        if self.distribution in {None, 'none', 'exp', 'exp-trunc'}:
            resulting_dim = output_size
        elif self.distribution in {'normal', 'lognormal', 'exp-trunc-high'}:
            resulting_dim = output_size * 2
        elif self.distribution in {'exp-trunc-bounds', }:
            resulting_dim = output_size * 3
        else:
            raise ValueError(f'Unknown distribution: {distribution}. '
                             'Only `none`, `normal`, `lognormal`, `exp`, `exp-trunc`, `exp-trunc-high`, '
                             '`exp-trunc-bounds` are supported')
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_size, resulting_dim),
        )

        for name, param in self.aggregator.named_parameters():
            if name.startswith('weight'):
                nn.init.xavier_normal_(param)

        self.storage = None

    def reset(self):
        self.storage = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: D1 x D2 x ... x Dn x input_size
        :return: D1 x D2 x ... x Dn x output_size
        """
        *dims, input_dim = tensor.shape
        tensor = tensor.contiguous().view(-1, input_dim)

        if self.storage is None:
            self.storage = self.cell(tensor)
        else:
            self.storage = self.cell(tensor, self.storage)

        if isinstance(self.storage, tuple):
            hidden = self.storage[0]
        else:
            hidden = self.storage

        aggregated = self.aggregator(hidden)

        if self.distribution in {None, 'none'}:
            out = self.activ(aggregated)
        elif self.distribution in {'normal', 'lognormal'}:
            mu, presigma = aggregated.chunk(2, dim=-1)
            sigma = F.softplus(presigma)
            if self.distribution == 'normal':
                noise = torch.randn_like(mu) if self.training else 0.
                out = mu + noise * sigma
            elif self.distribution == 'lognormal':
                noise = torch.randn_like(mu) if self.training else 0.5 * sigma
                out = torch.exp(mu + noise * sigma)
        elif self.distribution == 'exp':
            noise = -torch.rand_like(aggregated).log() if self.training else 1.
            out = F.softplus(aggregated) * noise / 10
        elif self.distribution in {'exp-trunc', 'exp-trunc-high', 'exp-trunc-bounds'}:
            aggregated = F.softplus(aggregated)
            if self.distribution == 'exp-trunc':
                low = 0.
                lmbda = aggregated
                high = math.sqrt(3)  # sqrt(3) == diagonal of the unit cube
            elif self.distribution == 'exp-trunc-high':
                low = 0.
                lmbda, high = aggregated.chunk(2, dim=-1)
            elif self.distribution == 'exp-trunc-bounds':
                lmbda, support, low = aggregated.chunk(3, dim=-1)
                high = low + support + 1e-7

            if self.training:
                noise = torch.rand_like(lmbda)
                e_low = torch.exp(- low * lmbda)
                e_high = torch.exp(- high * lmbda)
                out = - torch.log(e_low - noise * (e_low - e_high)) / lmbda
            else:
                support = high - low
                out = high + 1 / lmbda - support / (1 - torch.exp(- support * lmbda))
        else:
            raise ValueError(f'Unknown distribution: {self.distribution}')

        return out.contiguous().view(*dims, out.shape[-1])
