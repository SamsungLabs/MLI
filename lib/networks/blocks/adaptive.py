__all__ = ['assign_adaptive_params',
           'get_num_adaptive_params',
           'AdaptiveInstanceNorm',
           ]

from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveModuleBase(nn.Module):
    @property
    @abstractmethod
    def num_adaptive_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def assign_adaptive_params(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError


def get_num_adaptive_params(model: nn.Module) -> int:
    """Count the number of adaptive parameters needed by the model"""
    num_adaptive_params = 0
    for m in model.modules():
        if isinstance(m, AdaptiveModuleBase):
            num_adaptive_params += m.num_adaptive_params
    return num_adaptive_params


def assign_adaptive_params(adaptive_params: torch.Tensor,
                           model: nn.Module,
                           ) -> None:
    """Assign the adain_params to the AdaIN layers inside the model"""
    for m in model.modules():
        if isinstance(m, AdaptiveModuleBase):
            assert adaptive_params.shape[1]
            m.assign_adaptive_params(adaptive_params[:, :m.num_adaptive_params])
            if adaptive_params.shape[1] >= m.num_adaptive_params:
                adaptive_params = adaptive_params[:, m.num_adaptive_params:]


class AdaptiveInstanceNorm(AdaptiveModuleBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self._num_adaptive_params = 2 * num_features
        self.eps = eps
        self.momentum = momentum

        # weight and bias are dynamically assigned
        self._adaptive_params = None

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.float))
        self.register_buffer('running_var', torch.ones(num_features, dtype=torch.float))

    @property
    def num_adaptive_params(self) -> int:
        return self._num_adaptive_params

    def assign_adaptive_params(self, tensor: torch.Tensor) -> None:
        assert tensor.dim() == 2
        self._adaptive_params = tensor

    def _get_weight_bias(self):
        """Assign adaptive weight and bias"""
        assert self._adaptive_params is not None, \
            f'Please assign adaptive params before calling {self.__class__.__name__}!'
        bias = self._adaptive_params[:, :self._num_adaptive_params // 2].contiguous().view(-1)
        weight = self._adaptive_params[:, self._num_adaptive_params // 2:].contiguous().view(-1)
        return weight, bias

    def forward(self, x):
        b, c = x.shape[:2]
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        weight, bias = self._get_weight_bias()

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps
        )
        return out.view(b, c, *x.shape[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self._num_adaptive_params) + ')'
