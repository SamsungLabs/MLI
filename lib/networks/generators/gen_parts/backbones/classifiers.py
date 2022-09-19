__all__ = ['resnet_backbone']

from typing import Any, Callable, Optional

from torch import nn
from torchvision import models

from lib.networks.blocks.base import Identity


def resnet_backbone(arch: int = 18,
                    input_dim: int = 3,
                    output_dim: int = 1000,
                    norm_layer: Optional[Callable[[Any], nn.Module]] = Identity,
                    ) -> nn.Sequential:
    model_func = getattr(models, f'resnet{arch:d}')
    backbone = model_func(pretrained=False,
                          num_classes=output_dim,
                          norm_layer=norm_layer,
                          )
    modules = list(backbone.children())
    initial_conv: nn.Conv2d = modules[0]
    final_fc: nn.Linear = modules[-1]
    result = nn.Sequential(
        nn.Conv2d(in_channels=input_dim,
                  out_channels=initial_conv.out_channels,
                  kernel_size=initial_conv.kernel_size,
                  stride=initial_conv.stride,
                  padding=initial_conv.padding,
                  bias=initial_conv.bias is not None,
                  ),
        *modules[1:-1],
        nn.Conv2d(final_fc.in_features, output_dim, 1, 1, 0),
    )
    return result
