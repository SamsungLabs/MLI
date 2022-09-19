__all__ = ['MultiNet',
           ]

from torch import nn

import lib.networks.generators.gen_parts


class MultiNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        optimizer_base = kwargs['optimizer_base']
        architecture = optimizer_base.pop('architecture')
        self.iterations = kwargs['iterations']
        net = getattr(lib.networks.generators.gen_parts, architecture)

        self.nets = nn.ModuleList()
        for i in range(self.iterations):
            self.nets += [net(**optimizer_base)]

    def forward(self, x, it):
        x = self.nets[it](x)
        return x
