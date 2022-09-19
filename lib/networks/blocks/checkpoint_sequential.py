__all__ = ['CheckpointSequential']

from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


class CheckpointSequential(nn.Sequential):
    def __init__(self, n_chunks, *args):
        super().__init__(*args)
        self.n_chunks = n_chunks

    def forward(self, *args, **kwargs):
        return checkpoint_sequential(self, self.n_chunks, *args, **kwargs)
