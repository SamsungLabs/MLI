__all__ = ['DEQModel',
           'DEQMultiHeadAttention',
           ]

from abc import abstractmethod

import torch
from torch import nn

from lib.networks.blocks import MultiHeadAttention
from lib.utils.solvers import anderson, broyden


class DEQModel(nn.Module):
    def __init__(self,
                 forward_threshold: float,
                 backward_threshold: float,
                 solver: str = None,
                 ):
        super().__init__()

        self.forward_threshold = forward_threshold
        self.backward_threshold = backward_threshold
        self.hook = None

        if solver == 'anderson':
            self.solver = anderson
        elif solver == 'broyden':
            self.solver = broyden
        assert self.solver is not None, 'Unsupported solver!'

    @abstractmethod
    def layer_forward(self, z, *x):
        raise NotImplementedError

    def implicit_forward(self, z0, x):

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.layer_forward(z, *x), z0,
                                 threshold=self.forward_threshold)['result']  # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass
        if self.training:
            z_star.requires_grad_()
            new_z_star = self.layer_forward(z_star, *x)

            # Have to use a copy here because a hook will be applied to new_z_star (which could otherwise 
            # cause infinite recursion)
            # z_star_copy = z_star.clone().detach()
            # z_star_copy.requires_grad = True
            # new_z_star_copy = self.layer_forward(z_star_copy, *x)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = \
                    self.solver(
                        lambda y: torch.autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad, \
                        torch.zeros_like(grad), threshold=self.backward_threshold)['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)

        return new_z_star


class DEQMultiHeadAttention(DEQModel):
    def __init__(self,
                 forward_threshold: float,
                 backward_threshold: float,
                 num_heads: int,
                 dim_input_v: int,
                 dim_input_k: int,
                 dim_hidden_v: int,
                 dim_hidden_k: int,
                 dim_output: int,
                 residual: bool = True,
                 solver: str = None,
                 ):
        super().__init__(forward_threshold, backward_threshold, solver)

        self.multi_head_attention = MultiHeadAttention(num_heads,
                                                       dim_input_v,
                                                       dim_input_k,
                                                       dim_hidden_v,
                                                       dim_hidden_k,
                                                       dim_output,
                                                       residual
                                                       )

    def layer_forward(self, z, *x):
        q, v, k = x
        z_star, _ = self.multi_head_attention(q + z, k + z, v + z)
        return z_star

    def forward(self, q, v, k):
        z0 = torch.zeros_like(q)
        z_star = self.implicit_forward(z0, (q, v, k))
        return z_star, z_star
