__all__ = ['RendererBase']

import torch
from torch import nn


class RendererBase(nn.Module):
    def _init_modules(self, *args, **kwargs):
        pass

    @staticmethod
    @torch.no_grad()
    def get_raymarch_start_end_steps(camera_position, ray_direction, edge_half_length=1):
        """
        :param camera_position: Bc x 3
        :param ray_direction: Bc x D1 x ... x Dn x 3
        :param edge_half_length: cube edge length
        :return: 2 tensors of Bc x D1 x ... x Dn
        """
        _, *dims, _ = ray_direction.shape
        camera_position = camera_position.contiguous().view(camera_position.shape[0],
                                                            *([1] * len(dims)),
                                                            camera_position.shape[-1],
                                                            )
        t1 = (-edge_half_length - camera_position) / ray_direction
        t2 = (edge_half_length - camera_position) / ray_direction
        tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                         torch.max(torch.min(t1[..., 1], t2[..., 1]),
                                   torch.min(t1[..., 2], t2[..., 2])))
        tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                         torch.min(torch.max(t1[..., 1], t2[..., 1]),
                                   torch.max(t1[..., 2], t2[..., 2])))

        intersections = tmin < tmax
        tmin = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
        tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        return tmin, tmax

    @staticmethod
    def validate_world_position(position, valid_fn=None):
        """
        Check if the point is in the space that we are rendering.
        You can use your own validation function.
        If  you want use different function for each batch element, you can pass list of functions.
        Args:
            position (Tensor): B x D1 x ... x Dn x 3
            valid_fn (Function or list of functions): you can use your own validation function

        Returns:
            bool tensor: B x D1 x ... x Dn
        """

        result = torch.prod(torch.gt(position, -1.0) * torch.lt(position, 1.0),
                            dim=-1).float()

        if valid_fn is not None:
            if isinstance(valid_fn, list):
                fn_result = []
                for i, fn in enumerate(valid_fn):
                    fn_result.append(result[i] * fn(position[i]).float())
                result = torch.stack(fn_result)
            else:
                result = result * valid_fn(position).float()

        return result

    def validate_integration_point(self, t):
        return t >= self.tmax

    def generate_integration_startpoint(self):
        return self.tmin - self.dt * torch.rand_like(self.tmin)

    def generate_step_length(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def compute_alpha_contribution(cumulated_alpha, sampled_alpha, integration_step):
        updated_alpha = torch.clamp(cumulated_alpha + sampled_alpha * integration_step.unsqueeze(1),
                                    max=1.)
        contribution = updated_alpha - cumulated_alpha
        return contribution

    def do_raymarching(self, *args, **kwargs):
        raise NotImplementedError
