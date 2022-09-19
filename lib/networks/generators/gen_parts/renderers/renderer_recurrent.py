__all__ = ['RendererRecurrent']

import torch

from lib.networks.blocks.rnn_raymarcher import RNNRaymarcher
from .renderer_base import RendererBase


class RendererRecurrent(RendererBase):
    def __init__(self,
                 dt=1.,
                 rnn_type='gru',
                 latent_dim=256,
                 hidden_dim=256,
                 n_steps=10,
                 distribution=None,
                 ):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.n_color_channels = 3
        self.n_alpha_channels = 1
        self.tmin = None
        self.tmax = None

        input_dim = latent_dim + 2 * 3 + 2 * (self.n_color_channels + self.n_alpha_channels)
        self._init_modules(rnn_type, input_dim, hidden_dim, distribution)

    def _init_modules(self, rnn_type, input_dim, hidden_dim, distribution):
        self.step_length_generator = RNNRaymarcher(input_size=input_dim,
                                                   hidden_size=hidden_dim,
                                                   rnn_type=rnn_type,
                                                   distribution=distribution,
                                                   )

    def forward(self,
                sampling_fn,
                pixel_coords,
                cameras,
                backgrounds,
                z,
                color_corrs_scene=None,
                color_corrs_background=None,
                valid_fns=None,
                ):

        self.step_length_generator.reset()

        ray_direction = cameras.pixel_to_world_ray_direction(pixel_coords)  # B x H x W x 3
        self.tmin, self.tmax = self.get_raymarch_start_end_steps(cameras.world_position, ray_direction)  # B x H x W

        t = self.generate_integration_startpoint()  # B x H x W
        ray_position = cameras.world_position[:, None, None, :] + ray_direction * t.unsqueeze(-1)  # B x H x W x 3
        cumulated_rgb, cumulated_alpha, step_length = self.do_raymarching(sampling_fn,
                                                                          z,
                                                                          ray_position,
                                                                          ray_direction,
                                                                          t,
                                                                          valid_fns=valid_fns
                                                                          )

        if color_corrs_scene is not None:
            color_corrs_scene = color_corrs_scene[:, :, None, None]
            cumulated_rgb = cumulated_rgb * color_corrs_scene[:, :3] + color_corrs_scene[:, 3:]
        if color_corrs_background is not None:
            color_corrs_background = color_corrs_background[:, :, None, None]
            backgrounds = backgrounds * color_corrs_background[:, :3] + color_corrs_background[:, 3:]

        backgrounds = backgrounds.clamp(min=0.)

        cumulated_rgb = cumulated_rgb + (1 - cumulated_alpha) * backgrounds.clamp(min=0.)
        return cumulated_rgb, cumulated_alpha, step_length

    def generate_step_length(self,
                             z,
                             current_position,
                             ray_direction,
                             cumulated_rgb,
                             cumulated_alpha,
                             sampled_rgb,
                             sampled_alpha,
                             ):
        """
        :param z: B x latent_dim
        :param current_position: B x H x W x 3
        :param ray_direction: B x H x W x 3
        :param cumulated_rgb: B x 3 x H x W
        :param cumulated_alpha: B x 3 x H x W
        :param sampled_rgb: B x 3 x H x W
        :param sampled_alpha: B x 3 x H x W
        :return: B x H x W
        """
        input_data = torch.cat([
            z[:, None, None, :].expand(-1, *current_position.shape[1:3], -1),
            current_position,
            ray_direction,
            cumulated_rgb.permute(0, 2, 3, 1) / 255,
            cumulated_alpha.permute(0, 2, 3, 1),
            sampled_rgb.permute(0, 2, 3, 1) / 255,
            sampled_alpha.permute(0, 2, 3, 1),
        ], dim=-1)
        out = self.step_length_generator(input_data)
        return out.squeeze(-1) * self.dt

    def do_raymarching(self,
                       sampling_fn,
                       z,
                       start_position,
                       ray_direction,
                       start_t=None,
                       cumulated_rgb=None,
                       cumulated_alpha=None,
                       valid_fns=None,
                       ):
        batch_size, height, width, *_ = start_position.shape
        if cumulated_rgb is None:
            cumulated_rgb = torch.zeros(batch_size, self.n_color_channels, height, width,
                                        dtype=torch.float32, device=start_position.device)  # B x 3 x H x W
        if cumulated_alpha is None:
            cumulated_alpha = torch.zeros(batch_size, self.n_alpha_channels, height, width,
                                          dtype=torch.float32, device=start_position.device)  # B x 1 x H x W
        if start_t is None:
            start_t = torch.zeros_like(start_position[..., 0])

        # raymarch loop
        done = torch.zeros_like(start_t, dtype=torch.bool)
        t = start_t
        ray_position = start_position
        step_number = 0
        step_length = None

        # while step_number <= self.n_steps and not done.all().item():
        while step_number <= self.n_steps:
            validf = self.validate_world_position(ray_position, valid_fns)  # B x H x W

            sampled_rgb, sampled_alpha = sampling_fn(ray_position[:, None, :, :, :])  # added fake D dim to H and W
            # now squeeze that fake depth dimension
            sampled_rgb = sampled_rgb.squeeze(2)  # B x color x H x W
            sampled_alpha = sampled_alpha.squeeze(2)  # B x alpha x H x W

            step_length = self.generate_step_length(z,
                                                    ray_position,
                                                    ray_direction,
                                                    cumulated_rgb,
                                                    cumulated_alpha,
                                                    sampled_rgb,
                                                    sampled_alpha,
                                                    )  # B x H x W

            contrib = self.compute_alpha_contribution(cumulated_alpha, sampled_alpha, step_length)
            contrib = contrib * validf.unsqueeze(1)

            cumulated_rgb = cumulated_rgb + sampled_rgb * contrib
            cumulated_alpha = cumulated_alpha + contrib

            ray_position = ray_position + ray_direction * step_length.unsqueeze(-1)
            t = t + step_length
            step_number += 1
            done = done | self.validate_integration_point(t)

        # if done.all():
        #     step_length = torch.zeros_like(step_length)

        return cumulated_rgb, cumulated_alpha, step_length
