__all__ = ['RendererNeuralVolumes']

import torch

from .renderer_base import RendererBase


class RendererNeuralVolumes(RendererBase):
    def __init__(self, dt, step_jitter):
        super().__init__()
        self.dt = dt
        self.step_jitter = step_jitter
        self.n_color_channels = 3
        self.n_alpha_channels = 1
        self.tmin = None
        self.tmax = None

    def forward(self,
                sampling_fn,
                pixel_coords,
                cameras,
                backgrounds,
                color_corrs_scene=None,
                color_corrs_background=None,
                valid_fns=None,
                ):

        ray_direction = cameras.pixel_to_world_ray_direction(pixel_coords)  # B x H x W x 3
        self.tmin, self.tmax = self.get_raymarch_start_end_steps(cameras.world_position, ray_direction)  # B x H x W

        t = self.generate_integration_startpoint()  # B x H x W
        ray_position = cameras.world_position[:, None, None, :] + ray_direction * t.unsqueeze(-1)  # B x H x W x 3

        cumulated_rgb, cumulated_alpha = self.do_raymarching(sampling_fn,
                                                             ray_position,
                                                             ray_direction,
                                                             t,
                                                             valid_fns=valid_fns,
                                                             )

        if color_corrs_scene is not None:
            color_corrs_scene = color_corrs_scene[:, :, None, None]
            cumulated_rgb = cumulated_rgb * color_corrs_scene[:, :3] + color_corrs_scene[:, 3:]
        if color_corrs_background is not None:
            color_corrs_background = color_corrs_background[:, :, None, None]
            backgrounds = backgrounds * color_corrs_background[:, :3] + color_corrs_background[:, 3:]

        backgrounds = backgrounds.clamp(min=0.)

        cumulated_rgb = cumulated_rgb + (1 - cumulated_alpha) * backgrounds
        return cumulated_rgb, cumulated_alpha

    def generate_step_length(self, current_position):
        step = self.dt * torch.exp(self.step_jitter * torch.randn_like(current_position[..., 0]))
        return step

    def do_raymarching(self,
                       sampling_fn,
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

        while not done.all():
            validf = self.validate_world_position(ray_position, valid_fns)  # B x H x W

            sampled_rgb, sampled_alpha = sampling_fn(ray_position[:, None, :, :, :])  # added fake D dim to H and W

            # now squeeze that fake depth dimension
            sampled_rgb = sampled_rgb.squeeze(2)  # B x color x H x W
            sampled_alpha = sampled_alpha.squeeze(2)  # B x alpha x H x W

            step_length = self.generate_step_length(ray_position)  # B x H x W

            contrib = self.compute_alpha_contribution(cumulated_alpha, sampled_alpha, step_length)
            contrib = contrib * validf.unsqueeze(1)

            cumulated_rgb = cumulated_rgb + sampled_rgb * contrib
            cumulated_alpha = cumulated_alpha + contrib

            ray_position = ray_position + ray_direction * step_length.unsqueeze(-1)
            t = t + step_length
            done = done | self.validate_integration_point(t)

        return cumulated_rgb, cumulated_alpha
