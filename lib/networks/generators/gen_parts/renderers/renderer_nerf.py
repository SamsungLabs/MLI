__all__ = ['RendererNeRF']

from typing import Any, Callable, Optional, Tuple

import torch

from lib.modules.cameras import CameraPinhole
from .renderer_base import RendererBase


SamplingFuncNeRF = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class RendererNeRF(RendererBase):
    class _Decorators:
        @classmethod
        def batchify_raymarching(cls, chunk_size=2 ** 12, batching_dims=3, n_outputs=3):
            def decorator(func):
                def wrapper(instance, sampling_fn, *args, **kwargs):
                    assert len(set((a.shape[:batching_dims] for a in args))) == 1
                    batching_shape = args[0].shape[:batching_dims]
                    batching_length = torch.tensor(batching_shape).prod().int().item()

                    args = [a.contiguous().view(batching_length, *([1] * (len(batching_shape) - 1)), -1
                                                ).split(chunk_size) for a in args]
                    outs = [[] for _ in range(n_outputs)]

                    for args_chunk in zip(*args):
                        outs_chunk = func(instance, sampling_fn, *args_chunk, valid_fn=kwargs['valid_fn'])
                        for out, out_chunk in zip(outs, outs_chunk):
                            out.append(out_chunk)

                    outs = [torch.cat(out) for out in outs]
                    remaining_shapes = [out.shape[batching_dims:] for out in outs]
                    outs = [out.contiguous().view(*batching_shape, *remaining_shape)
                            for out, remaining_shape in zip(outs, remaining_shapes)]

                    return outs

                return wrapper

            return decorator

    def __init__(self,
                 n_coarse_locations: int = 64,
                 n_fine_locations: int = 128,
                 coarse_apply_background: bool = False,
                 fine_apply_background: bool = False,
                 ):
        super().__init__()
        self.n_coarse_locations = n_coarse_locations
        self.n_fine_locations = n_fine_locations
        self.tmin = self.tmax = None
        self.coarse_apply_background = coarse_apply_background
        self.fine_apply_background = fine_apply_background

    def forward(self,
                coarse_sampling_fn: SamplingFuncNeRF,
                fine_sampling_fn: SamplingFuncNeRF,
                pixel_coords: torch.Tensor,
                cameras: CameraPinhole,
                backgrounds: Optional[torch.Tensor],
                valid_fns: Any = None,
                ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor]]:
        backgrounds = backgrounds.permute(0, 2, 3, 1).clamp(min=0.)

        ray_direction = cameras.pixel_to_world_ray_direction(pixel_coords)  # B x H x W x 3
        camera_position = cameras.world_position  # B x 3
        self.tmin, self.tmax = self.get_raymarch_start_end_steps(camera_position, ray_direction)  # B x H x W

        camera_position = camera_position[:, None, None, :].expand_as(ray_direction)  # B x H x W x 3
        coarse_t_segments_bounds, coarse_t_integration_locations = self._generate_coarse_t_locations()
        coarse_cumulated_rgb, weights, coarse_transparency = self.do_raymarching(
            coarse_sampling_fn,
            camera_position,
            ray_direction,
            coarse_t_integration_locations,
            valid_fn=valid_fns,
        )

        if self.coarse_apply_background:
            coarse_cumulated_rgb = coarse_cumulated_rgb + coarse_transparency * backgrounds

        fine_t_integration_locations = self.weighted_sampling(weights.detach(),
                                                              coarse_t_segments_bounds,
                                                              self.n_fine_locations,
                                                              )
        fine_t_integration_locations = torch.cat([fine_t_integration_locations, coarse_t_integration_locations],
                                                 dim=-1)
        fine_t_integration_locations, _ = torch.sort(fine_t_integration_locations, dim=-1)
        fine_cumulated_rgb, _, fine_transparency = self.do_raymarching(
            fine_sampling_fn,
            camera_position,
            ray_direction,
            fine_t_integration_locations,
            valid_fn=valid_fns,
        )

        if self.fine_apply_background:
            fine_cumulated_rgb = fine_cumulated_rgb + fine_transparency * backgrounds
        return (
            (coarse_cumulated_rgb.permute(0, 3, 1, 2), 1. - coarse_transparency.permute(0, 3, 1, 2)),
            (fine_cumulated_rgb.permute(0, 3, 1, 2), 1. - fine_transparency.permute(0, 3, 1, 2)),
        )

    @_Decorators.batchify_raymarching(chunk_size=2**12, batching_dims=3, n_outputs=3)
    def do_raymarching(self,
                       sampling_fn: SamplingFuncNeRF,
                       camera_position: torch.Tensor,  # B x H x W x 3
                       ray_direction: torch.Tensor,  # B x H x W x 3
                       t_integration_locations: torch.Tensor,  # B x H x W x n_locations
                       valid_fn: Any = None,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        world_integration_locations = (
            camera_position.unsqueeze(-2)
            + ray_direction.unsqueeze(-2) * t_integration_locations.unsqueeze(-1)
        )  # B x H x W x n_locations x 3

        rgb, density = sampling_fn(
            world_integration_locations,
            ray_direction.unsqueeze(-2).expand_as(world_integration_locations),
        )
        density = density * self.validate_world_position(world_integration_locations,
                                                         valid_fn=valid_fn).unsqueeze(-1)
        cumulated_rgb, weights, transparency = self.accumulate_colors(rgb,
                                                                      density,
                                                                      t_integration_locations,
                                                                      )
        return cumulated_rgb, weights, transparency

    @torch.no_grad()
    def _generate_coarse_t_locations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build equal segments and sample random locations from them uniformly.
        Returns:
            t_segments_bounds: B x H x W x (n_coarse_locations + 1)
            t_integration_locations: B x H x W x n_coarse_locations
        """
        batch_size = self.tmin.shape[0]
        nearest_t_location, _ = self.tmin.reshape(batch_size, -1).min(-1, keepdim=True)  # B x 1
        farthest_t_location, _ = self.tmax.reshape(batch_size, -1).max(-1, keepdim=True)
        coef = torch.linspace(0, 1, self.n_coarse_locations + 1,
                              device=self.tmin.device)
        t_segments_bounds = farthest_t_location * coef + nearest_t_location * (1 - coef)  # B x (1 + n_coarse_locations)

        t_segments_length = t_segments_bounds[:, 1:] - t_segments_bounds[:, :-1]  # B x n_coarse_location
        noise = torch.rand(*self.tmin.shape, self.n_coarse_locations,
                           device=self.tmin.device)
        t_integration_locations = t_segments_bounds[:, None, None, :-1] + noise * t_segments_length[:, None, None, :]

        return t_segments_bounds[:, None, None, :].expand(*self.tmin.shape, -1), t_integration_locations

    @staticmethod
    @torch.no_grad()
    def weighted_sampling(weights: torch.Tensor,
                          segments_bounds: torch.Tensor,
                          n_locations: int,
                          ) -> torch.Tensor:
        """
        Generate `n_locations` points according to the provided weights per segments.

        Warning: our implementation relies on Gumbel-max trick, while the original codebase uses inverse empirical cdf
        https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L157

        Args:
            weights: D1 x ... x Dn x n_segments x 1
            segments_bounds: D1 x ... x Dn x (1 + n_segments)
            n_locations: number of locations to generate

        Returns:
            sampled_locations: D1 x ... x Dn x n_locations
        """

        # 1. Sample segments indices using Gumbel-max trick
        noise = torch.rand(*weights.shape[:-1], n_locations,
                           device=weights.device)
        gumbel_noise = noise.log().mul(-1.).log().mul(-1.)
        logits = weights.add(1e-5).log()

        sampled_segment_idx = torch.argmax(logits + gumbel_noise, dim=-2)  # D1 x ... x Dn x n_locations
        del noise, gumbel_noise, logits

        # 2. Sample uniform noise and transform it to the selected segments
        noise = torch.rand(*sampled_segment_idx.shape, device=sampled_segment_idx.device)
        segment_lengths = segments_bounds[..., 1:] - segments_bounds[..., :-1]  # D1 x ...x Dn x n_segments
        left_bounds = segments_bounds[..., :-1]

        # D1 x ... x Dn x n_locations
        sampled_locations = (
            torch.gather(left_bounds, dim=-1, index=sampled_segment_idx)
            + noise * torch.gather(segment_lengths, dim=-1, index=sampled_segment_idx)
        )
        del noise

        return sampled_locations

    @staticmethod
    def accumulate_colors(rgb: torch.Tensor,
                          density: torch.Tensor,
                          t_locations: torch.Tensor,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate colors along the rays.
        Warning: the actual upper limit of integration equals +Infinity.

        Args:
            rgb:  RGB color at each location, D1 x ... x Dn x n_locations x 3
            density: hazard function at each location, D1 x ... x Dn x n_locations x 1
            t_locations: integration locations in terms of time (with ray velocity of unit length in world coordinates),
                D1 x ... x Dn x n_locations

        Returns:
            cumulated_rgb: the resulting RGB color, D1 x ... x Dn x 3
            weights: weight of each location in the resulting color, D1 x ... x Dn x n_locations x 1
            transparency_proba: the remaining weight for background, D1 x ... x Dn x 1

        """
        t_segment_length = t_locations[..., 1:] - t_locations[..., :-1]
        infinity = torch.tensor([1e10], device=t_segment_length.device).expand_as(t_segment_length[..., :1])
        t_segment_length = torch.cat([t_segment_length, infinity], dim=-1)

        alpha = 1 - torch.exp(- density * t_segment_length.unsqueeze(-1))  # D1 x ... x Dn x n_locations x 1
        unit = torch.tensor([1.], device=alpha.device).expand_as(alpha[..., :1, :])
        survival_value = torch.cumprod(1 - alpha[..., :-1, :], dim=-2)
        survival_value = torch.cat([unit, survival_value], dim=-2)
        weights = alpha * survival_value
        transparency_proba = 1 - weights.sum(dim=-2)  # D1 x ... x Dn x 1

        cumulated_rgb = torch.sum(rgb * weights, dim=-2)
        return cumulated_rgb, weights, transparency_proba
