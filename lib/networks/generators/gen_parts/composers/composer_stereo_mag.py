__all__ = ['ComposerStereoMagnification']

from typing import Optional, Tuple, NamedTuple, Union

import torch

from .composer_base import ComposerBase


class ComposerResult(NamedTuple):
    color: torch.Tensor
    opacity: torch.Tensor
    weight: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None


class ComposerStereoMagnification(ComposerBase):

    # TODO Add the ability to compose along the axis by specifying it in the parameters.
    def __init__(self,
                 black_background: bool = False,
                 multi_reference_cams: bool = False,
                 masked_black_background: bool = True,
                 ):
        super().__init__()
        self.black_background = black_background
        self.multi_reference_cams = multi_reference_cams
        self.masked_black_background = masked_black_background

    @staticmethod
    def precompute_features(features: torch.Tensor,
                            black_background: bool,
                            masked_black_background: bool,
                            ) -> Tuple[torch.Tensor, ...]:
        """
        Pre-compute transmittance and colorization from features
        """
        color = features[:, :, :, :, :-1]
        opacity = features[:, :, :, :, -1:].clamp(min=0, max=1)  # B x n_ref x n_cams x n_intersections x 1 x ...

        eps = 1e-6
        if not black_background:
            if masked_black_background:
                real_intersection_mask = opacity.sum(dim=3, keepdim=True) > eps
            else:
                real_intersection_mask = 1
            opacity[:, :, :, [0]] = torch.ones_like(opacity[:, :, :, [0]]) * real_intersection_mask

        transmittance = torch.cumprod(1 - opacity.flip(3) + eps, dim=3)
        total_transmittance = transmittance[:, :, :, -1:]
        transmittance = torch.cat([torch.ones_like(transmittance[:, :, :, [0]]), transmittance[:, :, :, :-1]],
                                  dim=3).flip(3)

        return color, opacity, transmittance, total_transmittance

    def forward(self,
                features: torch.Tensor,
                use_uncertainty: bool = False,
                timestamps: Optional[torch.Tensor] = None,
                inside_frustum_mask: Optional[torch.BoolTensor] = None,
                black_background: Optional[bool] = None,
                detach_opacity_for_uncertainty: bool = False,
                return_namedtuple: bool = False,
                masked_black_background: bool = True,
                ) -> Union[Tuple[torch.Tensor, ...], ComposerResult]:
        """
        Compose-over from back to front. The last channel is treated as opacity.

        Args:
            features: B x n_ref x n_cams x n_intersections x C x ...
            use_uncertainty: bool
            timestamps: B x n_ref x n_cams x n_intersections x ...
                If None, features are treated as they are already properly ordered
            inside_frustum_mask: B x n_ref x n_cams x 1 x ...
            black_background:
            detach_opacity_for_uncertainty:
            return_namedtuple:
            masked_black_background:
        Returns:
            out: B x n_ref x n_cams x C-1 x ... if uncertainty is None,
                 B x n_ref x n_cams x C x ... otherwise
            opacity: B x n_ref x n_cams x 1 x ...
            weight: B x n_ref x n_cams x n_intersections x 1 x ...
        """

        if self.multi_reference_cams is False:
            features = features.unsqueeze(1)
            if timestamps is not None:
                timestamps = timestamps.unsqueeze(1)

        if timestamps is not None:
            features = self.sort_intersections(features, timestamps, dim=3)

        if use_uncertainty:
            uncertainty = features[:, :, :, :, [-2]]
            features = torch.cat([features[:, :, :, :, :-2], features[:, :, :, :, -1:]], dim=4)
        else:
            uncertainty = None

        color, opacity, transmittance, total_transmittance = self.precompute_features(
            features=features,
            black_background=self.black_background if black_background is None else black_background,
            masked_black_background=self.masked_black_background if masked_black_background is None else masked_black_background,
        )
        weight = opacity * transmittance
        out = torch.sum(color * weight, dim=3)
        opacity = 1 - total_transmittance.squeeze(3)

        if use_uncertainty:
            features_for_uncertainty = torch.cat([
                torch.clamp(
                    transmittance.detach() if detach_opacity_for_uncertainty else transmittance,
                    min=0, max=1),
                uncertainty
            ], dim=4)
            if not self.multi_reference_cams:
                features_for_uncertainty = features_for_uncertainty.squeeze(1)
            aggregated_uncertainty = self.forward(features=features_for_uncertainty,
                                                  use_uncertainty=False,
                                                  inside_frustum_mask=inside_frustum_mask,
                                                  black_background=True,
                                                  return_namedtuple=True).color.clamp(min=0, max=1)
            if not self.multi_reference_cams:
                aggregated_uncertainty = aggregated_uncertainty.unsqueeze(1)
            if not return_namedtuple:
                out = torch.cat([out, aggregated_uncertainty], dim=3)
                del aggregated_uncertainty
                torch.cuda.empty_cache()
                aggregated_uncertainty = None
        else:
            aggregated_uncertainty = None

        if self.multi_reference_cams is False:
            out = out.squeeze(1)
            opacity = opacity.squeeze(1)
            weight = weight.squeeze(1)
            if aggregated_uncertainty is not None:
                aggregated_uncertainty = aggregated_uncertainty.squeeze(1)

        if not return_namedtuple:
            return out, opacity
        else:
            return ComposerResult(
                color=out,
                opacity=opacity,
                uncertainty=aggregated_uncertainty,
                weight=weight,
            )
