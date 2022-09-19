__all__ = ['TBlock']

from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from lib.modules.cameras import CameraMultiple
from lib.networks.blocks.positional_encoders import PositionalEncoderSelfAttention
from lib.networks.blocks.attention import MultiHeadAttention
from lib.networks.blocks.resnet import DoubleConvResnet
from lib.networks.generators.gen_parts.surfaces import RaySampler
from lib.utils.base import get_grid


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var


class TBlock(nn.Module):
    def __init__(self,
                 num_surfaces: int,
                 input_size: int = 3,
                 input_surfaces_feats_size: int = 3,
                 output_surfaces_feats_size: int = 3,
                 post_aggregate_block_dims: list = [24, 6],
                 post_aggregate_block_conv_dim: int = 2,
                 surfaces_type: str = 'mpi',
                 surfaces_sampling_mode: str = 'disparity',
                 min_distance: float = 1.,
                 max_distance: float = 100.,
                 interpolate_input_feats: bool = False,
                 agg_with_mean_var: bool = True,
                 agg_processor_hidden_size: int = None,
                 aggregation_with_weights: bool = False,
                 ):
        super().__init__()

        self.num_surfaces = num_surfaces
        self.input_surfaces_feats_size = input_surfaces_feats_size
        self.output_surfaces_feats_size = output_surfaces_feats_size
        self.surfaces_type = surfaces_type
        self.interpolate_input_feats = interpolate_input_feats
        self.agg_processor_hidden_size = agg_processor_hidden_size
        self.agg_with_mean_var = agg_with_mean_var
        self.aggregation_with_weights = aggregation_with_weights
        self.post_aggregate_block_conv_dim = post_aggregate_block_conv_dim

        self.ray_sampler = RaySampler(
            mode=surfaces_sampling_mode,
            surfaces_type=self.surfaces_type,
            n_steps=num_surfaces,
            min_distance=min_distance,
            max_distance=max_distance,
        )

        post_aggregate_block_input_size = self.input_surfaces_feats_size
        if self.agg_with_mean_var:
            post_aggregate_block_input_size += input_size * 2
        if self.agg_processor_hidden_size is not None:
            post_aggregate_block_input_size += input_size * 2

        assert self.post_aggregate_block_conv_dim in [2, 3], 'post_aggregate_block_conv_dim supports only 3d and 2d convs'
        self.post_aggregate_block = DoubleConvResnet(
            input_dim=post_aggregate_block_input_size,
            dims=post_aggregate_block_dims + [[post_aggregate_block_dims[-1][-1], self.output_surfaces_feats_size]],
            use_residual=True,
            output_activation=False,
            activation='elu',
            conv_dim=self.post_aggregate_block_conv_dim,
        )

        self.position_encoder = PositionalEncoderSelfAttention(
            dim=self.input_surfaces_feats_size
        )

        self.attention_agg = None

        if self.agg_processor_hidden_size is not None:
            self.attention_agg = MultiHeadAttention(
                        num_heads=1,
                        dim_input_v=input_size + 1,
                        dim_input_k=input_size + 1,
                        dim_hidden_v=agg_processor_hidden_size,
                        dim_hidden_k=agg_processor_hidden_size,
                        dim_output=input_size,
                        residual=False,
                    )

    def forward(self,
                source_features,
                source_cameras: CameraMultiple,
                reference_cameras: CameraMultiple,
                input_surfaces: torch.Tensor = None,
                output_surfaces_resolution: Tuple[int, int] = None,
                feats_resolution: Tuple[int, int] = None,
                layered_depth: torch.Tensor = None,
                source_weights: torch.Tensor = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
           Args:
               source_features: B x n_source x C x H x W or B x n_source x num_surfaces x C x H x W
               source_cameras: B x 1 x n_source x KRT
               reference_cameras: B x 1 x 1 x KRT
               input_surfaces: B x num_surfaces x input_surfaces_feats_size x H x W
               output_surfaces_resolution:
               source_weights: B x n_source x num_surfaces x C x H x W
           Returns:
               surfaces: B x num_surfaces x output_surfaces_feats_size x H x W
        """
        if self.input_surfaces_feats_size and input_surfaces is not None:
            assert input_surfaces.shape[2] == self.input_surfaces_feats_size, \
                f'self.input_surfaces_feats_size {self.input_surfaces_feats_size} ' \
                f'must be equal to input surfaces features size {input_surfaces.shape[2]} '
        if source_weights is not None:
            assert self.aggregation_with_weights is not False, 'flag aggregation_with_weights must be set!!!'

        self.ray_sampler.set_position(reference_cameras, mpi_camera=reference_cameras)

        batch_size, n_source, *_, input_feat_size, h, w = source_features.shape

        if feats_resolution is not None:
            h_rescale, w_rescale = feats_resolution
            if h != h_rescale or w != w_rescale:
                source_features_2d = F.interpolate(source_features.reshape(-1, *source_features.shape[-3:]),
                                                   size=(h_rescale, w_rescale), mode='bicubic')
                source_features = source_features_2d.reshape(*source_features.shape[:2],
                                                             *source_features_2d.shape[1:])

        h, w = output_surfaces_resolution


        ref_pixel_coords = get_grid(batch_size=batch_size,
                                    height=h,
                                    width=w,
                                    relative=True,
                                    values_range='sigmoid',
                                    align_corners=True if (layered_depth is not None) else False,
                                    device=source_features.device,
                                    )

        rescale_input_surfaces = False
        if input_surfaces is not None:
            rescale_input_surfaces = ((h, w) != input_surfaces.shape[-2:])

        if input_surfaces is not None:
            input_surfaces_feats = input_surfaces
            if rescale_input_surfaces:
                input_surfaces_feats_2d = F.interpolate(input_surfaces_feats.reshape(-1, *input_surfaces_feats.shape[-3:]), size=(h, w), mode='bicubic')
                input_surfaces_feats = input_surfaces_feats_2d.reshape(*input_surfaces_feats.shape[:2], *input_surfaces_feats_2d.shape[1:])
        else:
            input_surfaces_feats = torch.zeros([batch_size * h * w, self.num_surfaces,
                                                self.input_surfaces_feats_size],
                                   device=source_features.device)
            input_surfaces_feats = self.position_encoder(input_surfaces_feats)
            input_surfaces_feats = input_surfaces_feats.reshape(batch_size, h, w, self.num_surfaces,
                                                                self.input_surfaces_feats_size).permute(0, 3, 4, 1, 2)

        # for surface_idx in surfaces_indices:
        #  B x n_source x num_surfaces x C x H x W  /
        #  B x n_source x num_surfaces x 1 x H x W  /
        #  B x n_source x 1 x num_surfaces x H x W x XYZ
        process_all_surfaces_together = self.training or (self.post_aggregate_block_conv_dim == 3)
        if process_all_surfaces_together:
            surfaces_indices = [None]
            new_surfaces_feats = None
        else:
            surfaces_indices = range(self.num_surfaces)
            new_surfaces_feats = torch.empty([batch_size, self.num_surfaces, self.output_surfaces_feats_size, h, w],
                                                device=source_features.device)

        for surface_idx in surfaces_indices:
            unprojected_feats, mask, source_vectors_descriptors = \
                self.ray_sampler.project_on(source_features=source_features,
                                            source_camera=source_cameras,
                                            reference_pixel_coords=ref_pixel_coords,
                                            relative_intrinsics=True,
                                            return_source_vectors_displacement=False,
                                            surface_idx=surface_idx,
                                            return_displacement_namedtuple=True,
                                            depth_tensor=layered_depth,
                                            )

            if source_weights is not None and self.aggregation_with_weights:
                unprojected_weights, mask, source_vectors_descriptors = \
                    self.ray_sampler.project_on(source_features=source_weights,
                                                source_camera=source_cameras,
                                                reference_pixel_coords=ref_pixel_coords,
                                                relative_intrinsics=True,
                                                return_source_vectors_displacement=False,
                                                surface_idx=surface_idx,
                                                return_displacement_namedtuple=True,
                                                depth_tensor=layered_depth,
                                                )

            #   B x n_source x num_surfaces x 1 x H x W
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-7)
            if self.aggregation_with_weights:
                weight = weight * unprojected_weights

            surfaces_state = []
            if surface_idx is None:
                surfaces_state.append(input_surfaces_feats)
            else:
                surfaces_state.append(input_surfaces_feats[:, [surface_idx]])

            if self.agg_with_mean_var:
                #   B x 1 x num_surfaces x 2 * n_features x H x W
                surfaces_mean_var = torch.cat([*fused_mean_variance(unprojected_feats, weight)], dim=3)
                surfaces_state.append(surfaces_mean_var.squeeze(1))

            if self.attention_agg:
                feats = torch.cat([unprojected_feats, weight], axis=3)
                feats = feats.permute(0, 2, 4, 5, 1, 3)
                _, num_surfaces_curr, _, _, _, _ = feats.shape
                #  B * num_surfaces * H * W x n_source x C + 1
                feats = feats.reshape(-1, *feats.shape[-2:])
                feats, _ = self.attention_agg(feats, feats, feats)
                feats = feats.reshape(batch_size, num_surfaces_curr, h, w, n_source, -1)
                feats = feats.permute(0, 4, 1, 5, 2, 3)
                #   B x 1 x num_surfaces x 2 * n_features x H x W
                feats_mean_var = torch.cat([*fused_mean_variance(feats, weight)], dim=3)
                surfaces_state.append(feats_mean_var.squeeze(1))

            # B x num_surfaces x post_aggregate_block_input_size x H_novel x W_novel
            surfaces_feats = torch.cat(surfaces_state, dim=2)

            if self.post_aggregate_block_conv_dim == 3:
                # B x num_surfaces x output_surfaces_feats_size x H_novel x W_novel
                surfaces_feats = self.post_aggregate_block(surfaces_feats.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            else:
                # B * num_surfaces x output_surfaces_feats_size x H_novel x W_novel
                surfaces_feats = surfaces_feats.reshape(-1, *surfaces_feats.shape[2:])
                surfaces_feats = self.post_aggregate_block(surfaces_feats)
                surfaces_feats = surfaces_feats.reshape(batch_size, -1, *surfaces_feats.shape[1:])

            if not process_all_surfaces_together:
                new_surfaces_feats[:, [surface_idx]] = surfaces_feats
                del surfaces_feats, surfaces_state
                torch.cuda.empty_cache()
            else:
                new_surfaces_feats = surfaces_feats

        return new_surfaces_feats
