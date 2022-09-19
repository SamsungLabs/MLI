__all__ = ['GeneratorDeformedPSV',
           ]

import logging
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.modules.mesh_layers import build_layers_from_view
from lib.modules.cameras.utils import (convert_to_camera_pytorch3d,
                                       interpolate_cameras_pytorch3d,
                                       sample_camera_with_pixel_offset,
                                       )
from .gen_parts import SurfacesMPI, RendererMeshLayers
try:
    from .gen_parts import RasterizerFFrast
except:
    pass
from .gen_mpi_base import GeneratorMPIBase

logger = logging.getLogger(__name__)


class GeneratorDeformedPSV(GeneratorMPIBase):
    def __init__(self, params: dict):
        super().__init__(params=params)
        self.num_layers: int = self.params.get('num_layers', 1)

        self.independent_planes = self.params.get('independent_planes', False)
        self.use_opacity = self.params.get('use_opacity', True)
        self.reverse_compose_over = self.params.get('reverse_compose_over', False)
        self.use_color_activation = self.params.get('use_color_activation', True)
        self.calc_blending_weight_from_opacity = self.params.get('calc_blending_weight_from_opacity', False)
        self.predict_foreground = self.params.get('predict_foreground', False)
        self.predict_background = self.params.get('predict_background', True)
        self.mix_projected_source = self.params.get('mix_projected_source', False)
        self.use_3d_psv = self.params.get('use_3d_psv', False)

        self.depth_t = 1
        self.depth_postprocessing: Optional[str] = self.params.get('depth_postprocessing')
        self.detach_depth_from_color = self.params.get('detach_depth_from_color', False)
        self.detach_depth_from_verts = self.params.get('detach_depth_from_verts', False)
        self.annealing_depth_rate: float = self.params.get('annealing_depth_rate', 0.)
        self.default_background_depth: Optional[float] = self.params.get('default_background_depth')
        self.impact_key = self.params.get('impact_key', 'texel_weights')

        self.align_grid_corners = self.params.get('align_grid_corners', True)
        self.use_cloud_proxy = self.params.get('use_cloud_proxy', False)

        # novel_interpolation | offset_circle
        self.virtual_frame_mode: str = self.params.get('virtual_frame_mode', 'novel_interpolation')
        self.max_offset: float = self.params.get('max_offset', 0.2)
        self.annealing_virtual_frame_iterations: Optional[int] = self.params.get('annealing_virtual_frame_iterations')

        self.detach_before_decoder = self.params.get('detach_before_decoder', False)
        self.feed_opacity_to_decoder = self.params.get('feed_opacity_to_decoder', False)
        self.blend_decoder_input_output = self.params.get('blend_decoder_input_output', False)
        self.detach_decoder_input_while_blend = self.params.get('detach_decoder_input_while_blend', False)
        self.no_sort_depth = self.params.get('no_sort_depth', False)

        self.frozen_mesh_resolution: Optional[Tuple[int, int]] = self.params.get('frozen_mesh_resolution')

    def _init_modules(self,
                      rasterizer=None,
                      shader=None,
                      depth_estimator=None,
                      psv_net=None,
                      surfaces=None,
                      decoder=None
                      ):
        self.depth_predictor = depth_estimator
        self.psv_net = psv_net

        self.renderer = RendererMeshLayers(rasterizer, shader, self.params['num_layers'])

        self.surfaces: SurfacesMPI = surfaces
        self.decoder = decoder

    def calculate_opacity(self, psv: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Args:
            psv: B x n_layers x C x H x W
        Returns:
            opacity: B x n_layers x 1 x H x W
        """
        if not self.use_opacity:
            return None

        if self.reverse_compose_over:
            opacity = self._calc_opacity_from_reverse_compose_over(psv[..., -1:, :, :])
        else:
            opacity = torch.sigmoid(psv[..., -1:, :, :])
        return opacity

    @staticmethod
    def _calc_opacity_from_reverse_compose_over(tensor: torch.Tensor, mask_value: float = 0.5) -> torch.Tensor:
        """
        Calculate opacity from reverse compose-over op, in back-to-front manner.
        Args:
            tensor: B x n_layers x 1 x H x W
            mask_value: float

        Returns:
            opacity: B x n_layers x 1 x H x W
        """
        eps = 1e-6
        impact = torch.softmax(tensor, dim=1)
        cumsum = impact.cumsum(dim=1)
        opacity = impact / cumsum.add(eps)

        mask_of_hidden_verts = opacity.gt(eps).long().cumsum(dim=1).le(0)
        opacity = torch.where(mask_of_hidden_verts,
                              torch.ones_like(opacity) * mask_value,
                              opacity
                              )
        return opacity

    def postprocess_psv(self,
                        raw_output: torch.Tensor,
                        features: torch.Tensor,
                        projected_source: Optional[torch.Tensor] = None,
                        ) -> dict:
        """
        if  use_opacity == True, the last feature of output , i.e. out[..., -1:, :, :] is treated as
        opacity and has value between 0 and 1

        Args:
            raw_output: B x some_channels x H x W
            features: B x RGB x H x W
            projected_source: B x n_layers x RGB x H x W

        Returns:
            mpi: B x n_layers x RGBA x H x W
        """
        batch_size = features.shape[0]

        # ### Case of independent planes ###

        if self.independent_planes:
            if self.use_3d_psv:
                psv = raw_output.transpose(1, 2).contiguous()
            else:
                psv = raw_output.view(batch_size, self.num_layers, -1, *raw_output.shape[-2:])
            if self.use_opacity:
                colors, opacity = psv[..., :-1, :, :], psv[..., -1:, :, :]
                if self.reverse_compose_over:
                    opacity = self._calc_opacity_from_reverse_compose_over(opacity)
                else:
                    opacity = torch.sigmoid(opacity)
            else:
                colors, opacity = psv, None
            if self.use_color_activation:
                colors = torch.tanh(colors)
            if opacity is not None:
                mpi = torch.cat([colors, opacity], dim=-3)
            else:
                mpi = colors
            return dict(mpi=mpi)

        # ### Case of foreground-background mixing weights ###

        if self.mix_projected_source and self.predict_background:
            planes_multiplier = 4  # logits for bg, fg, source + opacity
        else:
            planes_multiplier = int(not self.calc_blending_weight_from_opacity) + 1
        features_multiplier = int(self.predict_foreground) + int(self.predict_background)
        if not self.use_3d_psv:
            if not features_multiplier:
                n_features = None
            else:
                n_features = (raw_output.shape[1] - self.num_layers * planes_multiplier) // features_multiplier
        else:
            n_features = projected_source.shape[-3]

        # Take foreground from raw_output
        if not self.predict_foreground:
            foreground = features[..., :n_features, :, :].unsqueeze(1)
        else:
            if not self.use_3d_psv:
                foreground = raw_output[:, :n_features].view(batch_size, 1, -1, *raw_output.shape[-2:])
                raw_output = raw_output[:, n_features:]
            else:
                foreground = raw_output[:, :, -1]
                raw_output = raw_output[:, :, :-1]
            if self.use_color_activation:
                foreground = torch.tanh(foreground)

        # Take background from raw_output
        if self.predict_background:
            if not self.use_3d_psv:
                background = raw_output[:, :n_features].view(batch_size, 1, -1, *raw_output.shape[-2:])
                raw_output = raw_output[:, n_features:]
            else:
                background = raw_output[:, :, 0]
                raw_output = raw_output[:, :, 1:]
            if self.use_color_activation:
                background = torch.tanh(background)
        else:
            background = None

        # Take opacities and weights per each plane from raw_output
        if not self.use_3d_psv:
            raw_output = raw_output.view(batch_size, self.num_layers, -1, *raw_output.shape[-2:])
        else:
            raw_output = raw_output.transpose(1, 2).contiguous()
            # Currently this may lead to unused channels below: we use just one or two features from each channel
            # If fg or bg were predicted before, then each layer has 3 channels.

        if not self.calc_blending_weight_from_opacity:
            if not (self.mix_projected_source and self.predict_background):
                weight = torch.sigmoid(raw_output[..., :1, :, :])
                raw_output = raw_output[..., 1:, :, :]
            else:
                weight = torch.softmax(raw_output[..., :3, :, :], dim=-3)
                raw_output = raw_output[..., 3:, :, :]

        opacity = self.calculate_opacity(raw_output)

        if self.calc_blending_weight_from_opacity:  # treat PSV as back-to-front
            transmittance = 1 - opacity + 1e-8
            weight = transmittance.flip(1)[:, :-1].cumprod(dim=1)
            weight = torch.cat([torch.ones_like(weight[:, :1]), weight], dim=1).flip(1)

        if self.mix_projected_source:
            if not self.predict_background:
                feature = foreground * weight + projected_source * (1 - weight)
            else:
                feature = (weight[:, :, :1] * foreground
                           + weight[:, :, 1:2] * background
                           + weight[:, :, 2:3] * projected_source
                           )
        else:
            feature = foreground * weight + background * (1 - weight)

        if opacity is not None:
            mpi = torch.cat([feature, opacity], dim=-3)
        else:
            mpi = feature
        return dict(mpi=mpi)

    @staticmethod
    def _sort_layered_depth(depth: torch.Tensor,
                            back_to_front: bool = True,
                            return_order: bool = False,
                            no_sort: bool = False,
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            depth: B x N x H x W
            back_to_front: back-to-front of front-to-back

        Returns:
            out: B x N x H x W
        """
        if no_sort:
            return depth
        avg_per_layer = depth.contiguous().view(*depth.shape[:-2], -1).mean(dim=-1)  # B x N
        order = avg_per_layer.argsort(dim=-1, descending=back_to_front)  # B x N
        sorted_depth = depth.gather(dim=1, index=order[..., None, None].expand_as(depth))
        if return_order:
            return sorted_depth, order
        else:
            return sorted_depth

    def _postprocess_depth(self,
                           depth: torch.Tensor,
                           mode: Optional[str] = None,
                           ) -> torch.Tensor:
        """
        Postprocess depth and return layers sorted in back-to-front manner (in average).
        This format better suits visualisation purposes.

        Args:
            depth: B x N x H x W
            mode: None | 'none' | 'cumsum' | 'offset' | 'softmax' | 'sigmoid'

        Returns:
            out: B x N x H x W
        """
        if mode is None:
            mode = self.depth_postprocessing

        if mode in {None, 'none'}:
            final_depth = self._sort_layered_depth(depth, no_sort=self.no_sort_depth)
        elif mode == 'cumsum':
            final_depth = depth.cumsum(dim=1).flip(1)
        elif mode == 'offset':
            final_depth = self._sort_layered_depth(depth + self.surfaces.mpi_depths[None, :, None, None],
                                                   no_sort=self.no_sort_depth)
        elif mode == 'softmax':
            n_planes = len(self.surfaces.mpi_depths)
            depth = depth.contiguous().view(depth.shape[0], -1, n_planes, *depth.shape[2:])
            depth = torch.sum(torch.softmax(depth, dim=2) * self.surfaces.mpi_depths[None, None, :, None, None], dim=2)
            final_depth = self._sort_layered_depth(depth, no_sort=self.no_sort_depth)
        elif mode == 'sigmoid':
            # MPI surfaces are listed in back-to-front manner
            low_bound = torch.cat([self.surfaces.mpi_depths[1:],
                                   torch.zeros_like(self.surfaces.mpi_depths[:1]),
                                   ])
            segment_len = self.surfaces.mpi_depths[:-1] - self.surfaces.mpi_depths[1:]
            segment_len = torch.cat([segment_len, self.surfaces.mpi_depths[-1:]])
            depth = torch.sigmoid(depth) / self.depth_t
            depth = depth * segment_len[None, :, None, None] + low_bound[None, :, None, None]
            final_depth = self._sort_layered_depth(depth, no_sort=self.no_sort_depth)
        else:
            raise ValueError(f'Unknown depth postprocessing mode {mode}')

        if self.default_background_depth is not None:
            # keep back-to-front order
            final_depth = torch.cat([torch.ones_like(final_depth[:, :1]) * self.default_background_depth,
                                     final_depth,
                                     ], dim=1)

        return final_depth

    def decode_and_refine(self,
                          image: torch.Tensor,
                          opacity: torch.Tensor,
                          ) -> Optional[torch.Tensor]:
        """
        Apply decoder/enhancer to the shaded view.

        Args:
            image: B x C x H x W
            opacity: B x 1 x H x W

        Returns:
            enhanced: B x C' x H x W
        """
        if self.decoder is None:
            return None

        if self.feed_opacity_to_decoder:
            decoder_input = torch.cat([image, opacity], dim=-3)
        else:
            decoder_input = image
        if self.detach_before_decoder:
            decoder_input = decoder_input.detach()

        enhanced = self.decoder(decoder_input)
        if self.blend_decoder_input_output:
            assert image.shape[1] == enhanced.shape[1], \
                f'Number of channel should be equal. Got {image.shape[1]} and {enhanced.shape[1]} instead.'
            enhanced = ((1 - opacity) * enhanced
                        + opacity * (image.detach() if self.detach_decoder_input_while_blend else image)
                        )
        return enhanced

    def render_with_proxy_geometry(self,
                                    novel_cameras: CameraMultiple,
                                    proxy_geometry: dict,
                                    ) -> dict:
        mpi = proxy_geometry['mpi']
        novel_cameras = convert_to_camera_pytorch3d(novel_cameras)
        height, width = mpi.shape[-2:]
        render_output: dict = self.renderer(novel_cameras,
                                            proxy_geometry['verts'],
                                            proxy_geometry['faces'],
                                            verts_rgb=proxy_geometry['verts_rgb'],
                                            custom_resolution=(height, width),
                                            )
        novel_image, novel_depth = render_output['images'][:, 1:], render_output['images'][:, :1]
        novel_impact = render_output['texel_weights'].unsqueeze(2)
        opacity = render_output['opacity']
        decoded_novel_image = self.decode_and_refine(novel_image, opacity)

        mpi_impact = self.renderer.shader.do_blending(
            mpi.flip(1).permute(0, 3, 4, 1, 2).contiguous()
        )['texel_weights'].permute(0, 3, 4, 1, 2).flip(1)  # B x N x 1 x H x W

        return dict(
            novel_image=novel_image,
            decoded_novel_image=decoded_novel_image,
            novel_opacity=opacity,
            novel_depth=novel_depth,
            mpi=mpi,
            mpi_impact=mpi_impact,
            novel_impact=novel_impact,
            proxy_geometry=proxy_geometry,
        )

    def forward(self, *args, **kwargs):
        action = kwargs.pop('action', 'forward')
        if action == 'forward':
            return self.manual_forward(*args, **kwargs)
        elif action == 'render_with_proxy_geometry':
            return self.render_with_proxy_geometry(*args, **kwargs)
        elif action == 'get_grid':
            return self.grid
        else:
            raise ValueError(f'Unknown action={action}')

    def manual_forward(self,
                reference_images,
                source_images,
                reference_cameras: CameraMultiple,
                source_cameras: CameraMultiple,
                novel_cameras: CameraMultiple,
                iteration: Optional[int] = None,
                virtual_frame_interpolation_coef: Optional[float] = None,
                ):

        # current pipeline handles the only reference, the only source and the only novel camera
        assert reference_images.shape[1:3] == (1, 1)
        assert source_images.shape[1:3] == (1, 1)
        assert reference_cameras.cameras_shape[1:] == (1, 1)
        assert source_cameras.cameras_shape[1:] == (1, 1)
        assert novel_cameras.cameras_shape[1:] == (1, 1)

        if iteration is None:
            self.depth_t = 1
        else:
            self.depth_t = 1 + self.annealing_depth_rate / (iteration + 1e-6)

        self.surfaces.set_position(reference_cameras)
        batch_size, n_ref, n_source, *_, height, width = reference_images.shape
        if self.frozen_mesh_resolution is not None:
            height, width = self.frozen_mesh_resolution
        reference_pixel_coords = self._get_grid(features=None,
                                                height=height,
                                                width=width,
                                                relative=True,
                                                values_range='sigmoid',
                                                align_corners=self.align_grid_corners,
                                                device=reference_images.device,
                                                ).expand(batch_size, *([-1] * 5))

        psv = self.surfaces.project_on(source_features=source_images,
                                       source_camera=source_cameras,
                                       reference_pixel_coords=reference_pixel_coords,
                                       relative_intrinsics=True,
                                       )
        if self.frozen_mesh_resolution is not None:
            resampled_reference_images = F.interpolate(reference_images[:, 0, 0],
                                                       size=(height, width),
                                                       mode='bilinear',
                                                       )[:, None, None, :, :, :]
        else:
            resampled_reference_images = reference_images
        psv = torch.cat([psv, resampled_reference_images.unsqueeze(-4)], dim=-4)
        # B x n_layers x H x W
        need_depth_interpolation = (height % 128) != 0 or (width % 128) != 0
        if need_depth_interpolation:
            new_size = (min([height, 256]), min([width, 256]))
            interpolated_psv = F.interpolate(psv.contiguous().view(batch_size, -1, *psv.shape[-2:]), size=new_size)
        else:
            interpolated_psv = psv.contiguous().view(batch_size, -1, *psv.shape[-2:])
        depth_tensor = self.depth_predictor(interpolated_psv)
        depth_tensor = self._postprocess_depth(depth_tensor)
        if need_depth_interpolation:
            depth_tensor = F.interpolate(depth_tensor, size=(height, width), mode='bilinear')
        # B x 1 x 1 x n_layers x H x W x XYZ
        verts = reference_cameras.pixel_to_world(
            reference_pixel_coords.unsqueeze(-4).expand(-1, -1, -1, self.num_layers, -1, -1, -1),
            depth=depth_tensor[:, None, None, :, :, :, None],
        )

        # B x 1 x 1 x n_layers x H x W x 1   -->   N x n_layers x H x W
        verts_depth_for_novel = novel_cameras.world_to_depth(verts)[:, 0, 0, :, :, :, 0]

        # B x 1 x 1 x n_layers x H x W x UV
        verts_projected = source_cameras.world_to_pixel(verts)  # relative pixel coords in range [0, 1]
        if self.detach_depth_from_color:
            verts_projected = verts_projected.detach()

        # B x n_layers x C x H x W
        psv_deformed = F.grid_sample(
            source_images[:, 0, 0]
                .unsqueeze(1)
                .expand(-1, self.num_layers, -1, -1, -1)
                .contiguous()
                .view(batch_size * self.num_layers, -1, *source_images.shape[-2:]),  # B*n_layers x C x H x W
            grid=verts_projected.view(-1, *verts_projected.shape[-3:]) * 2 - 1,  # B*n_layers x H x W x UV
            mode='bilinear', padding_mode='zeros',
        ).view(batch_size, self.num_layers, -1, height, width)

        if not self.use_3d_psv:
            psv_deformed_and_reference = torch.cat([
                psv_deformed,
                resampled_reference_images[:, 0, 0].unsqueeze(1)
            ], dim=1).transpose(1, 2).contiguous()  # B x C x (n_layers + 1) x H x W

            # B x C*(n_layers+1) x H x W
            psv_deformed_and_reference = psv_deformed_and_reference.view(batch_size, -1, *psv_deformed.shape[-2:])
        else:
            psv_deformed_and_reference = torch.cat([
                psv_deformed,
                resampled_reference_images[:, 0, 0].unsqueeze(1).expand(*psv_deformed.shape[:2], -1, -1, -1),
            ], dim=2).transpose(1, 2).contiguous()  # B x (C + RGB) x n_layers x H x W
        processed_psv_deformed = self.psv_net(psv_deformed_and_reference)
        verts_features = self.postprocess_psv(processed_psv_deformed, resampled_reference_images[:, 0, 0], psv_deformed)
        mpi = verts_features['mpi']  # B x N x RGBA x H x W
        mpi_with_depth = torch.cat([verts_depth_for_novel.unsqueeze(2), mpi], dim=2)
        reference_cameras = convert_to_camera_pytorch3d(reference_cameras)
        novel_cameras = convert_to_camera_pytorch3d(novel_cameras)

        # if height != width:
        #     interpolated_depth = F.interpolate(depth_tensor, size=(min(height, width)))
        #     interpolated_mpi = F.interpolate(mpi_with_depth,
        #                                      size=(mpi_with_depth.shape[-3], min(height, width), min(height, width)))
        # else:
        interpolated_depth = depth_tensor
        interpolated_mpi = mpi_with_depth
        verts, faces, _, verts_features_with_depth = build_layers_from_view(interpolated_mpi,
                                                                            interpolated_depth,
                                                                            reference_cameras,
                                                                            align_corners=self.align_grid_corners,
                                                                            only_cloud=self.use_cloud_proxy,
                                                                            )
        if self.detach_depth_from_verts:
            verts_for_rendering = verts.detach()
        else:
            verts_for_rendering = verts

        render_output: dict = self.renderer(novel_cameras, verts_for_rendering, faces,
                                            custom_resolution=(height, width),
                                            verts_rgb=verts_features_with_depth,
                                            )
        novel_image, novel_depth = render_output['images'][:, 1:], render_output['images'][:, :1]
        novel_opacity = render_output['opacity']
        decoded_novel_image = self.decode_and_refine(novel_image, novel_opacity)

        novel_impact = render_output['texel_weights'].unsqueeze(2)
        mpi_impact = self.renderer.shader.do_blending(
            mpi.flip(1).permute(0, 3, 4, 1, 2).contiguous()
        )[self.impact_key].permute(0, 3, 4, 1, 2).flip(1)  # B x N x 1 x H x W

        if iteration is not None and (self.annealing_virtual_frame_iterations is not None
                                      or virtual_frame_interpolation_coef is not None):
            if virtual_frame_interpolation_coef is None:
                virtual_frame_interpolation_coef = min(1, iteration / self.annealing_virtual_frame_iterations)
            if self.virtual_frame_mode == 'novel_interpolation':
                virtual_cameras = interpolate_cameras_pytorch3d(start=reference_cameras,
                                                                end=novel_cameras,
                                                                timestamp=virtual_frame_interpolation_coef,
                                                                )
            elif self.virtual_frame_mode == 'offset_circle':
                virtual_cameras = sample_camera_with_pixel_offset(
                    camera=reference_cameras,
                    offset=virtual_frame_interpolation_coef * self.max_offset,
                    scale_y=1,
                )
            else:
                raise ValueError(f'Unknown mode: {self.virtual_frame_mode}')
            virtual_render_output: dict = self.renderer(
                virtual_cameras, verts_for_rendering, faces,
                verts_rgb=verts_features_with_depth,  # the depth is not for virtual but for novel camera
            )
            virtual_image = virtual_render_output['images'][:, 1:]
            virtual_opacity = virtual_render_output['opacity']
            decoded_virtual_image = self.decode_and_refine(virtual_image, virtual_opacity)
        else:
            virtual_image = virtual_opacity = decoded_virtual_image = None

        return dict(
            novel_image=novel_image,
            novel_opacity=novel_opacity,
            novel_depth=novel_depth,
            mpi=mpi,
            mpi_impact=mpi_impact,
            novel_impact=novel_impact,
            mesh_verts=verts,
            mesh_faces=faces,
            decoded_novel_image=decoded_novel_image,
            layered_depth=depth_tensor,
            mesh_verts_rgb=verts_features_with_depth,
            virtual_image=virtual_image,
            virtual_opacity=virtual_opacity,
            decoded_virtual_image=decoded_virtual_image,
        )