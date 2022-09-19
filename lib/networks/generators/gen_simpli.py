__all__ = ['GeneratorSIMPLI']

import math

import torch
import torch.nn.functional as F
from typing import Optional

from lib.modules.cameras import CameraMultiple
from lib.utils.base import get_grid
from lib.modules.mesh_layers import build_layers_from_view
from lib.modules.cameras.utils import convert_to_camera_pytorch3d
from lib.modules.cameras import CameraPytorch3d
from .gen_parts import RendererMeshLayers
from .gen_base import GeneratorBase

try:
    from .gen_parts import RasterizerFFrast
except:
    pass


class GeneratorSIMPLI(GeneratorBase):
    def _init_modules(self,
                      features_extractor,
                      composer,
                      rasterizer,
                      shader,
                      sampi_block_lvl0,
                      alpha_decoder_lvl0,
                      sampi_block_lvl1,
                      planes_deformer_lvl1=None,
                      depth_decoder_lvl1=None,
                      rgba_decoder_lvl1=None,
                      rgba_decoder_lvl2=None,
                      alpha_decoder_lvl1=None,
                      samli_block_lvl2=None,
                      samli_block_lvl3=None,
                      rgba_decoder_lvl3=None,
                      samli_block_lvl4=None,
                      rgba_decoder_lvl4=None,
                      last_layer_processor=None,
                      composer_mpi=None,
                      ):

        self.num_layers: int = self.params.get('num_layers', 4)
        self.grid_generator: str = self.params.get('grid_generator', 'gen_quad_planes')
        self.disable_error: bool = self.params.get('disable_error', False)
        self.max_depth: int = self.params.get('max_depth', 100)
        self.min_depth: int = self.params.get('min_depth', 1)
        self.reprocess_upsampled_depth: bool = self.params.get('reprocess_upsampled_depth', False)
        self.use_intersected_groups: int = self.params.get('use_intersected_groups', False)

        self.scale_feat_extractor: float = self.params.get('scale_feat_extractor', 4)
        self.scale_lvl0: float = self.params.get('scale_lvl0', 4)
        self.scale_lvl1: float = self.params.get('scale_lvl1', 4)
        self.scale_lvl2: float = self.params.get('scale_lvl2', 4)
        self.scale_repr_lvl0 = self.params.get('scale_repr_lvl0', [self.scale_lvl0, self.scale_lvl0])
        self.scale_repr_lvl1 = self.params.get('scale_repr_lvl1', [self.scale_lvl1, self.scale_lvl1])
        self.scale_repr_lvl2 = self.params.get('scale_repr_lvl2', [self.scale_lvl2, self.scale_lvl2])

        self.lvl2_error_with_image = self.params.get('lvl2_error_with_image', False)
        self.lvl3_error_with_image = self.params.get('lvl3_error_with_image', False)
        self.lvl4_error_with_image = self.params.get('lvl4_error_with_image', False)
        # self.num_correction_after_stage_2 = self.params.get('num_correction_after_stage_2', 0)

        self.mpi_mode = False
        self.composer_mpi = composer_mpi
        if composer_mpi:
            self.mpi_mode = True

        self.sampi_block_lvl0 = sampi_block_lvl0
        self.alpha_decoder_lvl0 = alpha_decoder_lvl0

        self.sampi_block_lvl1 = sampi_block_lvl1
        self.planes_deformer_lvl1 = planes_deformer_lvl1
        self.depth_decoder_lvl1 = depth_decoder_lvl1
        self.rgba_decoder_lvl1 = rgba_decoder_lvl1

        self.alpha_decoder_lvl1 = alpha_decoder_lvl1

        self.samli_block_lvl2 = samli_block_lvl2
        self.rgba_decoder_lvl2 = rgba_decoder_lvl2

        self.samli_block_lvl3 = samli_block_lvl3
        self.rgba_decoder_lvl3 = rgba_decoder_lvl3

        self.samli_block_lvl4 = samli_block_lvl4
        self.rgba_decoder_lvl4 = rgba_decoder_lvl4

        self.last_layer_processor = last_layer_processor

        self.features_extractor = features_extractor
        self.composer = composer
        self.renderer = RendererMeshLayers(rasterizer, shader, self.num_layers)

    def forward(self, *args, **kwargs):
        action = kwargs.pop('action', 'forward')
        if action == 'forward':
            return self.manual_forward(*args, **kwargs)
        elif action == 'render_with_proxy_geometry':
            return self.render_with_proxy_geometry(*args, **kwargs)
        else:
            raise ValueError(f'Unknown action={action}')

    def compute_error_to_sources(self,
                                 batch_size,
                                 h,
                                 w,
                                 mpi,
                                 ray_sampler,
                                 source_cameras,
                                 source_features,
                                 reference_cameras,
                                 error_with_weight=False,
                                 disable_error=False,
                                 ):

        if disable_error:
            return F.interpolate(source_features.reshape(-1, *source_features.shape[-3:]),
                                 size=(h, w)).reshape(*source_features.shape[:3], h, w)

        source_pixel_coords = get_grid(batch_size=batch_size,
                                       height=h,
                                       width=w,
                                       relative=True,
                                       values_range='sigmoid',
                                       align_corners=False,
                                       device=source_features.device,
                                       )[:, None, None]
        source_pixel_coords = source_pixel_coords.expand(*source_cameras.cameras_shape,
                                                         *source_pixel_coords.shape[-3:])

        mpi_reproject, timestamps, frustum_mask = ray_sampler.look_at(
            reference_features=mpi,
            reference_cameras=reference_cameras,
            novel_camera=source_cameras,
            novel_pixel_coords=source_pixel_coords,
            relative_intrinsics=True,
        )
        composer_result = self.composer(mpi_reproject,
                                        return_namedtuple=True,
                                        )
        source_views_projections = composer_result.color
        error = (source_views_projections
                 - F.interpolate(source_features.reshape(-1, *source_features.shape[-3:]),
                                 size=(h, w)).reshape(source_views_projections.shape)
                 )
        if error_with_weight:
            return error, source_views_projections, composer_result.weight

        return error, source_views_projections

    def compute_error_on_mesh(self,
                              h,
                              w,
                              mli,
                              source_cameras,
                              source_features,
                              err_with_image=False,
                              disable_error=False
                              ):
        num_source = source_cameras.cameras_shape[-1]
        opacity, feats, weights = [], [], []

        if disable_error:
            return F.interpolate(source_features.reshape(-1, *source_features.shape[-3:]),
                                 size=(h, w)).reshape(*source_features.shape[:3], h, w)
        for i in range(num_source):
            render_output: dict = self.renderer(
                self.convert_to_camera_pytorch3d(source_cameras, novel_index=i),
                mli['verts'], mli['faces'],
                custom_resolution=(h, w),
                verts_rgb=mli['verts_features'],
            )
            feats.append(render_output['images'])
        source_views_projections = torch.stack(feats, dim=1)
        error = (source_views_projections -
                 F.interpolate(source_features.reshape(-1, *source_features.shape[-3:]),
                               size=(h, w)).reshape(*source_features.shape[:3], h, w)
                 )

        if err_with_image:
            error = torch.cat([error, source_features], axis=-3)

        return error, source_views_projections

    def manual_forward(self,
                       source_images,
                       source_cameras: CameraMultiple,
                       reference_cameras: CameraMultiple,
                       ) -> dict:
        """
        Args:
            source_images: B x n_source x C x H x W
            source_cameras: B x 1 x n_source x KRT
            reference_cameras: B x 1 x 1 x KRT

        Returns:
            rgba: B x 1 x C x H x W
            opacity: B x 1 x 1 x H x W
            mpi: B x 1 x n_steps x RGBA x H x W
        """

        batch_size, num_source_images, _, source_h, source_w = source_images.shape

        h, w = reference_cameras.images_size

        fe_size = int(source_h * self.scale_feat_extractor), int(source_w * self.scale_feat_extractor)
        lvl0_size = int(source_h * self.scale_lvl0), int(source_w * self.scale_lvl0)
        lvl1_size = int(source_h * self.scale_lvl1), int(source_w * self.scale_lvl1)
        lvl2_size = int(source_h * self.scale_lvl2), int(source_w * self.scale_lvl2)

        lvl0_rep_size = int(h * self.scale_repr_lvl0[0]), int(w * self.scale_repr_lvl0[1])
        lvl1_rep_size = int(h * self.scale_repr_lvl1[0]), int(w * self.scale_repr_lvl1[1])
        lvl2_rep_size = int(h * self.scale_repr_lvl2[0]), int(w * self.scale_repr_lvl2[1])

        if fe_size != (h, w):
            fe_source_images = F.interpolate(
                source_images.view(batch_size * num_source_images, *source_images.shape[-3:]),
                size=fe_size,
                mode='bicubic')
        else:
            fe_source_images = source_images.view(batch_size * num_source_images, *source_images.shape[-3:])

        if self.training:
            source_features = self.features_extractor(fe_source_images)
        else:
            source_features_list = []
            for i in range(num_source_images):
                source_features_list.append(self.features_extractor(fe_source_images[[i]]))

            source_features = torch.cat(source_features_list, dim=0)
            del source_features_list
            torch.cuda.empty_cache()


        _, feats_dim, source_feats_h, source_feats_w = source_features.shape
        source_features = source_features.reshape(batch_size,
                                                  num_source_images,
                                                  feats_dim,
                                                  source_feats_h,
                                                  source_feats_w
                                                  )

        intermediate_errors = []
        intermediate_sources = []

        ## LEVEL 0
        ## ------------------------------------------------------------------------------------------------------------
        raw_mpi_feats = self.sampi_block_lvl0(
            source_features=source_features,
            source_cameras=source_cameras,
            reference_cameras=reference_cameras,
            output_surfaces_resolution=lvl0_rep_size,
            feats_resolution=lvl0_size
        )

        mpi_alpha = torch.sigmoid(self.alpha_decoder_lvl0(raw_mpi_feats.reshape(-1, *raw_mpi_feats.shape[-3:])))
        mpi_alpha = mpi_alpha.reshape(batch_size, -1, *mpi_alpha.shape[-3:])

        raw_mpi_feats[:, -3:] = torch.tanh(raw_mpi_feats[:, -3:]).clamp(min=-1, max=1)
        mpi_feats_a = torch.cat([raw_mpi_feats, mpi_alpha], dim=-3)

        feats_error_lvl1 = self.compute_error_to_sources(
            batch_size=batch_size,
            h=lvl1_size[0],
            w=lvl1_size[1],
            mpi=mpi_feats_a,
            ray_sampler=self.sampi_block_lvl1.ray_sampler,
            source_cameras=source_cameras,
            source_features=source_features,
            reference_cameras=reference_cameras,
            error_with_weight=self.sampi_block_lvl1.aggregation_with_weights,
            disable_error=self.disable_error
        )

        if not self.training:
            del source_features
            torch.cuda.empty_cache()

        source_psv_weights_lvl1 = None
        if self.sampi_block_lvl1.aggregation_with_weights:
            source_psv_weights_lvl1 = feats_error_lvl1[2]

        intermediate_errors.append(feats_error_lvl1[0][..., -3:, :, :])
        intermediate_sources.append(feats_error_lvl1[1][..., -3:, :, :])
        feats_error_lvl1 = feats_error_lvl1[0]

        ## LEVEL 1
        ## ------------------------------------------------------------------------------------------------------------
        raw_mpi_feats = self.sampi_block_lvl1(
            input_surfaces=mpi_feats_a,
            source_features=feats_error_lvl1,
            source_cameras=source_cameras,
            reference_cameras=reference_cameras,
            output_surfaces_resolution=lvl1_rep_size,
            feats_resolution=lvl1_size,
            source_weights=source_psv_weights_lvl1
        )

        if not self.training:
            del feats_error_lvl1
            torch.cuda.empty_cache()

        if self.mpi_mode:
            raw_mli_feats = raw_mpi_feats
        else:
            raw_mli_feats = self.planes_deformer_lvl1(
                raw_mpi_feats,
                positions=self.sampi_block_lvl1.ray_sampler.surfaces.disparities(normalize=True).unsqueeze(0)
            )
            # B * num_layers x num_surf_per_group x H x W
            raw_depth = self.depth_decoder_lvl1(raw_mli_feats.reshape(-1, *raw_mli_feats.shape[-3:]))
            # B x num_layers x num_surf_per_group x H x W
            raw_depth = raw_depth.reshape(batch_size, -1, 1, *raw_depth.shape[-2:])

            layered_depth = self.process_depth(
                raw_depth,
                self.sampi_block_lvl1.ray_sampler.surfaces.mpi_depths
            )

        if self.rgba_decoder_lvl1 is not None:
            raw_mli_predicted = self.rgba_decoder_lvl1(raw_mli_feats.reshape(-1, *raw_mli_feats.shape[-3:]))
            raw_mli_predicted = raw_mli_predicted.reshape(batch_size, -1, *raw_mli_predicted.shape[-3:])
            raw_mli_predicted[:, :, :3] = torch.tanh(raw_mli_predicted[:, :, :3]).clamp(min=-1, max=1)
            raw_mli_predicted[:, :, -1:] = torch.sigmoid(raw_mli_predicted[:, :, -1:])
        else:
            raw_mli_predicted = raw_mli_feats

        if self.mpi_mode:
            out = dict(
                mpi=raw_mli_feats[:, None],
                intermediate_error=intermediate_errors
            )
            return out

        ## LEVEL 2
        ## ------------------------------------------------------------------------------------------------------------

        if self.samli_block_lvl2 is not None:

            # B x N x C x H x W
            mesh_verts, mesh_faces, _, mesh_verts_rgb = build_layers_from_view(
                raw_mli_predicted,
                layered_depth,
                convert_to_camera_pytorch3d(reference_cameras),
                align_corners=True,
                grid_generator = self.grid_generator,
            )

            feats_error_lvl2, inter_sources_lvl2 = self.compute_error_on_mesh(
                h=lvl2_size[0],
                w=lvl2_size[1],
                mli={'verts': mesh_verts, 'faces': mesh_faces, 'verts_features': mesh_verts_rgb},
                source_cameras=source_cameras,
                source_features=source_images,
                err_with_image=self.lvl2_error_with_image,
                disable_error=self.disable_error
            )

            # intermediate_errors.append(feats_error_lvl2[..., :3, :, :])
            intermediate_sources.append(inter_sources_lvl2[..., :3, :, :])

            if lvl2_rep_size != lvl1_rep_size:
                raw_depth = raw_depth.reshape(batch_size, -1, *raw_depth.shape[-2:])
                raw_depth = F.interpolate(raw_depth,
                                          size=[lvl2_rep_size[0], lvl2_rep_size[1]], mode='bicubic')
                raw_depth = raw_depth.reshape(batch_size, -1, 1, *raw_depth.shape[-2:])
                layered_depth = self.process_depth(
                    raw_depth,
                    self.sampi_block_lvl1.ray_sampler.surfaces.mpi_depths
                )

            raw_mli_feats = self.samli_block_lvl2(
                source_features=feats_error_lvl2,
                source_cameras=source_cameras,
                reference_cameras=reference_cameras,
                input_surfaces=raw_mli_feats,
                output_surfaces_resolution=lvl2_rep_size,
                feats_resolution=lvl2_size,
                layered_depth=layered_depth,
            )

            if self.rgba_decoder_lvl2 is not None:
                raw_mli_predicted = self.rgba_decoder_lvl2(raw_mli_feats.reshape(-1, *raw_mli_feats.shape[-3:]))
                raw_mli_predicted = raw_mli_predicted.reshape(batch_size, -1, *raw_mli_predicted.shape[-3:])
            else:
                raw_mli_predicted = raw_mli_feats

            raw_mli_predicted[:, :, :3] = torch.tanh(raw_mli_predicted[:, :, :3]).clamp(min=-1, max=1)
            raw_mli_predicted[:, :, -1:] = torch.sigmoid(raw_mli_predicted[:, :, -1:])


        ## LEVEL 3
        ## ------------------------------------------------------------------------------------------------------------

        if self.samli_block_lvl3 is not None:

            # B x N x C x H x W
            mesh_verts, mesh_faces, _, mesh_verts_rgb = build_layers_from_view(
                raw_mli_predicted,
                layered_depth,
                convert_to_camera_pytorch3d(reference_cameras),
                align_corners=True,
                grid_generator = self.grid_generator,
            )


            feats_error_lvl3, inter_sources_lvl3 = self.compute_error_on_mesh(
                h=lvl2_size[0],
                w=lvl2_size[1],
                mli={'verts': mesh_verts, 'faces': mesh_faces, 'verts_features': mesh_verts_rgb},
                source_cameras=source_cameras,
                source_features=source_images,
                err_with_image=self.lvl3_error_with_image,
                disable_error=self.disable_error
            )
            # intermediate_errors.append(feats_error_lvl3[..., :3, :, :])
            intermediate_sources.append(inter_sources_lvl3[..., :3, :, :])

            raw_mli_feats = self.samli_block_lvl3(
                source_features=feats_error_lvl3,
                source_cameras=source_cameras,
                reference_cameras=reference_cameras,
                input_surfaces=raw_mli_feats,
                output_surfaces_resolution=lvl2_rep_size,
                feats_resolution=lvl2_size,
                layered_depth=layered_depth,
            )

            if self.rgba_decoder_lvl3 is not None:
                raw_mli_predicted = self.rgba_decoder_lvl3(raw_mli_feats.reshape(-1, *raw_mli_feats.shape[-3:]))
                raw_mli_predicted = raw_mli_predicted.reshape(batch_size, -1, *raw_mli_predicted.shape[-3:])
            else:
                raw_mli_predicted = raw_mli_feats

            raw_mli_predicted[:, :, :3] = torch.tanh(raw_mli_predicted[:, :, :3]).clamp(min=-1, max=1)
            raw_mli_predicted[:, :, -1:] = torch.sigmoid(raw_mli_predicted[:, :, -1:])

        ## LEVEL 4
        ## ------------------------------------------------------------------------------------------------------------

        if self.samli_block_lvl4 is not None:

            # B x N x C x H x W
            mesh_verts, mesh_faces, _, mesh_verts_rgb = build_layers_from_view(
                raw_mli_predicted,
                layered_depth,
                convert_to_camera_pytorch3d(reference_cameras),
                align_corners=True,
                grid_generator = self.grid_generator,
            )


            feats_error_lvl4, inter_sources_lvl4 = self.compute_error_on_mesh(
                h=lvl2_size[0],
                w=lvl2_size[1],
                mli={'verts': mesh_verts, 'faces': mesh_faces, 'verts_features': mesh_verts_rgb},
                source_cameras=source_cameras,
                source_features=source_images,
                err_with_image=self.lvl4_error_with_image,
                disable_error=self.disable_error
            )

            # intermediate_errors.append(feats_error_lvl4[..., :3, :, :])
            intermediate_sources.append(inter_sources_lvl4[..., :3, :, :])

            raw_mli_feats = self.samli_block_lvl4(
                source_features=feats_error_lvl4,
                source_cameras=source_cameras,
                reference_cameras=reference_cameras,
                input_surfaces=raw_mli_feats,
                output_surfaces_resolution=lvl2_rep_size,
                feats_resolution=lvl2_size,
                layered_depth=layered_depth,
            )

            if self.rgba_decoder_lvl3 is not None:
                raw_mli_predicted = self.rgba_decoder_lvl4(raw_mli_feats.reshape(-1, *raw_mli_feats.shape[-3:]))
                raw_mli_predicted = raw_mli_predicted.reshape(batch_size, -1, *raw_mli_predicted.shape[-3:])
            else:
                raw_mli_predicted = raw_mli_feats

            raw_mli_predicted[:, :, :3] = torch.tanh(raw_mli_predicted[:, :, :3]).clamp(min=-1, max=1)
            raw_mli_predicted[:, :, -1:] = torch.sigmoid(raw_mli_predicted[:, :, -1:])

        if self.last_layer_processor is not None:
            batch_size, num_layers, c, h, w = raw_mli_feats.shape
            raw_last_layer = self.last_layer_processor(raw_mli_feats.reshape(batch_size, num_layers * c, h, w))
            raw_mli_predicted[:, :1, :3] = torch.tanh(raw_last_layer[:, None, :]).clamp(min=-1, max=1)
            raw_mli_predicted[:, :1, -1:] = torch.ones_like(raw_mli_predicted[:, :1, -1:])

        # B x N x C x H x W
        mesh_verts, mesh_faces, _, mesh_verts_rgb = build_layers_from_view(
            raw_mli_predicted,
            layered_depth,
            convert_to_camera_pytorch3d(reference_cameras),
            align_corners=True,
            grid_generator = self.grid_generator,
        )


        out = dict(
            mpi=raw_mli_predicted[:, None],
            mli={'verts': mesh_verts, 'faces': mesh_faces, 'verts_features': mesh_verts_rgb},
            layered_depth=layered_depth.squeeze(2),  # B x num_layers x H x W
            intermediate_sources=intermediate_sources
        )

        return out

    def process_depth(self, raw_depth, mpi_depths):
        batch_size, *_, height, width = raw_depth.shape
        planes_groups_alphas = torch.cat(torch.sigmoid(raw_depth).chunk(self.num_layers, dim=1), dim=0)

        if self.use_intersected_groups:
            default_depths = mpi_depths.view(-1, 1, 1)
            num_planes = default_depths.shape[0]
            # B x num_layers x group_size + 1
            default_depths = F.unfold(default_depths[None, None, None, ..., 0, 0],
                                      kernel_size=(1, math.ceil(num_planes / self.num_layers)),
                                      stride=(1, math.floor(num_planes / self.num_layers)),
                                      padding=(0, 0), dilation=(1, 1)).transpose(-1, -2)
            group_size = default_depths.shape[-1]
            default_depths = default_depths[..., None, None].expand(1, *default_depths.shape[-2:], height, width)
            default_depths = default_depths.view(-1, group_size, height, width)
        else:
            default_depths = mpi_depths.view(-1, 1, 1).expand(batch_size, -1, height, width)
            default_depths = torch.cat(default_depths.chunk(self.num_layers, dim=1), dim=0)

        default_depths_origins = default_depths[:, [-1]]
        default_depths = default_depths - default_depths_origins
        default_depths = default_depths.unsqueeze(2)
        # B*n_groups x planes_per_group x 2 x H x W
        default_depths_with_alphas = torch.cat([default_depths,
                                                planes_groups_alphas], dim=2)

        # B*n_groups x 1 x H x W
        group_depth = self.composer(
            default_depths_with_alphas[:, None],
            black_background=False,
            masked_black_background=self.params.get('masked_black_background', True))[0][:, 0]

        group_depth = group_depth + default_depths_origins

        layered_depth = torch.cat(group_depth.chunk(self.num_layers, dim=0), dim=1)  # B x n_groups x H x W
        return layered_depth

    @staticmethod
    def convert_to_camera_pytorch3d(camera: CameraMultiple,
                                    convert_intrinsics_to_relative: bool = False,
                                    height: Optional[int] = None,
                                    width: Optional[int] = None,
                                    novel_index: Optional[int] = 0,
                                    ) -> CameraPytorch3d:
        scaling = 1
        if convert_intrinsics_to_relative:
            if height is not None and width is not None:
                scaling = torch.tensor([width, height, 1], dtype=torch.float, device=camera.device).view(-1, 1)
            else:
                logger.warning('Asked to convert intrinsics to relative, but did not provide resolution')

        return CameraPytorch3d(
            extrinsics=camera.extrinsics.view(*camera.cameras_shape,
                                              *camera.extrinsics.shape[-2:])[:, 0, novel_index],
            intrinsics=camera.intrinsics.view(*camera.cameras_shape,
                                              *camera.intrinsics.shape[-2:])[:, 0, novel_index] / scaling,
        )

    def render_with_proxy_geometry(self, novel_cameras, proxy_geometry):
        mpi = proxy_geometry['mpi']

        H, W = novel_cameras.images_size
        batch, _, n_surfaces, _, _, _ = mpi.shape

        if self.composer_mpi:
            novel_pixel_coords = get_grid(batch_size=batch,
                                          height=H,
                                          width=W,
                                          relative=True,
                                          values_range='sigmoid',
                                          align_corners=False,
                                          device=mpi.device,
                                          )[:, None, None]
            novel_pixel_coords = novel_pixel_coords.expand(*novel_cameras.cameras_shape,
                                                           *novel_pixel_coords.shape[-3:])
            ray_sampler = self.sampi_block_lvl1.ray_sampler

            mpi, timestamps, frustum_mask = ray_sampler.surfaces.look_at(
                reference_features=mpi,
                novel_camera=novel_cameras,
                novel_pixel_coords=novel_pixel_coords,
                relative_intrinsics=True,
            )

            composer_out = self.composer_mpi(mpi, return_namedtuple=True)

            feats, opacity = composer_out.color, composer_out.opacity

            out = dict(
                opacity=opacity,
                mpi=mpi,
                frustum_mask=frustum_mask,
                weight=composer_out.weight,
                disparities=ray_sampler.surfaces.disparities(normalize=True).unsqueeze(0),
                depths=ray_sampler.surfaces.depths(normalize=True).unsqueeze(0),
            )

            out['rgb'] = feats[:, :, :3].clamp(min=-1, max=1)

        else:

            mli = proxy_geometry['mli']
            # For novel only
            num_novels = novel_cameras.cameras_shape[-1]
            opacity, feats, weights, depth = [], [], [], []

            for i in range(num_novels):
                render_output: dict = self.renderer(
                    self.convert_to_camera_pytorch3d(novel_cameras, novel_index=i),
                    mli['verts'], mli['faces'],
                    custom_resolution=(H, W),
                    verts_rgb=mli['verts_features'],
                )
                feats.append(render_output['images'])
                opacity.append(render_output['opacity'])
                weights.append(render_output['texel_weights'])

                depth.append(
                    torch.sum(render_output['fragments'].zbuf * render_output['texel_weights'].permute(0, 2, 3, 1),
                              dim=3))

            feats = torch.stack(feats)
            opacity = torch.stack(opacity)
            weights = torch.stack(weights)
            depth = torch.stack(depth)

            out = dict(
                opacity=opacity,
                mpi=mpi,
                mli=mli,
                weight=weights.unsqueeze(2),
                layered_depth=proxy_geometry['layered_depth'],
                rgb=feats.clamp(min=-1, max=1),
                depth=depth,
            )

        return out
