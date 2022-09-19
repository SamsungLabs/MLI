__all__ = ['GeneratorStereoMagnification2Mesh']

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.modules.mesh_layers import build_layers_from_view
from lib.modules.cameras.utils import convert_to_camera_pytorch3d
from .gen_stereo_mag import GeneratorStereoMagnification
from .gen_deformed_psv import GeneratorDeformedPSV
from .gen_parts import SurfacesMPI, RendererMeshLayers
try:
    from .gen_parts import RasterizerFFrast
except:
    pass

class GeneratorStereoMagnification2Mesh(GeneratorStereoMagnification, GeneratorDeformedPSV):
    def __init__(self, params: dict):
        super().__init__(params=params)
        self.n_groups: int = self.params.get('n_groups', 8)
        self.n_samples: int = self.params.get('n_samples', 20)
        self.sigma: float = self.params.get('sigma', 0.1)
        self.freeze_size: int = self.params.get('freeze_size', 256)

    def _init_modules(self,
                      psv_network,
                      composer,
                      surfaces,
                      rasterizer,
                      shader,
                      ):
        self.psv_network = psv_network
        self.composer = composer
        self.surfaces: SurfacesMPI = surfaces
        self.renderer = RendererMeshLayers(rasterizer, shader, self.params['num_layers'])

        self.decoder = None

    @torch.no_grad()
    def convert_mpi_to_mesh(self,
                            camera: CameraMultiple,
                            mpi: torch.Tensor,
                            n_groups: int = 8,
                            n_samples: int = 20,
                            sigma: float = 0.1,
                            ) -> dict:
        """
        Args:
            camera: B x n_ref x 1 x KRT
            mpi: B x n_ref x n_intersections x RGBA x H x W (back-to-front order)
            n_groups: number of groups (mesh layers). If n_groups does not divide the number of mpi layers,
                 fake transparent planes are appended to the farthest plane
            n_samples: number of samples for Monte Carlo integration
            sigma: std (in relative pixel units, i.e. all pixels have coordinates between 0 and 1 in these units)
                for random sampling

        Returns:
            mpi: B x n_groups x RGBA x H x W
            verts:
            faces:
            verts_rgb:
        """
        assert camera.cameras_shape[1:3] == (1, 1)
        assert mpi.shape[1] == 1
        batch_size, *_, height, width = mpi.shape

        # 0. Append fake planes if necessary

        opacity = mpi[..., -1:, :, :]  # B x n_ref x n_planes x 1 x H x W
        planes_disparity = self.surfaces.mpi_depths.reciprocal()  # (n_layers, )
        disparity_volume = torch.cat([
            planes_disparity[None, None, :, None, None, None].expand_as(opacity),
            opacity,
        ], dim=-3)
        assert len(planes_disparity) == mpi.shape[2]
        n_planes = mpi.shape[2]
        remainder, planes_per_group = n_planes % n_groups, n_planes // n_groups
        if remainder != 0:
            fake_opacity = torch.zeros_like(opacity[:, :, :1]).expand(-1, -1, planes_per_group - remainder, -1, -1, -1)
            fake_mpi = torch.zeros_like(mpi[:, :, :1]).expand(-1, -1, planes_per_group - remainder, -1, -1, -1)
            fake_disparity = planes_disparity[None, None, :1, None, None, None].expand_as(fake_opacity)
            fake_disparity_volume = torch.cat([fake_disparity, fake_opacity], dim=-3)

            disparity_volume = torch.cat([fake_disparity_volume, disparity_volume], dim=2)
            mpi = torch.cat([fake_mpi, mpi], dim=2)
            planes_per_group += 1

        # 1. Compute layered depth

        planes_groups = torch.cat(disparity_volume.chunk(n_groups, dim=2),
                                  dim=0)  # B*n_groups x n_ref x planes_per_group x 2 x H x W
        # B*n_groups x n_ref x 1 x 1 x H x W
        group_disparity = self.composer(planes_groups.unsqueeze(2), black_background=False)[0]
        layered_disparity = torch.cat(group_disparity.chunk(n_groups, dim=0),
                                      dim=-3)  # B x n_ref x 1 x n_groups x H x W
        layered_depth = layered_disparity.reciprocal()  # B x n_ref x 1 x n_groups x H x W
        # B x n_ref x 1 x H x W x UV
        pixel_coords = self._get_grid(mpi.unsqueeze(2), relative=True, values_range='sigmoid')
        pixel_coords = pixel_coords.unsqueeze(-4).expand(*camera.cameras_shape, n_groups, *pixel_coords.shape[-3:])
        # B x n_ref x 1 x n_groups x H x W x XYZ
        world_layers = camera.pixel_to_world(pixel_coords, layered_depth.unsqueeze(-1))

        # 2. Compute layered RGBA
        # 2.1. Sample random viewpoints per each mesh vertex and compute rays

        # B x n_ref x n_samples x n_groups x H x W x UV
        random_translation = torch.randn(*world_layers.shape[:2], n_samples, *world_layers.shape[3:-1], 2,
                                         device=camera.device)
        sample_weight = random_translation.pow(2).sum(-1).mul(-0.5).exp()  # B x n_ref x n_samples x n_groups x H x W
        sample_weight.unsqueeze_(-4)  # B x n_ref x n_samples x 1 x n_groups x H x W
        random_translation = random_translation * sigma + 0.5
        random_viewpoint_xyz = camera.pixel_to_world(random_translation[:, :, None], 
                                                     torch.zeros_like(random_translation[:, :, None, ..., :1])
                                                     ).squeeze(2)
        # B x n_ref x n_samples x n_groups x H x W x XYZ
        ray_direction = F.normalize(world_layers - random_viewpoint_xyz, dim=-1)

        # 2.2. Collect RGBA from MPI planes for the sampled rays and apply compose-over within each group of planes

        # B x n_ref x n_samples x n_surfaces x RGBA x n_groups x H x W
        rendered_features = self.surfaces.look_at_rays(reference_features=mpi,
                                                       rays=ray_direction,
                                                       start_points=random_viewpoint_xyz,
                                                       relative_intrinsics=True,
                                                       )[0]
        if remainder != 0:
            fake_rendered_features = torch.zeros_like(rendered_features[:, :, :, :1]).expand(
                -1, -1, -1, planes_per_group - remainder, -1, -1, -1, -1)
            rendered_features = torch.cat([fake_rendered_features, rendered_features], dim=3)
        # n_groups x B x n_ref x n_samples x planes_per_group x RGBA x n_groups x H x W
        rendered_features = torch.stack(rendered_features.chunk(n_groups, dim=3), dim=0)
        # now we should consider for each group only intersections within the same group
        idx = torch.arange(n_groups, device=rendered_features.device) \
            .view(1, 1, 1, 1, 1, 1, -1, 1, 1) \
            .expand(1, *rendered_features.shape[1:])
        # B x n_ref x n_samples x planes_per_group x RGBA x n_groups x H x W
        rendered_features = rendered_features.gather(dim=0, index=idx).squeeze(0)
        # B x n_ref x n_samples x C x n_groups x H x W
        rendered_color, rendered_opacity = self.composer(rendered_features, black_background=True)
        rendered_log_density = rendered_opacity.clamp(min=1e-6, max=1-1e-6).neg().log1p()

        # 2.3. Monte-Carlo integration for each mesh vertex

        # B x n_ref x C x n_groups x H x W
        normalizer = torch.sum(sample_weight * rendered_log_density, dim=2).clamp(max=-1e-6)
        color = torch.sum(sample_weight * rendered_log_density * rendered_color, dim=2) / normalizer
        log_density = torch.sum(sample_weight * rendered_log_density * rendered_log_density, dim=2) / normalizer
        opacity = torch.clamp(1 - log_density.exp(), min=0, max=1)
        layered_rgba = torch.cat([color, opacity], dim=-4)  # B x n_ref x RGBA x n_groups x H x W
        layered_rgba = layered_rgba.transpose(2, 3)  # B x n_ref x n_groups x RGBA x H x W

        # 2.4. Build the layered mesh

        camera = convert_to_camera_pytorch3d(camera=camera,
                                             convert_intrinsics_to_relative=False,
                                             )
        layered_rgbda = torch.cat([layered_depth.squeeze(1).squeeze(1).unsqueeze(-3),
                                   layered_rgba.squeeze(1)], dim=-3)
        verts, faces, _, verts_features = build_layers_from_view(features_tensor=layered_rgbda,
                                                                 depth_tensor=layered_depth.squeeze(1).squeeze(1),
                                                                 source_cameras=camera,
                                                                 align_corners=True,
                                                                 only_cloud=False,
                                                                 )
        return dict(
            mpi=layered_rgba.squeeze(1),
            verts=verts,
            faces=faces,
            verts_rgb=verts_features,
            layered_depth=layered_depth.squeeze(1).squeeze(1).unsqueeze(-3),
        )

    def render_with_proxy_geometry(self, *args, **kwargs):
        result = GeneratorDeformedPSV.render_with_proxy_geometry(self, *args, **kwargs)
        result['novel_image'] = F.interpolate(result['novel_image'], size=self.custom_input_resolution)
        return result

    def forward(self, *args, **kwargs):
        action = kwargs.pop('action', 'forward')
        if action == 'forward':
            return self.manual_forward(*args, **kwargs)
        elif action == 'render_with_proxy_geometry':
            return self.render_with_proxy_geometry(*args, **kwargs)
        else:
            raise ValueError(f'Unknown action={action}')

    @torch.no_grad()
    def manual_forward(self,
                reference_images,
                source_images,
                reference_cameras: CameraMultiple,
                source_cameras: CameraMultiple,
                novel_cameras: CameraMultiple,
                ):
        """
        Args:
            reference_images: B x n_reference x 1 x C x H x W
            source_images: B x n_reference x n_source x C x H x W
            reference_cameras: B x n_reference x 1 x KRT
            source_cameras: B x n_reference x n_source x KRT
            novel_cameras: B x n_reference x n_novel x KRT
        """
        self.surfaces.set_position(reference_cameras)
        reference_pixel_coords = self._get_grid(reference_images, relative=True, values_range='sigmoid')

        psv = self.surfaces.project_on(source_features=source_images,
                                       source_camera=source_cameras,
                                       reference_pixel_coords=reference_pixel_coords,
                                       relative_intrinsics=True,
                                       )
        raw_output = self.psv_network(
            self.preprocess_psv(psv, reference_images)
        )
        self.custom_input_resolution = reference_images.shape[-2:]
        if self.freeze_size != min(self.custom_input_resolution ):
            raw_output = F.interpolate(raw_output,
                                           size=(self.freeze_size, self.freeze_size,))
            reference_images = F.interpolate(reference_images.squeeze(1).squeeze(1),
                                           size=(self.freeze_size, self.freeze_size,)).unsqueeze(1).unsqueeze(1)
        postprocessed_psv: dict = self.postprocess_psv(raw_output, reference_images)
        mpi = postprocessed_psv['mpi']
        proxy_geometry = self.convert_mpi_to_mesh(camera=reference_cameras,
                                                  mpi=mpi,
                                                  n_groups=self.n_groups,
                                                  n_samples=self.n_samples,
                                                  sigma=self.sigma,
                                                  )
        # for key in ['mpi', ]:
        #     proxy_geometry[key] = F.interpolate(proxy_geometry[key], size=(height, width))
        return self.render_with_proxy_geometry(novel_cameras=novel_cameras, proxy_geometry=proxy_geometry)
