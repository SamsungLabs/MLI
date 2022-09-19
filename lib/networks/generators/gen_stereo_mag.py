__all__ = ['GeneratorStereoMagnification',
           'GeneratorStereoFeaturedMagnification',
           ]

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from lib.modules.cameras import CameraMultiple
from lib.networks.generators.gen_parts.surfaces import SurfacesBase
from lib.networks.generators.gen_parts.composers import ComposerBase
from .gen_mpi_base import GeneratorMPIBase


class GeneratorStereoMagnification(GeneratorMPIBase):
    def __init__(self, params: dict):
        super().__init__(params)

        # resolution, at which PSV-net operates
        self.psv_net_resolution: Union[None, int, Tuple[int, int]] = self.params.get('psv_net_resolution')

    def _init_modules(self, psv_network, composer, surfaces):
        self.psv_network = psv_network
        self.composer: ComposerBase = composer
        self.surfaces: SurfacesBase = surfaces

    def preprocess_psv(self,
                       psv: torch.Tensor,
                       features: Optional[torch.Tensor] = None,
                       ) -> torch.Tensor:
        """
        Preprocess projected PSV before processing by the network.

        The Stereo Magnification paper concatenates all the projected planes along the channels axis and afterwards
        concatenates with the image from the reference camera.

        Args:
            psv: PSV-projection of reference cameras B x n_ref x n_source x n_intersections x n_features x H x W
            features: features observed by reference camera, B x n_ref x 1 x C x H x W

        Returns:
            out: B * n_ref x (C + n_source * n_intersections * n_features) x H x W
        """
        batch_size, n_ref, n_source, = psv.shape[:3]

        # B*n_ref x n_source*n_intersections*n_features x H x W
        out = psv.contiguous().view(batch_size * n_ref, -1, *psv.shape[-2:])

        if features is not None:
            features = features.contiguous().view(batch_size * n_ref, -1, *features.shape[-2:])
            out = torch.cat([out, features], dim=1)

        if self.psv_net_resolution is not None:
            out = F.interpolate(out, size=self.psv_net_resolution, mode='bilinear')
        return out

    def postprocess_psv(self,
                        raw_output: torch.Tensor,
                        features: torch.Tensor,
                        ) -> dict:
        """
        Postprocess the reference PSV after the neural refinement.

        The Stereo Magnification paper proposed to treat the network output as a single background, opacities and
        weights, that mix the predicted background with the features from the reference POV.

        Args:
            raw_output: refined PSV, B*n_ref x n_features x H x W
            features: features observed by reference camera, B x n_ref x 1 x C x H x W

        Returns:
            mpi: postprocessed PSV in RGBA, B x n_ref x n_intersections x RGBA x H x W
        """
        batch_size, n_ref = features.shape[:2]
        n_channels = features.shape[-3]  # RGB

        raw_output = torch.tanh(raw_output)

        # upsample to real resolution
        if self.psv_net_resolution is not None and raw_output.shape[-2:] != features.shape[-2:]:
            real_resolution = tuple(features.shape[-2:])
            raw_output = F.interpolate(raw_output, size=real_resolution, mode='bilinear')

        # B x n_ref x n_features x H x W
        raw_output = raw_output.contiguous().view(batch_size, n_ref, *raw_output.shape[1:])

        # B x n_ref x n_intersections x 1 x H x W
        weight = raw_output[:, :, :self.surfaces.n_intersections].unsqueeze(3)

        # B x n_ref x n_intersections x 1 x H x W
        opacity = raw_output[:, :, self.surfaces.n_intersections : -n_channels].unsqueeze(3)

        # B x n_ref x 1 x RGB x H x W
        background = raw_output[:, :, -n_channels:].unsqueeze(2)

        # convert from Tanh output to [0, 1] range
        weight = weight.add(1).div(2)
        opacity = opacity.add(1).div(2)

        mixed = features * weight + background * (1 - weight)  # B x n_ref x n_intersections x RGB x H x W
        mpi = torch.cat([mixed, opacity], dim=3)

        out = {'mpi': mpi}
        return out

    def render_with_proxy_geometry(self, novel_cameras, proxy_geometry):
        mpi = proxy_geometry['mpi']
        novel_pixel_coords = proxy_geometry.get('novel_pixel_coords')
        if novel_pixel_coords is None:
            novel_pixel_coords = self._get_grid(mpi.unsqueeze(2))
            novel_pixel_coords = novel_pixel_coords.expand(*novel_cameras.cameras_shape,
                                                           *novel_pixel_coords.shape[-3:])

        novel_features, timestamps, frustum_mask = self.surfaces.look_at(reference_features=mpi,
                                                                         novel_camera=novel_cameras,
                                                                         novel_pixel_coords=novel_pixel_coords,
                                                                         )
        novel_image, opacity = self.composer(features=novel_features,
                                             timestamps=timestamps, inside_frustum_mask=frustum_mask)

        out = dict(
            novel_image=novel_image,
            novel_opacity=opacity,
            frustum_mask=frustum_mask,
            mpi=mpi,
        )
        return out

    def forward(self,
                reference_images,
                source_images,
                reference_cameras: CameraMultiple,
                source_cameras: CameraMultiple,
                novel_cameras: CameraMultiple,
                reference_pixel_coords=None,
                novel_pixel_coords=None,
                ) -> dict:
        """
        Args:
            reference_images: B x n_reference x 1 x C x H x W
            source_images: B x n_reference x n_source x C x H x W
            reference_cameras: B x n_reference x 1 x KRT
            source_cameras: B x n_reference x n_source x KRT
            novel_cameras: B x n_reference x n_novel x KRT
            reference_pixel_coords:
            novel_pixel_coords:

        Returns:
            novel_images: B x n_reference x n_novel x C x H x W
            opacity: B x n_reference x n_novel x 1 x H x W
            frustum_mask: B x n_reference x n_novel x 1 x H x W
            mpi: B x n_ref x n_intersections x RGBA x H x W
        """
        self.surfaces.set_position(reference_cameras)
        if reference_pixel_coords is None:
            reference_pixel_coords = self._get_grid(reference_images)

        psv = self.surfaces.project_on(source_features=source_images,
                                       source_camera=source_cameras,
                                       reference_pixel_coords=reference_pixel_coords,
                                       )
        raw_output = self.psv_network(
            self.preprocess_psv(psv, reference_images)
        )
        postprocessed_psv: dict = self.postprocess_psv(raw_output, reference_images, psv)
        mpi = postprocessed_psv['mpi']

        if novel_pixel_coords is None:
            novel_pixel_coords = self._get_grid(mpi.unsqueeze(2))
            novel_pixel_coords = novel_pixel_coords.expand(*novel_cameras.cameras_shape, *novel_pixel_coords.shape[-3:])

        novel_features, timestamps, frustum_mask = self.surfaces.look_at(reference_features=mpi,
                                                                         novel_camera=novel_cameras,
                                                                         novel_pixel_coords=novel_pixel_coords,
                                                                         )

        novel_image, opacity = self.composer(features=novel_features,
                                             timestamps=timestamps, inside_frustum_mask=frustum_mask)
        out = dict(
            novel_image=novel_image,
            novel_opacity=opacity,
            frustum_mask=frustum_mask,
            mpi=mpi,
        )
        return out


class GeneratorStereoFeaturedMagnification(GeneratorStereoMagnification):

    def postprocess_psv(self,
                        raw_output: torch.Tensor,
                        features: torch.Tensor,
                        ) -> dict:
        """
        Postprocess the reference PSV after the neural refinement.
        Transpose reference PSV

        Args:
            raw_output: refined PSV, B*n_ref x n_features x H x W
            features: features observed by reference camera, B x n_ref x 1 x C x H x W

        Returns:
            mpi: postprocessed PSV B x n_ref x n_intersections x n_features x H x W
        """
        b, n_ref, n_cam, _, h, w = features.size()

        # upsample to real resolution
        if self.psv_net_resolution is not None and raw_output.shape[-2:] != features.shape[-2:]:
            raw_output = F.interpolate(raw_output, size=(h, w), mode='bilinear')

        mpi_features = raw_output.contiguous().view(b, n_ref, self.surfaces.n_intersections, -1, h, w)
        return dict(mpi=mpi_features)
