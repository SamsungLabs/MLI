__all__ = ['AllCamerasModule']

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from lib.modules.cameras import CameraFrustum

logger = logging.getLogger(__name__)


class CameraModule(nn.Module, CameraFrustum):
    """
    Class which manage set of cameras for particular scene.
    """

    def __init__(self,
                 use_learnable_color_correction: bool = False,
                 use_learnable_background: bool = False,
                 use_learnable_cam_fix: bool = False
                 ):
        """
        Args:
            use_learnable_color_correction (Bool): train color correction for each camera,
                six-params (r_scale, g_scale, b_scale, r_bias, g_bias, b_bias)
            use_learnable_background (Bool): train background tensor for each camera, tensor have same dims as image
            use_learnable_cam_fix (Bool): train extrinsics fix for each camera
        """

        nn.Module.__init__(self)
        CameraFrustum.__init__(self)

        self.use_learnable_color_correction = use_learnable_color_correction
        self.use_learnable_background = use_learnable_background
        self.use_learnable_cam_fix = use_learnable_cam_fix

    @classmethod
    def from_data(cls,
                  extrinsics: torch.Tensor,
                  intrinsics: torch.Tensor,
                  images_sizes: torch.Tensor,
                  use_learnable_color_correction: bool = False,
                  use_learnable_background: bool = False,
                  use_learnable_cam_fix: bool = False,
                  device: torch.device = None):
        """
        Init CamerasModule from data.

        Args:
            extrinsics (torch.Tensor): Bc x 3 x 4 or Bc x 4 x 4
            intrinsics (torch.Tensor): Bc x 3 x 3
            images_sizes (torch.Tensor): Bc x 2 or 1 x 2 or 2
            use_learnable_color_correction (bool):
            use_learnable_background (bool):
            use_learnable_cam_fix (bool):
            device (torch.device):

        Returns:
            CamerasModule
        """

        if device is None:
            device = extrinsics.device

        instance = cls(use_learnable_color_correction,
                       use_learnable_background,
                       use_learnable_cam_fix)
        instance.to(device)

        if extrinsics.dim() != 3:
            extrinsics = extrinsics.unsqueeze(0)
        assert extrinsics.dim() == 3
        assert extrinsics.shape[1:] in {(3, 4), (4, 4)}, \
            f'Expected B x 3 x 4 or B x 4 x 4 shape, but obtained {extrinsics.shape}'
        instance.register_buffer('_extrinsics', extrinsics.to(device))

        if intrinsics.dim() != 3:
            intrinsics = intrinsics.unsqueeze(0)
        assert intrinsics.dim() == 3
        assert intrinsics.shape[1:] == (3, 3)
        instance.register_buffer('_intrinsics', intrinsics.to(device))

        if instance.use_learnable_cam_fix:
            instance.extrinsics_param = nn.Parameter(instance._extrinsics.to(device))
            instance.intrinsics_param = nn.Parameter(instance._intrinsics.to(device))

        if images_sizes.dim() == 1:
            images_sizes = images_sizes.unsqueeze(1)
        if images_sizes.shape[0] < len(instance):
            images_sizes = images_sizes.expand(len(instance), -1)
        instance.register_buffer('images_sizes', images_sizes.float().to(device))

        instance.color_corrections_background = None
        if instance.use_learnable_color_correction == 'both' or instance.use_learnable_color_correction == 'background':
            instance.color_corrections_background = nn.Parameter(
                torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float, device=device)
                    .view(1, -1)
                    .expand(len(instance), -1)
            )

        instance.color_corrections_scene = None
        if instance.use_learnable_color_correction == 'both' or instance.use_learnable_color_correction == 'scene':
            instance.color_corrections_scene = nn.Parameter(
                torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float, device=device)
                    .view(1, -1)
                    .expand(len(instance), -1)
            )

        instance.backgrounds = None
        if instance.use_learnable_background:
            instance.backgrounds = nn.Parameter(
                torch.ones(len(instance), 3, images_sizes[0][0], images_sizes[0][1],
                           dtype=torch.float, device=device) * 255
            )

        return instance

    @classmethod
    def from_krt(cls,
                 krt,
                 use_learnable_color_correction: bool = False,
                 use_learnable_background: bool = False,
                 use_learnable_cam_fix: bool = False,
                 device: torch.device = None
                 ):
        return cls.from_data(torch.tensor(np.array([el['extrin'] for el in krt])),
                             torch.tensor(np.array([el['intrin'] for el in krt])),
                             torch.tensor(np.array([el['image_size'] for el in krt])),
                             use_learnable_color_correction=use_learnable_color_correction,
                             use_learnable_background=use_learnable_background,
                             use_learnable_cam_fix=use_learnable_cam_fix,
                             device=device)

    @classmethod
    def from_slice(cls,
                   extrinsics: torch.Tensor,
                   intrinsics: torch.Tensor,
                   images_sizes: torch.Tensor,
                   use_learnable_color_correction: bool = False,
                   use_learnable_background: bool = False,
                   use_learnable_cam_fix: bool = False,
                   color_corrections_background: nn.Parameter = None,
                   color_corrections_scene: nn.Parameter = None,
                   backgrounds: nn.Parameter = None
                   ):

        instance = cls(use_learnable_color_correction,
                       use_learnable_background,
                       use_learnable_cam_fix)

        instance._extrinsics = extrinsics
        instance._intrinsics = intrinsics
        instance.images_sizes = images_sizes
        instance.color_corrections_background = color_corrections_background
        instance.color_corrections_scene = color_corrections_scene
        instance.backgrounds = backgrounds

        return instance

    def __getitem__(self, key):

        if np.isscalar(key):
            key = torch.LongTensor([key])
        elif isinstance(key, torch.Tensor) and key.dim() == 0:
            key = key.unsqueeze(0)

        color_corrections_background = None
        if self.color_corrections_background is not None:
            color_corrections_background = self.color_corrections_background[key]

        color_corrections_scene = None
        if self.color_corrections_scene is not None:
            color_corrections_scene = self.color_corrections_scene[key]

        backgrounds = None
        if self.backgrounds is not None:
            backgrounds = self.backgrounds[key]

        return CameraModule.from_slice(extrinsics=self.extrinsics[key],
                                       intrinsics=self.intrinsics[key],
                                       images_sizes=self.images_sizes[key],
                                       use_learnable_color_correction=self.use_learnable_color_correction,
                                       use_learnable_background=self.use_learnable_background,
                                       color_corrections_background=color_corrections_background,
                                       color_corrections_scene=color_corrections_scene,
                                       backgrounds=backgrounds)

    def __len__(self):
        return self.extrinsics.shape[0]

    def forward(self,
                mode: str,
                **kwargs):
        if mode == 'slice':
            return self[kwargs['idx']]

    @property
    def extrinsics(self):
        if self.use_learnable_cam_fix:
            return self.extrinsics_param
        return self._extrinsics

    @property
    def intrinsics(self):
        if self.use_learnable_cam_fix:
            return self.intrinsics_param
        return self._intrinsics


class AllCamerasModule(nn.Module):
    """ Supporting class for working with several scene/time camera sets.
        For Example you have 2 scene with 4 different times in each scene.
    It's 8 camera sets (scene1_time1, scene2_time_1, ... , scene1_time4, scene2_time4) which managed by this class.
    Camera set is :class: '~CamerasModule'.
    """

    def __init__(self,
                 params: dict
                 ):
        """
        Args:
            params (dict): Must contain 'krt' item which stores extrinsics and intrinsics for all cameras.
                Additional params: 'use_learnable_color_correction'
                                   'use_learnable_background'
                                   'use_learnable_cam_fix'

        """
        super().__init__()

        self.all_cameras = nn.ModuleDict({
            scene_time: CameraModule.from_krt(krt,
                                              params.get('use_learnable_color_correction'),
                                              params.get('use_learnable_background'),
                                              params.get('use_learnable_cam_fix'))
            for scene_time, krt in params['krt'].items()
        })

        self.check_frustums_params = params.get('check_frustums')

    def __getitem__(self,
                    scene_time: str
                    ) -> CameraModule:
        return self.all_cameras[scene_time]

    def forward(self,
                mode: str,
                **kwargs
                ) -> Tuple[CameraFrustum,
                           Optional[torch.Tensor],
                           Optional[torch.Tensor],
                           Optional[torch.Tensor],
                           Optional[torch.Tensor]]:
        """
        Some operation with AllCamerasModule during training.
        You can pick operation with mode parameter.

        modes:
            get_scene_cam_feats:
                In this mode, function returned all CamerasModule stuff
                which you need for train step for each scene_camera index.
                All stuff concatenate along camera dim.

                ModeArgs:
                    camera_scene_idx (List[Tuple[str, str]]): list with scene/camera indexes.

                ModeReturn:
                    Cameras: cameras for each scene_camera index.
                    torch.Tensor or None:
                        If use_learnable_background is enable, backgrounds for each scene_camera index, else None.
                    torch.Tensor or None:
                        If color_corrections_background is enable, backgrounds for each scene_camera index, else None.
                    torch.Tensor or None:
                        If color_corrections_scene is enable, backgrounds for each scene_camera index, else None.
                    torch.Tensor or None:
                        If check_frustums_params is enable, backgrounds for each scene_camera index, else None.

        Args:
            mode:
            **kwargs:

        Returns:

        """

        if mode == 'get_scene_cam_feats':
            backgrounds = []
            color_corrs_background = []
            color_corrs_scene = []
            valid_fns = []
            out_cameras = []

            for scene_idx, camera_idx in kwargs['camera_scene_idx']:
                cameras = self.all_cameras[scene_idx]
                camera = cameras.forward(mode='slice', idx=camera_idx)
                if camera.backgrounds is not None:
                    backgrounds.append(camera.backgrounds)
                if camera.color_corrections_background is not None:
                    color_corrs_background.append(camera.color_corrections_background)
                if camera.color_corrections_scene is not None:
                    color_corrs_scene.append(camera.color_corrections_scene)
                if (self.check_frustums_params is not None) and \
                        (self.check_frustums_params.get('resolution') is not None) and \
                        (self.check_frustums_params.get('min_cams') is not None):
                    valid_fns.append(
                        cameras.build_grid_sample_checker_function(resolution=self.check_frustums_params['resolution'],
                                                                   min_cams=self.check_frustums_params['min_cams']))
                out_cameras.append(camera)

            if len(valid_fns) == 0:
                valid_fns = None
            elif len(set(kwargs['camera_scene_idx'])) == 1:
                valid_fns = valid_fns[0]

            backgrounds = torch.cat(backgrounds) if len(backgrounds) else None
            color_corrs_background = torch.cat(color_corrs_background) if len(color_corrs_background) else None
            color_corrs_scene = torch.cat(color_corrs_scene) if len(color_corrs_scene) else None

            return CameraFrustum.from_cameras(out_cameras), backgrounds, color_corrs_background, color_corrs_scene, valid_fns

        else:
            logger.error(f"Wrong forward mode for AllCamerasModule: '{mode}', only support 'get_scene_cam_feats'")
