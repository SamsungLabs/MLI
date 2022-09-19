"""
Validation dataset with structure:
| root/:
|---- scenes/
    |---- 0
        |---- scene.csv
        |---- initial
            |---- frame1.jpg
            |---- frame2.jpg
        |---- reference
            |---- frame1.jpg
            |---- frame2.jpg
        |---- novel
            |---- frame1.jpg
            |---- frame2.jpg
    |---- 1
        |---- scene.csv
        |---- initial
            |---- frame1.jpg
            |---- frame2.jpg
        |---- reference
            |---- frame1.jpg
            |---- frame2.jpg
        |---- novel
            |---- frame1.jpg
            |---- frame2.jpg
    ...

scene.csv fields:
    General:
        frame_id - frame id
        type - initial / reference / novel

    File paths:
        frame_path - frame image path if None, frame is virtual

    Frame data:
        extrinsic_re - frame extrinsic in realestate format
        intrinsics_re - frame intrinsic in realestate format
"""

__all__ = ['ValidationDataset']

from typing import Dict, Tuple, Union, List
import logging
import os
from glob import glob
from ast import literal_eval

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from lib.modules.cameras.utils import average_extrinsics

DUMPED_FILE_NAME = 'dump.pkl'
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ValidationDataset(Dataset):
    def __init__(self,
                 root,
                 image_scale=1,
                 relative_intrinsics=True,
                 scene_scale_constant=1,
                 set_reference_as_origin=False,
                 skip_each_ith_frame: int = 0,
                 virtual_average_ref_extrinsics=True,
                 ):

        super().__init__()
        self.transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)

        self.scenes_dfs = []
        num_scenes = len(glob(os.path.join(root, '*', 'scene.csv')))
        for i in range(num_scenes):
            scenes_df_path = os.path.join(root, str(i), 'scene.csv')
            scene_path = '/'.join(scenes_df_path.split("/")[:-1])
            scene_dfs = pd.read_csv(scenes_df_path)
            scene_dfs['extrinsic_re'] = scene_dfs['extrinsic_re'].apply(literal_eval).apply(
                lambda x: np.array(x).reshape(3, 4))
            scene_dfs['intrinsics_re'] = scene_dfs['intrinsics_re'].apply(literal_eval).apply(
                lambda x: np.array(x).reshape(3, 3))
            scene_dfs['frame_size'] = scene_dfs['frame_size'].apply(literal_eval).apply(
                lambda x: np.array(x).reshape(2))
            scene_dfs['frame_path'] = scene_dfs['frame_path'].apply(
                lambda x: os.path.join(scene_path, x) if x != 'virtual' else 'virtual'
            )
            self.scenes_dfs.append(scene_dfs)

        self.image_scale = image_scale
        self.skip_each_ith_frame = skip_each_ith_frame
        self.relative_intrinsics = relative_intrinsics
        self.scene_scale_constant = scene_scale_constant
        self.set_reference_as_origin = set_reference_as_origin
        self.virtual_average_ref_extrinsics = virtual_average_ref_extrinsics
        # self.novel_trajectory = novel_trajectory

    def __len__(self):
        return len(self.scenes_dfs)

    @staticmethod
    def _relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic * scaling

    def _transform_extrinsics(self,
                              extrinsics: torch.Tensor,
                              reference_extrinsics: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.scene_scale_constant is not None:
            extrinsics[..., -1:] = extrinsics[..., -1:] / self.scene_scale_constant
            reference_extrinsics[..., -1:] = reference_extrinsics[..., -1:] / self.scene_scale_constant

        if not self.set_reference_as_origin:
            return extrinsics, reference_extrinsics

        inverse_reference_extrinsics = torch.inverse(
            reference_extrinsics.expand_as(extrinsics)[..., :3]
        )
        transformed_extrinsics = extrinsics.clone()
        transformed_extrinsics[..., :3] = torch.einsum('...ij,...jk->...ik',
                                                       [extrinsics[..., :3], inverse_reference_extrinsics])
        reference_translation = reference_extrinsics[..., 3]
        reference_translation = torch.matmul(inverse_reference_extrinsics, reference_translation.unsqueeze(-1))
        reference_translation = torch.matmul(extrinsics[..., :3], reference_translation).squeeze(-1)
        transformed_extrinsics[..., 3] = extrinsics[..., 3] - reference_translation

        return (transformed_extrinsics,
                torch.cat([torch.eye(3, 3), torch.zeros(3, 1)], dim=1).expand_as(reference_extrinsics),
                )

    def _read_image(self,
                    image_path: str,
                    current_extr: torch.Tensor,
                    relative_intr: torch.Tensor,
                    image_scale: float = 1,
                    image_size: Union[List, Tuple] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_path == 'virtual':
            image_resize_size = image_size
            current_image = torch.zeros(3, 1, 1)
        else:
            try:
                with Image.open(image_path) as img:
                    img_size = np.array(img.size)
                    image_resize_size = [int(image_scale * img_size[1]), int(image_scale * img_size[0])]
                    if image_scale != 1:
                        img = transforms.Resize(image_resize_size)(img)
                    current_image = self.transforms(img)
            except OSError as e:
                logger.error(f'Possibly, image file is broken: {image_path}')
                raise e

        if not self.relative_intrinsics is not None:
            absolute_intr = self._relative_intrinsic_to_absolute(*image_resize_size, relative_intr)

        if self.relative_intrinsics:
            current_intr = relative_intr
        else:
            current_intr = absolute_intr

        return current_image, current_extr, current_intr

    def _read_cameras(self,
                      scene_df,
                      cameras_type: str
                      ) -> Dict[str, torch.Tensor]:
        images, extrinsics, intrinsics = [], [], []
        cameras_df = scene_df[scene_df['type'] == cameras_type]
        if cameras_type == 'initial' and self.skip_each_ith_frame > 0:
            # Skip every ith frame
            cameras_df = cameras_df[cameras_df.index % self.skip_each_ith_frame == 0]
            print('shape', len(cameras_df))

        for i, camera_data in enumerate(cameras_df.iterrows()):
            camera_data = camera_data[1]
            image_path = camera_data['frame_path']
            current_extr = torch.from_numpy(camera_data['extrinsic_re']).float()
            relative_intr = torch.from_numpy(camera_data['intrinsics_re']).float()
            current_image, current_extr, current_intr = self._read_image(
                image_path,
                current_extr,
                relative_intr,
                self.image_scale,
                image_size=camera_data['frame_size']
            )

            images.append(current_image)
            extrinsics.append(current_extr)
            intrinsics.append(current_intr)

        num_cameras = len(images)
        images = torch.stack(images).view(1, num_cameras, *images[0].shape)
        extrinsics = torch.stack(extrinsics).view(1, num_cameras, *extrinsics[0].shape)
        intrinsics = torch.stack(intrinsics).view(1, num_cameras, *intrinsics[0].shape)

        return {
            'timestamp': torch.from_numpy(np.array(range(num_cameras))[None, :]),
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

    def __getitem__(self, idx):
        scene_df = self.scenes_dfs[idx]
        init_data = self._read_cameras(scene_df, cameras_type='initial')
        novel_data = self._read_cameras(scene_df, cameras_type='novel')

        if self.virtual_average_ref_extrinsics:
            ref_data = {}
            ref_data['extrinsics'] = average_extrinsics(init_data['extrinsics'])
            ref_data['image'] = init_data['image'][:, [0]]
            ref_data['intrinsics'] = init_data['intrinsics'][..., [0], :, :]
        else:
            ref_data = init_data

        time_id = 0
        output = {
            'time_id': time_id,
            'scene_time': f'{idx}//{time_id}',
            'scene_id': idx,
            'initial': init_data,
            'reference': ref_data,
            'novel': novel_data,
        }

        if 'initial_cams' in scene_df.columns:
            initial_cams_for_novels = []
            for el in list(scene_df[scene_df['type'] == 'novel']['initial_cams']):
                if el is not None:
                    initial_cams_for_novels.append(torch.tensor([int(x.split('_')[-1]) for x in el[2:-3].replace("'", '').split(", ")]))
                else:
                    initial_cams_for_novels.append([])

            output['initial_cams_for_novels'] = initial_cams_for_novels

        return output
