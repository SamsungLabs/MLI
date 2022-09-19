"""
This dataset assumes that you have a frozen camera rig.
All the scenes are shot using this rig.
The numeration of cameras is consistent across the scenes.
Therefore, the background for each camera (if known) is the same across all the scenes.
Every scene has images taken from all the available cameras.

The dataset files are expected to have the following structure:

| root/
|---- KRT.txt
|---- pose.txt
|---- background_0.jpg
|---- background_1.jpg
|---- scene_0/
     |---- time_0/
          |---- image_0.jpg
          |---- depth_0.jpg
          |---- image_1.jpg
          |---- depth_1.jpg
     |---- time_1/
     |---- time_2/
|---- scene_1/
     |---- time_0/
     |---- time_1/
     |---- time_2/


The file `roles.yaml` should contain the ids of cameras, used to extract the latent variable, and to reconstruct.
Example:

```

- scene_0:
  - time_0:
    - condition_train: [5, 7, 200]
    - reconstruction_train: [0, 2, 4]
    - condition_vis: [100, 19]
    - reconstruction_vis: [0, 1, 3, 11, 15, 17]
  - time_1:
    - condition_train: [6, 8, 201]
    - reconstruction_train: [2, 4, 7]
    - condition_vis: [100, 19, 58]
    - reconstruction_vis: [0, 1,, 11, 15, 17]
- scene_1:
  - time_2:
    - ...
  - time_1:
    - ...
- ...

```
"""

__all__ = ['CameraRigDataset']

import itertools
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from lib.utils.io import load_krt, load_pose, load_roles
from .multiview_dataset import MultiviewDataset

logger = logging.getLogger(__name__)
ROLES_PATH = 'roles.yaml'


class CameraRigDataset(MultiviewDataset):
    def __init__(self,
                 root,
                 mode,  # train | val | vis | ...
                 roles_path=None,
                 background_type=None,
                 condition_sampling_type=None,  # random | deterministic
                 condition_sampling_number=None,  # 3
                 condition_data_types=None,  # [image, depth]
                 data_types=('image',),  # [image, background, depth]
                 world_scale=1.,
                 subsample_type=None,
                 subsample_size=0,
                 image_size=(1024, 667),
                 resize_scale=1,
                 fix_intrisics_coef=1,  # TODO: rewrite dryice KRT and get rid of this redundant parameter
                 change_orientation=False,
                 img_ext='jpg',
                 crop_size=None):

        super(torch.utils.data.Dataset, self).__init__()

        self.root = root
        self.img_ext = img_ext
        self.mode = mode
        if roles_path is None:
            roles_path = ROLES_PATH
        if os.path.isabs(roles_path):
            self.roles_path = roles_path
        else:
            self.roles_path = os.path.join(self.root, roles_path)

        self.background_type = background_type
        self.subsample_type = self.parse_subsample_type(subsample_type)
        self.subsample_size = subsample_size

        self.condition_sampling_type = condition_sampling_type
        self.condition_sampling_number = condition_sampling_number
        self.condition_data_types = condition_data_types
        self.data_types = data_types

        # firstly, the image is cropped. Secondly, the picked crop is resized.
        self.image_size = image_size
        self.crop_size = crop_size
        self.resize_scale = resize_scale
        if self.crop_size is not None:
            self.image_size = self.crop_size
        self.image_size = (int(self.image_size[0] / self.resize_scale),
                           int(self.image_size[1] / self.resize_scale))

        transf_path = os.path.join(self.root, 'pose.txt')
        transf = load_pose(transf_path)
        krt_path = os.path.join(self.root, 'KRT.txt')
        self.krt = load_krt(krt_path,
                            image_rescale_ratio=(1 / resize_scale) * fix_intrisics_coef,
                            world_center=transf[:, -1:],
                            world_scale=world_scale,
                            change_orientation=change_orientation,
                            image_size=image_size,
                            center_crop=crop_size,
                            )

        for val in self.krt.values():
            val.update({"image_size": np.array(self.image_size)})

        self.camera_ids = sorted(self.krt.keys())

        scene_prefix = 'scene_'
        time_prefix = 'time_'

        self.scene_time_links = dict()
        self.scene_time_list = []
        self.cumulated_number_of_views = []
        roles = load_roles(self.roles_path)

        for dirpath, subdirnames, filenames in os.walk(self.root, followlinks=True):
            possible_scene, possible_time = dirpath.split(os.sep)[-2:]
            if possible_scene.startswith(scene_prefix) and possible_time.startswith(time_prefix):
                scene_id = possible_scene[len(scene_prefix):]
                time_id = possible_time[len(time_prefix):]

                data = dict()
                try:
                    roles_data = roles[possible_scene][possible_time]
                except KeyError:
                    continue

                self.scene_time_list.append((scene_id, time_id))

                if self.condition_sampling_type not in {'none', None}:
                    data['condition_cameras'] = roles_data[f'condition_{self.mode}']
                else:
                    data['condition_cameras'] = []
                data['reconstruction_cameras'] = roles_data[f'reconstruction_{self.mode}']
                data['dirpath'] = dirpath
                self.scene_time_links[(scene_id, time_id)] = data
                self.cumulated_number_of_views.append(len(data['reconstruction_cameras']))
                self._check_constistency(data)

        self.cumulated_number_of_views = np.cumsum(self.cumulated_number_of_views)

    @property
    def all_cameras_krt(self):
        krt = []
        for index in self.camera_ids:
            camera = self.krt[index]
            camera['camera_name'] = index
            krt.append(camera)

        return {f'frozen_camera_rig': krt}

    def _check_constistency(self, data):
        if len(self.camera_ids) < len(data['condition_cameras']) + len(data['reconstruction_cameras']):
            intersecting_cameras = list(set(data['condition_cameras']) & set(data['reconstruction_cameras']))
            logger.warning(f'Some cameras are used both for conditioning and reconstruction in the folder '
                           f'{data["dirpath"]} -- '
                           + ', '.join(intersecting_cameras)
                           )

        for cam_id, dtype in itertools.product(data['condition_cameras'],
                                               self.condition_data_types,
                                               ):
            if dtype == 'background':
                filepath = os.path.join(self.root, f'{dtype}_{cam_id}.{self.img_ext}')
            else:
                filepath = os.path.join(data['dirpath'], f'{dtype}_{cam_id}.{self.img_ext}')
            if not os.path.exists(filepath):
                logger.warning(f'The condition camera {cam_id} is specified, however file {filepath} is missing')
            if cam_id not in self.krt:
                logger.warning(f'The condition camera {cam_id} is specified, however it is missing in KRT.txt')

        for cam_id, dtype in itertools.product(data['reconstruction_cameras'],
                                               self.data_types,
                                               ):
            if dtype == 'background':
                filepath = os.path.join(self.root, f'{dtype}_{cam_id}.{self.img_ext}')
            else:
                filepath = os.path.join(data['dirpath'], f'{dtype}_{cam_id}.{self.img_ext}')
            if not os.path.exists(filepath):
                logger.warning(f'The reconstruction camera {cam_id} is specified, however file {filepath} is missing')
            if cam_id not in self.krt:
                logger.warning(f'The reconstruction camera {cam_id} is specified, however it is missing in KRT.txt')

    def __getitem__(self, idx):
        position = np.searchsorted(self.cumulated_number_of_views, idx + 1)
        scene_id, time_id = self.scene_time_list[position]
        data = self.scene_time_links[(scene_id, time_id)]

        result = {'scene_id': scene_id,
                  'time_id': time_id,
                  'scene_time': 'frozen_camera_rig',
                  }

        # condition images and cameras
        if self.condition_sampling_type in {None, 'none'}:
            condition_ids = None
        elif self.condition_sampling_type == 'deterministic':
            condition_ids = data['condition_cameras']
        else:
            condition_ids = random.sample(data['condition_cameras'], self.condition_sampling_number)

        if condition_ids is not None:
            condition_keys = ['condition_cameras_ids',
                              'condition_extrinsics',
                              'condition_intrinsics',
                              'condition_images_sizes'
                              ]
            condition_keys.extend([f'condition_{dtype}' for dtype in self.condition_data_types])
            for key in condition_keys:
                result[key] = []
            for cond_id in condition_ids:
                result['condition_cameras_ids'].append(self.camera_ids.index(cond_id))
                result['condition_extrinsics'].append(self.krt[cond_id]['extrin'])
                result['condition_intrinsics'].append(self.krt[cond_id]['intrin'])
                result['condition_images_sizes'].append(self.krt[cond_id]['image_size'])

                for dtype in self.condition_data_types:
                    filepath = os.path.join(data['dirpath'], f'{dtype}_{cond_id}.{self.img_ext}')
                    image, _ = self._read_image(filepath)
                    result[f'condition_{dtype}'].append(self._preprocess_condition_image(image))
            for key in condition_keys:
                result[key] = np.stack(result[key], axis=0)

        # reconstruction camera, images and coords
        if position > 0:
            reconstruction_id = idx - self.cumulated_number_of_views[position - 1]
        else:
            reconstruction_id = idx
        reconstruction_id = data['reconstruction_cameras'][reconstruction_id]

        result['camera_id'] = self.camera_ids.index(reconstruction_id)
        result['extrinsics'] = self.krt[reconstruction_id]['extrin']
        result['intrinsics'] = self.krt[reconstruction_id]['intrin']
        result['images_sizes'] = self.krt[reconstruction_id]['image_size']

        for dtype in self.data_types:
            if dtype == 'background':
                filepath = os.path.join(self.root, f'{dtype}_{reconstruction_id}.{self.img_ext}')
            else:
                filepath = os.path.join(data['dirpath'], f'{dtype}_{reconstruction_id}.{self.img_ext}')
            image, _ = self._read_image(filepath)
            result[dtype] = image

        if self.background_type == 'real':
            assert 'background' in result, "There are no real background. " \
                                           "Probably your forgot add background option in " \
                                           "general_dataset_constants/data_types field in your config."
        elif self.background_type == 'white':
            result['background'] = np.ones_like(result['image']) * 255
        elif self.background_type == 'black':
            result['background'] = np.zeros_like(result['image'])
        elif self.background_type == 'self':
            result['background'] = result['image']

        height, width = result['image'].shape[1:3]
        result['pixel_coords'], result['relative_coords'] = self._sample_pixel_grid(height,
                                                                                    width,
                                                                                    image=result.get('image'),
                                                                                    background=result.get('background'),
                                                                                    )
        if result.get('image') is not None and result['pixel_coords'].shape[:2] != result['image'].shape[1:3]:
            result['image'] = F.grid_sample(torch.from_numpy(result['image']).float().unsqueeze(0),
                                            torch.from_numpy(result['relative_coords']).float().unsqueeze(0),
                                            ).squeeze(0)
            if 'background' in result:
                result['background'] = F.grid_sample(torch.from_numpy(result['background']).float().unsqueeze(0),
                                                    torch.from_numpy(result['relative_coords']).float().unsqueeze(0),
                                                    ).squeeze(0)

        return result
