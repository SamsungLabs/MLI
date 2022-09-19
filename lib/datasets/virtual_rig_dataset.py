"""
This dataset assumes that you generate a virtual camera rig.
All the scenes are shot using this rig.
The numeration of cameras is consistent across the scenes.
The background may be either black or white for all the shots.

The dataset files are expected to have the following structure:

| root/
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


The file `roles.yaml` should contain the ids of cameras, used to extract the latent variable.
Example:

```

- scene_0:
  - time_0:
    - condition_render: [5, 7, 200]
  - time_1:
    - condition_render: [6, 8, 201]
- scene_1:
  - time_2:
    - ...
  - time_1:
    - ...
- ...

```
"""

__all__ = ['VirtualRigDataset']

import itertools
import os
import logging
import random

import numpy as np
import torch.utils.data

from lib.utils.io import load_roles
from .multiview_dataset import MultiviewDataset

logger = logging.getLogger(__name__)
ROLES_PATH = 'roles.yaml'


def normalize(tensor: np.ndarray, axis=-1):
    return tensor / np.linalg.norm(tensor, axis=axis, keepdims=True)


class VirtualRigDataset(MultiviewDataset):
    def __init__(self,
                 root,
                 mode,  # train | val | vis | ...
                 scene,
                 time=None,
                 sort_timestamps=False,
                 n_cameras=200,
                 roles_path=None,
                 background_type='black',
                 condition_sampling_type=None,  # random | deterministic
                 condition_sampling_number=None,  # 3
                 condition_data_types=None,  # [image, depth]
                 image_size=(1024, 667),
                 resize_scale=1,
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
        self.subsample_type = None

        self.condition_sampling_type = condition_sampling_type
        self.condition_sampling_number = condition_sampling_number
        self.condition_data_types = condition_data_types

        # firstly, the image is cropped. Secondly, the picked crop is resized.
        self.image_size = image_size
        self.crop_size = crop_size
        self.resize_scale = resize_scale
        if self.crop_size is not None:
            self.image_size = self.crop_size
        self.image_size = (int(self.image_size[0] / self.resize_scale),
                           int(self.image_size[1] / self.resize_scale))

        self.krt = self._generate_krt(n_cameras)
        for val in self.krt.values():
            val.update({"image_size": np.array(self.image_size)})

        self.camera_ids = sorted(self.krt.keys())

        scene_prefix = 'scene_'
        time_prefix = 'time_'

        self.scene_time_links = dict()
        self.scene_time_list = []
        self.cumulated_number_of_views = []
        roles = load_roles(self.roles_path)

        scene = str(scene)
        if time is not None:
            if not isinstance(time, list):
                time = [time]
            time = [str(timestamp) for timestamp in time]

        for dirpath, subdirnames, filenames in os.walk(self.root, followlinks=True):
            possible_scene, possible_time = dirpath.split(os.sep)[-2:]
            if possible_scene.startswith(scene_prefix) and possible_time.startswith(time_prefix):
                scene_id = possible_scene[len(scene_prefix):]
                time_id = possible_time[len(time_prefix):]
                if scene_id == scene and (time is None or time_id in time):
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
                    # data['reconstruction_cameras'] = roles_data[f'reconstruction_{self.mode}']
                    data['dirpath'] = dirpath
                    # self.cumulated_number_of_views.append(len(data['reconstruction_cameras']))
                    self.scene_time_links[(scene_id, time_id)] = data
                    self._check_constistency(data)

        if time is None:
            time = list(set([time_id for scene_id, time_id in self.scene_time_list]))
        if sort_timestamps:
            time = sorted(time)

        self.scene_time_list = [(scene, timestamp)
                                for timestamp in time
                                if (scene, timestamp) in self.scene_time_list
                                ]

        n_cameras_per_timestamp = n_cameras // len(time)
        residual = list(map(str, np.arange(n_cameras_per_timestamp * len(time), n_cameras)))
        self.cumulated_number_of_views = []
        for i, (scene_id, time_id) in enumerate(self.scene_time_list):
            data = self.scene_time_links[(scene_id, time_id)]
            data['reconstruction_cameras'] = list(map(str, range(i * n_cameras_per_timestamp,
                                                                 (i + 1) * n_cameras_per_timestamp)
                                                      ))
            if i == len(time) - 1:
                data['reconstruction_cameras'].extend(residual)
            self.cumulated_number_of_views.append(len(data['reconstruction_cameras']))

        self.cumulated_number_of_views = np.cumsum(self.cumulated_number_of_views)

    def _generate_krt(self, n_cameras):
        # angles = np.linspace(0.75 * np.pi, 0.25 * np.pi, n_cameras)
        angles = np.linspace(0. * np.pi, 2 * np.pi, n_cameras)

        distance_xy = 5
        distance_z = 0.5
        xs = distance_xy * np.cos(angles)
        ys = distance_xy * np.sin(angles)
        zs = distance_z * np.ones_like(angles)

        # distance_xz = 5
        # distance_y = 0.5
        # xs = distance_xz * np.cos(angles)
        # zs = distance_xz * np.sin(angles)
        # ys = distance_y * np.ones_like(angles)

        cam_poses = np.stack([xs, ys, zs], axis=-1)

        cam_z_dir = - normalize(cam_poses)
        cam_y_dir = - np.stack([np.zeros_like(xs), np.zeros_like(ys), np.ones_like(zs)], axis=-1)
        # cam_y_dir = np.stack([np.zeros_like(xs), np.ones_like(ys), np.zeros_like(zs)], axis=-1)
        cam_x_dir = np.cross(cam_y_dir, cam_z_dir)
        cam_x_dir = normalize(cam_x_dir)
        cam_y_dir = np.cross(cam_z_dir, cam_x_dir)
        cam_y_dir = normalize(cam_y_dir)

        # cam_x_dir = np.stack([-ys, xs, np.zeros_like(zs)], axis=-1)  # keep the horizon line
        # cam_x_dir = normalize(cam_x_dir)
        # cam_y_dir = np.cross(cam_z_dir, cam_x_dir)
        # multiplier = 1 - 2 * (cam_y_dir[:, -1:] > 0).astype(np.float)  # cam Y axis should go towards the ground
        # print(multiplier.reshape(-1))
        # cam_x_dir *= multiplier
        # cam_y_dir *= multiplier

        rotations = np.stack([cam_x_dir, cam_y_dir, cam_z_dir], axis=1)
        translations = - np.einsum('bij,bj->bi', rotations, cam_poses)
        extrinsics = np.concatenate([rotations, translations[:, :, None]], axis=-1).astype(np.float32)
        intrinsic = np.array([[1e3 * self.image_size[0] / 480, 0., self.image_size[0] // 2],
                              [0., 1e3 * self.image_size[1] / 480, self.image_size[1] // 2],
                              [0., 0., 1]]).astype(np.float32)
        out = dict()
        for cam, extrinsic in enumerate(extrinsics):
            out[str(cam)] = {'extrin': extrinsic,
                             'intrin': intrinsic,
                             }
        return out

    def save_krt(self, path):
        with open(path, 'w') as f:
            n_cameras = len(self.krt)
            for name, data in self.krt.items():
                # camera name
                f.write(name)
                f.write('\n')

                # simulated intrinsics
                for line in data['intrin']:
                    f.write(' '.join(str(x) for x in line))
                    f.write('\n')

                # fake dist
                f.write(str(0))
                f.write('\n')

                # simulated extrinsics
                for line in data['extrin']:
                    f.write(' '.join(str(x) for x in line))
                    f.write('\n')

                if name != n_cameras - 1:
                    f.write('\n')

    @property
    def all_cameras_krt(self):
        krt = []
        for index in self.camera_ids:
            camera = self.krt[index]
            camera['camera_name'] = index
            krt.append(camera)

        return {f'virtual_camera_rig': krt}

    def _check_constistency(self, data):
        for cam_id, dtype in itertools.product(data['condition_cameras'],
                                               self.condition_data_types,
                                               ):
            if dtype == 'background':
                filepath = os.path.join(self.root, f'{dtype}_{cam_id}.{self.img_ext}')
            else:
                filepath = os.path.join(data['dirpath'], f'{dtype}_{cam_id}.{self.img_ext}')
            if not os.path.exists(filepath):
                logger.warning(f'The condition camera {cam_id} is specified, however file {filepath} is missing')

    def __getitem__(self, idx):
        position = np.searchsorted(self.cumulated_number_of_views, idx + 1)
        scene_id, time_id = self.scene_time_list[position]
        data = self.scene_time_links[(scene_id, time_id)]

        result = {'scene_id': scene_id,
                  'time_id': time_id,
                  'scene_time': 'virtual_camera_rig',
                  }

        # condition images and cameras
        if self.condition_sampling_type in {None, 'none'}:
            condition_ids = None
        elif self.condition_sampling_type == 'deterministic':
            condition_ids = data['condition_cameras']
        else:
            condition_ids = random.sample(data['condition_cameras'], self.condition_sampling_number)

        if condition_ids is not None:
            condition_keys = [  # 'condition_cameras_ids',
                # 'condition_extrinsics',
                # 'condition_intrinsics',
            ]
            condition_keys.extend([f'condition_{dtype}' for dtype in self.condition_data_types])
            for key in condition_keys:
                result[key] = []
            for cond_id in condition_ids:
                # result['condition_cameras_ids'].append(self.camera_ids.index(cond_id))
                # result['condition_extrinsics'].append(self.krt[cond_id]['extrin'])
                # result['condition_intrinsics'].append(self.krt[cond_id]['intrin'])

                for dtype in self.condition_data_types:
                    filepath = os.path.join(data['dirpath'], f'{dtype}_{cond_id}.{self.img_ext}')
                    image, _ = self._read_image(filepath)
                    result[f'condition_{dtype}'].append(self._preprocess_condition_image(image))
            for key in condition_keys:
                result[key] = np.stack(result[key], axis=0)

        # reconstruction camera and background
        if position > 0:
            reconstruction_id = idx - self.cumulated_number_of_views[position - 1]
        else:
            reconstruction_id = idx
        reconstruction_id = data['reconstruction_cameras'][reconstruction_id]

        result['camera_id'] = self.camera_ids.index(reconstruction_id)
        result['extrinsics'] = self.krt[reconstruction_id]['extrin']
        result['intrinsics'] = self.krt[reconstruction_id]['intrin']
        result['images_sizes'] = self.krt[reconstruction_id]['image_size']

        height, width = self.image_size
        result['pixel_coords'], result['relative_coords'] = self._sample_pixel_grid(height, width)

        if self.background_type == 'white':
            result['background'] = np.ones((1,) + self.image_size, dtype=np.float32) * 255
        elif self.background_type == 'black':
            result['background'] = np.zeros((1,) + self.image_size, dtype=np.float32)

        return result
