"""
The dataset files are expected to have the following structure:

| root/
|---- scene_0/
     |---- time_0/
          |---- KRT.txt
          |---- pose.txt
          |---- image_0.jpg
          |---- depth_0.jpg
          |---- background_0.jpg
          |---- image_1.jpg
          |---- depth_1.jpg
          |---- background_1.jpg
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

__all__ = ['MultiviewDataset']

import itertools
import logging
import os
import random

import numpy as np
from scipy.ndimage import gaussian_filter
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import cv2 as cv

from lib.utils.io import load_krt, load_pose, load_roles

logger = logging.getLogger(__name__)
ROLES_PATH = 'roles.yaml'


class MultiviewDataset(torch.utils.data.Dataset):
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
                 fix_intrisics_coef=1,
                 change_orientation=False,
                 img_ext='jpg',
                 crop_size=None):

        super().__init__()

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

                transf_path = os.path.join(dirpath, 'pose.txt')
                krt_path = os.path.join(dirpath, 'KRT.txt')

                transf = load_pose(transf_path)
                data['krt'] = load_krt(krt_path,
                                       image_rescale_ratio=(1 / resize_scale) * fix_intrisics_coef,
                                       world_center=transf[:, -1:],
                                       world_scale=world_scale,
                                       change_orientation=change_orientation,
                                       image_size=image_size,
                                       center_crop=crop_size,
                                       )
                for key, item in data['krt'].items():
                    item.update({"image_size": np.array(self.image_size)})

                if self.condition_sampling_type not in {'none', None}:
                    data['condition_cameras'] = roles_data[f'condition_{self.mode}']
                else:
                    data['condition_cameras'] = []
                data['reconstruction_cameras'] = roles_data[f'reconstruction_{self.mode}']
                data['cameras_ids'] = sorted(list(
                    set(data['condition_cameras']) | set(data['reconstruction_cameras'])
                ))
                data['dirpath'] = dirpath
                self.scene_time_links[(scene_id, time_id)] = data
                self.cumulated_number_of_views.append(len(data['reconstruction_cameras']))
                self._check_constistency(data)

        self.cumulated_number_of_views = np.cumsum(self.cumulated_number_of_views)

    @property
    def all_cameras_krt(self):
        out = {}
        for (scene_id, time_id), data in self.scene_time_links.items():
            krt = []
            for index in data['cameras_ids']:
                camera = data['krt'][index]
                camera['camera_name'] = index
                krt.append(camera)

            out[f'{scene_id}//{time_id}'] = krt

        return out

    def _check_constistency(self, data):
        if len(data['cameras_ids']) < len(data['condition_cameras']) + len(data['reconstruction_cameras']):
            intersecting_cameras = list(set(data['condition_cameras']) & set(data['reconstruction_cameras']))
            logger.warning(f'Some cameras are used both for conditioning and reconstruction in the folder '
                           f'{data["dirpath"]} -- '
                           + ', '.join(intersecting_cameras)
                           )

        for cam_id, dtype in itertools.product(data['condition_cameras'],
                                               self.condition_data_types,
                                               ):
            filepath = os.path.join(data['dirpath'], f'{dtype}_{cam_id}.{self.img_ext}')
            if not os.path.exists(filepath):
                logger.warning(f'The condition camera {cam_id} is specified, however file {filepath} is missing')
            if cam_id not in data['krt']:
                krt_path = os.path.join(data['dirpath'], 'KRT.txt')
                logger.warning(f'The condition camera {cam_id} is specified, however it is missing in {krt_path}')

        for cam_id, dtype in itertools.product(data['reconstruction_cameras'],
                                               self.data_types,
                                               ):
            filepath = os.path.join(data['dirpath'], f'{dtype}_{cam_id}.{self.img_ext}')
            if not os.path.exists(filepath):
                logger.warning(f'The reconstruction camera {cam_id} is specified, however file {filepath} is missing')
            if cam_id not in data['krt']:
                krt_path = os.path.join(data['dirpath'], 'KRT.txt')
                logger.warning(f'The reconstruction camera {cam_id} is specified, however it is missing in {krt_path}')

    def __getitem__(self, idx):
        position = np.searchsorted(self.cumulated_number_of_views, idx + 1)
        scene_id, time_id = self.scene_time_list[position]
        data = self.scene_time_links[(scene_id, time_id)]

        result = {'scene_id': scene_id,
                  'time_id': time_id,
                  'scene_time': f'{scene_id}//{time_id}',
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
                result['condition_cameras_ids'].append(data['cameras_ids'].index(cond_id))
                result['condition_extrinsics'].append(data['krt'][cond_id]['extrin'])
                result['condition_intrinsics'].append(data['krt'][cond_id]['intrin'])
                result['condition_images_sizes'].append(data['krt'][cond_id]['image_size'])

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

        result['camera_id'] = data['cameras_ids'].index(reconstruction_id)
        result['extrinsics'] = data['krt'][reconstruction_id]['extrin']
        result['intrinsics'] = data['krt'][reconstruction_id]['intrin']
        result['images_sizes'] = data['krt'][reconstruction_id]['image_size']

        for dtype in self.data_types:
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

    def __len__(self):
        return self.cumulated_number_of_views[-1]

    def _sample_pixel_grid(self, height, width, image=None, background=None):
        subsample_type, subsample_params = self.get_subsample_type()

        if subsample_type == 'patch':
            indx = np.random.randint(0, width - self.subsample_size + 1)
            indy = np.random.randint(0, height - self.subsample_size + 1)

            px, py = np.meshgrid(np.arange(indx, indx + self.subsample_size),
                                 np.arange(indy, indy + self.subsample_size))
            adjustment = -1
        elif subsample_type == 'random-int':
            px = np.random.randint(0, width, size=(self.subsample_size, self.subsample_size))
            py = np.random.randint(0, height, size=(self.subsample_size, self.subsample_size))
            adjustment = -1
        elif subsample_type == 'random-uniform':
            px = np.random.uniform(0, width - 1e-5, size=(self.subsample_size, self.subsample_size))
            py = np.random.uniform(0, height - 1e-5, size=(self.subsample_size, self.subsample_size))
            adjustment = 0
        elif subsample_type in {'bbox', 'bbox-cv',
                                'patch-bbox', 'patch-bbox-cv',
                                }:
            mask = self._get_object_mask(image,
                                         background,
                                         mode='cv' if subsample_type in {'bbox-cv', 'patch-bbox-cv'} else 'diff',
                                         )
            h, w = mask.shape
            odds_ratio = float(subsample_params.get('odds_ratio', 100))
            # pixel of an object is odds_ratio (e.g., 100) times more likely chosen than a background pixel
            probas = odds_ratio * mask + 1. * (1 - mask)
            if subsample_type in {'bbox', 'bbox-cv'}:
                probas /= probas.sum()
                probas = probas.reshape(-1)
                selected_pixels_indices = np.random.choice(h * w, self.subsample_size ** 2, p=probas, replace=False)
                selected_flags = np.isin(np.arange(h * w), selected_pixels_indices)
                selected_flags = selected_flags.reshape(h, w)
                px, py = np.nonzero(selected_flags)
                px = px.reshape(self.subsample_size, self.subsample_size)
                py = py.reshape(self.subsample_size, self.subsample_size)
            elif subsample_type in {'patch-bbox', 'patch-bbox-cv'}:
                probas[:self.subsample_size, :] = 0
                probas[-self.subsample_size:, :] = 0
                probas[:, :self.subsample_size] = 0
                probas[:, -self.subsample_size:] = 0
                probas /= probas.sum()
                probas = probas.reshape(-1)
                selected_pixel_index = np.random.choice(h * w, 1, p=probas)
                selected_flags = np.zeros(h * w, dtype=np.float32)
                selected_flags[selected_pixel_index] = 1
                selected_flags = selected_flags.reshape(h, w)
                indx, indy = np.nonzero(selected_flags)
                indx = np.arange(indx - np.floor(self.subsample_size / 2),
                                 indx + np.ceil(self.subsample_size / 2))
                indy = np.arange(indy - np.floor(self.subsample_size / 2),
                                 indy + np.ceil(self.subsample_size / 2))
                px, py = np.meshgrid(indx, indy)
            adjustment = -1
        else:
            px, py = np.meshgrid(np.arange(width),
                                 np.arange(height))
            adjustment = -1

        pixel_coords = np.stack((px, py), axis=-1).astype(np.float32)
        grid_normalizer = np.array([width, height]).reshape((1, 1, -1)).astype(np.float32) + adjustment
        assert pixel_coords.ndim == grid_normalizer.ndim, \
            f'{pixel_coords.shape}\t{grid_normalizer.shape}'
        relative_coords = -1 + 2 * pixel_coords / grid_normalizer

        return pixel_coords, relative_coords

    @staticmethod
    def _get_object_mask(image, background, mode='cv'):
        """
        Return a binary mask of an object vs its background. (1 for object, 0 for background)

        :param image: C x H x W
        :param background: C x H x W
        :param mode: diff | cv
        :return: H x W
        """
        if mode == 'cv':
            back_sub = cv.createBackgroundSubtractorMOG2()
            back_sub.apply(background.transpose((1, 2, 0)), learningRate=1)
            diff = back_sub.apply(image.transpose((1, 2, 0)), learningRate=0) > 1e-3
            mask = gaussian_filter(diff * 255, sigma=4, mode='nearest') > 1e-3
        elif mode == 'diff':
            mask = 1 - np.isclose(image, background, atol=5).all(0)
        else:
            raise ValueError(f'Unknown mode {mode}')

        return mask

    def _read_image(self, path):
        with Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.crop_size:
                image = TF.center_crop(image, self.crop_size)
            if self.resize_scale != 1:
                image = TF.resize(image, self.image_size)
            image = np.asarray(image, dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)  # C x H x W
        valid = np.float32(image.sum() != 0)
        return image, valid

    @staticmethod
    def _preprocess_condition_image(img):
        """[0, 255] -> [-1, 1]"""
        return img / 255 * 2 - 1

    @staticmethod
    def parse_subsample_type(subsample_type):
        """
        Input is expected to be a dict similar to this one:
        ```
        random-uniform:
            proba: 0.4
        bbox-cv:
            proba: 0.6
            odds_ratio: 100
        ```

        However, some other cases are also supported.
        """
        if subsample_type is None:
            return subsample_type

        elif isinstance(subsample_type, str):
            subsample_type = dict(types=[subsample_type], probas=[1.], params=[{}])
            return subsample_type

        elif isinstance(subsample_type, dict):
            out = dict(types=[], probas=[], params=[])
            without_proba_idx = []
            for i, (type_name, config) in enumerate(subsample_type.items()):
                out['types'].append(type_name)
                out['params'].append({k: v for k, v in config.items() if v != 'proba'})
                if 'proba' in config:
                    out['probas'].append(config['proba'])
                else:
                    without_proba_idx.append(i)

            if len(without_proba_idx) > 0:
                residual = 1 - sum(out['probas'])
                equal_proba = residual / len(without_proba_idx)
                for idx in without_proba_idx:
                    out['probas'].insert(idx, equal_proba)

            return out

        else:
            raise ValueError(f'Unknown kind of input: {subsample_type}')

    def get_subsample_type(self):
        if self.subsample_type is None:
            return None, None
        idx = np.random.choice(len(self.subsample_type['types']), 1, p=self.subsample_type['probas'])[0]
        type_name = self.subsample_type['types'][idx]
        params = self.subsample_type['params'][idx]
        return type_name, params
