"""
https://github.com/augmentedperception/spaces_dataset
"""

__all__ = ['SpacesDataset']

import json
import logging
import math
import os
import random
from copy import copy
from glob import glob
from typing import Dict, Tuple

import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lib.modules.cameras.utils import average_extrinsics
from lib.utils.coord_conversion import coords_pixel_to_film

logger = logging.getLogger(__name__)

_EPS = numpy.finfo(float).eps * 4.0


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3), dtype=numpy.float64)
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1.0])
    1.0
    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=numpy.float64)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True
    """
    quaternion = numpy.zeros((4,), dtype=numpy.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[3] = math.cos(angle / 2.0)
    return quaternion


class Camera(object):
    """Represents a Camera with intrinsics and world from/to camera transforms.
    Attributes:
      w_f_c: The world from camera 4x4 matrix.
      c_f_w: The camera from world 4x4 matrix.
      intrinsics: The camera intrinsics as a 3x3 matrix.
      inv_intrinsics: The inverse of camera intrinsics matrix.
    """

    def __init__(self, intrinsics, w_f_c):
        """Constructor.
        Args:
          intrinsics: A numpy 3x3 array representing intrinsics.
          w_f_c: A numpy 4x4 array representing wFc.
        """
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        self.w_f_c = w_f_c
        self.c_f_w = np.linalg.inv(w_f_c)


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


def _WorldFromCameraFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.
    Args:
      view_json: A dictionary of view parameters.
    Returns:
       A 4x4 transform matrix representing the world from camera transform.
    """

    # The camera model transforms the 3d point X into a ray u in the local
    # coordinate system:
    #
    #  u = R * (X[0:2] - X[3] * c)
    #
    # Meaning the world from camera transform is [inv(R), c]

    transform = np.identity(4)
    position = view_json['position']
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json['orientation']
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform

    axis = angle_axis / angle
    rot_mat = quaternion_matrix(
        quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return transform


def _IntrinsicsFromViewDict(view_params):
    """Fills the intrinsics matrix from view_params.
    Args:
      view_params: Dict view parameters.
    Returns:
       A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = np.identity(3)
    intrinsics[0, 0] = view_params['focal_length']
    intrinsics[1, 1] = (
            view_params['focal_length'] * view_params['pixel_aspect_ratio'])
    intrinsics[0, 2] = view_params['principal_point'][0]
    intrinsics[1, 2] = view_params['principal_point'][1]
    return intrinsics


def ReadView(base_dir, view_json):
    return View(
        image_path=os.path.join(base_dir, view_json['relative_path']),
        shape=(int(view_json['height']), int(view_json['width'])),
        camera=Camera(
            _IntrinsicsFromViewDict(view_json),
            _WorldFromCameraFromViewDict(view_json)))


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, 'models.json')) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views


cameras_configurations = {
    'small_5': {
        'init': [
            [3, 2, 5, 12, 11],
            [2, 1, 6, 11, 10],
            [1, 0, 7, 10, 9],
            [5, 6, 11, 13, 14],
            [6, 7, 10, 14, 15],
        ],
        'novel': [
            [4, 6],
            [5, 7],
            [6, 8],
            [12, 10],
            [11, 9],
        ],

    },
    'large_5': {
        'init': [
            [3, 0, 6, 12, 9],
        ],
        'novel': [
            [2, 1, 4, 5, 7, 8, 11, 10],
        ]
    },
    'big_5': {
        'init': [
            [3, 0, 6, 13, 15],
            [3, 0, 11, 13, 15],
            [3, 0, 10, 13, 15],
        ],
        'novel': [
            [2, 1, 4, 5, 7, 8, 12, 11, 10, 9, 14],
            [2, 1, 4, 5, 6, 7, 8, 12, 10, 9, 14],
            [2, 1, 4, 5, 6, 7, 8, 12, 11, 9, 14],
        ]
    },
    'big_8': {
        'init': [
            [3, 6, 0, 8, 15, 14, 13, 4],
        ],
        'novel': [
            [2, 1, 5, 7, 12, 11, 10, 9],
        ]
    },
    'large_quad': {
        'init': [
            [3, 0, 12, 9],
        ],
        'novel': [
            [2, 1, 4, 5, 6, 7, 8, 11, 10],
        ],
    },
    'large_quad_same': {
        'init': [
            [3, 0, 12, 9],
        ],
        'novel': [
            [3, 0, 12, 9],
        ],
    },
    'medium_quad': {
        'sub_scenes': [0],
        'init': [
            [3, 1, 12, 10],
            [2, 0, 11, 9],
            [5, 7, 13, 15],
        ],
        'novel': [
            [2, 4, 5, 6, 7, 11],
            [1, 5, 6, 7, 8, 10],
            [6, 12, 11, 10, 9, 14],
        ],
    },
    'small_quad': {
        'sub_scenes': [0],
        'init': [
            [3, 2, 12, 11],
            [2, 1, 11, 10],
            [1, 0, 10, 9],
            [5, 6, 13, 4],
            [6, 7, 14, 15],
        ],
        'novel': [
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [12, 11, 10],
            [11, 10, 9],
        ],
    },
    'large_quad_test': {
        'sub_scenes': [0],
        'init': [
            [3, 0, 12, 9]
        ],
        'novel': [
            [1, 2, 4, 5, 6, 7, 8, 10, 11],
        ],
    },
    'medium_quad_test': {
        'sub_scenes': [0],
        'init': [
            [3, 1, 12, 10]
        ],
        'novel': [
            [2, 4, 5, 6, 7, 11],
        ],
    },
    'small_quad_test': {
        'sub_scenes': [0],
        'init': [
            [2, 1, 11, 10]
        ],
        'novel': [
            [5, 6, 7],
        ],
    }
}


class SpacesDataset(Dataset):
    def __init__(self,
                 root,
                 image_size,
                 crop_size=None,
                 relative_intrinsics=True,
                 sample_modes='small_5',
                 novel_cameras_number=None,
                 sample_first_nrow=None,
                 virtual_average_ref_extrinsics=False,
                 scale_constant=1,
                 scenes_list=None,
                 except_scenes_list=None,
                 random_cameras_sub_scene_ids=False,
                 enable_interpolation_for_predefine_configurations=True,
                 transforms_params=None,
                 image_pre_crop=None,
                 ):

        super().__init__()

        if except_scenes_list is None:
            except_scenes_list = []
        if transforms_params is None:
            transforms_params = {'num_cams_resolution_crop': {}}
        self.transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)
        self.relative_intrinsics = relative_intrinsics
        self.sample_modes = sample_modes
        if not isinstance(self.sample_modes, list):
            self.sample_modes = [self.sample_modes]
        self.enable_interpolation_for_predefine_configurations = enable_interpolation_for_predefine_configurations
        self.random_cameras_sub_scene_ids = random_cameras_sub_scene_ids
        self.image_pre_crop = image_pre_crop
        self.transforms_params = transforms_params

        self.novel_cameras_number = novel_cameras_number
        self.virtual_average_ref_extrinsics = virtual_average_ref_extrinsics
        self.scale_constant = scale_constant
        if isinstance(self.scale_constant, list) and len(self.scale_constant) == 2:
            self.scale_constant = sorted(self.scale_constant)
        # if self.novel_cameras_number is None:
        #     self.novel_cameras_number = 16

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.image_size = image_size

        if scenes_list is not None:
            scenes_paths = [os.path.join(root, scene_name) for scene_name in scenes_list]
        else:
            scenes_paths = glob(os.path.join(root, 'scene_*'))

        self.scenes_meta = []
        for scene_path in scenes_paths:
            scene_name = scene_path.split('/')[-1]
            if scene_name not in except_scenes_list:
                scene_meta = {'id': scene_name,
                              'scene_path': scene_path,
                              'scene': ReadScene(scene_path)
                              }
                self.scenes_meta.append(scene_meta)

        if sample_first_nrow is not None:
            self.scenes_meta = self.scenes_meta[:sample_first_nrow]

    def __len__(self):
        return len(self.scenes_meta)

    @staticmethod
    def _relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic * scaling

    @staticmethod
    def _absolute_intrinsic_to_relative(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic / scaling

    @staticmethod
    def _resize_abs_intrinsic(
            intrinsic: torch.Tensor,
            img_size,
            img_resize_size,
    ):
        height, width = img_size
        relative_intrinsic = SpacesDataset._absolute_intrinsic_to_relative(height, width, intrinsic.clone())
        height, width = img_resize_size
        abs_intrinsic = SpacesDataset._relative_intrinsic_to_absolute(height, width, relative_intrinsic)
        return abs_intrinsic

    @staticmethod
    def _center_crop_abs_intrinsic(
            intrinsic: torch.Tensor,
            img_size,
            crop_size,
    ):
        height, width = img_size
        crop_height, crop_width = crop_size
        assert (crop_height <= height) and (crop_width <= width), f"Crop size {crop_height, crop_width} " \
                                                                  f"must be less then image size {height, width}! "

        crop_x = math.floor((width - crop_width) / 2)
        crop_y = math.floor((height - crop_height) / 2)

        pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float).view(1, 1, -1)
        film_coords = coords_pixel_to_film(pixel_coords, intrinsic.unsqueeze(0))[0, 0]
        new_principal_point = - film_coords * torch.diagonal(intrinsic[:-1, :-1], dim1=0, dim2=1)
        cropped_intrinsic = intrinsic.clone()
        cropped_intrinsic[:-1, -1] = new_principal_point

        return cropped_intrinsic

    @staticmethod
    def _crop_data(
            image: torch.Tensor,
            intrinsic: torch.Tensor,
            crop_size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Central crop the image and transform the absolute intrinsics

        Args:
            image: C x H x W
            intrinsic: 3 x 3
            crop_size: [h, w]

        Returns:
            cropped_image: C x H_crop x W_crop
            cropped_intrinsic: 3 x 3
        """
        height, width = image.shape[1:]
        crop_height, crop_width = crop_size
        assert (crop_height <= height) and (crop_width <= width), f"Crop size {crop_height, crop_width} " \
                                                                  f"must be less then image size {height, width}! "

        crop_x = math.floor((width - crop_width) / 2)
        crop_y = math.floor((height - crop_height) / 2)

        cropped_image = image[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float).view(1, 1, -1)
        film_coords = coords_pixel_to_film(pixel_coords, intrinsic.unsqueeze(0))[0, 0]
        new_principal_point = - film_coords * torch.diagonal(intrinsic[:-1, :-1], dim1=0, dim2=1)
        cropped_intrinsic = intrinsic.clone()
        cropped_intrinsic[:-1, -1] = new_principal_point

        return cropped_image, cropped_intrinsic

    def _read_image(self,
                    image_path: str,
                    current_extr: torch.Tensor,
                    absolute_intr: torch.Tensor,
                    camera_shape,
                    image_resize_size,
                    image_crop_size=None,
                    image_pre_crop=None,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        try:
            with Image.open(image_path) as img:
                w, h = img.size
                img_size = np.array([h, w])

                absolute_intr = self._resize_abs_intrinsic(intrinsic=absolute_intr,
                                                           img_size=camera_shape,
                                                           img_resize_size=img_size)

                if image_pre_crop is not None:
                    img = transforms.CenterCrop(image_pre_crop)(img)
                    absolute_intr = self._center_crop_abs_intrinsic(intrinsic=absolute_intr,
                                                                    img_size=img_size,
                                                                    crop_size=image_pre_crop)
                    img_size = image_pre_crop

                if isinstance(image_resize_size, int):
                    shortest_idx = np.argsort(img_size)[0]
                    resize_ratio = image_resize_size / img_size[shortest_idx]
                    image_resize_size = [int(resize_ratio * img_size[0]), int(resize_ratio * img_size[1])]
                img = transforms.Resize(image_resize_size)(img)
                absolute_intr = self._resize_abs_intrinsic(intrinsic=absolute_intr,
                                                           img_size=img_size,
                                                           img_resize_size=image_resize_size)
                current_image = self.transforms(img)
        except OSError as e:
            logger.error(f'Possibly, image file is broken: {image_path}')
            raise e

        if image_crop_size is not None:
            current_image, absolute_intr = self._crop_data(image=current_image,
                                                           intrinsic=absolute_intr,
                                                           crop_size=image_crop_size)

        if self.relative_intrinsics:
            if image_crop_size is not None:
                current_intr = self._absolute_intrinsic_to_relative(*image_crop_size, absolute_intr)
            else:
                current_intr = absolute_intr
        else:
            current_intr = absolute_intr

        return current_image, current_extr, current_intr

    def _read_cameras(self,
                      scene_meta,
                      sub_scene_ids,
                      cameras_ids,
                      image_size,
                      crop_size,
                      scale_constant,
                      image_pre_crop=None,
                      ) -> Dict[str, torch.Tensor]:
        images, extrinsics, intrinsics = [], [], []

        for i, (camera_id, sub_scene_id) in enumerate(zip(cameras_ids, sub_scene_ids)):
            cam_4_pos = scene_meta[sub_scene_id][4].camera.w_f_c[:3, -1:]
            cam_8_pos = scene_meta[sub_scene_id][8].camera.w_f_c[:3, -1:]
            dist = np.sqrt(np.sum((cam_4_pos - cam_8_pos) ** 2))

            camera_data = scene_meta[sub_scene_id][camera_id]
            pose = camera_data.camera.c_f_w[:3, :]  # camera from world
            current_extr = torch.from_numpy(pose[:3, :]).float()
            current_extr[:, -1] = current_extr[:, -1] / dist * scale_constant
            absolute_intr = torch.from_numpy(camera_data.camera.intrinsics).float()
            camera_shape = camera_data.shape
            # relative_intr = self._absolute_intrinsic_to_relative(*scene_meta[sub_scene_id][camera_id].shape,
            #                                                      absolute_intr)

            current_image, current_extr, current_intr = self._read_image(
                camera_data.image_path,
                current_extr,
                absolute_intr=absolute_intr,
                camera_shape=camera_shape,
                image_resize_size=image_size,
                image_crop_size=crop_size,
                image_pre_crop=image_pre_crop,
            )

            images.append(current_image)
            extrinsics.append(current_extr)
            intrinsics.append(current_intr)

        num_cameras = len(images)
        images = torch.stack(images).view(1, num_cameras, *images[0].shape)
        extrinsics = torch.stack(extrinsics).view(1, num_cameras, *extrinsics[0].shape)
        intrinsics = torch.stack(intrinsics).view(1, num_cameras, *intrinsics[0].shape)

        return {
            'timestamp': torch.from_numpy((np.array(sub_scene_id) * 1000 + np.array(cameras_ids))[None, :]),
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

    def get_item_cameras_parameters(self, scene_meta):
        num_sub_scenes = len(scene_meta)
        sample_mode = random.choice(self.sample_modes)
        item_cameras_parameters = {}

        if sample_mode.split('_')[0] == 'random':
            num_init_cams = int(sample_mode.split('_')[-1])
            cameras_ids = random.sample(list(range(16)), k=16)
            cameras_configuration = {
                'init': [cameras_ids[:num_init_cams]],
                'novel': [cameras_ids[num_init_cams:]],
            }
        else:
            cameras_configuration = cameras_configurations[sample_mode]

        if len(cameras_configuration['init']) > 1:
            scene_cameras_configuration_id = random.randint(0, len(cameras_configuration['init']) - 1)
        else:
            scene_cameras_configuration_id = 0

        init_cams_ids = copy(cameras_configuration['init'][scene_cameras_configuration_id])

        if self.enable_interpolation_for_predefine_configurations:
            novel_cams_ids = copy(cameras_configuration['novel'][scene_cameras_configuration_id])
        else:
            novel_cams_ids = list(set(range(16)).difference(set(init_cams_ids)))

        if self.novel_cameras_number is not None:
            num_novel_cameras = min(self.novel_cameras_number, len(novel_cams_ids))
            novel_cams_ids = random.sample(novel_cams_ids, num_novel_cameras)

        sub_scenes_ids = cameras_configuration.get('sub_scenes', list(range(0, num_sub_scenes - 1)))
        if not self.random_cameras_sub_scene_ids:
            sub_scenes_ids = [random.choice(sub_scenes_ids)]

        item_cameras_parameters['init_cameras_ids'] = init_cams_ids
        item_cameras_parameters['novel_cameras_ids'] = novel_cams_ids
        item_cameras_parameters['init_scene_ids'] = random.choices(sub_scenes_ids, k=len(init_cams_ids))
        item_cameras_parameters['novel_scene_ids'] = random.choices(sub_scenes_ids, k=len(novel_cams_ids))

        transform_params = self.transforms_params['num_cams_resolution_crop'].get(len(init_cams_ids), None)
        resize_scales = self.transforms_params.get('resize_scales', [1])
        if len(resize_scales) == 3:
            resize_scales = np.linspace(resize_scales[0], resize_scales[1], resize_scales[2])

        if transform_params:
            scale_factor = random.choice(resize_scales)
            image_size = [int(transform_params[0][0] * scale_factor),
                          int(transform_params[0][1] * scale_factor)]
            crop_size = transform_params[1]
        else:
            image_size = self.image_size
            crop_size = self.crop_size

        item_cameras_parameters['image_size'] = image_size
        item_cameras_parameters['crop_size'] = crop_size

        return item_cameras_parameters

    def __getitem__(self, idx):
        scene_meta = copy(self.scenes_meta[idx]['scene'])
        scene_id = self.scenes_meta[idx]['id']
        item_cameras_parameters = self.get_item_cameras_parameters(scene_meta)

        if isinstance(self.scale_constant, list) and len(self.scale_constant) == 2:
            scale_constant = random.uniform(0, 1) \
                             * (self.scale_constant[1] - self.scale_constant[0]) \
                             + self.scale_constant[0]
        else:
            scale_constant = self.scale_constant

        init_data = self._read_cameras(scene_meta,
                                       sub_scene_ids=item_cameras_parameters['init_scene_ids'],
                                       cameras_ids=item_cameras_parameters['init_cameras_ids'],
                                       image_size=item_cameras_parameters['image_size'],
                                       crop_size=item_cameras_parameters['crop_size'],
                                       scale_constant=scale_constant,
                                       image_pre_crop=self.image_pre_crop,
                                       )
        novel_data = self._read_cameras(scene_meta,
                                        sub_scene_ids=item_cameras_parameters['novel_scene_ids'],
                                        cameras_ids=item_cameras_parameters['novel_cameras_ids'],
                                        image_size=item_cameras_parameters['image_size'],
                                        crop_size=item_cameras_parameters['crop_size'],
                                        scale_constant=scale_constant,
                                        image_pre_crop=self.image_pre_crop,
                                        )

        if self.virtual_average_ref_extrinsics:
            ref_data = {}
            ref_data['extrinsics'] = average_extrinsics(init_data['extrinsics'])
            ref_data['intrinsics'] = init_data['intrinsics'][..., [0], :, :]
        else:
            ref_data = init_data

        output = {
            'time_id': 0,
            'scene_time': f'{scene_id}//{0}',
            'scene_id': scene_id,
            'initial': init_data,
            'reference': ref_data,
            'novel': novel_data
        }

        return output
