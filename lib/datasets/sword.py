"""
For non-commercial research purposes. See license of the SWORD dataset.

SWORD dataset should have the following structure:
| root/:
|---- dataset.csv
|---- videos/
    |---- scene_id1
        |---- frame1.jpg
        |---- frame2.jpg
    |---- scene_id2
        |---- frame1.jpg
        |---- frame2.jpg
|---- views/
    |---- scene_id1.txt
    |---- scene_id2.txt

dataset.csv fields:

    General:
        scene_id - scene id

    File paths:
        images_path - scene images path in full resolution
        images_path_x - images path in x resolution in short side, for example:
            images_path_256 - it's images with 455x256 resolution if source was in full HD (1920x1080)
        colmap_path - path to colmap scene data (points3D.bin, images.bin, cameras.bin)
        views_path - path to txt file with scene RealEstate10k like views params (intrinsics, extrinsics).

    Scene params:
        num_points - number of 3d points in colmap scene
        num_views - number of views
        p_x - depth for x percentile, calculated via all scene views
        error_mean - mean of colmap 3d point error
        error_std - std of colmap 3d point error

    Intrinsic (relative):
        f_x - focal distance x
        f_y - focal distance y
        c_x - central point x
        c_y - central point y
        original_resolution_x - original image x resolution (which used for colmap)
        original_resolution_y - original image y resolution (which used for colmap)

"""

__all__ = ['SWORD', 'RealEstate10k']

import logging
import math
import os
import pickle
import random
import sys
from glob import glob
import copy

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from utils import *

DUMPED_FILE_NAME = 'dump.pkl'
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class RealEstate10k(Dataset):
    def __init__(self,
                 root,
                 image_size,
                 crop_size=None,
                 max_len_sequence=10,
                 set_reference_as_origin=False,
                 init_cameras_number_per_ref=1,
                 ref_cameras_number=1,
                 novel_cameras_number_per_ref=1,
                 cameras_sample_mode='extrapolation_only',
                 dump_path=None,
                 use_cached=False,
                 relative_intrinsics=False,
                 translation_fix=None,
                 use_point_clouds=False,
                 number_of_points=8000,
                 num_samples=None,
                 virtual_average_ref_extrinsics=False,
                 transforms_params=None,
                 sample_first_nrow=-1,
                 scale_constant=1,
                 fake_image=False,
                 image_scale=1.0,
                 ):
        """

        Args:
            root: root folder
            image_size (tuple): height, width for resize
            crop_size (int, tuple, None): height, width for crop after resize
            transforms_params (dict): transform params for generating random crops and random number of views on train.
            max_len_sequence (int): the maximum distance (in frames) might be between cameras
            set_reference_as_origin: whether to set reference camera pose as identity matrix,
                                     align other poses correspondingly
            init_cameras_number_per_ref: number of cameras, treated as initial, per one reference camera
            ref_cameras_number: number of cameras, treated as reference
            novel_cameras_number_per_ref: number of cameras, treated as novel, per one reference camera
            cameras_sample_mode:
                'simple': fully random camera samling.
                'extrapolation_only': novel cameras cannot be placed between any source or reference cameras.
                                        That is, the following consecutive triplets of cameras are prohibited:
                                            initial - novel - initial,
                                            initial - novel - reference.
                                        However, note, that a novel camera may be equal to the last source (i.e. initial or reference)
                                        camera in a row.
                'interpolation_only': initial cameras are sampled evenly along random subsequence with length max_len_sequence,
                                      then reference and novel cameras are sampled between them.
.
            dump_path: path with saved scene data dict
            relative_intrinsics: enable relative intrinsics format
            translation_fix: coefficient for translation vector, needed for MannequinChallenge dataset
            virtual_average_ref_extrinsics: generates average reference extrinsic based on source extrinsics
            use_point_clouds: load point clouds saved by Colmap
        Returns:
            dict of camera types with dicts of torch.Tensor for each camera, stacked along dim=0
        """
        super().__init__()
        self.transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)
        self.transforms_params = transforms_params
        self.image_size = image_size
        self.path = root
        self.scenes_data = {}
        self.video_folders = glob(os.path.join(self.path, 'videos/*'))
        self.set_reference_as_origin = set_reference_as_origin
        self.init_cameras_number_per_ref = init_cameras_number_per_ref
        self.ref_cameras_number = ref_cameras_number
        self.novel_cameras_number_per_ref = novel_cameras_number_per_ref
        self.total_cameras_needed = self.ref_cameras_number * (self.init_cameras_number_per_ref
                                                               + self.novel_cameras_number_per_ref
                                                               + 1)
        if virtual_average_ref_extrinsics:
            self.total_cameras_needed = self.total_cameras_needed - 1

        self.max_len_sequence = max(max_len_sequence, self.total_cameras_needed)
        # if cameras_sample_mode == 'interpolation_only':
        #     self.total_cameras_needed = self.max_len_sequence
        self.cameras_sample_mode = cameras_sample_mode
        self.use_point_clouds = use_point_clouds
        self.dump_path = dump_path
        self.relative_intrinsics = relative_intrinsics
        self.translation_fix = translation_fix
        self.number_of_points_in_cloud = number_of_points
        self.virtual_average_ref_extrinsics = virtual_average_ref_extrinsics
        self.num_samples = num_samples

        self.fake_image = fake_image
        self.image_scale = image_scale

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        if self.dump_path is not None and os.path.exists(os.path.join(self.dump_path, DUMPED_FILE_NAME)):
            with open(os.path.join(self.dump_path, DUMPED_FILE_NAME), 'rb') as fin:
                self.scenes_data = pickle.load(fin)
            if use_cached:
                logger.info('Load from cached data')
                if self.use_point_clouds:
                    self.filter_scenes_with_point_clouds()
                return

        if sample_first_nrow > 0:
            self.video_folders = self.video_folders[:sample_first_nrow]

        scale = 1 / scale_constant

        for video_folder in tqdm(self.video_folders):
            scene_id = video_folder.split('/')[-1].strip()
            txt_path = os.path.join(self.path, f'{scene_id}.txt')
            if not os.path.exists(txt_path) or \
                    use_point_clouds and not \
                    os.path.exists(os.path.join(self.path, 'colmap', scene_id)):
                continue
            images = glob(os.path.join(video_folder, '*.jpg'))
            if len(images) >= self.total_cameras_needed:
                with open(txt_path) as f:
                    text = f.read()
                rows = text.split('\n')[1:-1]
                intrinsics = {}
                extrinsics = {}
                images_dict = {}

                for i, row in enumerate(rows):
                    timestamp = int(row.split(' ')[0])
                    if not any([timestamp == int(im.split('/')[-1].split('.')[0]) for im in images]):
                        break
                    (focal_length_x,
                     focal_length_y,
                     principal_point_x,
                     principal_point_y) = np.array(row.split(' ')[1:5]).astype(np.float32)

                    intrinsics[timestamp] = np.array([[focal_length_x, 0, principal_point_x],
                                                      [0, focal_length_y, principal_point_y],
                                                      [0, 0, 1]])
                    extrs = np.array(row.split(' ')[7:19]).astype(np.float32).reshape((3, 4), order='C')
                    extrs[:, -1] = extrs[:, -1] / scale

                    extrinsics[timestamp] = extrs
                    images_dict[timestamp] = [s for s in images if
                                              timestamp == int(s.split('/')[-1].split('.')[0])]

                self.scenes_data[scene_id] = {'images_paths': images_dict,
                                              'txt_path': txt_path,
                                              'extrinsics': extrinsics,
                                              'intrinsics': intrinsics}

            if self.num_samples is not None and len(self.scenes_data) > self.num_samples:
                break

        if self.dump_path is not None:
            os.makedirs(self.dump_path, exist_ok=True)
            with open(os.path.join(self.dump_path, DUMPED_FILE_NAME), 'wb') as fout:
                pickle.dump(self.scenes_data, fout)
                logger.info(f'Update dumped dataset with size: {sys.getsizeof(self.scenes_data)}')
            os.chmod(os.path.join(self.dump_path, DUMPED_FILE_NAME), 0o777)

        if self.use_point_clouds:
            self.filter_scenes_with_point_clouds()

    def __len__(self):
        return len(self.scenes_data)

    def filter_scenes_with_point_clouds(self):
        colmap_folders = glob(os.path.join(self.path, 'colmap', '*'))
        colmap_folders = set([folder.split('/')[-1].strip() for folder in colmap_folders])
        self.scenes_data = {scene: data for scene, data in self.scenes_data.items() if scene in colmap_folders}

    def __getitem__(self, idx):
        scene_id = list(self.scenes_data.keys())[idx]

        init_cameras_number_per_ref, image_resize_size, image_crop_size, max_len_sequence = self.get_item_params()
        ref_ts, init_ts, novel_ts = self.sample_timestamps(
            init_cameras_number_per_ref=init_cameras_number_per_ref,
            txt_path=self.scenes_data[scene_id]['txt_path'],
            timestamps_poses=self.scenes_data[scene_id]['extrinsics'],
            max_len_sequence=max_len_sequence
        )

        ref_data = self._read_frame(scene_id, ref_ts, image_resize_size, image_crop_size)
        init_data = self._read_frame(scene_id, init_ts, image_resize_size, image_crop_size)

        init_data['extrinsics'], _ = self._transform_extrinsics(init_data['extrinsics'],
                                                                ref_data['extrinsics'])
        if novel_ts is None:
            novel_data = {'timestamp': [],
                          'image': [],
                          'extrinsics': [],
                          'intrinsics': ref_data['intrinsics'],
                          }
        else:
            novel_data = self._read_frame(scene_id, novel_ts, image_resize_size, image_crop_size)
            novel_data['extrinsics'], ref_data['extrinsics'] = self._transform_extrinsics(novel_data['extrinsics'],
                                                                                          ref_data['extrinsics'])


        time_id = 0
        output = {
            'time_id': time_id,
            'scene_time': f'{scene_id}//{time_id}',
            'scene_id': scene_id,
            'initial': init_data,
            'reference': ref_data,
            'novel': novel_data,
        }

        if self.use_point_clouds:
            point_cloud_path = os.path.join(self.path, 'colmap', scene_id, 'triangulated', 'points3D.bin')
            point_cloud_coords = load_colmap_binary_cloud(point_cloud_path)
            vs = point_cloud_coords.shape[0]
            rand_perm = random.sample(range(point_cloud_coords.shape[0]), min(self.number_of_points_in_cloud, vs))
            output['point_cloud'] = [torch.from_numpy(point_cloud_coords[rand_perm]).float()]

        return output

    def sample_timestamps(self,
                          init_cameras_number_per_ref,
                          txt_path,
                          timestamps_poses,
                          max_len_sequence=None
                          ):
        legacy_smaple_modes = ['extrapolation_only', 'interpolation_only', 'randomly', 'simple']
        timestamps = list(timestamps_poses.keys())

        if max_len_sequence is None:
            max_len_sequence = self.max_len_sequence

        if self.cameras_sample_mode in legacy_smaple_modes:
            if self.cameras_sample_mode == 'extrapolation_only':
                sample_function = sample_from_timestamps_extrapolation
            elif self.cameras_sample_mode == 'interpolation_only':
                sample_function = sample_from_timestamps_interpolation
            elif self.cameras_sample_mode == 'randomly':
                sample_function = sample_from_timestamps_randomly
            else:
                sample_function = sample_from_timestamps_base

            ref_ts, init_ts, novel_ts, too_short_sequence = sample_function(
                timestamps=timestamps,
                ref_cameras_number=self.ref_cameras_number,
                novel_cameras_number_per_ref=self.novel_cameras_number_per_ref,
                init_cameras_number_per_ref=init_cameras_number_per_ref,
                max_len_sequence=max_len_sequence
            )
        else:
            assert self.ref_cameras_number == 1, "self.ref_cameras_number must be 1"
            ref_ts, init_ts, novel_ts, too_short_sequence = sample_nearest(
                timestamps_poses=timestamps_poses,
                novel_cameras_number_per_ref=self.novel_cameras_number_per_ref,
                init_cameras_number_per_ref=init_cameras_number_per_ref,
                max_len_sequence=max_len_sequence
            )
        if too_short_sequence:
            logger.debug(f'{txt_path} max len sequence {self.max_len_sequence}, '
                         f'diff {len(timestamps) - self.max_len_sequence}, '
                         f'len_timestamps {len(timestamps)}'
                         )

        return ref_ts, init_ts, novel_ts

    def get_item_params(self):
        if self.transforms_params is not None:
            resize_scales = self.transforms_params.get('resize_scales', None)
            if resize_scales is not None:
                resize_scale = resize_scales[np.random.choice(len(resize_scales), 1)[0]]
            else:
                resize_scale = 1
            transform_combinations = self.transforms_params['num_cams_resolution_crop_lensq_weight']
            weights = np.array([el[-1] for el in transform_combinations])
            weights = weights / np.sum(weights)
            transform_combination = transform_combinations[np.random.choice(len(transform_combinations), 1, p=weights)[0]]
            init_cameras_number_per_ref = transform_combination[0]
            resolution = [int(transform_combination[1][0] * resize_scale * self.image_scale),
                          int(transform_combination[1][1] * resize_scale * self.image_scale)]
            # crop_size = transform_combination[2]
            crop_size = [int(transform_combination[2][0] * self.image_scale),
                         int(transform_combination[2][1] * self.image_scale)]
            if self.image_scale != 1:
                crop_size = [crop_size[0] // 16 * 16, crop_size[1] // 16 * 16]

            image_resize_size, image_crop_size = self._get_transform_params(
                resolution,
                crop_size
            )
            max_len_sequence = transform_combination[3]
        else:
            image_resize_size = self.image_size
            image_crop_size = self.crop_size
            init_cameras_number_per_ref = self.init_cameras_number_per_ref
            max_len_sequence = self.max_len_sequence

        return init_cameras_number_per_ref, image_resize_size, image_crop_size, max_len_sequence

    @staticmethod
    def _get_transform_params(
            image_size,
            crop_size
    ):
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        if isinstance(image_size, int):
                    image_size = [image_size, image_size]

        return image_size, crop_size

    def _transform_extrinsics(self,
                              extrinsics: torch.Tensor,
                              reference_extrinsics: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.translation_fix is not None:
            extrinsics[..., -1:] = extrinsics[..., -1:] / self.translation_fix
            reference_extrinsics[..., -1:] = reference_extrinsics[..., -1:] / self.translation_fix

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

    @staticmethod
    def _relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic * scaling

    @staticmethod
    def _absolute_intrinsic_to_relative(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic / scaling

    def _read_image(self,
                    image_path: str,
                    current_extr: torch.Tensor,
                    relative_intr: torch.Tensor,
                    image_resize_size,
                    image_crop_size=None,
                    fake_image=False,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.fake_image is not None:
            fake_image = self.fake_image

        if not fake_image:
            try:
                with Image.open(image_path) as img:
                    if isinstance(image_resize_size, int):
                        img_size = np.array(img.size)
                        shortest_idx = np.argsort(img_size)[0]
                        resize_ratio = image_resize_size / img_size[shortest_idx]
                        image_resize_size = [int(resize_ratio * img_size[1]), int(resize_ratio * img_size[0])]
                    img = transforms.Resize(image_resize_size)(img)
                    current_image = self.transforms(img)
            except OSError as e:
                logger.error(f'Possibly, image file is broken: {image_path}')
                raise e
        else:
            img = np.ones([image_resize_size[0], image_resize_size[1], 3]) * rgbFromStr(image_path)[None, None, :]
            img = Image.fromarray(img.astype(np.uint8))
            img = transforms.Resize(image_resize_size)(img)
            current_image = self.transforms(img)

        if not self.relative_intrinsics or image_crop_size is not None:
            absolute_intr = self._relative_intrinsic_to_absolute(*image_resize_size, relative_intr)

        if image_crop_size is not None:
            current_image, absolute_intr = self._crop_data(image=current_image,
                                                           intrinsic=absolute_intr,
                                                           crop_size=image_crop_size)

        if self.relative_intrinsics:
            if image_crop_size is not None:
                current_intr = self._absolute_intrinsic_to_relative(*image_crop_size, absolute_intr)
            else:
                current_intr = relative_intr
        else:
            current_intr = absolute_intr

        return current_image, current_extr, current_intr

    def _read_frame(self,
                    scene_id: str,
                    timestamps: np.ndarray,
                    image_resize_size,
                    image_crop_size,
                    ) -> Dict[str, torch.Tensor]:
        images, extrinsics, intrinsics = [], [], []

        for time in timestamps.reshape(-1):
            image_path: str = self.scenes_data[scene_id]['images_paths'][time][0]
            current_extr = torch.from_numpy(self.scenes_data[scene_id]['extrinsics'][time]).float()
            relative_intr = torch.from_numpy(self.scenes_data[scene_id]['intrinsics'][time]).float()
            current_image, current_extr, current_intr = self._read_image(
                image_path,
                current_extr,
                relative_intr,
                image_resize_size,
                image_crop_size
            )

            images.append(current_image)
            extrinsics.append(current_extr)
            intrinsics.append(current_intr)

        images = torch.stack(images).view(*timestamps.shape, *images[0].shape)
        extrinsics = torch.stack(extrinsics).view(*timestamps.shape, *extrinsics[0].shape)
        intrinsics = torch.stack(intrinsics).view(*timestamps.shape, *intrinsics[0].shape)

        return {
            'timestamp': torch.from_numpy(timestamps),
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

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


class SWORD(RealEstate10k):
    def __init__(self,
                 root,
                 image_size,
                 dataset_file='dataset.csv',
                 crop_size=None,
                 max_len_sequence=10,
                 set_reference_as_origin=False,
                 init_cameras_number_per_ref=1,
                 ref_cameras_number=1,
                 novel_cameras_number_per_ref=1,
                 cameras_sample_mode='extrapolation_only',
                 min_percentile_depth=15,
                 relative_intrinsics=False,
                 sample_first_nrow=-1,
                 scale_constant=1.25,
                 use_point_clouds=False,
                 number_of_points=8000,
                 one_scene=None,
                 precompute_resize_size=None,
                 virtual_average_ref_extrinsics=False,
                 virtual_average_novel_extrinsics=False,
                 transforms_params=None,
                 fake_image=False,
                 min_ammount_points=None,
                 scale_percentile='p_10',
                 image_scale=1.0,
                 ):
        """
        Improved RealEstate dataset.
        1) Now scenes indexed by dataset.csv, which allowed simply filtering scenes
           by several parameters such like : percentile depth/point error and e.t.c.
        2) Scene rescaling processed while training individually for each scene,
           it's make easy for changing rescale coeff on another percentile and
           preserves original cameras params and point clouds (which maybe important for another applications).
        3) Potentially you could have several datasets files in one root folder,
           it's helpful for merging different datasets.

        Args:
            root: root folder
            image_size (tuple): height, width for resize
            crop_size (int, tuple, None): height, width for crop after resize
            max_len_sequence (int): the maximum distance (in frames) might be between cameras
            set_reference_as_origin: whether to set reference camera pose as identity matrix,
                                     align other poses correspondingly
            init_cameras_number_per_ref: number of cameras, treated as initial, per one reference camera
            ref_cameras_number: number of cameras, treated as reference
            novel_cameras_number_per_ref: number of cameras, treated as novel, per one reference camera
            extrapolation_only: if True, novel cameras cannot be placed between any source or reference cameras.
                That is, the following consecutive triplets of cameras are prohibited:
                    initial - novel - initial,
                    initial - novel - reference.
                However, note, that a novel camera may be equal to the last source (i.e. initial or reference)
                camera in a row.
            fixed_ref_camera: int camera idx for fixed reference position
            one_scene: str with scene name '20200910_133225'
            precompute_resize_size: for using pre resized images put them in folder videos_<size> and pass size in precompute_resize_size arg.
            virtual_average_ref_extrinsics: generates average reference extrinsic based on source extrinsics
            virtual_average_novel_extrinsics: generates average novel extrinsic based on source extrinsics
        Returns:
            dict of camera types with dicts of torch.Tensor for each camera, stacked along dim=0
        """
        torch.utils.data.Dataset.__init__(self)
        self.transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)
        self.transforms_params = transforms_params
        self.image_size = image_size
        self.path = root
        self.scenes_data = {}
        self.dataset_df = pd.read_csv(os.path.join(self.path, dataset_file))
        self.use_point_clouds = use_point_clouds
        self.number_of_points_in_cloud = number_of_points

        self.set_reference_as_origin = set_reference_as_origin
        self.init_cameras_number_per_ref = init_cameras_number_per_ref
        self.ref_cameras_number = ref_cameras_number
        self.novel_cameras_number_per_ref = novel_cameras_number_per_ref
        self.total_cameras_needed = self.ref_cameras_number * (self.init_cameras_number_per_ref
                                                               + self.novel_cameras_number_per_ref
                                                               + 1)
        if virtual_average_ref_extrinsics:
            self.total_cameras_needed = self.total_cameras_needed - 1

        self.max_len_sequence = max(max_len_sequence, self.total_cameras_needed)
        # if cameras_sample_mode == 'interpolation_only':
        #     self.total_cameras_needed = self.max_len_sequence

        self.cameras_sample_mode = cameras_sample_mode
        self.relative_intrinsics = relative_intrinsics
        self.translation_fix = 1.0

        self.return_novel_trajectory = False
        self.virtual_average_ref_extrinsics = virtual_average_ref_extrinsics
        self.virtual_average_novel_extrinsics = virtual_average_novel_extrinsics

        self.precompute_resize_size = precompute_resize_size
        self.fake_image = fake_image
        self.image_scale = image_scale

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.dataset_df = self.dataset_df[self.dataset_df['p_10'] > min_percentile_depth]

        if one_scene:
            self.dataset_df = self.dataset_df[self.dataset_df['scene_name'] == one_scene]

        if sample_first_nrow > 0:
            self.dataset_df = self.dataset_df.head(sample_first_nrow)

        for scene_row in tqdm(self.dataset_df.iterrows()):
            scene_id = os.path.join(self.path, scene_row[1]['scene_name'])
            txt_path = os.path.join(self.path, scene_row[1]['views_path'])
            sparse_pointcloud_path = None
            if 'sparse_pointcloud_path' in scene_row[1]:
                sparse_pointcloud_path = os.path.join(self.path, scene_row[1]['sparse_pointcloud_path'])

            extensions = ['png', 'jpg', 'gif', 'JPG', 'PNG', 'GIF']  # Add image formats here
            images = []
            for extension in extensions:
                if self.precompute_resize_size:
                    temp = scene_row[1]['images_path']
                    temp = temp.replace("videos", "videos_" + str(precompute_resize_size))
                    images.extend(glob(os.path.join(self.path, temp, '*' + extension)))
                else:
                    images.extend(glob(os.path.join(self.path, scene_row[1]['images_path'], '*' + extension)))

            if not os.path.exists(txt_path):
                print(f'view path dont exist {txt_path}')
                continue

            if not (os.path.exists(f"{self.path}/point_clouds/{scene_row[1]['scene_name']}/points3D.bin")
                    or os.path.exists(f"{self.path}/colmap/{scene_row[1]['scene_name']}/points3D.bin") or
                    'sparse_pointcloud_path' in scene_row[1]):
                continue

            if min_ammount_points is not None:
                if scene_row[1]['num_points'] < min_ammount_points:
                    continue

            scale = scene_row[1][scale_percentile] / scale_constant
            if len(images) >= self.total_cameras_needed:
                with open(txt_path) as f:
                    text = f.read()
                rows = text.split('\n')[1:-1]
                intrinsics = {}
                extrinsics = {}
                images_dict = {}

                for timestamp, row in enumerate(rows):
                    row_name = row.split(' ')[0]
                    if not any([row_name == im.split('/')[-1].split('.')[0] for im in images]):
                        break
                    (focal_length_x,
                     focal_length_y,
                     principal_point_x,
                     principal_point_y) = np.array(row.split(' ')[1:5]).astype(np.float32)

                    intrinsics[timestamp] = np.array([[focal_length_x, 0, principal_point_x],
                                                      [0, focal_length_y, principal_point_y],
                                                      [0, 0, 1]])
                    extrs = np.array(row.split(' ')[7:19]).astype(np.float32).reshape((3, 4), order='C')
                    extrs[:, -1] = extrs[:, -1] / scale

                    extrinsics[timestamp] = extrs
                    images_dict[timestamp] = [s for s in images if
                                              row_name == s.split('/')[-1].split('.')[0]]

                self.scenes_data[scene_id] = {'images_paths': images_dict,
                                              'txt_path': txt_path,
                                              'scale': scale,
                                              'extrinsics': extrinsics,
                                              'intrinsics': intrinsics}

                if sparse_pointcloud_path:
                    self.scenes_data[scene_id]['sparse_pointcloud_path'] = sparse_pointcloud_path

    def _get_novels_cameras_near_ref(self, scene_id, timestamps, ref_ts, image_resize_size, image_crop_size):
        ref_idx = timestamps.index(ref_ts)
        novel_ts_arr = timestamps[ref_idx - self.max_len_sequence // 2:ref_idx + self.max_len_sequence // 2]
        return torch.stack(
            [self._read_frame(scene_id, np.array([[camera]]), image_resize_size, image_crop_size)['extrinsics'].squeeze(0).squeeze(0) for camera in
             novel_ts_arr])

    def __getitem__(self, idx):
        scene_id = list(self.scenes_data.keys())[idx]
        timestamps = list(self.scenes_data[scene_id]['extrinsics'].keys())

        init_cameras_number_per_ref, image_resize_size, image_crop_size, max_len_sequence = self.get_item_params()
        ref_ts, init_ts, novel_ts = self.sample_timestamps(
            init_cameras_number_per_ref=init_cameras_number_per_ref,
            txt_path=self.scenes_data[scene_id]['txt_path'],
            timestamps_poses=self.scenes_data[scene_id]['extrinsics'],
            max_len_sequence=max_len_sequence
        )

        ref_data = self._read_frame(scene_id, ref_ts, image_resize_size, image_crop_size)
        init_data = self._read_frame(scene_id, init_ts, image_resize_size, image_crop_size)
        if novel_ts is not None:
            novel_data = self._read_frame(scene_id, novel_ts, image_resize_size, image_crop_size)
        else:
            novel_data = copy.deepcopy(ref_data)

        init_data['extrinsics'], _ = self._transform_extrinsics(init_data['extrinsics'],
                                                                ref_data['extrinsics'])
        novel_data['extrinsics'], ref_data['extrinsics'] = self._transform_extrinsics(novel_data['extrinsics'],
                                                                                      ref_data['extrinsics'])

        time_id = 0
        output = {
            'time_id': time_id,
            'scene_time': f'{scene_id}//{time_id}',
            'scene_id': scene_id,
            'initial': init_data,
            'reference': ref_data,
            'novel': novel_data
        }

        if self.use_point_clouds:
            if 'sparse_pointcloud_path' in self.scenes_data[scene_id]:
                point_cloud_path = os.path.join(self.path, self.scenes_data[scene_id]['sparse_pointcloud_path'])
            else:
                point_cloud_path = os.path.join(self.path, f"point_clouds/{scene_id.split('/')[-1]}/points3D.bin")
            point_cloud_coords = load_colmap_binary_cloud(point_cloud_path) / float(self.scenes_data[scene_id]['scale'])
            vs = point_cloud_coords.shape[0]
            rand_perm = random.sample(range(point_cloud_coords.shape[0]), min(self.number_of_points_in_cloud, vs))
            output['point_cloud'] = [torch.from_numpy(point_cloud_coords[rand_perm]).float()]
        if self.return_novel_trajectory:
            output['novel_trajectory'] = self._get_novels_cameras_near_ref(scene_id, timestamps, ref_ts)

        return output
