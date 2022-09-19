"""
RealEstateStereo dataset should have the following structure:
| root/:
dataset.csv
|---- videos/
    |---- scene_id1
        |---- frame1.jpg
        |---- frame2.jpg
    |---- scene_id2
        |---- frame1.jpg
        |---- frame2.jpg
|---- colmap/
    |---- scene_id1/
    |---- scene_id2/
|---- views/ (or undistorted_views)
    |---- scene_id1.txt
    |---- scene_id2.txt
"""

__all__ = ['RealEstateStereo']

import logging
import os
import random
from collections import defaultdict
from glob import glob
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from lib.datasets import RealEstate2
from lib.modules.cameras.utils import average_extrinsics
from lib.utils.io import load_colmap_binary_cloud, load_colmap_fused_ply, load_colmap_bin_array

DUMPED_FILE_NAME = 'dump.pkl'
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class RealEstateStereo(RealEstate2):
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
                 relative_intrinsics=False,
                 filter_key=None,
                 min_percentile_depth=15,
                 measurement_value='p_10',
                 scale_constant=1.25,
                 sample_first_nrow=-1,
                 use_dense_clouds=False,
                 use_point_clouds=False,
                 number_of_points=8000,
                 use_undistorted_images=True,
                 default_stereo_side='left',
                 use_fixed_novel_ref_pos=False,
                 use_fixed_ref_source_pos=False,
                 use_depth_map=False,
                 one_scene=False,
                 dump_path=None,
                 virtual_average_ref_extrinsics=False,
                 virtual_average_novel_extrinsics=False,
                 transforms_params=None,
                 fake_image=False,
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

            relative_intrinsics: if True, dataset returns relative intrinsics
            filter_key: Enables scene filtering based on depth value percentile for the scene.
                        You need pick percentile p_5/p_10/p_20.. and set min_percentile_depth arg.
            min_percentile_depth: used with filter_key
            measurement_value: which value use for XYZ dimensions size. stereo_baseline_mean/p_10/p_20...
            scale_constant: used same as in stereomag
            use_fixed_ref_source_pos: use reference as left and source as a right cameras in the stereo dataset
            one_scene: fixes one scene from dataset for example: '20201103_102703'
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

        self.depth_transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

        self.transforms = transforms.Compose(self.transforms)
        self.transforms_params = transforms_params
        self.image_size = image_size
        self.path = root
        self.scenes_data = {}
        self.dataset_df = pd.read_csv(os.path.join(self.path, dataset_file))
        self.use_point_clouds = use_point_clouds

        self.set_reference_as_origin = set_reference_as_origin
        self.init_cameras_number_per_ref = init_cameras_number_per_ref
        self.ref_cameras_number = ref_cameras_number
        self.novel_cameras_number_per_ref = novel_cameras_number_per_ref
        self.total_cameras_needed = self.ref_cameras_number * (self.init_cameras_number_per_ref
                                                               + self.novel_cameras_number_per_ref
                                                               + 1)
        self.max_len_sequence = max(max_len_sequence, self.total_cameras_needed)
        if cameras_sample_mode == 'interpolation_only':
            self.total_cameras_needed = self.max_len_sequence

        self.cameras_sample_mode = cameras_sample_mode
        self.relative_intrinsics = relative_intrinsics
        self.translation_fix = 1.0

        self.use_dense_clouds = use_dense_clouds
        self.number_of_points_in_cloud = int(number_of_points)

        self.default_stereo_side = default_stereo_side

        self.use_fixed_ref_source_pos = use_fixed_ref_source_pos
        self.use_fixed_novel_ref_pos = use_fixed_novel_ref_pos
        self.return_novel_trajectory = False
        self.virtual_average_ref_extrinsics = virtual_average_ref_extrinsics
        self.virtual_average_novel_extrinsics = virtual_average_novel_extrinsics

        self.dump_path = dump_path
        self.fake_image = fake_image
        self.image_scale = image_scale

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        if filter_key is not None:
            self.dataset_df = self.dataset_df[self.dataset_df[filter_key] > min_percentile_depth]

        self.dataset_df = self.dataset_df[self.dataset_df['p_10'] > 1]

        if sample_first_nrow > 0:
            self.dataset_df = self.dataset_df.head(sample_first_nrow)

        if one_scene:
            self.dataset_df = self.dataset_df[self.dataset_df['scene_name'] == one_scene]

        cloud_dir_name = 'colmap_path_dense' if use_undistorted_images else 'colmap_path'
        distortion_image_path = 'undistorted_images_path' if use_undistorted_images else 'images_path'
        cloud_file_name = '/points3D.bin' if use_undistorted_images else '/sparse/points3D.bin'

        for scene_row in tqdm(self.dataset_df.iterrows()):
            scene_id = os.path.join(self.path, scene_row[1]['scene_name'])
            txt_path = os.path.join(self.path, scene_row[1]['views_sparse_path'])
            images = glob(os.path.join(self.path, scene_row[1][distortion_image_path], f'*.jpg'))
            depth_images = glob(os.path.join(self.path, scene_row[1]['depth_maps'], f'*geometric.bin'))

            if not os.path.exists(txt_path):
                continue

            if self.use_dense_clouds:
                point_cloud_path = os.path.join(self.path, f"colmap/{scene_id.split('/')[-1]}/dense/fused.ply")
            else:
                point_cloud_path = os.path.join(self.path, f"{scene_row[1][f'{cloud_dir_name}']}{cloud_file_name}")

            scale = scene_row[1][measurement_value] / scale_constant

            if not os.path.exists(point_cloud_path) and use_point_clouds:
                continue

            if len(images) >= self.total_cameras_needed * 2:
                with open(txt_path) as f:
                    text = f.read()
                rows = text.split('\n')[1:-1]
                intrinsics = defaultdict(dict)
                extrinsics = defaultdict(dict)
                images_dict = defaultdict(dict)
                depth_dict = defaultdict(dict)

                if len(rows) < self.total_cameras_needed * 2:
                    continue

                for i, row in enumerate(rows):
                    row = ' '.join(row.split())  # remove extra spaces (colmap bug?)
                    timestamp_str, direction = row.split(' ')[0].split('_')
                    timestamp = int(timestamp_str)
                    if not any([timestamp == int(im.split('/')[-1].split('.')[0].split('_')[0]) for im in images]):
                        break
                    (focal_length_x,
                     focal_length_y,
                     principal_point_x,
                     principal_point_y) = np.array(row.split(' ')[1:5]).astype(np.float32)

                    intrinsics[timestamp][direction] = np.array([[focal_length_x, 0, principal_point_x],
                                                                 [0, focal_length_y, principal_point_y],
                                                                 [0, 0, 1]])
                    extrs = np.array(row.split(' ')[7:19]).astype(np.float32).reshape((3, 4), order='C')
                    extrs[:, -1] = extrs[:, -1] / scale

                    extrinsics[timestamp][direction] = extrs

                    images_dict[timestamp][direction] = [s for s in images
                                                         if
                                                         timestamp == int(s.split('/')[-1].split('.')[0].split('_')[0])
                                                         and direction == s.split('/')[-1].split('.')[0].split('_')[1]]

                    depth_dict[timestamp][direction] = [s for s in depth_images
                                                        if
                                                        timestamp == int(s.split('/')[-1].split('.')[0].split('_')[0])
                                                        and direction == s.split('/')[-1].split('.')[0].split('_')[1]]

                bad_keys = []
                for key, value in images_dict.items():
                    left_exist = 'left' in value
                    right_exist = 'right' in value
                    if not left_exist:
                        logger.warning(f'Scene {scene_id} wrong timestamp {key}, left do not exist')
                    if not right_exist:
                        logger.warning(f'Scene {scene_id} wrong timestamp {key}, right do not exist')
                    if left_exist and right_exist:
                        continue
                    bad_keys.append(key)

                for key in bad_keys:
                    del (images_dict[key])
                    del (depth_dict[key])
                    del (extrinsics[key])
                    del (intrinsics[key])

                self.scenes_data[scene_id] = {'images_paths': images_dict,
                                              'depth_paths': depth_dict,
                                              'txt_path': txt_path,
                                              'point_cloud_path': point_cloud_path,
                                              'scale': scale,
                                              'extrinsics': extrinsics,
                                              'intrinsics': intrinsics}

    def __getitem__(self, idx):
        scene_id = list(self.scenes_data.keys())[idx]
        timestamps = list(self.scenes_data[scene_id]['extrinsics'].keys())

        init_cameras_number_per_ref, image_resize_size, image_crop_size, max_len_sequence = self.get_item_params()

        ref_ts, init_ts, novel_ts = self.sample_timestamps(
            init_cameras_number_per_ref=init_cameras_number_per_ref,
            txt_path=self.scenes_data[scene_id]['txt_path'],
            timestamps_poses={key: value['left'] for key, value in self.scenes_data[scene_id]['extrinsics'].items()},
            max_len_sequence=max_len_sequence,
        )

        ref_data = self._read_frame(scene_id, ref_ts, image_resize_size, image_crop_size)

        if self.use_fixed_ref_source_pos:
            init_data = self._read_frame(scene_id, ref_ts, image_resize_size, image_crop_size, stereo_index='right')
        else:
            init_data = self._read_frame(scene_id, init_ts, image_resize_size, image_crop_size)
        if self.use_fixed_novel_ref_pos and not self.use_fixed_ref_source_pos:
            novel_data = self._read_frame(scene_id, ref_ts, image_resize_size, image_crop_size, stereo_index='right')
        else:
            novel_data = self._read_frame(scene_id, novel_ts, image_resize_size, image_crop_size)

        init_data['extrinsics'], _ = self._transform_extrinsics(init_data['extrinsics'],
                                                                ref_data['extrinsics'])
        novel_data['extrinsics'], ref_data['extrinsics'] = self._transform_extrinsics(novel_data['extrinsics'],
                                                                                      ref_data['extrinsics'])
        if self.virtual_average_ref_extrinsics:
            ref_data['extrinsics'] = average_extrinsics(init_data['extrinsics']).unsqueeze(-3)

        if self.virtual_average_novel_extrinsics:
            novel_data['extrinsics'] = average_extrinsics(init_data['extrinsics']).unsqueeze(-3)

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
            point_cloud_path = self.scenes_data[scene_id]['point_cloud_path']
            if self.use_dense_clouds:
                # redefine path for sure
                point_cloud_path = os.path.join(self.path, f"colmap/{scene_id.split('/')[-1]}/dense/fused.ply")
                point_cloud_coords = load_colmap_fused_ply(point_cloud_path)
            else:
                point_cloud_coords = load_colmap_binary_cloud(point_cloud_path)
            vs = point_cloud_coords.shape[0]
            rand_perm = random.sample(range(point_cloud_coords.shape[0]), min(self.number_of_points_in_cloud, vs))
            output['point_cloud'] = [
                torch.from_numpy(point_cloud_coords[rand_perm] / float(self.scenes_data[scene_id]['scale'])).float()]
        if self.return_novel_trajectory:
            output['novel_trajectory'] = self._get_novels_cameras_near_ref(scene_id, timestamps, ref_ts,
                                                                           image_resize_size, image_crop_size)

        return output

    def _read_frame(self,
                    scene_id: str,
                    timestamps: np.ndarray,
                    image_resize_size,
                    image_crop_size,
                    stereo_index: str = 'left',
                    ) -> Dict[str, torch.Tensor]:
        images, extrinsics, intrinsics = [], [], []

        for time in timestamps.reshape(-1):
            image_path: str = self.scenes_data[scene_id]['images_paths'][time][stereo_index][0]
            current_extr = torch.from_numpy(self.scenes_data[scene_id]['extrinsics'][time][stereo_index]).float()
            relative_intr = torch.from_numpy(self.scenes_data[scene_id]['intrinsics'][time][stereo_index]).float()
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

        out = {
            'timestamp': torch.from_numpy(timestamps),
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

        return out
