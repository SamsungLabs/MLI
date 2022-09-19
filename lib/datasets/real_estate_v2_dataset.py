"""
RealEstate2 dataset should have the following structure:
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

__all__ = ['RealEstate2']

import logging
import math
import os
import random
from glob import glob
import copy

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from lib.datasets import RealEstate10k
from lib.modules.cameras.utils import average_extrinsics
from lib.utils.io import load_colmap_binary_cloud

DUMPED_FILE_NAME = 'dump.pkl'
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class RealEstate2(RealEstate10k):
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
                 fix_length_source_ref=None,
                 fixed_ref_camera=None,
                 fixed_view_cameras=None,
                 fix_length_source_source=None,
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
            fix_length_source_ref: fixed distance between source and reference cameras if extrapolation=False
            fixed_ref_camera: int camera idx for fixed reference position
            fixed_view_cameras: list of view [14, 25, 74] Note: the idx of camera from name
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

        if self.virtual_average_ref_extrinsics:
            ref_data['extrinsics'] = average_extrinsics(init_data['extrinsics'])

        if self.virtual_average_novel_extrinsics:
            novel_data['extrinsics'] = average_extrinsics(init_data['extrinsics'])

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
