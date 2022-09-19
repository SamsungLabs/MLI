import argparse
import functools
import shutil

import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

from lib.modules.cameras import CameraMultiple, trajectory_generators, camera_processors
from lib.utils.coord_conversion import get_cameras_world_positions
from lib.utils.data import get_dataloader_from_params
from lib.utils.io import (get_config)

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


def sort_nearest_initial_cams_ids(novel_extrinsic, initial_extrinsics, initial_ids):
    """
    novel_extrinsic: torch.tensor 1 x 3 x 4
    initial_extrinsics: torch.tensor N x 3 x 4
    initial_ids: list N
    """
    initial_global_positions = get_cameras_world_positions(initial_extrinsics).squeeze(1)
    novel_global_positions = get_cameras_world_positions(novel_extrinsic).squeeze(1)

    distances = ((novel_global_positions - initial_global_positions)**2).sum(dim=1)
    index = torch.argsort(distances, descending=False).numpy()
    return [initial_ids[idx] for idx in index]


def main():
    # DON'T SUPPORT VIRTUAL CAMERAS FROM DATASETS

    logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the config file.')
    parser.add_argument('--output-path', type=str, help='Path to outputs directory.')
    opts = parser.parse_args()

    config = get_config(opts.config)

    resolution = config['resolution']
    config_name = os.path.basename(opts.config).split('.')[0]
    new_config_name = config_name + '_' + 'x'.join([str(x) for x in resolution])
    output_path = os.path.join(opts.output_path, new_config_name)
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(opts.config, os.path.join(output_path, f'{new_config_name}.yaml'))

    config['dataloaders']['val']['batch_size'] = 1
    dataloader = get_dataloader_from_params(config, 'val')
    num_samples_per_epoch = len(dataloader.dataset)
    num_epoch = config.get('num_epoch', 1)

    cam_types = ['initial']
    novel_mode = config.get('novel_mode', 'dataset_gt')
    n_nearest_initial_cams_for_novel = config.get('n_nearest_initial_cams_for_novel', None)

    if novel_mode == 'dataset_gt':
        cam_types.append('novel')
    elif novel_mode == 'virtual_trajectory':
        virtual_resolution = config['virtual_trajectory']['resolution']
        virtual_trajectory_gen_input_cams = config['virtual_trajectory']['input_cams']
        pose_processor = getattr(camera_processors, config['virtual_trajectory']['pose_processor']['func'])
        pose_processor = functools.partial(pose_processor, **config['virtual_trajectory']['pose_processor']['params'])
        trajectory_gen = getattr(trajectory_generators, config['virtual_trajectory']['trajectory_gen']['func'])
        trajectory_gen = functools.partial(trajectory_gen, **config['virtual_trajectory']['trajectory_gen']['params'])
    else:
        logger.error('There are no novel_type: ', config['novel_mode'])

    for epoch in range(num_epoch):
        for idx, data in enumerate(tqdm(dataloader, desc='scenes', leave=True)):
            scene_path = os.path.join(output_path, str(epoch * num_samples_per_epoch + idx))
            os.makedirs(scene_path, exist_ok=True)

            frames_meta = {}
            initial_poses = []
            initial_ids = []

            for cam_type in cam_types:
                cam_type_frames_path = os.path.join(scene_path, cam_type)
                os.makedirs(cam_type_frames_path, exist_ok=True)
                num_cams = data[cam_type]['extrinsics'].shape[2]
                for i in range(num_cams):
                    frame_index = cam_type + '_' + f'{i:06d}'
                    frame_path = os.path.join(cam_type_frames_path, frame_index + '.jpg')
                    image_array = ((data[cam_type]['image'][0][0][i].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255)\
                        .astype(np.uint8)
                    image_size = image_array.shape[-3:-1]
                    image = Image.fromarray(image_array)
                    image.save(frame_path)

                    nearest_initial_ids = None
                    if n_nearest_initial_cams_for_novel is not None:
                        if cam_type == 'initial':
                            initial_poses.append(data[cam_type]['extrinsics'][0, 0, i])
                            initial_ids.append(frame_index)
                        if cam_type == 'novel':
                            nearest_initial_ids = sort_nearest_initial_cams_ids(data[cam_type]['extrinsics'][[0], 0, i],
                                                                                torch.stack(initial_poses, dim=0),
                                                                                initial_ids)
                            nearest_initial_ids = nearest_initial_ids[:n_nearest_initial_cams_for_novel]

                    frames_meta[frame_index] = [frame_index,
                                                cam_type,
                                                os.path.join(cam_type, frame_index + '.jpg'),
                                                list(image_size),
                                                list(data[cam_type]['extrinsics'][0, 0, i].flatten().numpy()),
                                                list(data[cam_type]['intrinsics'][0, 0, i].flatten().numpy()),
                                                # nearest_initial_ids,
                                                ]

            if novel_mode == 'virtual_trajectory':
                input_cams = CameraMultiple(extrinsics=data[virtual_trajectory_gen_input_cams]['extrinsics'][0, 0],
                                            intrinsics=data[virtual_trajectory_gen_input_cams]['intrinsics'][0, 0],
                                            )
                poses_for_trajectory_gen = pose_processor(input_cams)
                trajectory_poses = trajectory_gen(poses_for_trajectory_gen.get_extrinsics())
                processed_intrinsic = poses_for_trajectory_gen.get_intrinsics()[[-1], :, :]
                for i, trajectory_pose in enumerate(trajectory_poses):
                    if n_nearest_initial_cams_for_novel is not None:
                        nearest_initial_ids = sort_nearest_initial_cams_ids(trajectory_pose,
                                                                            torch.stack(initial_poses, dim=0),
                                                                            initial_ids)

                    frame_index = 'novel' + '_' + f'{i:06d}'
                    frames_meta[frame_index] = [frame_index,
                                                'novel',
                                                'virtual',
                                                list(virtual_resolution),
                                                list(trajectory_pose.flatten().numpy()),
                                                list(processed_intrinsic.flatten().numpy()),
                                                # nearest_initial_ids
                                                ]

            frames_meta_df = pd.DataFrame.from_dict(frames_meta, orient='index')
            # frames_meta_df.columns = ['frame_id', 'type', 'frame_path', 'frame_size', 'extrinsic_re', 'intrinsics_re', 'initial_cams']
            frames_meta_df.columns = ['frame_id', 'type', 'frame_path', 'frame_size', 'extrinsic_re', 'intrinsics_re']
            scene_path_csv = os.path.join(scene_path, 'scene.csv')
            frames_meta_df.to_csv(scene_path_csv, index=False)


if __name__ == '__main__':
    main()
