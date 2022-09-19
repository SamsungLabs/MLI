"""
The script has 2 usage scenarios.

Case 1. The only config with the specified iteration.

PYTHONPATH=$(pwd) python3 bin/render_mpi.py \
    --config configs/stylegan.yaml \
    --checkpoints-path /group-volume/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/outputs/stylegan/checkpoints \
    --iteration 100000 \
    --output-path outputs/ \
    --offset 10 \
    --pixel-units
    [--ema]

Case 2. A number of configs from the common parent directory with the latest weights loaded.

PYTHONPATH=$(pwd) python3 bin/render_mpi.py \
    --user-dir /group-volume/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/outputs \
    --model-names \
        stylegan \
        bedrooms_degree0 \
        dcgan \
    --output-path outputs/ \
    --offset 10 \
    --pixel-units \
    [--ema]
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import logging
import os
import re
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image

from lib import trainers
from lib.utils.base import get_latest_model_name
from lib.utils.coord_conversion import coords_cam_to_world, get_cameras_world_positions
from lib.utils.data import get_dataloader_from_params
from lib.utils.geometry import rotation_matrix_between_two_vectors, quanterion_between_two_vectors, \
    rotation_matrix_to_quaternion, quanterion_mult, quaternion_to_rotation_matrix
from lib.utils.io import (get_config,
                          save_mpi_binary,
                          save_pointcloud_to_ply,
                          modify_mpi_config_to_produce_mesh,
                          )
from lib.utils.visualise import VideoWriter
from lib.modules.cameras.utils import interpolate_extrinsics


def generate_cross_poses(ref_pose: torch.Tensor,
                         offset: Union[int, float] = 5,
                         direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                         direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                         focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                         scale_y: Union[int, float] = 0.5,
                         num_frames: int = 30,
                         ) -> List[torch.Tensor]:
    """
    Function generates poses for smooth video rendering
    Args:
        ref_pose: pose of the mpi reference camera
        offset: maximum offset the camera can shift along x axis
        direction_x: direction of the horizontal shift, camera coordinates
        direction_y: direction of the vertical shift, camera coordinates
        focal: focal values for rescaling the offsets
        scale_y: ratio between maximum vertical and horizontal shifts. Smaller vertical shift looks more plausible
        num_frames: number of frames to generate

    Returns:

    """
    offsets = np.linspace(-offset, offset, num_frames // 2)
    render_poses = []
    for idx, o in enumerate(offsets):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets[::-1]):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal + \
                              np.sin(np.pi * idx / (0.5 * num_frames)) * direction_y * offset * scale_y / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets[::-1]):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal - \
                              np.sin(np.pi * idx / (0.5 * num_frames)) * direction_y * offset * scale_y / focal
        render_poses.append(new_pose)

    return render_poses


def generate_spiral_poses(ref_pose: torch.Tensor,
                          offset: Union[int, float] = 5,
                          depth_limits: Tuple[float, float] = (-0.1, 0.2),
                          direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                          direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                          direction_z: torch.Tensor = torch.tensor([0, 0, 1], dtype=torch.float),
                          focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                          scale_y: Union[int, float] = 1,
                          num_frames: int = 30,
                          ) -> List[torch.Tensor]:
    timestamps = np.linspace(-2, 2, num_frames)
    min_depth, max_depth = depth_limits
    render_poses = []
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * offset / focal * np.cos(np.pi * t)
        new_pose[0, :3, 3] += direction_y * offset / focal * np.sin(np.pi * t) * scale_y
        new_pose[0, :3, 3] -= direction_z * ((0.5 * np.cos(np.pi * t / 2) + 0.5) * (max_depth - min_depth) + min_depth)
        render_poses.append(new_pose)
    return render_poses


def generate_spiral_poses_centered(ref_pose: torch.Tensor,
                          offset: Union[int, float] = 5,
                          depth_limits: Tuple[float, float] = (-0.5, 0.5),
                          direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                          direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                          direction_z: torch.Tensor = torch.tensor([0, 0, 1], dtype=torch.float),
                          focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                          scale_y: Union[int, float] = 1,
                          num_frames: int = 30,
                          ) -> List[torch.Tensor]:
    focus_point = torch.tensor([[0.0, 0.0, 1.0]]) * 4
    timestamps = np.linspace(-2, 2, num_frames)
    min_depth, max_depth = depth_limits
    render_poses = []
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * offset / focal * np.cos(np.pi * t)
        new_pose[0, :3, 3] += direction_y * offset / focal * np.sin(np.pi * t) * scale_y
        new_pose[0, :3, 3] -= direction_z * ((0.5 * np.cos(np.pi * t / 2) + 0.5) * (max_depth - min_depth) + min_depth)

        dir_vector = focus_point - new_pose[0, :3, 3]
        quanterion = quanterion_between_two_vectors(torch.tensor([[0.0, 0.0, 1.0]]), dir_vector)
        quanterion_ref = rotation_matrix_to_quaternion(ref_pose[0, :3, :3])
        new_pose[0, :3, :3] = quaternion_to_rotation_matrix(quanterion_mult(quanterion_ref, quanterion))

        render_poses.append(new_pose)
    return render_poses


def generate_poses_along_direction(ref_pose: torch.Tensor,
                                   limits: Tuple[float, float],
                                   direction: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                                   focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                                   num_frames: int = 30,
                                   ) -> List[torch.Tensor]:
    timestamps = np.linspace(-1, 1, num_frames)
    render_poses = []
    min_val, max_val = limits
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction / focal * ((0.5 * np.cos(np.pi * t) + 0.5) * (max_val - min_val) + min_val)
        render_poses.append(new_pose)
    return render_poses


def create_trainer_load_weights(config: dict,
                                checkpoints_dir: str,
                                iteration: Optional[int] = None,
                                use_ema: bool = False,
                                device: Union[str, torch.device] = 'cuda',
                                ):
    loaded_iteration = iteration
    trainer = getattr(trainers, config['trainer'])(config, eval_mode=True, device=device)

    state_dicts = []
    for model_part in config['models']:
        if model_part == 'dis':
            continue
        if iteration is None:
            model_part_weight_path = get_latest_model_name(checkpoints_dir, model_part)
            loaded_iteration = int(model_part_weight_path.split('_')[-1][:-3])
        else:
            model_part_weight_path = os.path.join(checkpoints_dir, f'model.{model_part}_{iteration:08d}.pt')
        current_state_dict = torch.load(model_part_weight_path, map_location=device)
        current_state_dict = {f'{model_part}.' + re.sub(r'^module\.', '', k): v for k, v in current_state_dict.items()}
        state_dicts.append(current_state_dict)

    if use_ema:
        for model_name in getattr(trainer, 'ema_models_list', []):
            current_state_dict = torch.load(os.path.join(checkpoints_dir,
                                                         f'model_ema.{model_name}.pt'),
                                            map_location=device)
            current_state_dict = {f'{model_name}.' + re.sub(r'^module\.', '', k): v
                                  for k, v in current_state_dict.items()}
            state_dicts.append(current_state_dict)

    full_state_dict = {}
    for current_state_dict in state_dicts:
        full_state_dict.update(current_state_dict)

    # TODO implement inference load state method in lib.trainers.trainer_base.TrainerBase
    trainer.load_state_dict(full_state_dict, strict=False)
    trainer.ema_inference = use_ema

    return trainer, loaded_iteration


def generate_interpolate_poses(ref_pose: torch.Tensor,
                               src_pose: torch.Tensor,
                               num_frames: int = 30,
                               offset: float = 0.,
                               ):
    offsets = torch.linspace(-offset, 1+offset, num_frames, device=ref_pose.device)
    return interpolate_extrinsics(start=ref_pose, end=src_pose, timestamp=offsets)


def generate_poses_seq_from_seq(sequence: List[torch.Tensor], frames_per_step):
    len_seq = len(sequence)
    res = []
    for i in range(1, len_seq):
        res.extend(generate_interpolate_poses(sequence[i - 1], sequence[i], frames_per_step))
    return res


def inference_video_clips(trainer,
                          dataloader,
                          clip_name_template,
                          use_pixel_units=False,
                          offset=5,
                          num_frames=120,
                          depth_limits=(0, 0.15),
                          save_mpi_bin=True,
                          save_mpi_png=True,
                          save_video=False,
                          reuse_geometry=False,
                          trajectory_type='cross',
                          save_geometry=True,
                          save_initial_frames=True
                          ):

    for idx, data in enumerate(tqdm(dataloader, desc='scenes', leave=True)):
        output_filename = clip_name_template.format(idx=idx,
                                                    ema='ema' if getattr(trainer, 'ema_inference', False) else '')
        video_writer = VideoWriter(output_filename)
        ref_pose = data['reference']['extrinsics']
        images = []
        trainer.prepared_noise = None

        if trajectory_type == 'novel':
            all_poses = data['novel_trajectory'].squeeze(0).unsqueeze(1)
            interpolate_frames = num_frames // len(all_poses)
            all_poses = generate_poses_seq_from_seq(all_poses, interpolate_frames)
        elif trajectory_type == 'interpolation':
            src_pose = data['initial']['extrinsics']
            # all_poses = generate_interpolate_poses(ref_pose[0, 0], src_pose[0, 0], num_frames, offset)
            # print('all_poses', all_poses.shape)
            all_poses = generate_interpolate_poses(src_pose[0, 0, [0]], src_pose[0, 0, [1]], num_frames, offset)
        else:
            direction_x = torch.tensor([1.0, 0, 0], dtype=torch.float)
            direction_y = torch.tensor([0, 1.0, 0], dtype=torch.float)
            direction_z = torch.tensor([0, 0, 1.0], dtype=torch.float)
            if use_pixel_units:
                focal_xy = torch.diagonal(data['reference']['intrinsics'][0, 0, 0, :-1, :-1],
                                          dim1=0, dim2=1)
                focal_xy = torch.cat([focal_xy, torch.tensor([1.])])
            else:
                focal_xy = torch.tensor([1., 1., 1.])

            if trajectory_type.startswith('along-'):
                axis = trajectory_type.split('-')[-1]
                if axis == 'x':
                    direction = direction_x
                    limits = (-offset, offset)
                elif axis == 'y':
                    direction = direction_y
                    limits = (-offset, offset)
                elif axis == 'z':
                    direction = - direction_z
                    limits = depth_limits
                else:
                    raise ValueError
                all_poses = generate_poses_along_direction(ref_pose[0, 0], limits=limits,
                                                           direction=direction, focal=focal_xy,
                                                           num_frames=num_frames)
            elif trajectory_type == 'spiral':
                all_poses = generate_spiral_poses(
                    ref_pose[0, 0], offset=offset, depth_limits=depth_limits,
                    direction_x=direction_x, direction_y=direction_y, direction_z=direction_z,
                    focal=focal_xy, scale_y=1, num_frames=num_frames,
                )
            elif trajectory_type == 'cross':
                all_poses = generate_cross_poses(ref_pose[0, 0], offset=offset,
                                                 direction_x=direction_x, direction_y=direction_y,
                                                 focal=focal_xy, num_frames=num_frames)
            else:
                raise ValueError(f'Unknown trajectory_type={trajectory_type}')

            all_poses.insert(0, ref_pose[0, 0])

        # if save_reference_frames:

        for i, pose in tqdm(enumerate(all_poses),
                            desc='camera poses', leave=False, total=len(all_poses)):
            data['novel']['extrinsics'] = pose[None, None]

            if not reuse_geometry:
                data['reference']['extrinsics'] = pose[None, None]
                data['proxy_geometry'] = None
            else:
                data['reference']['extrinsics'] = ref_pose

            result = trainer.inference(data)
            # if i > 0:
            images.append(result['images'].cpu())

            if reuse_geometry:
                data['proxy_geometry'] = result.get('proxy_geometry')

            if i == 0 and save_geometry:
                data['proxy_geometry'] = result.get('proxy_geometry')
                rgb = (data['proxy_geometry']['verts_rgb'][0, :, 1:4].cpu().float().numpy() + 1) / 2 * 255
                points = data['proxy_geometry']['verts'][0].cpu().float().numpy()
                output_filename = '.'.join(output_filename.split('.')[:-1] + ['ply'])
                save_pointcloud_to_ply(output_filename, points, rgb)

            if (save_mpi_bin or save_mpi_png or save_video) and not i:
                mpi = result.get('mpi')

                if mpi is not None:
                    mpi = mpi.squeeze(0).data.cpu().numpy()
                    path_to_mpi_layers = os.path.join('/'.join(output_filename.split('/')[:-1]), 'mpi')
                    os.makedirs(path_to_mpi_layers, exist_ok=True)
                    save_mpi_binary(mpi,
                                    focal=data['reference']['intrinsics'][0, 0, 0, 1, 1].item(),
                                    basedir=path_to_mpi_layers,
                                    tag=os.path.splitext(output_filename)[0].split('/')[-1],
                                    save_images=save_mpi_png,
                                    save_video=save_video,
                                    )

            if save_initial_frames and not i:
                path_to_initial_frames = os.path.join('/'.join(output_filename.split('/')[:-1]),
                                                        'initial_frames',
                                                        os.path.splitext(output_filename)[0].split('/')[-1])
                os.makedirs(path_to_initial_frames, exist_ok=True)

                num_init_frames = data['initial']['image'].shape[2]
                for j in range(num_init_frames):
                    print('save init_frame: ', j)
                    image = Image.fromarray(
                        ((data['initial']['image'][0][0][j].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255).astype(np.uint8))
                    image.save(os.path.join(path_to_initial_frames, str(j) + '.jpg'))

        images = torch.cat(images, dim=0)
        video_writer.process_batch(((images + 1.) / 2.) * 255)
        video_writer.finalize()

def update_models_settings_for_mpi2mesh(models_settings: dict,
                                        old_model_name: str,
                                        remove_old: bool = False,
                                        faces_per_pixel: int = 64,
                                        resolution: int = 256,
                                        ) -> None:
    status, new_config = modify_mpi_config_to_produce_mesh(models_settings[old_model_name]['config'],
                                                           faces_per_pixel=faces_per_pixel,
                                                           resolution=resolution,
                                                           )
    if not status:
        return
    new_model_name = old_model_name + '_mpi2mesh'
    models_settings[new_model_name] = deepcopy(models_settings[old_model_name])
    models_settings[new_model_name]['config'] = new_config
    if remove_old:
        del models_settings[old_model_name]


TRAJECTORY_TYPES = [
    'novel',  # Use for rendering real trajectory for reference
    'along-axis-x',  # Move along axis X'
    'along-axis-y',  # Move along axis Y'
    'along-axis-z',  # Move along axis Z'
    'interpolation',  # Use to interpolate between reference and source poses
    'spiral',  # Use a spiral trajectory for rendering
    'cross',
]

def main():
    # TODO add mode in which render use folder with images

    logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the config file.')
    parser.add_argument('--checkpoints-path', type=str, default=None,
                        help='Path to checkpoints directory.')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Iteration number')
    parser.add_argument('--num-frames', type=int, default=100,
                        help='Number of frames')
    parser.add_argument('--output-path', type=str, help='Path to outputs directory.')
    parser.add_argument('--offset', type=float, default=0.5,
                        help='Value of the offset.')
    parser.add_argument('--pixel-units', action='store_true',
                        help='Treat offset value in pixel units.')
    parser.add_argument('--trajectory-type', type=str, default=None,
                        help='Novel cameras trajectory type')
    parser.add_argument('--use-mean-reference', action='store_true',
                        help='Use the mean of source cameras as the start of trajectory')
    parser.add_argument('--depth-limits', nargs=2, default=(-0.5, 0.5), type=float,
                        help='limits of motion along camera z axis in spiral trajectory')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA generator for inference.')
    parser.add_argument('--model-names', type=str, default=[], nargs='+',
                        help='Names of the models, separated with spaces. E.g., bedrooms_multiscale_0degree')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device.')
    parser.add_argument('--user-dir', type=str, default=None,
                        help="Path to the user's directory with models. E.g., "
                             "/group-volume/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/outputs")
    parser.add_argument('--save-mpi', action='store_true',
                        help='Save mpi binary file.')
    parser.add_argument('--reuse-geometry', type=bool, default=True,
                        help='Infer proxy geometry only once per scene')
    parser.add_argument('--save-layers', action='store_true',
                        help='Save mpi binary file and png images of layers.')
    parser.add_argument('--save-layers-video', action='store_true',
                        help='Save layers video')
    parser.add_argument('--save-geometry', action='store_true',
                        help='Save proxy geometry like pointclouds.')
    parser.add_argument('--mpi2mesh', action='store_true',
                        help='Postprocess MPI planes to create a layered mesh')
    parser.add_argument('--resolution', type=int, default=256,
                        help='inference resolution for mpi2mesh modification')
    opts = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True

    # Load experiment setting
    models_settings = defaultdict(dict)
    if not opts.model_names:
        model_name = os.path.splitext(os.path.basename(opts.config))[0]
        setting = models_settings[model_name]
        setting['config'] = get_config(opts.config)
        if opts.checkpoints_path is not None:
            checkpoints_path = opts.checkpoints_path
        else:
            checkpoints_path = os.path.join(os.path.dirname(opts.config), 'checkpoints')
        setting['checkpoints_dir'] = checkpoints_path
        if opts.mpi2mesh:
            update_models_settings_for_mpi2mesh(models_settings, model_name,
                                                remove_old=True, resolution=opts.resolution)
    else:
        for model_name in opts.model_names:
            setting = models_settings[model_name]
            setting['config'] = get_config(os.path.join(opts.user_dir, model_name, f'{model_name}.yaml'))
            setting['checkpoints_dir'] = os.path.join(opts.user_dir, model_name, 'checkpoints')
            if opts.mpi2mesh:
                update_models_settings_for_mpi2mesh(models_settings, model_name,
                                                    remove_old=False, resolution=opts.resolution)

    failed_models = {}
    for model_name, setting in tqdm(models_settings.items(), desc='models'):
        try:
            os.makedirs(os.path.join(opts.output_path, model_name), exist_ok=True)

            config = setting['config']
            #  to use the fixed distance between ref and source, we specify sampling branch
            #  it doesn't affect on novel view due to the manual design of novel-cameras parameters
            dataloader = get_dataloader_from_params(config, 'render')

            trajectory_type = opts.trajectory_type
            assert trajectory_type in TRAJECTORY_TYPES, f'Unknown trajectory type {trajectory_type}'
            if trajectory_type == 'novel':
                dataloader.dataset.return_novel_trajectory = True

            trainer, loaded_iteration = create_trainer_load_weights(config=config,
                                                                    checkpoints_dir=setting['checkpoints_dir'],
                                                                    iteration=opts.iteration,
                                                                    use_ema=opts.ema,
                                                                    device=opts.device,
                                                                    )

            if opts.output_path is None:
                output_path = os.path.join(setting['checkpoint_dir'], 'videos')
                os.makedirs(output_path, exist_ok=True)
            else:
                os.makedirs(os.path.join(opts.output_path, model_name), exist_ok=True)
                output_path = os.path.join(opts.output_path, model_name)
            output_filename_template = os.path.join(output_path,
                                                    f'render_{loaded_iteration}_' + '{ema}_{idx}.mp4')
            inference_video_clips(trainer=trainer,
                                  dataloader=dataloader,
                                  clip_name_template=output_filename_template,
                                  use_pixel_units=opts.pixel_units,
                                  offset=opts.offset,
                                  num_frames=opts.num_frames,
                                  depth_limits=opts.depth_limits,
                                  save_mpi_bin=opts.save_mpi,
                                  save_mpi_png=(opts.save_layers or opts.save_layers_video),
                                  save_video=opts.save_layers_video,
                                  trajectory_type=trajectory_type,
                                  reuse_geometry=opts.reuse_geometry,
                                  save_geometry=opts.save_geometry,
                                  )
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception(e)
            failed_models[model_name] = e

    if failed_models:
        logger.warning('Could not produce videos for the following models:\n'
                       + '\n'.join(f'{name}: {reason}' for name, reason in failed_models.items())
                       )


if __name__ == '__main__':
    main()
