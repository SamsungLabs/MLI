import argparse
import gc
from timeit import default_timer as timer
import json

import logging
import numpy as np
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import torch
from PIL import Image

from lib.datasets import ValidationDataset
from lib.trainers.utils import create_trainer_load_weights_from_config
from lib.utils.io import get_config
from lib.utils.visualise import ImageWriter, image_folder_to_video, grayscale_to_cmap

from lib.utils.base import min_max_scale

logger = logging.getLogger('root')


def get_cam_data_with_index(cam_data, index):
    new_data = []
    for idx in index:
        new_data.append({'id': idx,
                         'extrinsics': cam_data['extrinsics'][:, :, [idx]],
                         'intrinsics': cam_data['intrinsics'][:, :, [idx]],
                         'image': cam_data['image'][:, :, [idx]]
                         }
                        )
    return new_data


def get_cam_data_with_another_index(cam_data, index):
    return {'id': sum(index),
            'extrinsics': torch.index_select(cam_data['extrinsics'], 2, index),
            'intrinsics': torch.index_select(cam_data['intrinsics'], 2, index),
            'image': torch.index_select(cam_data['image'], 2, index),
            }


def split_on_render_groups(data, mode='one_for_all'):
    render_groups = []

    _, _, num_poses, _, _ = data['novel']['extrinsics'].shape

    initial_cams_for_novels = data.get('initial_cams_for_novels', None)
    if initial_cams_for_novels is not None:
        if len(initial_cams_for_novels[0]) == 0:
            initial_cams_for_novels = None

    novels_data = get_cam_data_with_index(data['novel'], list(range(num_poses)))

    if mode == 'one_for_all' and initial_cams_for_novels is None:
        group_data = {'initial': data['initial'], 'novels': novels_data}
        if 'reference' in data:
            group_data['reference'] = data['reference']
        render_groups.append(group_data)
    elif mode == 'individual_for_all' or initial_cams_for_novels is not None:
        for i in range(num_poses):
            print(f"Render group: {i}, initial_poses: {initial_cams_for_novels[i]}")
            if initial_cams_for_novels is not None:
                initial_cams_data_for_novel = get_cam_data_with_another_index(data['initial'],
                                                                              initial_cams_for_novels[i][0])
            else:
                initial_cams_data_for_novel = data['initial']

            group_data = {'initial': initial_cams_data_for_novel, 'novels': [novels_data[i]]}
            render_groups.append(group_data)

    return render_groups


def save_layered_depth(layered_depth, path):
    num_layers, h, w = layered_depth.shape
    depth_meta_data = {}
    for i in range(num_layers):
        depth = layered_depth[i]
        low, high = np.min(depth), np.max(depth)
        scaled = (depth - low) / (high - low)
        ui = np.clip(scaled * 256.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(ui)

        tag = f'layer_depth_{i:02d}'
        img.save(os.path.join(path, tag + '.jpg'), quality=100)
        depth_meta_data[tag] = [low.item(), high.item()]

    return depth_meta_data


def write_meta(depth_meta_data, resolution, extrinsic, intrinsic, path):
    meta_data = depth_meta_data

    meta_data.update({'frame_size': resolution,
                      'extrinsic_re': extrinsic,
                      'intrinsics_re': intrinsic,
                      })

    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(meta_data, f, ensure_ascii=True)


def save_generated_proxy_geometry(save_path, proxy_geometry, identifier):
    dir_name = os.path.join(save_path, 'mpi_geom')
    os.makedirs(dir_name, exist_ok=True)
    layered_depth = proxy_geometry['layered_depth'].squeeze(0).squeeze(0).data.cpu()
    torch.save(proxy_geometry['mpi'].data.cpu(), f'{dir_name}/{identifier}.mpi')

    torch.save(proxy_geometry['mli']['verts'], f'{dir_name}/{identifier}.verts')
    torch.save(proxy_geometry['mli']['verts_features'], f'{dir_name}/{identifier}.verts_rgb')

    torch.save(layered_depth, f'{dir_name}/{identifier}.layered_depth')


def save_mpi(mpi, path, save_as_jpg=False):
    n_planes, channels, height, width = mpi.shape
    if channels != 4:
        logger.warning(f'Expected to get RGBA layers, but obtained {channels} channels, '
                       f'for visualizes will used first 3 as RGB and last as alpha')
        mpi = np.concatenate([mpi[:, :3], mpi[:, -1:]], axis=1)
    mpi = np.concatenate([mpi[:, :-1] * 0.5 + 0.5, mpi[:, -1:]], axis=1)
    mpi[0, -1:] = 1
    mpi = mpi.transpose((0, 2, 3, 1))
    mpi = (255 * np.clip(mpi, 0, 1)).astype(np.uint8)

    os.makedirs(path, exist_ok=True)
    for i, layer in enumerate(mpi):
        if save_as_jpg:
            Image.fromarray(layer[:, :, :3]).save(os.path.join(path, f'layer_{i:02d}.jpg'), optimize=True)
            Image.fromarray(np.repeat(layer[:, :, -1:], 3, axis=2)).save(os.path.join(path, f'layer_alpha_{i:02d}.jpg'), optimize=True)
        else:
            Image.fromarray(layer).save(os.path.join(path, f'layer_{i:02d}.png'), optimize=True)


def render(output_path, trainer, dataloader, save_gt_images=False,
           produce_videos=False, save_mpi_layers=False,
           save_mpi_geom=False, save_depth_images=True):
    output_images_path = os.path.join(output_path, 'images')
    os.makedirs(output_images_path, exist_ok=True)

    if produce_videos:
        output_videos_path = os.path.join(output_path, 'videos')
        os.makedirs(output_videos_path, exist_ok=True)
    else:
        output_videos_path = None

    time_profiling = {'infer': [], 'render': []}
    for idx, data in enumerate(tqdm(dataloader, desc='scenes', leave=True)):
        render_groups = split_on_render_groups(data, mode='one_for_all')

        print("num render groups: ", len(render_groups))

        scene_rendered_images = []
        scene_rendered_images_names = []
        scene_rendered_images_image_path = os.path.join(output_images_path, str(idx), 'rendered')
        if save_mpi_layers:
            scene_rendered_scenes_mpi_path = os.path.join(output_path, 'mpi', str(idx))
        scene_rendered_images_image_saver = ImageWriter(output_path=scene_rendered_images_image_path)

        if save_gt_images:
            scene_gt_images = []
            scene_gt_images_names = []
            scene_gt_images_image_path = os.path.join(output_images_path, str(idx), 'target')
            scene_gt_images_image_saver = ImageWriter(output_path=scene_gt_images_image_path)

        if save_depth_images:
            scene_rendered_depths = []
            scene_rendered_depths_names = []
            scene_rendered_depths_path = os.path.join(output_images_path, str(idx), 'rendered', 'depth')
            scene_rendered_depths_image_saver = ImageWriter(output_path=scene_rendered_depths_path)

        for render_group in tqdm(render_groups, desc='render_groups', leave=True):
            proxy_geometry = None
            first_novel = True
            for novel_data in tqdm(render_group['novels'], desc='novels', leave=True):
                start = timer()
                result = trainer.inference({'initial': render_group['initial'],
                                            'novel': novel_data,
                                            'proxy_geometry': proxy_geometry,
                                            'reference': render_group.get('reference', None)
                                            })
                measured_time = timer() - start
                if proxy_geometry is None:
                    proxy_geometry = result.get('proxy_geometry')
                    time_profiling['infer'].append(measured_time)
                else:
                    time_profiling['render'].append(measured_time)

                scene_rendered_images.append((((result['images'].cpu() + 1.) / 2.) * 255).permute(0, 2, 3, 1).numpy())
                scene_rendered_images_names.append(f"{novel_data['id']:06d}")

                if save_depth_images and 'depth' in result:
                    scene_rendered_depths.append(
                        result['depth'][:, 0].cpu()
                    )
                    scene_rendered_depths_names.append(f"{novel_data['id']:06d}")

                if save_gt_images:
                    gt_novel_image = novel_data['image'][0][0]
                    if list(gt_novel_image.shape[-2:]) != [1, 1]:
                        scene_gt_images.append(
                            (((gt_novel_image.cpu() + 1.) / 2.) * 255).permute(0, 2, 3, 1).numpy())
                        scene_gt_images_names.append(f"{novel_data['id']:06d}")
                    else:
                        save_gt_images = False

                if first_novel and save_mpi_geom:
                    first_novel = False
                    save_generated_proxy_geometry(output_path, proxy_geometry, identifier=idx, )

            if save_mpi_layers:

                mpi = result.get('mpi', None)
                if mpi is not None:
                    mpi = mpi.squeeze(0).data.cpu().numpy()
                    save_mpi(mpi, scene_rendered_scenes_mpi_path, save_as_jpg=True)

                if 'layered_depth' in proxy_geometry:
                    layered_depth = proxy_geometry['layered_depth'][0].cpu()
                    depth_meta_data = save_layered_depth(layered_depth.numpy(), scene_rendered_scenes_mpi_path)

                    intr = render_group.get('reference', None)['intrinsics'][:, :, 0].flatten().cpu().numpy()
                    extr = render_group.get('reference', None)['extrinsics'][:, :, 0].flatten().cpu().numpy()
                    write_meta(depth_meta_data,
                               [layered_depth.shape[1], layered_depth.shape[2]],
                               list(extr.astype(float)),
                               list(intr.astype(float)),
                               scene_rendered_scenes_mpi_path)

            del result
            torch.cuda.empty_cache()

        scene_rendered_images_image_saver.save_images(np.concatenate(scene_rendered_images, axis=0),
                                                      scene_rendered_images_names)

        if save_depth_images and len(scene_rendered_depths):
            scene_rendered_depths = torch.stack(scene_rendered_depths)
            scene_rendered_depths = min_max_scale(scene_rendered_depths, dim=[0, 2, 3]).numpy()
            p_90 = np.percentile(scene_rendered_depths, 90)
            p_05 = np.percentile(scene_rendered_depths, 5)
            scene_rendered_depths = (scene_rendered_depths - p_05) / (p_90 - p_05)

            scene_rendered_depths = 1 - scene_rendered_depths
            scene_rendered_depths_imgs = []
            for i in range(scene_rendered_depths.shape[0]):
                scene_rendered_depths_imgs.append(
                    (grayscale_to_cmap(
                        scene_rendered_depths[i],
                        cmap='magma',
                    ) * 255).astype(np.uint8)
                )

            scene_rendered_depths_image_saver.save_images(np.concatenate(scene_rendered_depths_imgs, axis=0),
                                                          scene_rendered_depths_names)

        if save_gt_images:
            scene_gt_images_image_saver.save_images(np.concatenate(scene_gt_images, axis=0), scene_gt_images_names)

        if output_videos_path is not None:
            image_folder_to_video(output_path=os.path.join(output_videos_path, str(idx) + '.mp4'),
                                  folder_path=scene_rendered_images_image_path
                                  )

            if save_depth_images and len(scene_rendered_depths):
                output_depth_videos_path = os.path.join(output_videos_path, 'depth')
                os.makedirs(output_depth_videos_path, exist_ok=True)
                image_folder_to_video(output_path=os.path.join(output_depth_videos_path, str(idx) + '_depth.mp4'),
                                      folder_path=scene_rendered_depths_path
                                      )

        del scene_rendered_depths
        del scene_rendered_depths_names
        del scene_rendered_images
        del scene_rendered_images_names
        del scene_rendered_images_image_path
        del scene_rendered_images_image_saver
        del data
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

    return time_profiling


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--res', help="resolution", nargs=2, type=int)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the config file.')
    parser.add_argument('--val-dataset', type=str, default=None,
                        help='Path to validation dataset.')
    parser.add_argument('--output-path', type=str, help='Path to outputs directory.')
    parser.add_argument('--checkpoints-path', type=str, default=None,
                        help='Path to checkpoints directory. If None extracted from config path')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Iteration number. If None - gets the latest.')
    parser.add_argument('--scene_scale_constant', type=float, default=1.0,
                        help='scene_scale_constant.')
    parser.add_argument('--skip_each_ith_frame', type=int, default=0, help='skip each')
    parser.add_argument('--use-ema', action='store_true', help='Use stereo as input.')
    parser.add_argument('--save-mpi', action='store_true',
                        help='Save mpi binary file.')
    parser.add_argument('--save-layers', action='store_true',
                        help='Save mpi binary file and png images of layers.')
    # parser.add_argument('--save-layers-video', action='store_true',
    #                     help='Save layers video')
    opts = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True

    # result_resolution = opts.res
    model_name = os.path.basename(opts.config).split('.')[0]
    config = get_config(opts.config)
    if opts.checkpoints_path is not None:
        checkpoints_path = opts.checkpoints_path
    else:
        checkpoints_path = os.path.join(os.path.dirname(opts.config), 'checkpoints')

    val_dataset_name = opts.val_dataset.split('/')[-1]
    if val_dataset_name == '':
        val_dataset_name = opts.val_dataset.split('/')[-2]

    val_dataset = ValidationDataset(opts.val_dataset,
                                    skip_each_ith_frame=opts.skip_each_ith_frame,
                                    scene_scale_constant=opts.scene_scale_constant)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                sampler=SequentialSampler(val_dataset)
                                )

    trainer, loaded_iteration = create_trainer_load_weights_from_config(config=config,
                                                                        checkpoints_dir=checkpoints_path,
                                                                        iteration=opts.iteration,
                                                                        device='cuda',
                                                                        use_ema=opts.use_ema,
                                                                        )
    total_params = sum(p.numel() for p in trainer.gen.parameters() if p.requires_grad)
    total_params_all = sum(p.numel() for p in trainer.gen.parameters())
    print('total_params: ', total_params)
    print('total_params_all: ', total_params_all)
    print(opts.output_path)
    if opts.output_path is None:
        output_path = os.path.dirname(opts.config)
    else:
        output_path = opts.output_path

    output_path = os.path.join(output_path, 'renders')
    print(output_path)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path,
                               val_dataset_name,
                               model_name + '_' + str(loaded_iteration))

    if val_dataset.skip_each_ith_frame > 0:
        output_path = output_path + f'_skip_frames_{val_dataset.skip_each_ith_frame}'
    print(output_path)
    os.makedirs(output_path, exist_ok=True)

    trainer.eval()
    time_profiling = render(output_path, trainer, val_dataloader,
                            save_gt_images=False,
                            produce_videos=True,
                            save_mpi_layers=True,
                            save_mpi_geom=opts.save_layers)

    time_profiling['infer_mean'] = np.mean(time_profiling['infer'])
    time_profiling['infer_std'] = np.std(time_profiling['infer'])
    time_profiling['render_mean'] = np.mean(time_profiling['render'])
    time_profiling['render_std'] = np.std(time_profiling['render'])
    time_profiling['total_params'] = total_params
    del time_profiling['infer']
    del time_profiling['render']
    with open(os.path.join(output_path, 'rendering_stats.json'), 'w') as fp:
        json.dump(time_profiling, fp)


if __name__ == '__main__':
    main()
