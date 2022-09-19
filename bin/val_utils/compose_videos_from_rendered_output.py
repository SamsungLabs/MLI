import argparse
import glob
import logging
import os

from lib.utils.visualise import image_folder_to_video


def main():
    logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_render_folder', type=str, default=None,
                        help='path_to_rendered.')
    opts = parser.parse_args()

    if os.path.isdir(os.path.join(opts.path_to_render_folder, 'images')):
        model_paths = [opts.path_to_render_folder]
        models_names = [opts.path_to_render_folder.split('/')[-1]]
    else:
        model_paths = sorted(glob.glob(os.path.join(opts.path_to_render_folder, '*', 'images')))
        model_paths = ['/'.join(x.split('/')[:-1]) for x in model_paths]
        models_names = [model_path.split('/')[-1] for model_path in model_paths]
        print(f'Found {len(models_names)} models in {model_paths}.')

    for model_name, model_path in zip(models_names, model_paths):
        scenes_paths = glob.glob(os.path.join(model_path, 'images', '*'))
        scenes_names = [x.split('/')[-1] for x in scenes_paths]
        video_folder_path = os.path.join(model_path, 'videos')
        os.makedirs(video_folder_path, exist_ok=True)
        for scenes_path, scene_name in zip(scenes_paths, scenes_names):
            scene_video_path = os.path.join(video_folder_path, scene_name + '.mp4')
            image_folder_to_video(scene_video_path, os.path.join(scenes_path, 'rendered'))


if __name__ == '__main__':
    main()