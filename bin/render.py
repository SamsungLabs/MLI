import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from lib.utils.data import get_dataloader_from_params
from lib.utils.io import get_config
from lib.utils.visualise import VideoWriter
from lib.trainers.utils import create_trainer_load_weights_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test/test.yaml',
                        help='Path to the config file.')
    parser.add_argument('--checkpoints-path', type=str, default='.',
                        help='Path to checkpoints directory.')
    parser.add_argument('--iteration', type=int, default=0,
                        help='Iteration number')
    parser.add_argument('--output-path', type=str, default='.',
                        help='Path to outputs directory.')
    parser.add_argument('--use-ema', action='store_true', help='Use ema')
    opts = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    os.makedirs(os.path.join(opts.output_path, model_name), exist_ok=True)
    output_filename = os.path.join(opts.output_path, model_name, f'render_{opts.iteration}.mp4')
    video_writer = VideoWriter(output_filename)

    # Setup loaders
    dataloader = get_dataloader_from_params(config['dataloaders'], 'render')
    # dataloader.dataset.save_krt(os.path.join(opts.output_path, model_name, 'virtual_KRT.txt'))

    if config['models'].get('all_cameras') is not None:
        config['models']['all_cameras']['krt'] = \
            dataloader.dataset.all_cameras_krt
        if config['dataloaders']['train']['dataset']['background_type'] == 'learnable':
            config['models']['all_cameras']['use_learnable_background'] = True

    trainer, loaded_iteration = create_trainer_load_weights_from_config(config=config, 
                                                                        checkpoints_dir=opts.checkpoints_path, iteration=opts.iteration,
                                                                        device='cuda',
                                                                        use_ema=opts.use_ema,
                                                                        )
    trainer.eval()

    for data in tqdm(dataloader):
        del data['reference']
        images = trainer.inference(data)['images']
        images = ((images + 1.) / 2.) * 255
        video_writer.process_batch(images)
    video_writer.finalize()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
