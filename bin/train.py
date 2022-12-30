import argparse
import logging
import os
import pdb
import shutil
import signal
import ssl
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
try:
    import horovod.torch as hvd
except ImportError:
    pass

from lib import trainers
from lib.utils.base import Timer, seed_freeze
from lib.utils.io import get_config
from lib.utils.visualise import write_loss, write_metrics, write_grad_norms
from lib.utils.data import get_dataloader_from_params


ssl._create_default_https_context = ssl._create_unverified_context
SOURCE_FOLDER = 'lib'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test/test.yaml', help='Path to the config file.')
    parser.add_argument('--output-path', type=str, default='.', help="outputs path")
    parser.add_argument('--resume-iteration', type=int, default=None, help="iteration for resuming")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--verbose', '-v', type=str, default='DEBUG', help="logging level")
    parser.add_argument('--device', '-d', type=str, default='cuda', help="type of the device")
    parser.add_argument('--local_rank', default=0, type=int)
    opts = parser.parse_args()

    n_gpu = 1
    if 'WORLD_SIZE' in os.environ:
        n_gpu = int(os.environ['WORLD_SIZE'])
        opts.distributed = n_gpu > 1
    else:
        opts.distributed = False

    # Load experiment setting
    config = get_config(opts.config)
    if opts.device == 'cpu':
        config['use_apex'] = False

    use_apex = config.get('use_apex', True)
    use_horovod = config.get('use_horovod', False)
    assert not (use_horovod and use_apex)

    # == FOR DISTRIBUTED ==
    # Set the device according to local_rank.
    local_rank = device_rank = opts.local_rank
    if use_horovod:
        hvd.init()
        device_rank = hvd.rank()
        local_rank = hvd.local_rank()
        n_gpu = hvd.size()
    if opts.device.lower() != 'cpu':
        torch.cuda.set_device(local_rank)

    # Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    if use_apex:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True

    # Set random seed
    seed_freeze(base_seed=config.get('seed'))

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0] + f'_{n_gpu:d}gpu'
    output_directory = os.path.join(opts.output_path, 'outputs', model_name)
    log_directory = os.path.join(output_directory, 'log')
    image_directory = os.path.join(output_directory, 'images')
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    copy_directory = os.path.join(output_directory, SOURCE_FOLDER)
    copy_tar_directory = os.path.join(output_directory, 'source')
    tensorboard_directory = os.path.join(opts.output_path, 'tensorboard', model_name)

    if opts.resume_iteration is not None:
        opts.resume = True

    fail_with_nan = False
    if os.environ.get('FAIL_WITH_NAN'):
        fail_with_nan = int(os.environ['FAIL_WITH_NAN'])

    if device_rank == 0:
        if not opts.resume and os.path.exists(tensorboard_directory):
            shutil.rmtree(tensorboard_directory)
        os.makedirs(tensorboard_directory, exist_ok=True)
        train_writer = SummaryWriter(tensorboard_directory)

        print(f'Logging directory: {log_directory}')
        os.makedirs(log_directory, exist_ok=True)
        logger = logging.getLogger('root')
        sys.excepthook = lambda exc_type, value, tb: logger.exception(f'Uncaught Exception',
                                                                      exc_info=(exc_type, value, tb))
        logging.basicConfig(filename=os.path.join(log_directory, 'log.txt'),
                            filemode='a+' if opts.resume else 'w',
                            format='%(levelname)s - %(name)s - %(asctime)s:\t%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=getattr(logging, opts.verbose.upper()))

        pil_logger = logging.getLogger('PIL.PngImagePlugin')
        pil_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger('PIL.TiffImagePlugin')
        pil_logger.setLevel(logging.ERROR)

        logger.debug(f'Image directory: {image_directory}')
        os.makedirs(image_directory, exist_ok=True)
        logger.debug(f'Checkpoints directory: {checkpoint_directory}')
        os.makedirs(checkpoint_directory, exist_ok=True)
        shutil.copy(opts.config, os.path.join(output_directory, f'{model_name}.yaml'))
        logger.debug(f'Source copy directory: {copy_directory}')
        if os.path.exists(f'{copy_tar_directory}.tar'):
            os.remove(f'{copy_tar_directory}.tar')
        shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, SOURCE_FOLDER),
                        copy_directory,
                        ignore=shutil.ignore_patterns('*__pycache__*'),
                        )
        shutil.make_archive(copy_tar_directory, 'tar', root_dir=output_directory, base_dir=SOURCE_FOLDER)
        shutil.rmtree(copy_directory)
        logger.info(f'FAIL_WITH_NAN value: {fail_with_nan}')

    # Setup loaders
    base_seed = config.get('seed')
    if base_seed is not None:
        base_seed = base_seed * (device_rank + 1)
    train_dataloader = get_dataloader_from_params(config, 'train', base_seed=base_seed, use_horovod=use_horovod)
    if device_rank == 0:
        vis_dataloader = get_dataloader_from_params(config, 'vis', base_seed=base_seed, use_horovod=False)
        val_dataloader = get_dataloader_from_params(config, 'val', base_seed=base_seed, use_horovod=False)

    if config['models'].get('all_cameras') is not None:
        config['models']['all_cameras']['krt'] = train_dataloader.dataset.all_cameras_krt
        if config['dataloaders']['train']['dataset']['background_type'] == 'learnable':
            config['models']['all_cameras']['use_learnable_background'] = True

    # Note: for multi-gpu mode device=cuda equal to .cuda() method
    device = torch.device(opts.device)
    trainer = getattr(trainers, config['trainer'])(config, device=device)

    # Start training
    iteration = trainer.resume(checkpoint_directory, opts.resume_iteration) if opts.resume else -1

    def urgent_checkpoint(*args, **kwargs):
        nonlocal iteration
        if device_rank == 0:
            logger.warning('Urgent checkpointing, iteration %d', iteration)
            trainer.save(checkpoint_directory, iteration)

    def urgent_debugger(*args, **kwargs):
        if device_rank == 0:
            pdb.set_trace()

    signal.signal(signal.SIGUSR1, urgent_checkpoint)  # kill -10 PID
    signal.signal(signal.SIGUSR2, urgent_debugger)  # kill -12 PID

    # TODO Add parametric iteration for visualise test

    while True:
        for data in train_dataloader:
            iteration += 1
            with Timer('Elapsed time in update: {:.3f}', iteration, config['log_iter']):
                # Main training code
                error_message = trainer.update(data, iteration)

            if device_rank == 0:
                #  Aggregate EMA module
                current_generator = trainer.gen.module if trainer.params.get('use_apex', True) else trainer.gen
                trainer.average_models(trainer.gen_ema, current_generator,
                                       alpha_decay=trainer.params.get('decay_rate', 0.9))

                # Dump trainer and data if NaN was caught
                if error_message is not None and fail_with_nan:
                    trainer.save(checkpoint_directory, iteration)
                    logger.error(error_message)
                    torch.save(data, os.path.join(checkpoint_directory, f'crush_data_{iteration:08d}'))
                    raise Exception(error_message)

                # Dump training stats in log file
                if iteration % config['log_iter'] == 0:
                    logger.info(f"Iteration: {iteration:08d}/{config['max_iter']:08d}")
                    write_loss(iteration, trainer.losses, train_writer)
                    write_grad_norms(iteration, trainer.gradient_info, train_writer)

                torch.cuda.empty_cache()
                # Write images
                if ('visualise_iter' in config) and (iteration % config['visualise_iter'] == 0) and vis_dataloader:
                    with Timer('Elapsed time in visualization: {:.3f}'):
                        image = trainer.visualise(vis_dataloader)
                    image.convert('RGB').save(os.path.join(image_directory, f'test_{iteration:08d}.jpg'))

                torch.cuda.empty_cache()
                # Validation
                if ('validation_iter' in config) \
                        and (iteration % config['validation_iter'] == 0) \
                        and val_dataloader:
                    with Timer('Elapsed time in validation: {:.3f}'):
                        metrics = trainer.evaluate(val_dataloader,
                                                   num_repeat=config.get('validation_num_repeat', 1))
                    write_metrics(iteration, metrics, train_writer)

                # Save network weights
                if iteration % config['snapshot_save_iter'] == 0:
                    trainer.save(checkpoint_directory, iteration)

            trainer.update_learning_rate()

            if iteration >= config['max_iter']:
                if device_rank == 0:
                    train_writer.close()
                    logger.info('Finish training')
                sys.exit()


if __name__ == '__main__':
    main()
