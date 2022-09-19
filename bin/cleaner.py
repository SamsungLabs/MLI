"""
Example:

python3 cleaner.py \
    --user-dir /dbstore/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/ \
    --ssh d.korzhenkov@75.17.101.11 \
    --n-images 3 \
    --n-checkpoints 1 \
    --tensorboard \
    --log \
    --models \
        kor_lsun_stylegan_totalmultiscale_0degree_1view \
        kor_lsun_stylegan_multiscale_0degree
"""

import argparse
from collections import defaultdict
import os
import subprocess

from tqdm import tqdm


def make_command_remote(command, ssh_cred=None):
    if ssh_cred is None:
        return command
    return f"ssh -o LogLevel=QUIET -t {ssh_cred} '{command}'"


def go_to_dir(directory):
    return f'cd {directory}'


def rename_remove_rename(directory, saved_files, ssh_cred=None):
    renaming_dict = {name: f'{name}.saved' for name in saved_files}
    command_rename = ' ; '.join(f'mv {old} {new}' for old, new in renaming_dict.items())

    command_remove = f'ls | grep -v .saved$ | xargs rm'

    renaming_dict = {new: old for old, new in renaming_dict.items()}
    command_rename_back = ' ; '.join(f'mv {old} {new}' for old, new in renaming_dict.items())

    command_full = ' ; '.join([go_to_dir(directory), command_rename, command_remove, command_rename_back])
    command_full = make_command_remote(command_full, ssh_cred)
    subprocess.call(command_full, shell=True)


def clean_images(directory, num_latest=3, ssh_cred=None):
    command_ls_images = f'{go_to_dir(directory)} && ls'
    command_ls_images = make_command_remote(command_ls_images, ssh_cred)
    list_of_images = subprocess.check_output(command_ls_images, shell=True).decode('utf-8').split()
    list_of_images = sorted([p for p in list_of_images if p], reverse=True)
    saved_images = list_of_images[:num_latest]
    rename_remove_rename(directory, saved_images, ssh_cred)


def clean_checkpoints(directory, num_latest=1, ssh_cred=None):
    command_ls_checkpoints = f'{go_to_dir(directory)} && ls'
    command_ls_checkpoints = make_command_remote(command_ls_checkpoints, ssh_cred)
    list_of_checkpoints = subprocess.check_output(command_ls_checkpoints, shell=True).decode('utf-8').split()
    list_of_checkpoints = [p for p in list_of_checkpoints if p]

    unique_model_parts = defaultdict(list)
    for checkpoint in list_of_checkpoints:
        checkpoint_type, part_iter = checkpoint.split('.')[:2]
        part_iter = part_iter.split('_')
        part, iter_num = '_'.join(part_iter[:-1]), part_iter[-1]
        unique_model_parts[f'{checkpoint_type}.{part}'].append(iter_num)

    common_iterations = None
    for key, value in unique_model_parts.items():
        if not key.startswith('model_ema.'):
            if common_iterations is None:
                common_iterations = set(value)
            else:
                common_iterations &= set(value)
    if common_iterations is None:
        return
    common_iterations = sorted(common_iterations, reverse=True)
    saved_iterations = set(common_iterations[:num_latest])
    if num_latest > 0:
        saved_iterations |= {'ema'}
    saved_checkpoints = [checkpoint for checkpoint in list_of_checkpoints
                         if checkpoint.split('.')[-2].split('_')[-1] in saved_iterations
                         ]
    rename_remove_rename(directory, saved_checkpoints, ssh_cred)


def clean_logs(directory, ssh_cred=None):
    command_remove = f'{go_to_dir(directory)} && rm -rf *'
    command_remove = make_command_remote(command_remove, ssh_cred)
    subprocess.call(command_remove, shell=True)

    
def clean_tensorboard(directory, model_name, ssh_cred=None):
    command_remove = f'{go_to_dir(directory)} && rm -rf {model_name}'
    command_remove = make_command_remote(command_remove, ssh_cred)
    subprocess.call(command_remove, shell=True)


def remove_folder(directory, ssh_cred=None):
    command_remove = f'rm -rf {directory}'
    command_remove = make_command_remote(command_remove, ssh_cred)
    subprocess.call(command_remove, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experiments logs cleaner')
    parser.add_argument('--user-dir', type=str, default='.',
                        help="Path to the user's model directory. E.g., "
                             "/dbstore/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/")
    parser.add_argument('--models', type=str, default=[], nargs='+',
                        help='Names of the models.')
    parser.add_argument('--n-images', type=int, default=3,
                        help='Number of latest images to save.')
    parser.add_argument('--n-checkpoints', type=int, default=1,
                        help='Number of latest checkpoints to save.')
    parser.add_argument('--ssh', type=str, default=None,
                        help='SSH connection for cleaning remote directories. E.g., d.korzhenkov@75.17.101.11')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Remove tensorboard logs as well.')
    parser.add_argument('--log', action='store_true',
                        help='Remove text logs as well.')
    parser.add_argument('--total', action='store_true',
                        help='Remove all the stuff related to the specified models.')
    opts = parser.parse_args()

    outputs_dir = os.path.join(opts.user_dir, 'outputs')
    tensorboard_dir = os.path.join(opts.user_dir, 'tensorboard')

    for model_name in tqdm(opts.models, desc='models'):
        model_dir = os.path.join(outputs_dir, model_name)

        if opts.total:
            remove_folder(model_dir, ssh_cred=opts.ssh)
            clean_tensorboard(tensorboard_dir, model_name, ssh_cred=opts.ssh)
        else:
            images_dir = os.path.join(model_dir, 'images')
            logs_dir = os.path.join(model_dir, 'log')
            checkpoints_dir = os.path.join(model_dir, 'checkpoints')

            clean_images(images_dir, num_latest=opts.n_images, ssh_cred=opts.ssh)
            clean_checkpoints(checkpoints_dir, num_latest=opts.n_checkpoints, ssh_cred=opts.ssh)
            if opts.log:
                clean_logs(logs_dir, ssh_cred=opts.ssh)
            if opts.tensorboard:
                clean_tensorboard(tensorboard_dir, model_name, ssh_cred=opts.ssh)
