import multiprocessing
import os
import random
import shutil
import subprocess
import glob
from typing import Callable, List

import numpy as np
from PIL import Image
import torch

try:
    from matplotlib import cm

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def multiprocessing_save_image(params):
    ndarr, file_name = params
    image = Image.fromarray(ndarr)
    image.save(file_name, format='JPEG', subsampling=0, quality=80)


def write_loss(iterations, losses, train_writer):
    losses_tb = {}

    for optimizer_group in losses.keys():
        for loss_name, value in losses[optimizer_group].items():
            losses_tb['loss_' + optimizer_group + '_' + loss_name] = value

    for loss_name, value in losses_tb.items():
        train_writer.add_scalar(loss_name, losses_tb[loss_name], iterations)


def write_metrics(iterations, metrics, train_writer):
    metrics_tb = {}

    for metric_name, value in metrics.items():
        metrics_tb['metric_' + metric_name] = value

    for metric_name, value in metrics_tb.items():
        train_writer.add_scalar(metric_name, metrics_tb[metric_name], iterations)


def write_grad_norms(iterations, gradient_info, train_writer):
    for name, gradient_norm in gradient_info.items():
        train_writer.add_scalar(name.replace('.', '/'), gradient_norm, iterations)


def plot_square_grid(images, cols, rows, padding=5, first_pad=0):
    widths, heights = images[0].size

    if first_pad == 0:
        first_pad = padding

    paddings_horizont = [padding] * (cols)
    paddings_vertical = [padding] * (rows)
    paddings_horizont[-1] = 0
    paddings_vertical[-1] = 0

    paddings_horizont[0] = first_pad

    table_width = widths * cols + sum(paddings_horizont)
    table_height = heights * rows + sum(paddings_vertical)

    new_im = Image.new('RGBA', (table_width, table_height))

    x = 0
    for i in range(rows):
        new_im.paste(Image.fromarray(np.ones((paddings_vertical[i], table_width, 4), dtype=np.uint8) * 0),
                     (x, 0))
        x += heights + paddings_vertical[i]

    y = 0
    for i in range(cols):
        new_im.paste(Image.fromarray(np.ones((table_height, paddings_horizont[i], 4), dtype=np.uint8) * 0),
                     (0, y))
        y += widths + paddings_horizont[i]

    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            new_im.paste(images[j + i * cols], (x, y))
            x += (widths + paddings_horizont[j])
        y += (heights + paddings_vertical[i])

    return new_im


def image_folder_to_video(output_path: str,
                          folder_path: str,
                          image_name_format: str = '%06d.jpg',
                          remove_image_folder: bool = False):
    """
    Convert folder with images to video, folder must contain only images which will be used in generating video.
    Args:
        output_path: output video path
        folder_path: folder with images
        image_name_format: name format for ffmpeg
        remove_image_folder: remove image folder after video generating, or not.

    """
    num_images = len(glob.glob(os.path.join(folder_path, '*.' + image_name_format.split('.')[-1])))
    command = (
            f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {folder_path}/{image_name_format} '
            + f'-vframes {num_images} '
            + '-vcodec libx264 -crf 18 '
            + '-pix_fmt yuv420p '
            + output_path
    ).split()
    subprocess.call(command)
    if remove_image_folder:
        shutil.rmtree(folder_path)


class ImageWriter:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        os.makedirs(self.output_path, exist_ok=True)

    def __del__(self):
        self.write_pool.close()

    @staticmethod
    def write_image(inputs):
        folder, name, numpy_image = inputs
        numpy_image = np.clip(numpy_image, 0., 255.).astype(np.uint8)
        if numpy_image.shape[1] % 2 != 0:
            # ffmpeg cannot process such frames
            numpy_image = numpy_image[:, :-1]
        filename = f'{name}.jpg'
        image = Image.fromarray(numpy_image)
        image.save(os.path.join(folder, filename))
        del image

    def save_images(self, images: np.array, names: List[str] = None):
        """
        Save array images to folder.
        Args:
            images: N x H x W x C
            names:  list of length N with names, if None, saver uses frame idx in array.

        Returns:

        """
        num_images = images.shape[0]
        if names is None:
            names = [f'{i:06d}' for i in range(num_images)]

        # for name, image in zip(names, images):
        #     self.write_image((self.output_path, name, image))
        self.write_pool.map(self.write_image,
                            zip([self.output_path] * num_images,
                                names,
                                images)
                            )


class VideoWriter:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 tmp_root='/tmp',
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        self.n_items = 0
        self.random_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        self.tmp_dir = os.path.join(tmp_root, self.random_id)

        self.image_writer = ImageWriter(output_path=self.tmp_dir, n_threads=n_threads)

    def process_batch(self, images: torch.Tensor):
        images = images.data.permute(0, 2, 3, 1).to('cpu').numpy()
        batch_size = images.shape[0]
        names = [f'{(self.n_items + i):06d}' for i in range(batch_size)]
        self.image_writer.save_images(images, names)
        self.n_items += batch_size

    def finalize(self):
        command = (
                f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {self.tmp_dir}/%06d.jpg '
                + f'-vframes {self.n_items} '
                + '-vcodec libx264 -crf 18 '
                + '-pix_fmt yuv420p '
                + self.output_path
        ).split()
        subprocess.call(command)
        shutil.rmtree(self.tmp_dir)


def grayscale_to_cmap(array: np.ndarray,
                      cmap: str = 'viridis'):
    if MATPLOTLIB_AVAILABLE:
        cmap = cm.get_cmap(cmap)
        return cmap(array)[..., :3]
    else:
        return np.stack([array] * 3, axis=-1)


class VideoWriterPool:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 tmp_root='/tmp',
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        self.n_items = 0
        self.random_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        self.tmp_dir = os.path.join(tmp_root, self.random_id)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def __del__(self):
        self.write_pool.close()

    @staticmethod
    def write_image(inputs):
        folder, frame_number, numpy_image = inputs
        numpy_image = np.clip(numpy_image, 0., 255.).astype(np.uint8)
        if numpy_image.shape[1] % 2 != 0:
            # ffmpeg cannot process such frames
            numpy_image = numpy_image[:, :-1]
        filename = f'{frame_number:06d}.jpg'
        Image.fromarray(numpy_image).save(os.path.join(folder, filename))

    @staticmethod
    def write_pil_image(inputs):
        folder, frame_number, pil_image = inputs
        filename = f'{frame_number:06d}.jpg'
        pil_image.save(os.path.join(folder, filename), quality=100)

    def process_batch(self, images: torch.Tensor):
        images = images.data.permute(0, 2, 3, 1).to('cpu').numpy()
        batch_size = images.shape[0]
        self.write_pool.map(self.write_image,
                            zip([self.tmp_dir] * len(images),
                                self.n_items + np.arange(batch_size),
                                images)
                            )
        self.n_items += batch_size

    def process_pil_list(self, images: List):
        batch_size = len(images)
        self.write_pool.map(self.write_pil_image,
                            zip([self.tmp_dir] * len(images),
                                self.n_items + np.arange(batch_size),
                                images)
                            )
        self.n_items += batch_size

    def finalize(self):
        command = (
            f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {self.tmp_dir}/%06d.jpg '
            + f'-vframes {self.n_items} '
            + '-vcodec libx264 -crf 18 '
            + '-pix_fmt yuv420p '
            + self.output_path
        ).split()
        subprocess.call(command)
        shutil.rmtree(self.tmp_dir)
