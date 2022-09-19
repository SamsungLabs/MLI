import argparse
from glob import glob

import logging
import os

import lpips
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from torchvision import transforms
from pytorch_msssim import SSIM, MS_SSIM

from lib.modules.losses import PSNRMetric, SSIM, HiFreqPSNRMetric, PSNRMetricColorShift, SSIMColorShift
from lib.modules.flip_loss import FLIPLoss

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def extract_image_id(path):
    scene_id, _, image_id, = path.split('/')[-3:]
    image_id = image_id.split('.')[0]
    image_id = image_id.split('_')[-1]
    return scene_id + '_' + image_id


class CropBorder:
    def __init__(self, crop_border_size):
        self.crop_border_size = crop_border_size

    def __call__(self, image):
        width, height = image.size

        width_crop_size = int(width * self.crop_border_size)
        height_crop_size = int(height * self.crop_border_size)
        return image.crop((width_crop_size,
                           height_crop_size,
                           width - width_crop_size,
                           height - height_crop_size))


class ScorerDataset(Dataset):
    def __init__(self,
                 rendered_folder: str,
                 val_dataset_folder: str,
                 crop_border_size: float = 0.15,
                 ):
        """

        Args:
            rendered_folder:
            val_dataset_folder:
        """
        super().__init__()
        self.transforms = [transforms.Lambda(CropBorder(crop_border_size).__call__),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)
        self.rendered_folder = rendered_folder
        self.val_dataset_folder = val_dataset_folder

        val_dataset_gt_images_paths = sorted(glob(os.path.join(self.val_dataset_folder, '*', 'novel', '*.jpg')))
        rendered_images_paths = sorted(glob(os.path.join(self.rendered_folder, 'images', '*', 'rendered', '*.jpg')))
        assert len(val_dataset_gt_images_paths) == len(rendered_images_paths), \
            f"num rendered images must be equal gt images {len(val_dataset_gt_images_paths)} != {len(rendered_images_paths)}"

        self.samples = []
        for val_dataset_gt_images_path, rendered_images_path in zip(val_dataset_gt_images_paths, rendered_images_paths):
            val_image_id = extract_image_id(val_dataset_gt_images_path)
            render_image_id = extract_image_id(rendered_images_path)
            assert val_image_id == render_image_id, \
                f"val and render image ids different {val_image_id} != {render_image_id}, " \
                f"{val_dataset_gt_images_path}, {rendered_images_path}"
            sample = {'id': val_image_id,
                      'gt_path': val_dataset_gt_images_path,
                      'rendered_path': rendered_images_path
                      }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        gt_image = self._read_image(sample['gt_path'])
        rendered_image = self._read_image(sample['rendered_path'])

        output = {
            'id': sample['id'],
            'gt_image': gt_image,
            'rendered_image': rendered_image,
            'gt_path': sample['gt_path'],
            'rendered_path': sample['rendered_path'],
        }

        return output

    def _read_image(self, image_path: str) -> torch.Tensor:
        try:
            with Image.open(image_path) as img:
                image = self.transforms(img)
        except OSError as e:
            logger.error(f'Possibly, image file is broken: {image_path}')
            raise e
        return image


class Scorer:
    def __init__(self, device='cuda', color_shift=False):
        self.device = device
        self.color_shift = color_shift
        self.psnr = PSNRMetric(input_range='tanh', batch_reduction=False).to(self.device)
        self.ssim = SSIM(input_range='tanh', batch_reduction=False).to(self.device)
        self.lpips_alex = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        self.flip_loss_module = FLIPLoss(input_range='tanh', batch_reduction=False, device=self.device)
        self.ms_ssim_score = MS_SSIM(data_range=1, size_average=False, channel=3)
        self.hifreq_psnr_20 = HiFreqPSNRMetric(input_range='tanh', batch_reduction=False, window_ratio=0.2)
        self.hifreq_psnr_30 = HiFreqPSNRMetric(input_range='tanh', batch_reduction=False, window_ratio=0.3)
        self.hifreq_psnr_40 = HiFreqPSNRMetric(input_range='tanh', batch_reduction=False, window_ratio=0.4)

    def color_shift_remove(self, fake_image, real_image):
        b, c, h, w = fake_image.shape
        fake_image = fake_image.reshape(c, -1)
        real_image = real_image.reshape(c, -1)
        fake_image_ext = torch.cat([fake_image[[0]] * fake_image[[1]],
                                    fake_image[[2]] * fake_image[[1]],
                                    fake_image[[2]] * fake_image[[0]],
                                    fake_image.pow(2),
                                    fake_image,
                                    torch.ones_like(fake_image[[0]])
                                    ], axis=0)

        x, _ = torch.lstsq(real_image.T, fake_image_ext.T)
        fake_image = torch.mm(fake_image_ext.T, x[:fake_image_ext.shape[0]]).T
        fake_image = fake_image.reshape(b, c, h, w)
        real_image = real_image.reshape(b, c, h, w)

        return fake_image

    def compute_score(self, images_output, images_target):
        images_output = images_output.to(self.device)
        images_target = images_target.to(self.device)

        if self.color_shift:
            images_output = self.color_shift_remove(images_output, images_target)

        lpips_alex_values = self.lpips_alex(images_output, images_target)
        lpips_alex_values = lpips_alex_values.reshape(-1)

        lpips_vgg_values = self.lpips_vgg(images_output, images_target)
        lpips_vgg_values = lpips_vgg_values.reshape(-1)

        out = {
            'ssim': self.ssim(images_output, images_target),
            # 'ms_ssim': self.ms_ssim_score(images_output.add(1).div(2),images_target.add(1).div(2)),
            'psnr': self.psnr(images_output, images_target),
            # 'hifreq_psnr_20': self.hifreq_psnr_20(images_output, images_target),
            # 'hifreq_psnr_30': self.hifreq_psnr_30(images_output, images_target),
            # 'hifreq_psnr_40': self.hifreq_psnr_40(images_output, images_target),
            'flip': self.flip_loss_module(images_output, images_target),
            'lpips_alex': lpips_alex_values,
            'lpips_vgg': lpips_vgg_values,
            'batch_length': images_output.shape[0],
        }
        return out


def compute_metrics_for_dataloader(dataloader, scorer):
    metrics = []
    for data in tqdm(dataloader, desc='samples', leave=True):
        scores = scorer.compute_score(data['rendered_image'], data['gt_image'])
        for i, id in enumerate(data['id']):
            metrics.append({
                'id': id,
                'ssim': scores['ssim'][i].item(),
                # 'ms_ssim': scores['ms_ssim'][i].item(),
                'psnr': scores['psnr'][i].item(),
                # 'hifreq_psnr_20': scores['hifreq_psnr_20'][i].item(),
                # 'hifreq_psnr_30': scores['hifreq_psnr_30'][i].item(),
                # 'hifreq_psnr_40': scores['hifreq_psnr_40'][i].item(),
                'flip': scores['flip'][i].item(),
                'lpips_alex': scores['lpips_alex'][i].item(),
                'lpips_vgg': scores['lpips_vgg'][i].item(),
                'gt_path': data['gt_path'][i],
                'rendered_path': data['rendered_path'][i],
            })
    return metrics


def extract_model_name_and_iteration(raw_model_name):
    if raw_model_name.split('_')[-1].isdigit():
        table_model_name = '_'.join(raw_model_name.split('_')[:-1])
        table_iteration = raw_model_name.split('_')[-1]
    else:
        table_model_name = raw_model_name
        table_iteration = None

    return table_model_name, table_iteration


def main():
    logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_render_folder', type=str, default=None,
                        help='Path to the folder with rendered images')
    parser.add_argument('--path_to_val_dataset', type=str, default=None,
                        help='Path to val dataset')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size')
    parser.add_argument('--crop_border_size', type=float, default=0.15,
                        help='crop_border_size')
    parser.add_argument('--recompute_all', type=bool, default=False,
                        help='recompute all metrics')
    parser.add_argument('--color_shift', type=bool, default=False,
                        help='color_shift')
    opts = parser.parse_args()
    if opts.crop_border_size == 0.15:
        scores_file_name = f'scores.csv'
    else:
        scores_file_name = f'scores_{int(opts.crop_border_size * 1000)}.csv'

    if opts.color_shift:
        scores_file_name = 'color_shift_' + scores_file_name

    score_df_path = os.path.join(opts.path_to_render_folder, scores_file_name)
    print(score_df_path)
    recompute_all = opts.recompute_all
    scores_df = None
    one_model_score = False

    models_for_scroring = {}
    if os.path.isdir(os.path.join(opts.path_to_render_folder, 'images')):
        models_for_scroring[opts.path_to_render_folder.split('/')[-1]] = opts.path_to_render_folder
        one_model_score = True
    else:
        exist_scores_models_names = []
        try:
            if os.path.exists(score_df_path):
                scores_df = pd.read_csv(score_df_path)
                exist_scores_models_names = list(scores_df['orig_model_name'])
        except:
            print("create new scenes.csv")

        for model_path in sorted(glob(os.path.join(opts.path_to_render_folder, '*', 'images'))):
            model_path = '/'.join(model_path.split('/')[:-1])
            model_name = model_path.split('/')[-1]
            if model_name not in exist_scores_models_names or recompute_all:
                models_for_scroring[model_name] = (model_path)

        print(f'Compute metrics for: {models_for_scroring.keys()} .')

    scores = []
    scorer = Scorer(color_shift=opts.color_shift)

    if opts.path_to_val_dataset[-1] == '/':
        opts.path_to_val_dataset = opts.path_to_val_dataset[:-1]
    dataset_name = opts.path_to_val_dataset.split('/')[-1]

    metric_names = [
        'ssim',
        # 'ms_ssim',
        'psnr',
        # 'hifreq_psnr_20',
        # 'hifreq_psnr_30',
        # 'hifreq_psnr_40',
        'flip',
        'lpips_alex',
        'lpips_vgg',
    ]

    for model_name, model_path in models_for_scroring.items():
        print(f'compute score for: {model_name}')
        try:
            dataset = ScorerDataset(rendered_folder=model_path,
                                    val_dataset_folder=opts.path_to_val_dataset,
                                    crop_border_size=opts.crop_border_size)

            val_dataloader = DataLoader(dataset,
                                        batch_size=opts.batch_size,
                                        sampler=SequentialSampler(dataset)
                                        )

            metrics = compute_metrics_for_dataloader(val_dataloader, scorer)
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(os.path.join(model_path, 'frames_scores.csv'), index=False)
            table_model_name, table_iteration = extract_model_name_and_iteration(model_name)

            mean_metrics = {'orig_model_name': model_name,
                            'dataset_name': '_'.join(dataset_name.split('_')[:-1]),
                            'model_name': table_model_name,
                            'iteration': table_iteration,
                            'resolution': dataset_name.split('_')[-1]
                            }
            for metric in metric_names:
                mean_metrics[metric] = metrics_df[metric].mean()
                mean_metrics[metric + '_std'] = metrics_df[metric].std()
            print(f'{mean_metrics}')
            scores.append(mean_metrics)
        except Exception as e:
            logger.exception(e)
            print(f'scene skipped')

    if one_model_score is False:
        metrics_df = pd.DataFrame(scores)
        if scores_df is not None:
            metrics_df = pd.concat([scores_df, metrics_df])
        metrics_df.to_csv(os.path.join(opts.path_to_render_folder, scores_file_name), index=False)


if __name__ == '__main__':
    main()
