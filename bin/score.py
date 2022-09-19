import argparse
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np

from lib import trainers
from lib.modules import FlowRAFT
from lib.trainers.trainer_base import TrainerBase
from lib.utils.base import seed_freeze
from lib.utils.io import get_config
from lib.utils.data import get_dataloader_from_params

SCORE_MODULE = {
    'ssim': 'ssim',
    'ssim_color_shift': 'ssim_color_shift',
    'psnr': 'psnr',
    'psnr_color_shift': 'psnr_color_shift',
    'lpips': 'lpips',
    'flip': 'flip_loss_module',
    'masked-ssim': 'ssim',
    'masked-psnr': 'psnr',
    'occlusion-ssim': 'ssim',
    'occlusion-psnr': 'psnr',
    'cropped-ssim': 'ssim',
    'cropped-psnr': 'psnr',
    'cropped-occlusion-psnr': 'psnr',
    'cropped-occlusion-ssim': 'ssim',
    'cropped-lpips': 'lpips',
    'cropped-flip': 'flip_loss_module',
}


def square_center_crop(tensor, crop_hw):
    im_hw = tensor.size()[-1]
    crop_pad = math.floor((im_hw - crop_hw) / 2.)
    return tensor[..., crop_pad: crop_pad + crop_hw, crop_pad:crop_pad + crop_hw]


class Scorer:
    def __init__(self,
                 trainer: TrainerBase,
                 metric_name_list: list,
                 crop_size=220,
                 raft_path: str = None):
        self.trainer = trainer
        self.metric_name_list = metric_name_list
        self.use_masked_occlusion = False

        self.crop_size = crop_size

        if len([x for x in self.metric_name_list if x.startswith('occlusion-')]):
            assert raft_path, 'The path to the weight of the RAFT must be specified to use the occlusion scores.'
            self.flow_raft = FlowRAFT(raft_path)
            self.use_masked_occlusion = True

    @staticmethod
    def aggregate_metrics(computed_metrics) -> dict:
        res = {name: np.array([dic[name] for dic in computed_metrics])
               for name in computed_metrics[0]}
        bs = res.pop('batch_length')
        out = {name: np.sum(res[name] * bs) / bs.sum() for name in res.keys()}
        return out

    @torch.no_grad()
    def compute_metrics_per_batch(self,
                                  result_images: Dict,
                                  ground_truth: Dict,
                                  exclude_repeated_frames=True):
        images_output: torch.Tensor = result_images['images']
        opacity: Optional[torch.Tensor] = result_images.get('opacity')
        mask: Optional[torch.BoolTensor] = opacity > 0 if opacity is not None else None

        images_target = ground_truth['novel']['image'].to(images_output.device)
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])
        images_reference = ground_truth['reference']['image'].to(images_output.device)
        images_reference = images_reference.contiguous().view(-1, *images_reference.shape[-3:])

        if exclude_repeated_frames:
            degenerate_mask = ~torch.eq(ground_truth['novel']['timestamp'],
                                        ground_truth['reference']['timestamp']
                                        ).view(-1)

            images_target = images_target[degenerate_mask]
            images_reference = images_reference[degenerate_mask]
            images_output = images_output[degenerate_mask]
            mask = mask[degenerate_mask] if mask is not None else None

        out = dict()
        if images_output.shape[0] == 0:
            out['batch_length'] = images_output.shape[0]
            return out

        mask_occlusion = None
        if self.use_masked_occlusion:
            res_flow = torch.abs(self.flow_raft.get_flow_displacement(images_target, images_reference))
            mask_occlusion = res_flow.gt(1).any(dim=1, keepdim=True)

            occlusion_area = mask_occlusion.float().mean(dim=[-1, -2, -3])

            occlusion_area_mask = ~(occlusion_area == 0)
            images_target = images_target[occlusion_area_mask]
            images_output = images_output[occlusion_area_mask]
            mask = mask[occlusion_area_mask] if mask is not None else None
            mask_occlusion = mask_occlusion[occlusion_area_mask]

            out['occlusion_area'] = occlusion_area.mean().item()

        if images_output.shape[0] == 0:
            out['batch_length'] = images_output.shape[0]
            return out

        for name in self.metric_name_list:
            scorer = getattr(self.trainer, SCORE_MODULE[name], None)
            if scorer is None:
                print(f'Warning: trainer does not have the scorer {SCORE_MODULE[name]}')
                continue
            if name.startswith('masked-') and mask is not None:
                mask_for_score = mask
            elif name.find('occlusion') != -1 and mask_occlusion is not None:
                mask_for_score = mask_occlusion
            else:
                mask_for_score = None

            if name.startswith('cropped-'):
                prediction = square_center_crop(images_output, self.crop_size)
                gt = square_center_crop(images_target, self.crop_size)
                if mask_for_score is not None:
                    mask_for_score = square_center_crop(mask_for_score, self.crop_size)
                    assert gt.shape[-2:] == mask_for_score.shape[-2:]
            else:
                prediction = images_output
                gt = images_target

            if mask_for_score is not None:
                score = scorer(prediction, gt, mask=mask_for_score)
            else:
                score = scorer(prediction, gt)

            out[name] = score.mean().item()

        out['batch_length'] = images_output.shape[0]
        mask_occlusion = square_center_crop(mask_occlusion, self.crop_size)
        out['cropped_occlusion_area'] = mask_occlusion.float().mean().item()

        return out

    @torch.no_grad()
    def evaluate(self, dataloader, num_repeat=1, without_aggregation=False):
        """
        Compute metrics for validation axis.
        :param dataloader: torch.utils.axis.Dataloader
        :param num_epochs: int number of epochs to compute metric
        :return: dict
        """
        metrics = []
        self.trainer.eval()
        for _ in range(num_repeat):
            for data in dataloader:
                result = self.trainer(*self.trainer._unpack_data(data, mode='eval'))
                result = self.compute_metrics_per_batch(result, data)
                if result['batch_length'] > 0:
                    metrics.append(result)

        if not without_aggregation:
            result = self.aggregate_metrics(metrics)
        else:
            result = metrics
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-config', type=str, default=None, help='Path to the config file with dataset spec.')
    parser.add_argument('--output-path', type=str, default='outputs', help="outputs path")
    parser.add_argument('--iteration', type=int, default=None, help="iteration")
    parser.add_argument('--crop-size', type=int, default=220, help="crop-size")
    parser.add_argument('--seed', type=int, default=None, help="seed")
    parser.add_argument('--not-aggregate', action="store_true", help="aggregate metrics over all batches")
    parser.add_argument('--use-custom-model', action="store_true", help="use model config specified in data config")
    parser.add_argument('--num-validation-repeat', type=int, default=10, help="iteration for resuming")
    parser.add_argument('--device', '-d', type=str, default='cuda', help="type of the device")
    parser.add_argument('--model-names', type=str, default=[], nargs='+',
                        help='Names of the models, separated with spaces. E.g., bedrooms_multiscale_0degree')
    parser.add_argument('--user-dir', type=str, default=None,
                        help="Path to the user's directory with models. E.g., "
                             "/group-volume/orc_srr/multimodal/PsinaVolumes/dkorzhenkov/outputs")
    parser.add_argument('--raft-path', type=str,
                        default="/group-volume/orc_srr/multimodal/pretrained/raft/raft-sintel.pth",
                        help="Path to the weights of raft model")
    parser.add_argument('--metric-save-path', type=str,
                        default="/gpfs-volume/psina_metrics_result.pkl",
                        help="Path where save dict")
    opts = parser.parse_args()
    seed_freeze(base_seed=opts.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(opts.device)

    metric_name_list = ['psnr', 'ssim', 'lpips', 'flip',
                        'masked-psnr', 'masked-ssim',
                        'cropped-psnr', 'cropped-ssim', 'cropped-lpips', 'cropped-flip',
                        'cropped-occlusion-psnr', 'cropped-occlusion-ssim',
                        'occlusion-psnr', 'occlusion-ssim']

    models_settings = defaultdict(dict)
    if opts.data_config is not None:
        data_config = get_config(opts.data_config)

    for model_name in opts.model_names:
        setting = models_settings[model_name]
        setting['config'] = get_config(os.path.join(opts.user_dir, model_name, f'{model_name}.yaml'))
        setting['checkpoints_dir'] = os.path.join(opts.user_dir, model_name, 'checkpoints')

    result_dict = {}
    val_dataloader = None
    for model_name, setting in tqdm(models_settings.items(), desc='models'):
        os.makedirs(os.path.join(opts.output_path, model_name), exist_ok=True)
        config = setting['config']
        output_directory = os.path.join(opts.output_path, 'outputs', model_name)
        if opts.data_config is None:
            data_config = config

        if val_dataloader is None or opts.data_config is None:
            # TODO add support of batch > 1 for area control
            data_config['dataloaders']['val']['params']['batch_size'] = 1
            val_dataloader = get_dataloader_from_params(data_config, 'val')
        model_config = config if not opts.use_custom_model else data_config
        trainer = getattr(trainers, model_config['trainer'])(model_config, eval_mode=True, device=device)
        iteration = trainer.resume(setting['checkpoints_dir'], opts.iteration)
        scorer = Scorer(trainer, metric_name_list,
                        raft_path=opts.raft_path, crop_size=opts.crop_size)
        metrics_res = {}
        metrics = scorer.evaluate(val_dataloader,
                                  without_aggregation=opts.not_aggregate,
                                  num_repeat=opts.num_validation_repeat)

        if not opts.not_aggregate:
            for metric_name, value in metrics.items():
                metrics_res[metric_name] = value
            print(f'Result metrics on {iteration} iteration for model {model_name}:')
            print("\n".join([f"{kk}: {vv}" for kk, vv in metrics_res.items()]))
        else:
            # List[dict]
            metrics_res = metrics
            print(f"Saved at {os.path.join(output_directory, f'metrics_{iteration}.pkl')}")

        os.makedirs(output_directory, exist_ok=True)
        with open(os.path.join(output_directory, f'metrics_{iteration}.pkl'), 'wb') as fwb:
            pickle.dump(metrics_res, fwb)
        result_dict[model_name] = metrics_res

    with open(opts.metric_save_path, 'wb') as fwb:
        pickle.dump(result_dict, fwb)


if __name__ == '__main__':
    main()
