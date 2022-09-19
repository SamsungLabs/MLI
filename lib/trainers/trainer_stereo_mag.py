__all__ = ['TrainerStereoMagnification']

import logging
import math
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from lib.modules.losses import PerceptualLoss, PSNRMetric, SSIM
from lib.modules.flip_loss import FLIPLoss
from lib.utils.visualise import plot_square_grid, grayscale_to_cmap
from lib.utils.augmentations import construct_kornia_aug
from lib.modules.cameras import CameraMultiple
from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class TrainerStereoMagnification(TrainerBase):
    def __init__(self, params, eval_mode=False, device='cuda'):
        super().__init__(params, eval_mode=eval_mode, device=device)

        self.augs_with_novel = None
        self.augs_without_novel = None

        if self.params.get('augmentations', None) is not None:
            same_on_batch = self.params['augmentations'].get('same_on_batch', True)

            augs_with_novel_params = self.params['augmentations'].get('augs_with_novel', None)
            if augs_with_novel_params is not None:
                self.augs_with_novel = construct_kornia_aug(
                    augs_with_novel_params,
                    same_on_batch=same_on_batch,
                    keepdim=True,
                    random_apply=1
                )

            augs_without_novel_params = self.params['augmentations'].get('augs_without_novel', None)
            if augs_without_novel_params is not None:
                self.augs_without_novel = construct_kornia_aug(
                    augs_without_novel_params,
                    same_on_batch=same_on_batch,
                    keepdim=True,
                    random_apply=1
                )

    def _unpack_data(self,
                     data: dict,
                     mode: str = 'eval',
                     ):
        # B x n_reference x n_source x ,,,
        source_images: torch.Tensor = data['initial']['image'].to(self.device)
        source_cameras = CameraMultiple(extrinsics=data['initial']['extrinsics'].to(self.device),
                                        intrinsics=data['initial']['intrinsics'].to(self.device),
                                        )

        # B x n_reference x 1 x ...
        reference_images: torch.Tensor = data['reference']['image'].to(self.device)
        reference_cameras = CameraMultiple(extrinsics=data['reference']['extrinsics'].to(self.device),
                                           intrinsics=data['reference']['intrinsics'].to(self.device),
                                           )
        reference_pixel_coords: Optional[torch.Tensor] = None

        # B x n_reference x n_novel x ...
        novel_cameras = CameraMultiple(extrinsics=data['novel']['extrinsics'].to(self.device),
                                       intrinsics=data['novel']['intrinsics'].to(self.device),
                                       )
        novel_pixel_coords: Optional[torch.Tensor] = None

        if mode == 'train':
            novel_images: torch.Tensor = data['novel']['image'].to(self.device)

            if self.params.get('augmentations', None) is not None:
                batch, n_ref, n_novel, C, H, W = novel_images.shape
                source_images = source_images.reshape(-1, C, H, W)
                source_images = (source_images + 1) / 2

                if self.augs_without_novel is not None:
                    source_images = self.augs_without_novel(source_images)

                if self.augs_with_novel is not None:
                    novel_images = novel_images.reshape(-1, C, H, W)
                    novel_images = (novel_images + 1) / 2
                    aug_images = self.augs_with_novel(torch.cat([novel_images, source_images], axis=0))
                    novel_images = aug_images[:batch * n_novel]
                    source_images = aug_images[batch * n_novel:]
                    novel_images = novel_images.reshape(batch, n_ref, n_novel, C, H, W)
                    novel_images = novel_images * 2 - 1

                source_images = source_images.reshape(batch, n_ref, -1, C, H, W)
                source_images = source_images * 2 - 1

            return (
                reference_images,
                source_images,
                novel_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                reference_pixel_coords,
                novel_pixel_coords,
            )

        elif mode == 'eval':
            return (
                reference_images,
                source_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                reference_pixel_coords,
                novel_pixel_coords,
                data.get('proxy_geometry'),
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def _init_auxiliary_modules(self):
        super()._init_auxiliary_modules()
        if self.params['weights']['generator'].get('perceptual', 0.) > 0:
            self.perceptual_loss = PerceptualLoss(input_range='tanh').to(self.device)

        self.psnr = PSNRMetric(input_range='tanh').to(self.device)
        self.ssim = SSIM(input_range='tanh').to(self.device)
        if LPIPS_AVAILABLE:
            # see https://github.com/richzhang/PerceptualSimilarity for details
            self.lpips = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips = None
        self.flip_loss_module = FLIPLoss(input_range='tanh', device=self.device)

    def update_step(self,
                    iteration,
                    reference_images,
                    source_images,
                    novel_images,
                    reference_cameras,
                    source_cameras,
                    novel_cameras,
                    reference_pixel_coords=None,
                    novel_pixel_coords=None,
                    ):
        gen_output = self.gen(reference_images,
                              source_images,
                              reference_cameras,
                              source_cameras,
                              novel_cameras,
                              reference_pixel_coords=reference_pixel_coords,
                              novel_pixel_coords=novel_pixel_coords,
                              )
        predicted_images = gen_output['novel_image']

        self.losses['generator']['l1'] = F.l1_loss(predicted_images, novel_images)
        if self.params['weights']['generator'].get('perceptual', 0.) > 0:
            self.losses['generator']['perceptual'] = self.perceptual_loss(
                predicted_images.contiguous().view(-1, *predicted_images.shape[-3:]),
                novel_images.contiguous().view(-1, *novel_images.shape[-3:])
            )

        nan_forward = self._aggregate_losses('generator')
        if nan_forward:
            return 'NaN in Forward'

        self.backward('generator')
        if iteration % self.gradient_update_period == 0:
            nan_backward = False
            for name, p in self.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any().item():
                        nan_backward = True
                        break
                    p.grad /= self.gradient_update_period
            if nan_backward:
                return f'NaN in Backward: {name}'
            self.optimizers['generator'].step()

        return None

    def forward(self,
                reference_images,
                source_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                reference_pixel_coords=None,
                novel_pixel_coords=None,
                proxy_geometry=None,
                ):
        generator = self.gen_ema if self.ema_inference else self.gen

        if proxy_geometry is not None:
            proxy_geometry['novel_pixel_coords'] = novel_pixel_coords
            gen_output = generator.render_with_proxy_geometry(
                novel_cameras,
                proxy_geometry,
            )
        else:
            gen_output = generator(reference_images,
                                  source_images,
                                  reference_cameras,
                                  source_cameras,
                                  novel_cameras,
                                  reference_pixel_coords=reference_pixel_coords,
                                  novel_pixel_coords=novel_pixel_coords,
                                  )
        novel_images = gen_output['novel_image']
        novel_opacity = gen_output['novel_opacity']
        mpi = gen_output['mpi']

        if not self.training:
            # a strange workaround instead of in-place detach_() op because of an error:
            # "RuntimeError: Can't detach views in-place. Use detach() instead",
            # observed in some pipelines
            novel_images = novel_images.detach()
            novel_opacity = novel_opacity.detach()

        out = {'images': novel_images.contiguous().view(-1, *novel_images.shape[-3:]),
               'opacity': novel_opacity.contiguous().view(-1, *novel_opacity.shape[-3:]),
               'mpi': mpi.contiguous().view(-1, *mpi.shape[-4:]),
               }

        if not self.training:
            out['proxy_geometry'] = {'mpi': mpi.detach()}

        return out

    @torch.no_grad()
    def inference(self, data) -> dict:
        training_status = self.training
        self.eval()
        result = self.forward(*self._unpack_data(data, 'eval'))
        self.train(training_status)
        return result

    def _compute_visualisation(self, output: dict, data: dict) -> List[Image.Image]:
        images_output = output['images'].data.add(1).div(2).to('cpu').numpy().transpose((0, 2, 3, 1))
        opacity = output['opacity'].data.to('cpu').numpy()[:, 0, :, :]
        opacity = grayscale_to_cmap(opacity)

        images_target = data['novel']['image']
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])
        images_target = images_target.data.add(1).div(2).to('cpu').numpy().transpose((0, 2, 3, 1))

        mse = np.power(images_output - images_target, 2).sum(-1, keepdims=True)

        mpi = output.get('mpi')
        if mpi is not None:
            mpi = mpi.data
            num_layers_to_show = 3  # show only this number of layers
            step = math.ceil((mpi.shape[-4] - 2) / (num_layers_to_show - 2))
            # we show the nearest plane (0), the farthest one (-1) and others with equal step,
            # starting from the index (1)
            layers_indices = [0] + list(range(1, mpi.shape[-4] - 1, step)) + [mpi.shape[-4] - 1]
            mpi = mpi[..., layers_indices, :, :, :]
            if getattr(self.gen, 'use_opacity', True):
                mpi_rgb, mpi_alpha = mpi[..., :-1, :, :], mpi[..., -1:, :, :]
            else:
                mpi_rgb, mpi_alpha = mpi, None
            if mpi_rgb.shape[-3] == 3:  # RGB channels
                mpi_rgb = mpi_rgb.add(1).div(2).cpu().numpy().transpose((1, 0, 3, 4, 2))
            else:
                mpi_rgb = None
            if mpi_alpha is not None:
                mpi_alpha = mpi_alpha[:, :, 0, :, :].transpose(0, 1).cpu().numpy()
                mpi_alpha = grayscale_to_cmap(mpi_alpha)
        else:
            mpi_rgb = mpi_alpha = None

        list_of_vis = [images_output,
                       images_target,
                       opacity,
                       np.repeat(mse, 3, axis=3),
                       ]
        if mpi_rgb is not None:
            list_of_vis.extend(mpi_rgb)
        if mpi_alpha is not None:
            list_of_vis.extend(mpi_alpha)
        images_out = np.concatenate(list_of_vis, axis=2) * 255

        pil_output_images = []
        for image in images_out:
            image = np.clip(image, 0., 255.).astype(np.uint8)
            if image.shape[1] % 2 != 0:
                image = image[:, :-1]
            pil_output_images.append(Image.fromarray(image))

        return pil_output_images

    def _aggregate_visualisation(self, visualisations: List[Image.Image]) -> Image.Image:
        return plot_square_grid(visualisations, 1, len(visualisations), padding=5)

    def _compute_metrics(self, output: dict, data: dict) -> dict:
        images_output = output['images']

        images_target = data['novel']['image'].to(self.device)
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])

        out = {
            'ssim': self.ssim(images_output, images_target),
            'psnr': self.psnr(images_output, images_target),
            'flip': self.flip_loss_module(images_output, images_target),
            'batch_length': images_output.shape[0],
        }
        if self.lpips is not None:
            out['lpips'] = self.lpips(images_output, images_target).mean()

        return out

    def _aggregate_metrics(self, metrics: List[dict]) -> dict:
        ssim, psnr, lpips_value, flip, batch_length = [], [], [], [], []
        for metrics_per_batch in metrics:
            ssim.append(metrics_per_batch['ssim'].item())
            psnr.append(metrics_per_batch['psnr'].item())
            flip.append(metrics_per_batch['flip'].item())
            batch_length.append(metrics_per_batch['batch_length'])
            if 'lpips' in metrics_per_batch:
                lpips_value.append(metrics_per_batch['lpips'].item())

        ssim, psnr, flip, batch_length = np.array(ssim), np.array(psnr), np.array(flip), np.array(batch_length)
        lpips_value = np.array(lpips_value) if lpips_value else None
        out = {
            'psnr': np.sum(psnr * batch_length) / batch_length.sum(),
            'ssim': np.sum(ssim * batch_length) / batch_length.sum(),
            'flip': np.sum(flip * batch_length) / batch_length.sum(),
        }
        if lpips_value is not None:
            out['lpips'] = np.sum(lpips_value * batch_length) / batch_length.sum()
        return out
