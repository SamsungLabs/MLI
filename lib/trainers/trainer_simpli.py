__all__ = ['TrainerSIMPLI']

import logging
import math
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..utils.base import min_max_scale

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from lib.modules.losses import PerceptualLoss, PSNRMetric, SSIM, photometric_masked_loss, \
    hinge_depth_regularizer, tv_loss
from lib.modules.flip_loss import FLIPLoss
from lib.utils.visualise import plot_square_grid, grayscale_to_cmap
from lib.utils.augmentations import construct_kornia_aug
from lib.modules.cameras.utils import average_extrinsics, get_median_extrinsic
from lib.modules.cameras import CameraMultiple
from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class TrainerSIMPLI(TrainerBase):
    def __init__(self, params, eval_mode=False, device='cuda'):
        super().__init__(params, eval_mode=eval_mode, device=device)

        self.turn_off_masks = self.params.get('turn_off_masks', False)
        self.render_every_novel_individually = self.params.get('render_every_novel_individually', False)
        if self.params['weights']['generator'].get('uncertain_mixture_likelihood', 0.) > 0:
            self.weight_uncertain_mixture_likelihood = \
                self.params['weights']['generator']['uncertain_mixture_likelihood']
            self.params['weights']['generator']['uncertain_mixture_likelihood'] = 1

        self.augs_with_novel = None
        self.augs_without_novel = None
        self.augs_mix = None

        if self.params.get('augmentations', None) is not None:
            same_on_batch = self.params['augmentations'].get('same_on_batch', True)

            augs_with_novel_params = self.params['augmentations'].get('augs_with_novel', None)
            if augs_with_novel_params is not None:
                self.augs_with_novel = construct_kornia_aug(
                    augs_with_novel_params,
                    same_on_batch=True,
                    keepdim=True,
                    random_apply=3
                )

            augs_without_novel_params = self.params['augmentations'].get('augs_without_novel', None)
            if augs_without_novel_params is not None:
                self.augs_without_novel = construct_kornia_aug(
                    augs_without_novel_params,
                    same_on_batch=same_on_batch,
                    keepdim=True,
                    random_apply=1
                )

            augs_mix_params = self.params['augmentations'].get('augs_mix', None)
            if augs_mix_params is not None:
                self.augs_mix = construct_kornia_aug(
                    augs_mix_params,
                    same_on_batch=False,
                    keepdim=True,
                    random_apply=False
                )

        self.ref_camera_scale = self.params.get('eval_mpi_scale', 1)

    def _unpack_data(self,
                     data: dict,
                     mode: str = 'eval',
                     ):

        # print('data[initial][image]', data['initial']['image'].shape)
        source_images: torch.Tensor = data['initial']['image'].squeeze(1).to(self.device)
        source_images_wo_aug = None

        source_images_size = tuple(source_images.shape[-2:])
        source_cameras = CameraMultiple(extrinsics=data['initial']['extrinsics'].to(self.device),
                                        intrinsics=data['initial']['intrinsics'].to(self.device),
                                        images_sizes=source_images_size,
                                        )

        novel_images_sizes = self.params.get('eval_novel_resolution', None)
        if novel_images_sizes is None or mode == 'train':
            novel_images_sizes = tuple(source_images_size)

        novel_cameras = CameraMultiple(extrinsics=data['novel']['extrinsics'].to(self.device),
                                       intrinsics=data['novel']['intrinsics'].to(self.device),
                                       images_sizes=novel_images_sizes,
                                       )
        source_extrinsics = source_cameras.get_extrinsics()
        source_intrinsics = source_cameras.get_intrinsics()

        average_camera_type = self.params.get('average_camera_type', 'virtual')

        if average_camera_type == 'virtual':
            reference_extrinsic = average_extrinsics(source_extrinsics)
        elif average_camera_type == 'real':
            reference_extrinsic = get_median_extrinsic(source_extrinsics)
        elif average_camera_type == 'ref':
            reference_cameras = CameraMultiple(extrinsics=data['reference']['extrinsics'][:, :, [0]].to(self.device),
                                               intrinsics=data['reference']['intrinsics'][:, :, [0]].to(self.device),
                                               images_sizes=source_images_size,
                                               )
            reference_extrinsic = reference_cameras.get_extrinsics()[..., [0], :, :]

        reference_images_sizes = self.params.get('eval_mpi_resolution', None)
        if reference_images_sizes is None or mode == 'train':
            reference_images_sizes = tuple(source_images_size)
            if self.ref_camera_scale != 1:
                reference_images_sizes = [int(reference_images_sizes[0] * self.ref_camera_scale),
                                          int(reference_images_sizes[1] * self.ref_camera_scale)]
                reference_images_sizes = [reference_images_sizes[0] // 16 * 16, reference_images_sizes[1] // 16 * 16]

        ref_intrinsic = source_intrinsics[0, :, :1].clone()
        if self.ref_camera_scale != 1 and mode == 'eval':
            ref_intrinsic[:, :, :2, :2] = ref_intrinsic[:, :, :2, :2] / self.ref_camera_scale

        if self.render_every_novel_individually:
            reference_cameras = CameraMultiple(extrinsics=data['novel']['extrinsics'].to(self.device),
                                               intrinsics=ref_intrinsic.expand_as(source_intrinsics[:, :, :1]),
                                               images_sizes=reference_images_sizes,
                                               )
        else:
            reference_cameras = CameraMultiple(extrinsics=reference_extrinsic,
                                               intrinsics=ref_intrinsic.expand_as(source_intrinsics[:, :, :1]),
                                               images_sizes=reference_images_sizes,
                                               )

        # assert novel_cameras.cameras_shape[2] == 1, "Don't support several novel cams"
        novel_pixel_coords: Optional[torch.Tensor] = None

        if mode == 'train':
            novel_images: torch.Tensor = data['novel']['image'].squeeze(1).to(self.device)

            if self.params.get('augmentations', None) is not None:
                if self.augs_without_novel or self.augs_mix is not None:
                    source_images_wo_aug = torch.clone(source_images)

                batch, n_novel, C, H, W = novel_images.shape
                source_images = source_images.reshape(-1, C, H, W)
                source_images = (source_images + 1) / 2

                if self.augs_without_novel is not None:
                    source_images = self.augs_without_novel(source_images)

                if self.augs_mix is not None:
                    source_images, _ = self.augs_mix(source_images,
                                                     torch.tensor(list(range(source_images.shape[0]))).to(self.device))

                if self.augs_with_novel is not None:
                    novel_images = novel_images.reshape(-1, C, H, W)
                    novel_images = (novel_images + 1) / 2
                    aug_images = self.augs_with_novel(torch.cat([novel_images, source_images], axis=0))
                    novel_images = aug_images[:batch * n_novel]
                    source_images = aug_images[batch * n_novel:]
                    novel_images = novel_images.reshape(batch, n_novel, C, H, W)
                    novel_images = novel_images * 2 - 1

                source_images = source_images.reshape(batch, -1, C, H, W)
                source_images = source_images * 2 - 1

            return (
                source_images,
                novel_images,
                source_cameras,
                novel_cameras,
                reference_cameras,
                source_images_wo_aug,
            )

        elif mode == 'eval':
            return (
                source_images,
                source_cameras,
                novel_cameras,
                reference_cameras,
                novel_pixel_coords,
                data.get('proxy_geometry')
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def _init_auxiliary_modules(self, eval_mode=False):
        super()._init_auxiliary_modules(eval_mode)

        if not eval_mode:
            if self.params['weights']['generator'].get('perceptual', 0.) > 0:
                self.perceptual_loss = PerceptualLoss(input_range='tanh').to(self.device)
                self.perceptual_loss.requires_grad_(False)

            self.psnr = PSNRMetric(input_range='tanh').to(self.device)
            self.ssim = SSIM(input_range='tanh').to(self.device)
            if LPIPS_AVAILABLE:
                # see https://github.com/richzhang/PerceptualSimilarity for details
                self.lpips = lpips.LPIPS(net='alex').to(self.device)
                self.lpips.requires_grad_(False)
            else:
                self.lpips = None
            self.flip_loss_module = FLIPLoss(input_range='tanh', device=self.device)

    def update_step(self,
                    iteration,
                    source_images,
                    novel_images,
                    source_cameras,
                    novel_cameras,
                    reference_cameras,
                    source_images_wo_aug,
                    ):

        batch, n_novel, C, _, _ = novel_images.shape
        # H, W = reference_cameras.images_size

        gen_output = self.gen(source_images,
                              source_cameras,
                              reference_cameras
                              )

        mpi = gen_output['mpi']
        # mpi = mpi.contiguous().view(*mpi.shape[:-2], H, W)
        proxy_geometry = {'mpi': mpi}
        proxy_geometry['mli'] = gen_output.get('mli')
        proxy_geometry['layered_depth'] = gen_output.get('layered_depth')

        reproject_output = self.gen(
            novel_cameras,
            proxy_geometry,
            action='render_with_proxy_geometry',
        )

        predicted_rgb = reproject_output['rgb']
        predicted_rgb = predicted_rgb.contiguous().view(-1, *predicted_rgb.shape[-3:])
        novel_images = novel_images.contiguous().view(-1, *novel_images.shape[-3:])
        source_images = source_images.contiguous().view(-1, *novel_images.shape[-3:])
        if source_images_wo_aug is not None:
            source_images_wo_aug = source_images_wo_aug.contiguous().view(-1, *novel_images.shape[-3:])

        predicted_opacity = reproject_output['opacity']
        # predicted_opacity_weight = reproject_output['weight']

        if self.turn_off_masks:
            mask = None
        else:
            mask = predicted_opacity.reshape(-1, *predicted_opacity.shape[-3:])

        if self.params['weights']['generator'].get('distortion', 0.) > 0:
            # self.losses['generator']['distortion'] = (reproject_output['weight'] ** 2).sum() / 3
            self.losses['generator']['distortion'] = reproject_output['opacity'].sum()
            # self.losses['generator']['distortion'] = 0
            _, _, num_layers, _, _, _ = reproject_output['weight'].shape
            for i in range(num_layers):
                depth_delta = torch.abs(reproject_output['depths'][:, [i]] - reproject_output['depths'])
                if i < num_layers - 1:
                    self.losses['generator']['distortion'] += (reproject_output['weight'][:, :, [i]] \
                                                               * reproject_output['weight'][:, :, i + 1:] \
                                                               * depth_delta[None, :, i + 1:, None,
                                                                 None, None]).sum()
                if i > 0:
                    self.losses['generator']['distortion'] += (reproject_output['weight'][:, :, [i]] \
                                                               * reproject_output['weight'][:, :, :i] \
                                                               * depth_delta[None, :, :i, None, None,
                                                                 None]).sum()

        self.losses['generator']['l1'] = photometric_masked_loss(predicted_rgb, novel_images,
                                                                 mask=mask, mode='l1')

        if 'layered_depth' in gen_output:
            self.losses['generator']['smooth_layers'] = tv_loss(gen_output['layered_depth'])
            if self.params['weights']['generator'].get('disp_smooth_layers', 0.) > 0:
                max_depth = gen_output['layered_depth'].max()
                min_depth = gen_output['layered_depth'].min() + 10 ** -5
                layered_dis = (gen_output['layered_depth'].reciprocal() - 1 / max_depth) / (
                            1 / min_depth - 1 / max_depth)
                self.losses['generator']['disp_smooth_layers'] = tv_loss(layered_dis)
            self.losses['generator']['depth_reg'] = hinge_depth_regularizer(gen_output['layered_depth'])

        if self.params['weights']['generator'].get('perceptual', 0.) > 0:
            self.losses['generator']['perceptual'] = self.perceptual_loss(predicted_rgb, novel_images,
                                                                          mask=mask)

        if 'intermediate_error' in gen_output:
            self.losses['generator']['intermediate_loss'] = 0
            for i, intermediate_error in enumerate(gen_output['intermediate_error']):
                self.losses['generator']['intermediate_loss_' + str(i)] = intermediate_error.abs().mean()
                self.losses['generator']['intermediate_loss'] += self.losses['generator']['intermediate_loss_' + str(i)]

        if 'intermediate_sources' in gen_output:
            self.losses['generator']['intermediate_loss_l1'] = 0
            sources_for_error = source_images
            if source_images_wo_aug is not None:
                sources_for_error = source_images_wo_aug

            for i, intermediate_sources in enumerate(gen_output['intermediate_sources']):
                self.losses['generator']['intermediate_loss_' + str(i) + '_l1'] = photometric_masked_loss(
                    intermediate_sources.view(-1, *intermediate_sources.shape[-3:]),
                    sources_for_error,
                    mode='l1',
                    allow_different_size_of_gt_and_pred=True
                )

        if 'intermediate_errors_dis' in gen_output:
            self.losses['generator']['intermediate_dis_loss'] = 0
            for i, intermediate_error in enumerate(gen_output['intermediate_errors_dis']):
                self.losses['generator']['intermediate_dis_loss_' + str(i)] = intermediate_error.abs().mean()
                self.losses['generator']['intermediate_dis_loss'] += self.losses['generator']['intermediate_dis_loss_' + str(i)]

        # if self.params['weights']['generator'].get('opacity_sparsity', 0.) != 0:
        #
        #     self.losses['generator']['opacity_sparsity'] = self._compute_sparsity_statistic(
        #         weights=predicted_opacity_weight.reshape(batch, -1, *predicted_opacity_weight.shape[-3:]).squeeze(2),
        #         p=2,
        #         criterion=1,
        #     )

        nan_forward = self._aggregate_losses('generator', iteration=iteration)
        if nan_forward:
            logger.error('NaN in Forward, iteration %d', iteration)
            return 'NaN in Forward'

        self.backward('generator')
        if iteration % self.gradient_update_period == 0:
            nan_backward = False
            name = None
            for name, p in self.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any().item():
                        nan_backward = True
                        break
                    p.grad /= self.gradient_update_period
            if nan_backward:
                logger.error('NaN in Backward, parameter %s, iteration %d', name, iteration)
                return f'NaN in Backward: {name}'
            self.save_gradient_norms_module('gen')
            self.optimizer_step('generator')

        return None

    def forward(self,
                source_images,
                source_cameras,
                novel_cameras,
                reference_cameras,
                novel_pixel_coords=None,
                proxy_geometry=None,
                ):

        if proxy_geometry is not None:
            proxy_geometry['novel_pixel_coords'] = novel_pixel_coords
            gen_output = self.gen(
                novel_cameras,
                proxy_geometry,
                action='render_with_proxy_geometry',
            )
        else:
            # TODO: add custom result resolution
            batch, n_novel, C, _, _ = source_images.shape
            # H, W = reference_cameras.images_size
            # patch_size = self.params['patch_size']

            gen_output = self.gen(source_images,
                                  source_cameras,
                                  reference_cameras
                                  )

            # #  rays_rgba: B x 1 x n_steps x RGBA x n_rays ->
            # #  mpi: postprocessed PS
            mpi = gen_output['mpi']
            # mpi = mpi.contiguous().view(*mpi.shape[:-2], H, W)
            mpi = mpi.detach()
            proxy_geometry = {'mpi': mpi}
            proxy_geometry['mli'] = gen_output.get('mli')
            if 'mli_d' in gen_output:
                proxy_geometry['mli_d'] = gen_output.get('mli_d', None)
            proxy_geometry['layered_depth'] = gen_output.get('layered_depth', None)
            # #
            gen_output = self.gen(
                novel_cameras,
                proxy_geometry,
                action='render_with_proxy_geometry',
            )

        novel_images = gen_output['rgb']
        novel_opacity = gen_output['opacity']
        novel_depth = gen_output.get('depth', None)

        novel_images = novel_images.detach()
        novel_opacity = novel_opacity.detach()

        if self.render_every_novel_individually:
            out = {'images': novel_images.contiguous().view(-1, *novel_images.shape[-3:]),
                   'opacity': novel_opacity.contiguous().view(-1, *novel_opacity.shape[-3:]),
                   }
        else:
            out = {'images': novel_images.contiguous().view(-1, *novel_images.shape[-3:]),
                   'opacity': novel_opacity.contiguous().view(-1, *novel_opacity.shape[-3:]),
                   'mpi': proxy_geometry['mpi'].contiguous().view(-1, *proxy_geometry['mpi'].shape[-4:]),
                   'proxy_geometry': proxy_geometry
                   }
            if novel_depth is not None:
                out['depth'] = novel_depth.contiguous().view(-1, *novel_depth.shape[-3:])

        return out

    @torch.no_grad()
    def inference(self, data) -> dict:
        training_status = self.training
        self.eval()
        result = self.forward(*self._unpack_data(data, 'eval'))
        self.train(training_status)
        return result

    def _compute_visualisation(self, output: dict, data: dict) -> List[Image.Image]:

        images_output = output['images'].add(1).div(2).cpu().numpy().transpose((0, 2, 3, 1))
        _, img_h, img_w, _, = images_output.shape
        opacity = output['opacity'].cpu().numpy()[:, 0, :, :]
        opacity = grayscale_to_cmap(opacity)

        proxy_geometry = output.get('proxy_geometry')
        depth = None
        if proxy_geometry is not None:
            depth = proxy_geometry.get('layered_depth')
            if depth is not None:
                depth = F.interpolate(depth, size=[img_h, img_w], mode='bicubic')
                depth = min_max_scale(depth, dim=[2, 3], mask=output['opacity'].data.gt(0.).expand_as(depth))
                depth = depth.data.permute(1, 0, 2, 3).to('cpu').numpy()
                depth = grayscale_to_cmap(depth, cmap='magma')

        images_target = data['novel']['image']
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])
        images_target = images_target.add(1).div(2).cpu().numpy().transpose((0, 2, 3, 1))

        mse = np.power(images_output - images_target, 2).sum(-1, keepdims=True)

        rays_rgba = output.get('mpi')
        if rays_rgba is not None:
            # num_layers_to_show = min(rays_rgba.shape[-4], 6)  # show only this number of layers
            num_layers_to_show = 6
            if rays_rgba.shape[-4] > num_layers_to_show:
                step = math.ceil((rays_rgba.shape[-4] - 2) / (num_layers_to_show - 2))
            else:
                step = 1
            # we show the nearest plane (0), the farthest one (-1) and others with equal step,
            # starting from the index (1)
            layers_indices = [0] + list(range(1, rays_rgba.shape[-4] - 1, step)) + [rays_rgba.shape[-4] - 1]
            rays_rgba = rays_rgba[..., layers_indices, :, :, :]
            rays_rgba_shape = rays_rgba.shape
            rays_rgba = F.interpolate(rays_rgba.reshape(-1, *rays_rgba_shape[-3:]),
                                      size=[img_h, img_w], mode='bicubic')
            rays_rgba = rays_rgba.reshape(*rays_rgba_shape[:-2], img_h, img_w)

        if rays_rgba is not None and not self.params['models']['gen'].get('rgb_after_compose', None):
            gen = self.gen.module if self.params.get('use_apex', True) else self.gen
            if getattr(gen, 'use_opacity', True):
                rays_rgb, rays_alpha = rays_rgba[..., :-1, :, :], rays_rgba[..., -1:, :, :]
            else:
                rays_rgb, rays_alpha = rays_rgba, None

            if rays_rgb.shape[-3] == 3:  # RGB channels
                rays_rgb = rays_rgb.add(1).div(2).cpu().numpy().transpose((1, 0, 3, 4, 2))
            else:
                rays_rgb = None

            if rays_alpha is not None:
                rays_alpha = rays_alpha[:, :, 0, :, :].transpose(0, 1).cpu().numpy()
                rays_alpha = grayscale_to_cmap(rays_alpha)
        else:
            rays_rgb = rays_alpha = None

        list_of_vis = [images_output,
                       images_target,
                       opacity,
                       np.repeat(mse, 3, axis=3),
                       ]
        if depth is not None:
            list_of_vis.extend(depth)
        if rays_rgb is not None:
            list_of_vis.extend(rays_rgb)
        if rays_alpha is not None:
            list_of_vis.extend(rays_alpha)

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
            'ssim': self.ssim(images_output, images_target).cpu(),
            'psnr': self.psnr(images_output, images_target).cpu(),
            'flip': self.flip_loss_module(images_output, images_target).cpu(),
            'batch_length': images_output.shape[0],
        }
        if self.lpips is not None:
            out['lpips'] = self.lpips(images_output, images_target).mean().cpu()

        del images_target

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

    @staticmethod
    def _compute_sparsity_statistic(weights: torch.Tensor,
                                    p=2,
                                    criterion=1,
                                    ) -> torch.Tensor:
        """
        Args:
            weights: B x n_layers x H x W
            p: which L_p norm to use for weights normalization
            criterion: use L1 or entropy sparsity criterion

        Returns:
            out: small value corresponds to sparse weights, high value - to dense ones
        """

        if p is not None:
            weights = F.normalize(weights, p=p, dim=-3)
        if criterion == 1:
            loss = weights.abs().sum(-3).mean()
        elif criterion == 'entropy':
            eps = 1e-6
            loss = weights * weights.add(eps).log()
            loss = - loss.sum(dim=-3).mean()
        else:
            raise ValueError(f'Unknown criterion={criterion}. Only 1 or entropy are supported')
        return loss
