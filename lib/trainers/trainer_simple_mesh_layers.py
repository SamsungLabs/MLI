__all__ = ['TrainerSimpleMeshLayers']

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.modules.cameras import CameraPytorch3d
from lib.modules.edge_filters.spatial_grad import SpatialGradient
from lib.modules.losses import PerceptualLoss, PSNRMetric, SSIM
from lib.utils.visualise import plot_square_grid
from .trainer_base import TrainerBase
from ..utils.base import min_max_scale

logger = logging.getLogger(__name__)


class TrainerSimpleMeshLayers(TrainerBase):
    def _unpack_data(self,
                     data: dict,
                     mode: str = 'eval',
                     ):
        assert list(data['initial']['image'].shape[1:3]) == [1, 1], f'Expected one image in one timestamp'
        source_images: torch.Tensor = data['initial']['image'].squeeze(1).squeeze(1).cuda()
        source_cameras = CameraPytorch3d(extrinsics=data['initial']['extrinsics'].squeeze(1).squeeze(1).cuda(),
                                         intrinsics=data['initial']['intrinsics'].squeeze(1).squeeze(1).cuda(),
                                         )

        novel_cameras = CameraPytorch3d(extrinsics=data['novel']['extrinsics'].squeeze(1).squeeze(1).cuda(),
                                        intrinsics=data['novel']['intrinsics'].squeeze(1).squeeze(1).cuda(),
                                        )

        if mode == 'train':
            novel_images: torch.Tensor = data['novel']['image'].squeeze(1).squeeze(1).cuda()
            return (
                source_images,
                novel_images,
                source_cameras,
                novel_cameras,
            )

        elif mode == 'eval':
            return (
                source_images,
                source_cameras,
                novel_cameras,
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def _init_auxiliary_modules(self):
        if self.params['weights']['generator'].get('perceptual', 0.) > 0:
            self.perceptual_loss = PerceptualLoss(
                feature_layers=(2, 7, 12, 21, 30),
                feature_weights=(1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0),
                use_avg_pool=False,
                use_input_norm=True,
                input_range='tanh').cuda()
        self.psnr = PSNRMetric(input_range='tanh').cuda()
        self.ssim = SSIM(input_range='tanh').cuda()
        self.spatial_grad = SpatialGradient()

    def update_step(self,
                    iteration,
                    a_images,
                    b_images,
                    a_cameras,
                    b_cameras
                    ):

        novel_image_b, novel_depth_b, depth_tuned_a, source_depth_a, novel_mask_b = self.gen(
            a_images,
            a_cameras,
            b_cameras,
        )

        novel_image_a, novel_depth_a, depth_tuned_b, source_depth_b, novel_mask_a = self.gen(
            b_images,
            b_cameras,
            a_cameras,
        )

        """
        GENERATOR UPDATE
        """
        self.losses['generator']['l1'] = 0
        self.losses['generator']['l1_depth'] = 0
        self.losses['generator']['depth_grad'] = 0

        """
        DEPTH LOSS
        """

        for predicted_depth, target_depth, frame_mask in [
            (novel_depth_b, source_depth_b, novel_mask_b),
            (novel_depth_a, source_depth_a, novel_mask_a),
        ]:
            self.losses['generator']['l1_depth'] += F.l1_loss(predicted_depth * frame_mask, target_depth * frame_mask)

        for predicted_depth, target_depth, frame_mask in [
            (novel_depth_b, source_depth_b, novel_mask_b),
            (novel_depth_a, source_depth_a, novel_mask_a),
        ]:
            self.losses['generator']['depth_grad'] += self.spatial_grad(predicted_depth).abs().mean()

        """
        IMAGE LOSS
        """

        for predicted_images, target_images, frame_mask in [
            (novel_image_b, b_images, novel_mask_b),
            (novel_image_a, a_images, novel_mask_a),
        ]:
            self.losses['generator']['l1'] += F.l1_loss(predicted_images * frame_mask,
                                                        target_images * frame_mask)

        self._aggregate_losses('generator')
        self.backward('generator')
        self.optimizers['generator'].step()

    def forward(self,
                source_images,
                source_cameras,
                novel_cameras
                ):

        novel_image, novel_depth, depth_tuned, depth_source, novel_mask = self.gen(
            source_images,
            source_cameras,
            novel_cameras,
        )

        out = {'images': novel_image,
               'depth': depth_tuned,
               'depth_novel': novel_depth,
               'depth_source': depth_source,
               'projection_mask': novel_mask,
               }

        return out

    @torch.no_grad()
    def inference(self, data):
        training_status = self.training
        self.eval()
        result = self.forward(*self._unpack_data(data, 'eval'))
        self.train(training_status)
        return result

    def _compute_visualisation(self, output: dict, data: dict) -> List[Image.Image]:

        novel_image = output['images'].data.add(1).div(2).to("cpu").numpy().transpose((0, 2, 3, 1))

        depth_output = output['depth'].data
        depth_output = min_max_scale(depth_output).to("cpu").numpy().transpose((0, 2, 3, 1))

        depth_novel = output['depth_novel'].data
        depth_novel = min_max_scale(depth_novel).to("cpu").numpy().transpose((0, 2, 3, 1))

        depth_source = output['depth_source'].data
        depth_source = min_max_scale(depth_source).to("cpu").numpy().transpose((0, 2, 3, 1))

        depth_grad = self.spatial_grad(output['depth_source']).abs().mean(2).data.to("cpu").numpy().transpose(
            (0, 2, 3, 1))

        projection_mask = output['projection_mask'].data.to("cpu").numpy().transpose((0, 2, 3, 1))

        mask_image = np.concatenate((projection_mask,
                                     1 - projection_mask,
                                     np.zeros_like(projection_mask)), axis=3)

        # TODO: this implementation contains squeezeing, suitable only for a single camera of each role per one scene
        images_target = data['novel']['image'].squeeze(1).squeeze(1).data.add(1).div(2).to("cpu").numpy().transpose(
            (0, 2, 3, 1))
        images_source = data['initial']['image'].squeeze(1).squeeze(1).data.add(1).div(2).to("cpu").numpy().transpose(
            (0, 2, 3, 1))

        depth_images = []
        for i in range(depth_output.shape[-1]):
            depth_images.append(np.repeat(depth_output[..., [i]], 3, axis=3))

        mse = np.power(novel_image - images_target, 2).sum(-1, keepdims=True)

        images_out = np.concatenate((images_source,
                                     images_target,
                                     np.repeat(mse, 3, axis=3),
                                     novel_image,
                                     np.repeat(depth_output, 3, axis=3),
                                     np.repeat(depth_source, 3, axis=3),
                                     np.repeat(depth_novel, 3, axis=3),
                                     mask_image,
                                     np.repeat(depth_grad, 3, axis=3),
                                     ),
                                    axis=2) * 255

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

        images_target = data['novel']['image'].cuda()
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])

        return {
            'ssim': self.ssim(images_output, images_target),
            'psnr': self.psnr(images_output, images_target),
            'batch_length': images_output.shape[0],
        }

    def _aggregate_metrics(self, metrics: List[dict]) -> dict:
        ssim, psnr, batch_length = [], [], []
        for metrics_per_batch in metrics:
            ssim.append(metrics_per_batch['ssim'].item())
            psnr.append(metrics_per_batch['psnr'].item())
            batch_length.append(metrics_per_batch['batch_length'])

        ssim, psnr, batch_length = np.array(ssim), np.array(psnr), np.array(batch_length)
        return {
            'psnr': np.sum(psnr * batch_length) / batch_length.sum(),
            'ssim': np.sum(ssim * batch_length) / batch_length.sum(),
        }
