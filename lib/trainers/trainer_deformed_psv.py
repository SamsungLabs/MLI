__all__ = ['TrainerDeformedPSV']

import itertools
import logging
import math
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import SSIM, MS_SSIM
except:
    pass
from lib.trainers import TrainerStereoMagnification
from lib.modules.edge_filters.spatial_grad import SpatialGradient
from lib.modules.cameras import CameraMultiple
from lib.modules.pytorch3d_structures.meshes import Meshes
from lib.modules.losses import tv_loss, point_cloud_loss,\
    disparity_loss, photometric_masked_loss, hinge_depth_regularizer
from lib.utils.base import min_max_scale
from lib.utils.visualise import grayscale_to_cmap

logger = logging.getLogger(__name__)


class TrainerDeformedPSV(TrainerStereoMagnification):
    """
    Options for this trainer:

    use_mpi_impact: True | False
    ground_truth_for_layered_depth_type: pointcloud | depthmap
    depth_loss_mode: None | none | disparity | log
    sparsity_normalize_p: int
    sparsity_criterion: entropy | 1
    r1_penalty_iter: int
    r1_penalty_scale_number: int | all
    use_offtheshelf_depth: True | False
    opacity_plain_sum_threshold: int
    auto_virtual_frame_interpolation_coef: True | False
    const_virtual_frame_interpolation_coef: float
    """
    def _init_auxiliary_modules(self):
        super()._init_auxiliary_modules()
        self.spatial_grad = SpatialGradient()
        try:
            self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        except:
            self.ms_ssim_loss = None

    def _unpack_data(self,
                     data: dict,
                     mode: str = 'eval',
                     ):
        if data['initial']['image'].shape[2] == 2:
            data['reference'] = {}
            for k in ['image', 'extrinsics', 'intrinsics']:
                data['reference'][k] = data['initial'][k][:, :, [1]]
                data['initial'][k] = data['initial'][k][:, :, [0]]

        source_images: torch.Tensor = data['initial']['image'].to(self.device)
        source_cameras = CameraMultiple(extrinsics=data['initial']['extrinsics'].to(self.device),
                                        intrinsics=data['initial']['intrinsics'].to(self.device),
                                        )
        reference_images: torch.Tensor = data['reference']['image'].to(self.device)
        reference_cameras = CameraMultiple(extrinsics=data['reference']['extrinsics'].to(self.device),
                                           intrinsics=data['reference']['intrinsics'].to(self.device),
                                           )

        novel_cameras = CameraMultiple(extrinsics=data['novel']['extrinsics'].to(self.device),
                                       intrinsics=data['novel']['intrinsics'].to(self.device),
                                       )

        if mode == 'train':
            novel_images: torch.Tensor = data['novel']['image'].to(self.device)
            if 'depth' in data['reference']:
                reference_depth = data['reference']['depth'].to(self.device)
                novel_depth = data['novel']['depth'].to(self.device)
            else:
                reference_depth = novel_depth = None

            return (
                reference_images,
                source_images,
                novel_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                reference_depth,
                novel_depth,
            )

        elif mode == 'eval':
            return (
                reference_images,
                source_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                data.get('proxy_geometry'),
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def update_step(self,
                    iteration,
                    reference_images,
                    source_images,
                    novel_images,
                    reference_cameras,
                    source_cameras,
                    novel_cameras,
                    point_clouds,
                    reference_depth,
                    novel_depth,
                    ):
        # Unpack outputs

        if self.params.get('auto_virtual_frame_interpolation_coef', False) and hasattr(self, 'dis'):
            overfit_rate = self.dis.get_overfit_heuristic()
            virtual_frame_interpolation_coef = 1 - max(0, overfit_rate)
        elif self.params.get('const_virtual_frame_interpolation_coef') is not None:
            virtual_frame_interpolation_coef = self.params['const_virtual_frame_interpolation_coef']
        else:
            virtual_frame_interpolation_coef = None

        gen_output = self.gen(
            reference_images,
            source_images,
            reference_cameras,
            source_cameras,
            novel_cameras,
            iteration,
            virtual_frame_interpolation_coef,
            action='forward',
        )
        predicted_images = gen_output['novel_image']
        decoded_predicted_images = gen_output.get('decoded_novel_image')
        layered_depth = gen_output['layered_depth']
        predicted_novel_depth = gen_output['novel_depth']
        novel_opacity = gen_output['novel_opacity']
        mask = novel_opacity > 0
        mpi = gen_output['mpi']
        layered_opacity = mpi[..., -1, :, :]  # B x n_layers x H x W
        if self.params.get('use_mpi_impact', True):
            impact = gen_output['mpi_impact']
        else:
            impact = gen_output['novel_impact']
            # TODO: need to handle impact out of frustum for correct work
        virtual_images = gen_output.get('virtual_image')
        decoded_virtual_images = gen_output.get('decoded_virtual_image')
        virtual_opacity = gen_output.get('virtual_opacity')

        # Losses on opacity values

        if self.params['weights']['generator'].get('opacity_prior', 0.) > 0:
            self.losses['generator']['opacity_prior'] = self._calc_alpha_prior_loss(layered_opacity)

        if self.params['weights']['generator'].get('opacity_plain_sum', 0.) > 0:
            threshold = self.params.get('opacity_plain_sum_threshold', 2)
            self.losses['generator']['opacity_plain_sum'] = \
                layered_opacity.sum(dim=1).clamp(min=threshold).sub(threshold).mean()

        # Loss on predicted layered depth and opacity of voxels close to real depth levels

        self.losses['generator']['smooth_layers'] = tv_loss(layered_depth)
        self.losses['generator']['depth_reg'] = hinge_depth_regularizer(layered_depth)

        ground_truth_for_layered_depth_type = self.params.get('ground_truth_for_layered_depth_type', 'pointcloud')
        if ground_truth_for_layered_depth_type == 'pointcloud' and self.params['weights']['generator'].get(
                'clouds', 0.) > 0:
            mesh_verts, mesh_faces = gen_output['mesh_verts'], gen_output['mesh_faces']
            meshes = Meshes(verts=mesh_verts, faces=mesh_faces)
            reference_pixel_coords = self.gen(action='get_grid').expand(reference_images.shape[0], -1, -1, -1, -1, -1)
            reference_rays = reference_cameras.pixel_to_world_ray_direction(reference_pixel_coords)[:, 0, 0]
            self._compute_depth_point_cloud_loss(reference_cameras.world_position[:, 0, 0],
                                                 reference_rays,
                                                 meshes.verts_padded(),
                                                 point_clouds.points_padded(),
                                                 layered_opacity,
                                                 )
        elif ground_truth_for_layered_depth_type == 'depthmap':
            self._compute_layered_depth_depthmap_loss(layered_depth,
                                                      reference_depth,
                                                      layered_opacity,
                                                      )
        elif ground_truth_for_layered_depth_type != 'none':
            raise ValueError(f'Unknown ground_truth_for_layered_depth_type={ground_truth_for_layered_depth_type}.'
                             'Please use pointcloud, depthmap or none.'
                             )

        # Losses on impact sparsity or density

        if self.params['weights']['generator'].get('impact_sparsity', 0.) != 0:
            self.losses['generator']['impact_sparsity'] = self._compute_sparsity_statistic(
                weights=impact.squeeze(2),
                p=self.params.get('sparsity_normalize_p', 1),
                criterion=self.params.get('sparsity_criterion', 'entropy'),
            )
        if self.params['weights']['generator'].get('average_layer_impact_sparsity', 0.) != 0:
            self.losses['generator']['average_layer_impact_sparsity'] = self._compute_sparsity_statistic(
                weights=impact.squeeze(2).mean(dim=[-1, -2], keepdim=True),
                p=self.params.get('sparsity_normalize_p', 1),
                criterion=self.params.get('sparsity_criterion', 'entropy'),
            )

        # Photometric losses

        predicted_images = predicted_images.contiguous().view(-1, *predicted_images.shape[-3:])
        novel_images = novel_images.contiguous().view(-1, *novel_images.shape[-3:])

        self.losses['generator']['l1'] = photometric_masked_loss(predicted_images, novel_images,
                                                                 mask=mask, mode='l1')
        if self.params['weights']['generator'].get('ms_ssim', 0.) > 0:
            # TODO think about masking in ssim
            self.losses['generator']['ms_ssim'] = 1 - self.ms_ssim_loss(predicted_images.add(1).div(2),
                                                                        (novel_images * mask).add(1).div(2))

        if self.params['weights']['generator'].get('perceptual', 0.) > 0:
            self.losses['generator']['perceptual'] = self.perceptual_loss(predicted_images, novel_images, mask=mask)

        # Photometric losses after decoding

        if decoded_predicted_images is not None:
            decoded_predicted_images = decoded_predicted_images.contiguous().view(-1, *predicted_images.shape[-3:])
            self.losses['generator']['l1_decoded'] = photometric_masked_loss(decoded_predicted_images,
                                                                             novel_images, mode='l1')
            if self.params['weights']['generator'].get('perceptual_decoded', 0.) > 0:
                self.losses['generator']['perceptual_decoded'] = self.perceptual_loss(decoded_predicted_images,
                                                                                      novel_images)

        # Losses on rendered novel depth

        if self.params.get('use_offtheshelf_depth', False):
            assert hasattr(self, 'guidance_depth')
            with torch.no_grad():
                novel_depth = self.guidance_depth(novel_images)

        if self.params['weights']['generator'].get('novel_depth', 0.) > 0:
            novel_depth = novel_depth.contiguous().view(-1, *novel_depth.shape[-3:])
            self.losses['generator']['novel_depth'] = disparity_loss(predicted_depth=predicted_novel_depth,
                                                                     target_depth=novel_depth,
                                                                     mask=mask,
                                                                     mode=self.params.get('depth_loss_mode'),
                                                                     )

        # Adversarial losses
        vfreq = self.params.get('virtual_frame_freq', 2)
        use_virtual = iteration % vfreq == 0 and virtual_images is not None
        if decoded_predicted_images is None:
            images_for_dis = virtual_images if use_virtual else predicted_images
            opacity_for_dis = virtual_opacity if use_virtual else novel_opacity
        else:
            images_for_dis = decoded_virtual_images if use_virtual else decoded_predicted_images
            opacity_for_dis = torch.ones_like(novel_opacity)

        images_for_dis = images_for_dis.reshape(-1, *images_for_dis.shape[-3:])
        opacity_for_dis = opacity_for_dis.reshape(-1, *opacity_for_dis.shape[-3:])
        use_feature_matching = self.params['weights']['generator'].get('feature_matching', 0.) > 0 and not use_virtual
        self.update_dis_and_get_adversarial_losses(iteration=iteration,
                                                   predicted_images=images_for_dis,
                                                   novel_images=novel_images,
                                                   predicted_opacity=opacity_for_dis,
                                                   mask=opacity_for_dis > 0,
                                                   use_feature_matching=use_feature_matching,
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
                logger.debug(f'NaN in Backward: {name}, skipping update')
                return f'NaN in Backward: {name}'
            else:
                self.save_gradient_norms_module('gen')
                self.optimizer_step('generator')
        # self.save_gradient_norms_module('gen')
        # clip_grad_norm_(self.gen.parameters(), 200)
        # self.optimizers['generator'].step()

    def update_dis_and_get_adversarial_losses(self,
                                              iteration,
                                              predicted_images,
                                              novel_images,
                                              predicted_opacity,
                                              mask=None,
                                              use_feature_matching=True,
                                              ):
        if not hasattr(self, 'dis'):
            return
        predicted_opacity = predicted_opacity.contiguous().view(-1, *predicted_opacity.shape[-3:]).gt(0)
        predicted_images = torch.where(predicted_opacity,
                                       predicted_images,
                                       torch.zeros_like(predicted_images))
        predicted_images = torch.cat([predicted_images, predicted_opacity], dim=-3)

        novel_images = torch.where(predicted_opacity,
                                   novel_images,
                                   torch.zeros_like(novel_images))
        novel_images = torch.cat([novel_images, predicted_opacity], dim=-3)

        apply_r1_penalty = (self.params['weights']['discriminator'].get('r1', 0.) > 0
                            and iteration % self.params.get('r1_penalty_iter', 1) == 0)
        if apply_r1_penalty:
            novel_images.requires_grad_()
            novel_images.retain_grad()

        self.losses['discriminator']['adversarial'], scores_real = \
            self.dis((predicted_images.detach(), novel_images), mode='calc_dis_loss')
        if apply_r1_penalty:
            r1_penalty = self._calc_r1_penalty(novel_images,
                                               scores_real,
                                               scale_number=self.params.get('r1_penalty_scale_number', 0),
                                               )
            novel_images.requires_grad_(False)
            self.losses['discriminator']['r1'] = r1_penalty * self.params.get('r1_penalty_iter', 1)

        self.losses['discriminator']['overfit_heuristic'] = self.dis.get_overfit_heuristic()
        self._aggregate_losses('discriminator')
        self.backward('discriminator')
        self.save_gradient_norms_module('dis')
        self.optimizers['discriminator'].step()

        gen_adversarial_loss, predicted_dis_features = self.dis(predicted_images, mode='calc_gen_loss')
        self.losses['generator']['adversarial'] = gen_adversarial_loss

        self.losses['generator']['feature_matching'] = 0
        if use_feature_matching:
            with torch.no_grad():
                _, original_dis_features = self.dis(novel_images, mode='calc_gen_loss')

            for feat_a, feat_b in zip(itertools.chain.from_iterable(predicted_dis_features),
                                      itertools.chain.from_iterable(original_dis_features)):
                self.losses['generator']['feature_matching'] += photometric_masked_loss(feat_a, feat_b,
                                                                                        mask=mask, mode='l2')

    @staticmethod
    def _compute_sparsity_statistic(weights: torch.Tensor,
                                    p: Optional[int] = 2,
                                    criterion: Union[int, str] = 1,
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

    def _compute_depth_point_cloud_loss(self,
                                        camera_position,
                                        frustum_rays,
                                        predicted_cloud,
                                        ground_truth_cloud,
                                        layered_opacity,
                                        ):
        """
        This function compute special losses for point cloud and depth tensor
        in order to produce mesh layers structure
        Args:
            camera_position: B x 3
            frustum_rays: rays from reference image
            predicted_cloud: vertices of predicted using depth cloud
            ground_truth_cloud: ground truth sparse (optional dense point cloud)
            layered_opacity: opacity of each pixel in each layer, B x num_layers x H x W
        """
        bs, _, xyz = ground_truth_cloud.size()
        self.losses['generator']['clouds'], pixel_idx, layer_idx, mask = point_cloud_loss(
            camera_position,
            frustum_rays.view(bs, -1, xyz),
            ground_truth_cloud,
            predicted_cloud.view(bs, self.gen.num_layers, -1, xyz),
        )

        if self.params['weights']['generator'].get('opacity_gt_cloud', 0.) > 0:
            layered_opacity = layered_opacity.view(*layered_opacity.shape[:-2], -1)  # B x num_layers x H*W
            opacity_of_gt_cloud = layered_opacity[torch.arange(bs).view(-1, 1), layer_idx, pixel_idx]  # B x V
            self.losses['generator']['opacity_gt_cloud'] = (torch.abs(1 - opacity_of_gt_cloud) * mask).mean()

    def _compute_layered_depth_depthmap_loss(self,
                                             estimated_depth: torch.Tensor,
                                             target_depth: torch.Tensor,
                                             layered_opacity: torch.Tensor,
                                             ):
        """
        Custom loss for the ground-true absolute depth map vs predicted layered depth map.

        Args:
            estimated_depth: B x num_layers x H x W
            target_depth: B x 1 x H x W
            layered_opacity: opacity of each pixel in each layer, B x num_layers x H x W

        Returns:
            loss: scalar
            layer_idx: which layer of layered depth contains value most close to the target, B x 1 x H x W
        """
        dist = (estimated_depth - target_depth).pow(2)
        loss_per_pixel, nearest_layer_idx = dist.min(dim=1)
        self.losses['generator']['clouds'] = loss_per_pixel.mean()

        if self.params['weights']['generator'].get('opacity_gt_cloud', 0.) > 0:
            opacity_of_gt_pixels = torch.gather(layered_opacity, dim=1, index=nearest_layer_idx.unsqueeze(1))
            self.losses['generator']['opacity_gt_cloud'] = torch.abs(1 - opacity_of_gt_pixels).mean()

    def forward(self,
                reference_images,
                source_images,
                reference_cameras,
                source_cameras,
                novel_cameras,
                proxy_geometry=None,
                ):
        if proxy_geometry is not None:
            gen_output = self.gen(
                novel_cameras,
                proxy_geometry,
                action='render_with_proxy_geometry',
            )
        else:
            gen_output = self.gen(reference_images,
                                  source_images,
                                  reference_cameras,
                                  source_cameras,
                                  novel_cameras,
                                  action='forward',
                                  )
        if gen_output['decoded_novel_image'] is None:
            novel_images = gen_output['novel_image']
        else:
            novel_images = gen_output['decoded_novel_image']
        novel_opacity = gen_output['novel_opacity']
        mpi = gen_output['mpi']
        novel_depth = gen_output['novel_depth']
        mpi_impact = gen_output['mpi_impact']
        if not self.training:
            # a strange workaround instead of in-place detach_() op because of an error:
            # "RuntimeError: Can't detach views in-place. Use detach() instead",
            # observed in some pipelines
            novel_images = novel_images.detach()
            novel_opacity = novel_opacity.detach()
            novel_depth = novel_depth.detach()
            mpi = mpi.detach()
            mpi_impact = mpi_impact.detach()

        out = {'images': novel_images.contiguous().view(-1, *novel_images.shape[-3:]),
               'opacity': novel_opacity.contiguous().view(-1, *novel_opacity.shape[-3:]),
               'depth': novel_depth.contiguous().view(-1, *novel_depth.shape[-3:]),
               'mpi': mpi.contiguous().view(-1, *mpi.shape[-4:]),
               'mpi_impact': mpi_impact.contiguous().view(-1, *mpi_impact.shape[-4:]),
               }
        if not self.training:
            if 'proxy_geometry' in gen_output:
                proxy_geometry = gen_output['proxy_geometry']
            else:
                proxy_geometry = {
                    'verts': gen_output['mesh_verts'],
                    'faces': gen_output['mesh_faces'],
                    'verts_rgb': gen_output['mesh_verts_rgb'],
                    'mpi': gen_output['mpi'],
                    'layered_depth': gen_output.get('layered_depth'),
                }
                proxy_geometry = {k: v.detach() for k, v in proxy_geometry.items() if v is not None}

            out['proxy_geometry'] = proxy_geometry

        return out

    @staticmethod
    def _calc_r1_penalty(real_images: torch.Tensor,
                         scores_real: List[torch.Tensor],
                         scale_number: Union[int, List[int], str] = 0,
                         ) -> torch.Tensor:
        assert real_images.requires_grad
        if isinstance(scale_number, int):
            scale_number = [scale_number]
        if isinstance(scale_number, str):
            if scale_number != 'all':
                raise ValueError(f'scale_number should be int, List[int] or literal "all". Got value: {scale_number}')
            scale_number = list(range(len(scores_real)))

        penalties = 0.
        for scale_idx in scale_number:
            scores = scores_real[scale_idx]
            gradients = torch.autograd.grad(scores.sum(), real_images, create_graph=True, retain_graph=True)[0]
            penalty = gradients.pow(2).view(gradients.shape[0], -1).sum(1).mean()
            penalties += penalty
        return penalties / len(scale_number)

    @staticmethod
    def _calc_alpha_prior_loss(tensor: torch.Tensor):
        lower_bound = -2.20727
        tensor = tensor.contiguous().view(tensor.shape[0], -1)
        loss = torch.log(0.1 + tensor) + torch.log(0.1 + 1 - tensor)
        return loss.sub(lower_bound).mean(-1).mean()

    def _compute_visualisation(self, output: dict, data: dict) -> List[Image.Image]:
        images_output = output['images'].data.add(1).div(2).to('cpu').numpy().transpose((0, 2, 3, 1))
        opacity = output['opacity'].data
        resolution = opacity.shape[-2:]

        depth = output['depth']
        depth = min_max_scale(depth, dim=[1, 2, 3], mask=opacity.gt(0.).expand_as(depth))
        depth = depth.data.to('cpu').numpy()[:, 0, :, :]
        depth = grayscale_to_cmap(depth)

        opacity = opacity.to('cpu').numpy()[:, 0, :, :]
        opacity = grayscale_to_cmap(opacity)

        images_target = data['novel']['image']
        images_target = images_target.contiguous().view(-1, *images_target.shape[-3:])
        images_target = images_target.data.add(1).div(2).to('cpu').numpy().transpose((0, 2, 3, 1))

        mse = np.power(images_output - images_target, 2).sum(-1, keepdims=True)

        mpi = output['mpi'].data
        mpi_impact = output.get('mpi_impact')
        if mpi.shape[-2:] != resolution:
            logger.debug(f'Need to interpolate MPI for visualisation: images have size {resolution}, '
                         f'while MPI is of size {mpi.shape[-2:]}')
            first_dims = mpi.shape[:-3]
            mpi = F.interpolate(mpi.reshape(-1, *mpi.shape[-3:]),
                                size=resolution,
                                mode='bilinear',
                                ).reshape(*first_dims, -1, *resolution)
            if mpi_impact is not None:
                mpi_impact = F.interpolate(mpi_impact.data.reshape(-1, *mpi_impact.shape[-3:]),
                                           size=resolution,
                                           mode='bilinear',
                                           ).reshape(*first_dims, -1, *resolution)

        num_layers_to_show = self.params.get('num_layers_to_vis', mpi.shape[-4])  # show only this number of layers
        step = math.ceil((mpi.shape[-4] - 2) / (num_layers_to_show - 2)) if num_layers_to_show > 2 else 2
        # we show the nearest plane (0), the farthest one (-1) and others with equal step, starting from the index (1)
        layers_indices = [0] + list(range(1, mpi.shape[-4] - 1, step)) + [mpi.shape[-4] - 1]
        mpi = mpi[..., layers_indices, :, :, :]
        if getattr(self.gen, 'use_opacity', True):
            mpi_rgb, mpi_alpha = mpi[..., :-1, :, :], mpi[..., -1:, :, :]
        else:
            mpi_rgb, mpi_alpha = mpi, None
        if mpi_impact is not None:
            mpi_alpha = mpi_impact[..., layers_indices, :, :, :]
        if mpi_rgb.shape[-3] == 3:  # RGB channels
            mpi_rgb = mpi_rgb.add(1).div(2).cpu().numpy().transpose((1, 0, 3, 4, 2))
        else:
            mpi_rgb = None
        if mpi_alpha is not None:
            mpi_alpha = mpi_alpha[:, :, 0, :, :].transpose(0, 1).cpu().numpy()
            mpi_alpha = grayscale_to_cmap(mpi_alpha)

        list_of_vis = [images_output,
                       images_target,
                       opacity,
                       np.repeat(mse, 3, axis=3),
                       depth,
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
