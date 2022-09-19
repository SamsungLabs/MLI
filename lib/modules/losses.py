__all__ = ['VGG19PerceptualLoss',
           'PerceptualLoss',
           'PSNRMetric',
           'SSIM',
           'point_cloud_loss',
           'tv_loss',
           'photometric_masked_loss',
           'uncertain_mixture_likelihood_loss',
           'disparity_loss',
           'HiFreqPSNRMetric',
           'PSNRMetricColorShift'
           ]

import math
from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


def color_shift_remove(fake_image, real_image):
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


def photometric_masked_loss(prediction: torch.Tensor,
                            target: torch.Tensor,
                            mask: Optional[torch.BoolTensor] = None,
                            multiplier: Optional[torch.Tensor] = None,
                            mode: str = 'l1',
                            eps: float = 1e-6,
                            ) -> torch.Tensor:
    """
    Args:
        prediction: B x C x H x W
        target: B x C x H x W
        mask: B x 1 x h x w
        multiplier: B x 1 x h x w or B x C x H x W
        mode: l1 | l2
        eps: const for numerical stability

    Returns:
        loss
    """
    assert prediction.shape == target.shape

    if mode == 'l1':
        func = lambda x, y: torch.abs(x - y)
    elif mode == 'l2':
        func = lambda x, y: (x - y) ** 2
    else:
        raise ValueError(f'Unknown mode={mode}')

    if multiplier is not None and multiplier.shape[-2:] != target.shape[-2:]:
        multiplier = F.interpolate(multiplier.expand(-1, target.shape[1], -1, -1),
                                   size=target.shape[-2:], mode='bilinear')

    if mask is None:
        pixelwise = func(prediction, target)
        if multiplier is None:
            return pixelwise.mean()
        else:
            return torch.mean(pixelwise * multiplier)
    else:
        mask = F.interpolate(mask.float(), size=target.shape[-2:], mode='bilinear')
        pixelwise = func(prediction, target) * mask
        if multiplier is None:
            return torch.mean(
                torch.sum(pixelwise, dim=[-1, -2]) / mask.sum(dim=[-1, -2]).add(eps)
            )
        else:
            return torch.mean(
                torch.sum(pixelwise * multiplier, dim=[-1, -2]) / mask.sum(dim=[-1, -2]).add(eps)
            )


def uncertain_mixture_likelihood_loss(prediction: torch.Tensor,
                                      target: torch.Tensor,
                                      uncertainty: torch.Tensor,
                                      mask: Optional[torch.Tensor] = None,
                                      weight: float = 1,
                                      mode: str = 'l1',
                                      input_range: str = 'tanh',
                                      eps: float = 1e-6,
                                      ) -> torch.Tensor:
    """
    Negative log-likelihood of the arithmetic mixture distribution

    loss = -log p(target | prediction; uncertainty), where
        p(target | prediction; uncertainty, weight)
            = uncertainty * Uniform(target; input_range) + (1 - uncertainty) * q(target | prediction; weight),
    and density q(target | prediction; weight) follows the Gaussian or  Laplacian law.

    The loss value is computed using the LogSumEXp operation, that is why we manipulate the log-densities.

    Args:
        prediction: B x C x H x W
        target: B x C x H x W
        uncertainty: B x 1 x H x W
        mask: B x 1 x h x w
        weight: 1/sigma^2 for l2 (Gaussian) or 1/b for l1 (Laplacian)
        mode: l1 | l2
        input_range: tanh | sigmoid
        eps:
    Returns:
        loss
    """

    ranges = {
        'sigmoid': 1,
        'tanh': 2,
    }
    uncertain_part = torch.log(uncertainty + eps) - math.log(ranges[input_range])

    if mode in ('l1', 'truncated-l1'):
        log_reciprocal_partition_constant = math.log(weight / 2)
        diff = torch.abs(prediction - target)
    elif mode == 'l2':
        log_reciprocal_partition_constant = 0.5 * math.log(weight) - 0.5 * math.log(2 * math.pi)
        diff = prediction - target
        diff = diff * diff
    else:
        raise ValueError(str(mode))

    confident_part = torch.log(-uncertainty + 1 + eps) - weight * diff + log_reciprocal_partition_constant
    if mode == 'truncated-l1':
        if input_range == 'sigmoid':
            raise NotImplementedError
        log_normalizer = torch.log(torch.relu(
            2. - torch.exp(-weight * (prediction + 1)) - torch.exp(weight * (prediction - 1))
        ) + eps) + math.log(0.5)
        confident_part = confident_part - log_normalizer

    log_likelihood = torch.stack([
        uncertain_part.expand_as(confident_part),
        confident_part
    ], dim=0).logsumexp(dim=0).sum(dim=1, keepdim=True)  # B x 1 x H x W

    del uncertain_part, confident_part
    torch.cuda.empty_cache()

    if mask is None:
        return -log_likelihood.mean()
    else:
        return -torch.mean(
            torch.sum(log_likelihood * mask, dim=[-1, -2]) / mask.sum(dim=[-1, -2]).add(eps)
        )


# ########################################################################################
# Out basic implementation of Perceptual Loss
# ########################################################################################

class VGG19BaseLoss(nn.Module):
    def __init__(self, comparison_levels):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True)
        feature_extractors = []
        prev_i = 0
        for chunk_end in comparison_levels:
            feature_extractors.append(nn.Sequential(
                *vgg.features[prev_i:chunk_end]
            ))
            prev_i = chunk_end
        self._feature_extractors = nn.ModuleList(feature_extractors).eval()
        for p in self._feature_extractors.parameters():
            p.requires_grad = False

    def forward(self, predicted, target):
        result = 0.
        for layer in self._feature_extractors:
            predicted = layer(predicted)
            target = layer(target)
            result = result + self._calc_diff(predicted, target)
        return result

    def _calc_diff(self, predicted_features, target_features):
        raise NotImplementedError()


class VGG19PerceptualLoss(VGG19BaseLoss):
    def __init__(self, comparison_levels=(3, 5, 11, 15)):
        super().__init__(comparison_levels)

    def _calc_diff(self, predicted_features, target_features):
        return F.mse_loss(predicted_features.clone(), target_features.clone())


# ########################################################################################
# More pleasant implementation of Perceptual Loss
# ########################################################################################

class MultiscaleVGG(nn.Module):
    def __init__(self,
                 feature_layers: Sequence[int],
                 use_bn: bool = True,
                 use_input_norm: bool = True,
                 use_avg_pool: bool = True,
                 input_range: str = 'tanh',
                 ):
        super().__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)

        if use_avg_pool:
            for idx, module in model.features._modules.items():
                if module.__class__.__name__ == 'MaxPool2d':
                    model.features._modules[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            if input_range == 'tanh':  # [-1, 1]
                mean = 2 * mean - 1
                std *= 2
            elif input_range == 'uint8':  # [0, 255]
                mean *= 255
                std *= 255
            elif input_range == 'sigmoid':  # [0, 1]
                pass
            else:
                raise ValueError(f'Unsupported input_range={input_range}. Use "sigmoid", "tanh" or "uint8".')
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        model.eval()
        self.features = nn.ModuleList()

        feature_layers = [0] + list(feature_layers)
        for i in range(len(feature_layers) - 1):
            feats = nn.Sequential(*list(model.features.children())[feature_layers[i]:feature_layers[i + 1]])
            for param in feats.parameters():
                param.requires_grad = False
            self.features.append(feats)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        outs = [x]
        for f in self.features:
            out = f(outs[-1])
            outs.append(out)
        return outs[1:]


class PerceptualLoss(nn.Module):
    def __init__(self,
                 feature_layers: Sequence[int] = (4, 9, 18, 27, 36),
                 feature_weights: Optional[Sequence[float]] = (1, 1, 2, 3, 40),
                 use_bn: bool = False,
                 use_input_norm: bool = True,
                 use_avg_pool: bool = True,
                 input_range: str = 'tanh',
                 loss_features: str = 'l1',
                 color_shift_elimination: bool = False,
                 ):
        super().__init__()
        self._feature_extractors = MultiscaleVGG(feature_layers=feature_layers,
                                                 use_bn=use_bn,
                                                 use_input_norm=use_input_norm,
                                                 use_avg_pool=use_avg_pool,
                                                 input_range=input_range,
                                                 )
        self._loss_features = loss_features
        self._feature_weights = feature_weights
        self.color_shift_elimination = color_shift_elimination

    def forward(self,
                fake_image: torch.Tensor,
                real_image: torch.Tensor,
                feature_weights: Optional[Sequence[float]] = None,
                mask: Optional[torch.Tensor] = None,
                multiplier: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        if self.color_shift_elimination:
            fake_image = color_shift_remove(fake_image, real_image)

        fake_features = self._feature_extractors(fake_image)
        real_features = self._feature_extractors(real_image)
        loss = 0.

        if feature_weights is None:
            feature_weights = self._feature_weights
        if feature_weights is None:
            feature_weights = [1] * len(fake_features)

        for fake_layer, real_layer, weight in zip(fake_features, real_features, feature_weights):
            loss += weight * photometric_masked_loss(fake_layer, real_layer,
                                                     mask=mask, multiplier=multiplier, mode=self._loss_features)
        return loss


class PerceptualLossColorShift(PerceptualLoss):
    def __init__(self,
                 feature_layers: Sequence[int] = (4, 9, 18, 27, 36),
                 feature_weights: Optional[Sequence[float]] = (1, 1, 2, 3, 40),
                 use_bn: bool = False,
                 use_input_norm: bool = True,
                 use_avg_pool: bool = True,
                 input_range: str = 'tanh',
                 loss_features: str = 'l1',):
        super().__init__(feature_layers=feature_layers,
                         feature_weights=feature_weights,
                         use_bn=use_bn,
                         use_input_norm=use_input_norm,
                         use_avg_pool=use_avg_pool,
                         input_range=input_range,
                         loss_features=loss_features,
                         color_shift_elimination=True,
                         )


class PSNRMetric(nn.Module):
    def __init__(self,
                 input_range: str = 'tanh',
                 batch_reduction: bool = True,
                 color_shift_elimination: bool = False):

        super().__init__()
        self.input_range = input_range
        self.batch_reduction = batch_reduction
        self.color_shift_elimination = color_shift_elimination

    def forward(self,
                fake_image: torch.Tensor,
                real_image: torch.Tensor,
                input_range: Optional[str] = None,
                mask: torch.BoolTensor = None,
                ) -> torch.Tensor:
        if input_range is None:
            input_range = self.input_range

        if input_range == 'tanh':  # [-1, 1]
            fake_image = fake_image.add(1).div(2)
            real_image = real_image.add(1).div(2)
        elif input_range == 'uint8':  # [0, 255]
            fake_image = fake_image.div(255)
            real_image = real_image.div(255)
        elif input_range == 'sigmoid':  # [0, 1]
            pass
        else:
            raise ValueError(f'Unsupported input_range={input_range}. Use "sigmoid", "tanh" or "uint8".')

        if mask is None:
            mask = torch.ones_like(fake_image)

        if self.color_shift_elimination:
            fake_image = color_shift_remove(fake_image, real_image)
            # b, c, h, w = fake_image.shape
            # fake_image = fake_image.reshape(c, -1)
            # real_image = real_image.reshape(c, -1)
            # fake_image_ext = torch.cat([fake_image[[0]] * fake_image[[1]],
            #                             fake_image[[2]] * fake_image[[1]],
            #                             fake_image[[2]] * fake_image[[0]],
            #                             fake_image.pow(2),
            #                             fake_image,
            #                             torch.ones_like(fake_image[[0]])
            #                             ], axis=0)
            #
            # x, _ = torch.lstsq(real_image.T, fake_image_ext.T)
            # fake_image = torch.mm(fake_image_ext.T, x[:fake_image_ext.shape[0]]).T
            # fake_image = fake_image.reshape(b, c, h, w)
            # real_image = real_image.reshape(b, c, h, w)

        if self.batch_reduction:
            mse = F.mse_loss(fake_image.clamp(0, 1) * mask, real_image.clamp(0, 1) * mask)
            mask = mask.float().mean()
        else:
            mse = F.mse_loss(fake_image.clamp(0, 1) * mask, real_image.clamp(0, 1) * mask, reduction='none')
            mask = mask.float().reshape(mask.shape[0], -1).mean(dim=1)
            mse = mse.reshape(mse.shape[0], -1).mean(dim=1)

        return - 10 * (mse.add(1e-6) / mask).log10()


class PSNRMetricColorShift(PSNRMetric):
    def __init__(self,
                 input_range: str = 'tanh',
                 batch_reduction: bool = True):
        super().__init__(input_range=input_range, batch_reduction=batch_reduction, color_shift_elimination=True)


class HiFreqPSNRMetric:
    def __init__(self,
                 input_range: str = 'tanh',
                 batch_reduction: bool = True,
                 window_ratio: float = 0.2,
                 ):
        self.input_range = input_range
        self.batch_reduction = batch_reduction
        self.window_ratio = window_ratio

    def fft_hi_pass(self, x: np.ndarray) -> np.ndarray:
        ffted = np.fft.fft2(x, axes=(-2, -1), norm='ortho')
        magnitude = np.abs(ffted)
        magnitude_shifted = np.fft.fftshift(magnitude, axes=(-2, -1))
        h, w = magnitude_shifted.shape[-2:]
        magnitude_shifted[...,
        int(h / 2 - h * self.window_ratio / 2): int(h / 2 + h * self.window_ratio / 2) + 1,
        int(w / 2 - w * self.window_ratio / 2): int(w / 2 + w * self.window_ratio / 2) + 1,
        ] = 0
        return magnitude_shifted

    @staticmethod
    def mse_to_psnr(x: torch.Tensor) -> torch.Tensor:
        return -10 * x.add(1e-6).log10()

    def __call__(self,
                 fake_image: torch.Tensor,
                 real_image: torch.Tensor,
                 input_range: Optional[str] = None,
                 ):
        if input_range is None:
            input_range = self.input_range

        if input_range == 'tanh':  # [-1, 1]
            fake_image = fake_image.add(1).div(2)
            real_image = real_image.add(1).div(2)
        elif input_range == 'uint8':  # [0, 255]
            fake_image = fake_image.div(255)
            real_image = real_image.div(255)
        elif input_range == 'sigmoid':  # [0, 1]
            pass
        else:
            raise ValueError(f'Unsupported input_range={input_range}. Use "sigmoid", "tanh" or "uint8".')

        fake_image = fake_image.cpu().numpy()
        real_image = real_image.cpu().numpy()
        height, width = fake_image.shape[-2:]
        remaining_area = height * width * (1 - self.window_ratio ** 2)

        fake_fft = self.fft_hi_pass(fake_image)
        real_fft = self.fft_hi_pass(real_image)
        mse = np.mean(np.sum((fake_fft - real_fft) ** 2, axis=(-2, -1)) / remaining_area, axis=1)
        mse = torch.from_numpy(mse).float()
        if self.batch_reduction:
            mse = mse.mean()
        return self.mse_to_psnr(mse)


# ########################################################################################
# SSIM. Based on implementation of https://github.com/Po-Hsun-Su/pytorch-ssim
# ########################################################################################

def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    x_space = torch.arange(window_size).float()
    gauss = x_space.sub(window_size // 2).pow(2).div(2 * sigma ** 2).neg().exp()  # tensor of shape (window_size,)
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).unsqueeze(0).unsqueeze(0)  # 1 x 1 x window_size x window_size
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(tensor1: torch.Tensor,
          tensor2: torch.Tensor,
          window: torch.Tensor,
          window_size: int,
          channel: int,
          size_average: bool = True,
          ):
    mu1 = F.conv2d(tensor1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(tensor2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(tensor1 * tensor1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(tensor2 * tensor2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(tensor1 * tensor2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    const1 = 0.01 ** 2
    const2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + const1) * (2 * sigma12 + const2)
                ) / ((mu1_sq + mu2_sq + const1) * (sigma1_sq + sigma2_sq + const2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.reshape(ssim_map.shape[0], -1).mean(dim=1)


class SSIM(nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 batch_reduction: bool = True,
                 input_range: str = 'tanh',
                 color_shift_elimination: bool = False,
                 ):
        super().__init__()
        self.window_size = window_size
        self.batch_reduction = batch_reduction
        self.register_buffer('window', _create_window(window_size, 1))
        self.input_range = input_range
        self.color_shift_elimination = color_shift_elimination

    def forward(self,
                tensor1: torch.Tensor,
                tensor2: torch.Tensor,
                mask: torch.BoolTensor = None,
                ):
        # TODO Mask without accounting in convolutions
        if self.input_range == 'tanh':  # [-1, 1]
            tensor1 = tensor1.add(1).div(2)
            tensor2 = tensor2.add(1).div(2)
        elif self.input_range == 'uint8':  # [0, 255]
            tensor1 = tensor1.div(255)
            tensor2 = tensor2.div(255)
        elif self.input_range == 'sigmoid':  # [0, 1]
            pass
        else:
            raise ValueError(f'Unsupported input_range={self.input_range}. Use "sigmoid", "tanh" or "uint8".')
        channel = tensor1.shape[1]
        window = self.window.expand(channel, -1, -1, -1)

        if self.color_shift_elimination:
            tensor1 = color_shift_remove(tensor1, tensor2)

        if mask is None:
            mask = torch.ones_like(tensor1)

        if self.batch_reduction:
            return _ssim(tensor1 * mask,
                         tensor2 * mask,
                         window,
                         self.window_size,
                         channel,
                         self.batch_reduction) * mask.float().mean()
        else:
            return _ssim(tensor1 * mask,
                         tensor2 * mask,
                         window,
                         self.window_size,
                         channel,
                         False) * mask.float().reshape(mask.shape[0], -1).mean(dim=1)


class SSIMColorShift(SSIM):
    def __init__(self,
                 window_size: int = 11,
                 batch_reduction: bool = True,
                 input_range: str = 'tanh'
                 ):
        super().__init__(window_size=window_size,
                         batch_reduction=batch_reduction,
                         input_range=input_range,
                         color_shift_elimination=True
                         )


# ########################################################################################
# Losses for predicted layered depth vs ground-true point cloud
# ########################################################################################

def point_cloud_loss(rays_origin, img_rays, gt_cloud, verts, padding_value=0.):
    """
    Custom point cloud loss
    Args:
        rays_origin: B x 3 - coordinate of ray origin (e.g., pinhole camera)
        img_rays: B x H*W x 3 coordinates of image rays
        gt_cloud: B x V x 3
        verts:  B x n_layers x H*W x 3
        padding_value: padding value for batch of point clouds (default: 0)
    Returns:
        loss: distance between gt_cloud and verts, scalar
        rays_idx: nearest ray per each point from the pointcloud, B x V
        layer_idx: which layer of layered depth contains value most close to the target, B x V
        mask: which values correspond to padding, B x V (1 for real value, 0 for padding)
    """
    n_rays = img_rays.shape[1]  # HW
    n_points_per_cloud = gt_cloud.shape[1]  # V
    bs, n_layers = verts.shape[:2]

    origin_to_cloud = gt_cloud - rays_origin.unsqueeze(1)  # B x V x 3
    origin_to_cloud = origin_to_cloud.unsqueeze(1).expand(-1, n_rays, -1, -1)  # B x HW x V x 3
    img_rays = F.normalize(img_rays, dim=-1).unsqueeze(-2).expand(-1, -1, n_points_per_cloud, -1)  # B x HW x V x 3
    dist = torch.cross(origin_to_cloud, img_rays, dim=-1).pow(2).sum(-1)  # B x HW x V

    idx_intersections = torch.argmin(dist, dim=1)  # B x V

    # B  x n_layers x V x 3
    selected_verts = torch.gather(verts, dim=2, index=idx_intersections.view(bs, 1, -1, 1).expand(-1, n_layers, -1, 3))

    new_dist = (selected_verts - gt_cloud.unsqueeze(1)).pow(2).sum(-1)  # B x n_layers x V
    loss_per_vertex, nearest_layer_idx = new_dist.min(dim=1)
    mask = 1 - gt_cloud.eq(padding_value).all(-1).float()  # B x V
    loss = (loss_per_vertex * mask).mean()  # ToDo: select only needed values wrt mask to participate in the loss
    return loss, idx_intersections, nearest_layer_idx, mask


def tv_loss(img: torch.Tensor, eps=1e-6):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w).add(eps).sqrt() / (bs_img * c_img * h_img * w_img)


def hinge_depth_regularizer(layered_depth: torch.Tensor, margin=0.0):
    # Note: default regularization on positive values.
    regularizer = torch.pow(torch.relu(layered_depth[:, 1:, :, :] - layered_depth[:, :-1, :, :]), 2)
    return regularizer.mean(-1).mean(-1).sum(-1).mean()


def disparity_loss(predicted_depth: torch.Tensor,
                   target_depth: torch.Tensor,
                   mask: Optional[torch.BoolTensor] = None,
                   mode: Optional[str] = None,
                   detach_adjustment: bool = False,
                   ) -> torch.Tensor:
    """
    Args:
        predicted_depth: B x 1 x H x W
        target_depth: B x 1 x H x W
        mask: B x 1 x H x W. True for predicted pixels, False for pixel outside the frustum
        mode: None | none | disparity | log | depth
            None and 'none' lead to L1 loss applied to disparities.
            'disparity' involves robust estimation of scale and shift in disparity domain. (see MiDaS paper)
            'log' estimates the scale in log-depth domain. (see Single view MPI paper)
        detach_adjustment: whether to detach calculated scale and bias

    Returns:
        out: scalar loss
    """
    assert predicted_depth.shape == target_depth.shape
    batch_size = predicted_depth.shape[0]

    eps = 1e-6
    predicted_disparity = predicted_depth.add(eps).reciprocal()
    target_disparity = target_depth.add(eps).reciprocal()

    if mode in (None, 'none'):
        loss_per_pixel = torch.abs(predicted_disparity - target_disparity)

    elif mode == 'depth':
        loss_per_pixel = torch.abs(predicted_depth - target_depth)

    elif mode == 'disparity':
        if mask is None:
            target_shift = target_disparity.view(batch_size, -1).median(dim=-1)[0][:, None, None, None]
            target_scale = torch.abs(target_disparity - target_shift).mean(dim=[-1, -2])[:, :, None, None]
            predicted_shift = predicted_disparity.view(batch_size, -1).median(dim=-1)[0][:, None, None, None]
            predicted_scale = torch.abs(predicted_disparity - predicted_shift).mean(dim=[-1, -2])[:, :, None, None]
        else:
            # here we should provide an unpleasant procedure of computing the median for a masked tensor.
            # future versions of pytorch have `nanmedian` function for such a case, but now we need to write
            # a slow and inefficient sorting-based implementation
            disparity = torch.cat([predicted_disparity, target_disparity], dim=0)
            disparity_sorted, order = disparity.view(2 * batch_size, -1).sort(dim=-1)
            mask_sorted = mask \
                .repeat(2, 1, 1, 1) \
                .view(2 * batch_size, -1)[torch.arange(2 * batch_size).view(-1, 1), order]
            mask_sorted_cumsum = mask_sorted.long().cumsum(dim=-1)
            boolean_position_indicator = mask_sorted_cumsum == torch.floor(mask_sorted_cumsum[:, -1:].float() / 2)
            hw = boolean_position_indicator.shape[-1]
            all_indices = torch.arange(hw, 0, -1, device=boolean_position_indicator.device).view(1, -1)
            selected_indices = torch.argmax(all_indices * boolean_position_indicator, dim=1, keepdim=True)
            shift = torch.gather(disparity_sorted, dim=-1, index=selected_indices)[:, :, None, None]

            diff = torch.abs(disparity - shift)
            scale = torch.sum(diff * mask.repeat(2, 1, 1, 1), dim=[-1, -2]) / mask.sum(dim=[-1, -2]).repeat(2, 1)
            scale = scale[:, :, None, None]

            predicted_shift, target_shift = shift[:batch_size], shift[batch_size:]
            predicted_scale, target_scale = scale[:batch_size], scale[batch_size:]

        target_disparity_normalized = (target_disparity - target_shift) / target_scale
        if detach_adjustment:
            predicted_shift.detach_()
            predicted_scale.detach_()
        predicted_disparity_normalized = (predicted_disparity - predicted_shift) / predicted_scale
        loss_per_pixel = torch.abs(target_disparity_normalized - predicted_disparity_normalized)

    elif mode == 'log':
        diff = target_depth.add(eps).log() - predicted_depth.add(eps).log()
        if mask is None:
            logscale = diff.mean(dim=[-1, -2])[:, :, None, None]
        else:
            logscale = torch.sum(diff * mask, dim=[-1, -2]) / mask.sum(dim=[-1, -2])
            logscale = logscale[:, :, None, None]
        if detach_adjustment:
            logscale.detach_()
        loss_per_pixel = torch.pow(predicted_disparity.add(eps).log() - target_disparity.add(eps).log() - logscale, 2)

    else:
        raise ValueError(f'Unknown value {mode}')

    if mask is None:
        return loss_per_pixel.mean()
    else:
        return torch.mean(
            torch.sum(loss_per_pixel * mask, dim=[-1, -2]) / mask.sum(dim=[-1, -2])
        )
