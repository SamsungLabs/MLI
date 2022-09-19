import functools
import logging
import math
import os
import random
import time
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

logger = logging.getLogger(__name__)

NONLINEARITIES = dict(
    tanh=torch.tanh,
    sigmoid=torch.sigmoid,
    clamp=lambda x: torch.clamp(x, -1., 1.),
    logsoftmax=lambda x: torch.log_softmax(x, dim=1),
    none=lambda x: x,
)


class Timer:
    def __init__(self, msg, current_iter=1, period=1):
        self.msg = msg
        self.start_time = None
        self.logging_flag = not (current_iter % period)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.logging_flag:
            logger.info(self.msg.format(time.time() - self.start_time))


def get_latest_model_name(dir_name, name):
    if os.path.exists(dir_name) is False:
        return None
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if
             os.path.isfile(os.path.join(dir_name, f))
             and f.startswith('model.')
             and f.endswith('.pt')
             and '_'.join(f.split('.')[1].split('_')[:-1]) == name
             ]
    if not len(files):
        return None
    files.sort()
    return files[-1]


def weights_init(init_type='gaussian'):
    def init_fun(module):
        class_name = module.__class__.__name__
        if (class_name.find('Conv') == 0 or class_name.find('Linear') == 0) and hasattr(module, 'weight'):
            if init_type == 'gaussian':
                init.normal_(module.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(module.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(module.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(module, 'bias') and module.bias is not None:
                init.constant_(module.bias.data, 0.0)

    return init_fun


def stack_list_of_dicts_along_dim(list_of_dicts: list, dim=0):
    # TODO: deal with this ugly functoin on list of dicts of dicts

    result = {}
    for level_1_key in list_of_dicts[0].keys():
        level_2_keys = set.intersection(*[set(el[level_1_key].keys()) for el in list_of_dicts])
        if 'label' in level_2_keys:
            level_2_keys.remove('label')
        result[level_1_key] = {}

        for elem in list_of_dicts:
            for level_2_key in level_2_keys:
                item = elem[level_1_key][level_2_key]
                if isinstance(item, torch.Tensor):
                    if level_2_key not in result[level_1_key]:
                        result[level_1_key][level_2_key] = []
                    result[level_1_key][level_2_key].append(item.unsqueeze(dim))
                else:
                    logger.debug(f'{item} is not a tensor => dont add')

        for level_2_key in level_2_keys:
            tensors = result[level_1_key][level_2_key]
            result[level_1_key][level_2_key] = torch.cat(tensors, dim=dim)

    return result


def get_total_data_dim(data_dict):
    """
    :param data_dict: axis dict (input_data/output_data) from config, looks like:
         'output_dims': {
            'images': {'color_space': 'lab', 'dim': 3},
            'depth_map': {'color_space': 'grayscale', 'dim': 1}
            }

    :return: num of dims for example above 3 + 1 = 4
    """
    return sum(data_type['dim'] for data_type in data_dict.values())


def module_list_forward(module_list, tensor, spade_input=None):
    if spade_input:
        for layer in module_list:
            tensor = layer(tensor, spade_input)
    else:
        for layer in module_list:
            tensor = layer(tensor)

    return tensor


def split_tensor_to_maps(tensor, maps_types: dict) -> dict:
    output = {}
    i = 0
    for name, properties in maps_types.items():
        func = properties.get('func', 'tanh')
        output[name] = NONLINEARITIES[func](tensor[:, i:i + properties['dim']])
        i += properties['dim']
    return output


def atanh(x, eps=1e-5):
    """A poor realization of tanh^-1 (x)"""
    x = x.clamp(-1., 1.)
    out = 0.5 * (torch.log1p(x + eps) - torch.log1p(-x + eps))  # .clamp(-100., 100.)
    return out


def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                    nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]


def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


def init_weights(module: Union[torch.Tensor, nn.Module],
                 mode: str = 'xavier',
                 nonlinearity: str = 'relu',
                 gain: float = 1.0
                 ) -> None:
    name = module.__class__.__name__

    if name == 'Linear':
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.01)
    elif name.lower().startswith(('conv1d', 'conv2d', 'conv3d')):
        nn.init.constant_(module.bias, 0.01)
        if mode == 'xavier':
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif mode == 'kaiming':
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        else:
            raise ValueError(f'Unknown mode {mode}')
    elif name != 'LayerNormTotal' and name.startswith(('LayerNorm', 'BatchNorm')):
        nn.init.constant_(module.bias, 0.)
        nn.init.constant_(module.weight, 1.)


class Rodrigues(nn.Module):
    def __init__(self):
        super(Rodrigues, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3)


class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack((
            1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
        ), dim=1).view(-1, 3, 3)


def get_grid(batch_size: int,
             height: int,
             width: int,
             relative: bool = False,
             values_range: str = 'tanh',
             align_corners: bool = True,
             device='cpu',
             extend_scale: float = 1,
             ) -> torch.Tensor:
    """
    Build 2D grid with pixel coordinates UV,
    0 <= U < width and 0 <= V < height.

    Returns:
        grid: batch_size x height x width x UV
    """
    if not relative:
        xgrid = torch.arange(width, device=device)
        ygrid = torch.arange(height, device=device)
    else:
        if values_range == 'tanh':
            min_v, max_v = -1 * extend_scale, 1 * extend_scale
            if align_corners:
                min_val_x, max_val_x = min_val_y, max_val_y = min_v, max_v
            else:
                min_val_x, max_val_x = min_v + 1 / width,  max_v - 1 / width
                min_val_y, max_val_y = min_v + 1 / height, max_v - 1 / height
        elif values_range == 'sigmoid':
            extend_v = 1 * extend_scale - 1
            min_v, max_v = 0 - extend_v, 1 + extend_v
            if align_corners:
                min_val_x, max_val_x = min_val_y, max_val_y = min_v, max_v
            else:
                min_val_x, max_val_x = min_v + 0.5 / width, max_v - 0.5 / width
                min_val_y, max_val_y = min_v + 0.5 / height, max_v - 0.5 / height
        else:
            raise ValueError(str(values_range))
        xgrid = torch.linspace(min_val_x, max_val_x, width, device=device)
        ygrid = torch.linspace(min_val_y, max_val_y, height, device=device)

    grid = torch.stack(torch.meshgrid(ygrid, xgrid), dim=-1).flip(-1).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1)


def seed_freeze(x=0, base_seed=None, total_fix_seed=False):
    if base_seed is not None:
        seed = base_seed + x
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if total_fix_seed:
            np.random.seed(seed)
        else:
            np.random.seed()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        np.random.seed()


def min_max_scale(x: torch.Tensor,
                  dim: Union[None, int, Sequence[int]] = None,
                  mask: Optional[torch.BoolTensor] = None,
                  eps: float = 1e-9,
                  ):
    if mask is not None:
        assert mask.shape == x.shape
        x_for_min = torch.where(mask, x, torch.empty_like(x).fill_(float('inf')))
        x_for_max = torch.where(mask, x, torch.empty_like(x).fill_(float('-inf')))
    else:
        x_for_min = x_for_max = x

    if dim is None:
        min_value = x_for_min.min()
        max_value = x_for_max.max()
    else:
        if isinstance(dim, int):
            min_value = x_for_min.min(dim=dim, keepdim=True)[0]
            max_value = x_for_max.max(dim=dim, keepdim=True)[0]
        else:
            ndim = x.ndim
            arange = list(range(ndim))
            other_dims = [i for i in arange if i not in dim]
            forward_perm = other_dims + list(dim)
            backward_perm = [pair[0] for pair in sorted(zip(arange, forward_perm), key=lambda el: el[1])]

            y_for_min = x_for_min.permute(*forward_perm)
            y_for_min = y_for_min.contiguous().view(*y_for_min.shape[:len(other_dims)], -1)
            y_for_max = x_for_max.permute(*forward_perm)
            y_for_max = y_for_max.contiguous().view(*y_for_max.shape[:len(other_dims)], -1)

            min_value = y_for_min.min(dim=-1)[0][(...,) + (None,) * len(dim)].permute(*backward_perm)
            max_value = y_for_max.max(dim=-1)[0][(...,) + (None,) * len(dim)].permute(*backward_perm)
    return torch.clamp((x - min_value) / (max_value - min_value + eps), 0, 1)


def is_broadcastable(shp1, shp2):
    """https://stackoverflow.com/q/24743753"""
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

def product(args):
    """Like sum but product"""
    return functools.reduce(lambda x, y: x * y, args, 1)


def polynomialRollingHash(str, m=255, p=31):
    power_of_p = 1
    hash_val = 0

    for i in range(len(str)):
        hash_val = ((hash_val + (ord(str[i]) -
                                 ord('a') + 1) *
                     power_of_p) % m)

        power_of_p = (power_of_p * p) % m

    return int(hash_val)


def rgbFromStr(str):
    hash_r = polynomialRollingHash(str, m=255, p=401)
    hash_g = polynomialRollingHash(str, m=255, p=601)
    hash_b = polynomialRollingHash(str, m=255, p=701)
    return np.array([hash_r, hash_g, hash_b])


def load_class_from_config(module, config):
    class_type = config.pop('type')
    return getattr(module, class_type)(**config)


def crop_center_from_tensor(
        tensor: torch.Tensor,
        crop_size,
):
    """
    Central crop the tensor

    Args:
        tensor: B x C x H x W
        crop_size: [h, w]

    Returns:
        cropped_tensor: B x C x H_crop x W_crop
    """

    height, width = tensor.shape[-2:]

    crop_height, crop_width = crop_size
    assert (crop_height <= height) and (crop_width <= width), f"Crop size {crop_height, crop_width} " \
                                                              f"must be less then image size {height, width}! "

    crop_x = math.floor((width - crop_width) / 2)
    crop_y = math.floor((height - crop_height) / 2)

    cropped_tensor = tensor[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    return cropped_tensor


def print_tensor_anomaly(name, tensor):
    print(f'{name} :')
    print('nan:', torch.isnan(tensor).sum())
    print('inf:', torch.isinf(tensor).sum())
    print('shape:', tensor.shape)

def clone_module(module, memo=None) -> nn.Module:
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.cloned_module().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    """
    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # Create a copy of the module.
    if not isinstance(module, torch.nn.Module):
        return module
    cloned_module = module.__new__(type(module))
    cloned_module.__dict__ = module.__dict__.copy()
    cloned_module._parameters = cloned_module._parameters.copy()
    cloned_module._buffers = cloned_module._buffers.copy()
    cloned_module._modules = cloned_module._modules.copy()

    # Re-write all parameters
    if hasattr(cloned_module, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    cloned_module._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    cloned_module._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Handle the buffers if necessary
    if hasattr(cloned_module, '_buffers'):
        for buffer_key in module._buffers:
            if cloned_module._buffers[buffer_key] is not None and \
                    cloned_module._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    cloned_module._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    cloned_module._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    if hasattr(cloned_module, '_modules'):
        for module_key in cloned_module._modules:
            cloned_module._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    return cloned_module


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)
