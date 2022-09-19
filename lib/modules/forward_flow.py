"""
Current implementation of forward optical flow  is mostly borrowed from
https://github.sec.samsung.net/a-grigorev/PoseResynthesis/blob/master/modules/grid_sampler.py
"""

__all__ = ['forward_flow']

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def integer_meshgrid_tensor(*sizes: int,
                            device: Union[torch.device, str] = 'cpu',
                            ) -> torch.IntTensor:
    """
    Compute integer meshgrid.

    Args:
        *sizes: size per each axes, D_0 ..., D_{n-1}
        device:

    Returns:
        grid: meshgrid of shape (D_0, ..., D_{n-1}, n). Values of grid[..., k] are integers from range [0, D_k)

    """
    aranges = [torch.arange(cur_size, device=device) for cur_size in sizes]
    grids = torch.meshgrid(aranges)
    grid = torch.stack(grids, dim=-1)
    return grid


def ravel_multi_index(indices: Sequence[torch.IntTensor],
                      shape: Sequence[int],
                      ) -> torch.IntTensor:
    """
    Compute indices in the flattened tensor.

    Consider a tensor `x` of shape (D_0, ..., D_{n-1}).
    Assume `m` positions from this tensor with `n`-dimensional indices [i_10, ..., i_{1,n-1}], ...,
    [i_m0, ..., i_{m,n-1}].

    Let `y` be the flattened view of `x`. The output of this function is a 1-dimensional tensor with `m` integer values
    - indices for those `m` positions inside `y`.

    To understand the calculations, recap that the `flatten` operation starts from the last dim
    >>> x = torch.arange(2 * 3 * 4).view(2, 3, 4)
    >>> y = x.view(-1)
    >>> print(x); print(y)

    Therefore, if you look for the position [i_0, ..., i_{n-1}] of tensor `x` in the tensor `y`,
    the desired index j equals

    j = i_0*D_1*...*D_{n-1} + i_1*D_2*...*D_{n-1} + ... + i_{n-2}*D_{n-1} + i_{n-1}
      = (((i_0 * D_1 + i_1) * D_2 + i_2) * ... ) * D_{n-1} + i_{n-1}

    Args:
        indices: tuple of `n` 1-dimensional indices of length `m`:
            ([i_10, ..., i_m0], ..., [i_{1,n-1}, ..., i_{m,n-1}])
            The first element of the tuple contatins indices for the first axis of all the `m` positions, etc.
        shape: shape of the whole tensor:
            (D_0, ..., D_{n-1}).

    Returns:
        indices: 1-dimensional tensor of length `m`:
            [j_1, ..., j_m]
    """
    indices_ravel = indices[0]
    for i in range(1, len(indices)):
        indices_ravel = indices_ravel * shape[i] + indices[i]
    return indices_ravel


def add_repeated_(t: torch.Tensor,
                  indices: Sequence[torch.IntTensor],
                  values: torch.Tensor,
                  ) -> None:
    """
    Performs an in-place operation t[indices[0], indices[1], ..., indices[-1]] += values
    such that for each repeated cell in multi-index `indices` all values will be accounted
    in the summation. E.g. see:

    >>> A = torch.zeros(5)
    >>> A[[1, 1, 2]] += 1
    >>> A
    tensor([ 0.,  1.,  1.,  0.,  0.])

    >>> A = torch.zeros(5)
    >>> add_repeated_(A, (torch.tensor([1, 1, 2]),), torch.tensor([1., 1., 1.]))
    >>> A
    tensor([ 0.,  2.,  1.,  0.,  0.])

    PyTorch has a function torch.Tensor.index_add_() which solves this task only for flattened arrays.
    This is an adaptation for multi-dimensional arrays.

    Args:
        t: tensor to be edited, shape (D1, ..., Dn)
        indices: tuple of `m` LongTensors, all of shape (n,); indices[i] must have values in {0, 1, ..., Di - 1}
        values: torch Tensor of shape (m,)

    Returns:
        None.
    """

    shape = t.shape
    t_ravel = t.view(-1)
    indices_ravel = ravel_multi_index(indices, shape)
    t_ravel.index_add_(0, indices_ravel, values)


def forward_flow(x: Optional[torch.Tensor],
                 inv_grid: torch.Tensor,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 fill_value: float = 0.,
                 ) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Wrap tensor `x` with forward flow `inv_grid`.
    Pass x = None to compute only the mask of given warping.

    This function works as follows (a bit simplified):
    0. Switch from coordinates of range (-1, 1) to ranges (0, W-1) and (0, H-1) respectively.
    1. Initialize the output tensor wth zeros.
    2. Look for the nearest integer pixel indices for all the (generally, non-integer) values of the inv_grid and push
        the features from `x` to those integer pixel with the corresponding weight. The weights are computed from the
        bilinear kernel h(x, y) := max(0, 1 - |x|) * max(0, 1 - |y|).
    3. Normalize those weights for all the integer pixels and make the weights sum to one.

    Args:
        x: tensor of shape (B, C, H, W)
        inv_grid: tensor of shape (B, H, W, 2), values from range [-1, 1]. Note, that the coordinate inv_grid[..., 0]
            corresponds to the x-axis, and inv_grid[..., 1] - to the y-axis, respectively
            (as in torch.nn.functional.grid_sample function).
        height: height of the resulting tensor. If None, equal to the height of x.
        width: width of the resulting tensor. If None, equal to the width of x.
        fill_value: value to fill the uncovered pixels of the output tensor.

    Returns:
        output: wrapped tensor of shape (B, C, H_out, W_out).
        mask: boolean mask of covered pixels. True values correspond to the covered values,
            False - to the uncovered ones.
    """
    eps = 1e-10

    if x is not None:
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
    else:
        x = torch.ones(*inv_grid.shape[:-1], dtype=torch.float, device=inv_grid.device).unsqueeze(1)

    batch_size, n_channels, h, w = x.shape
    if width is not None:
        w = width
    if height is not None:
        h = height

    inv_grid = inv_grid.flip(-1)  # flip standard (x, y) grid coords to (y, x)
    inv_grid = (inv_grid.clone() + 1) / 2.0  # values in [0, 1]
    inv_grid[..., 0] *= h
    inv_grid[..., 1] *= w

    inv_grid += 1  # we convert [0, h] and [0, w] coordinate ranges to [1, h + 1], [1, w + 1]
    inv_grid = torch.stack([inv_grid[..., 0].clamp(0, h + 1 - 2 * eps),
                            inv_grid[..., 1].clamp(0, w + 1 - 2 * eps),
                            ], dim=-1)
    inv_grid = inv_grid.unsqueeze(1).expand(-1, n_channels, -1, -1, -1)
    output_cells = inv_grid.contiguous().view(-1, inv_grid.shape[-1])  # (B*(C+1)*H*W, 2)

    mgrid = integer_meshgrid_tensor(batch_size, n_channels, x.shape[2], x.shape[3], device=inv_grid.device)
    input_cells = mgrid.view(-1, mgrid.shape[-1])  # B*(C+1)*H*W x 4
    output_inds_b, output_inds_ch, _, _ = input_cells.unbind(-1)

    output = torch.zeros((batch_size, n_channels, h + 3, w + 3), device=x.device)
    for func_i in (torch.floor, torch.ceil):
        output_inds_i = func_i(output_cells[..., 0])
        bilinear_weights_i = F.relu(1 - torch.abs((output_cells[..., 0] - output_inds_i)))
        for func_j in (torch.floor, torch.ceil):
            output_inds_j = func_j(output_cells[..., 1])
            bilinear_weights_j = F.relu(1 - torch.abs((output_cells[..., 1] - output_inds_j)))

            add_repeated_(output,
                          (output_inds_b, output_inds_ch, output_inds_i.long(), output_inds_j.long()),
                          x.view(-1) * bilinear_weights_i * bilinear_weights_j)

    output = output[..., 1 : h + 1, 1 : w + 1]  # cutting out the border

    normalization_weights = output[:, -1:]
    mask = normalization_weights.ge(eps)

    if output.shape[1] == 1:
        return normalization_weights, mask

    unnormalized_features = output[:, :-1]
    output = torch.where(
        mask,
        unnormalized_features / normalization_weights.add(eps),
        torch.zeros_like(unnormalized_features) + fill_value,
    )
    return output, mask
