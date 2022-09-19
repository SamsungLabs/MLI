__all__ = ['Textures']

from typing import List, Optional, Union

import torch
from torch.nn.functional import interpolate

"""
This file has functions for interpolating textures after rasterization and utils
from https://github.com/facebookresearch/pytorch3d/blob/3d7dea58e150dae8070b3fea2f1deaf6f8c55106/pytorch3d/structures/textures.py
"""


def padded_to_list(x: torch.Tensor, split_size: Union[list, tuple, None] = None):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a list of N tensors
    of shape (Mi, Ki) where (Mi, Ki) is specified in split_size(i), or of shape
    (M, K) if split_size is None.
    Support only for 3-dimensional input tensor.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: a list of tensors
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        elif len(split_size[i]) == 2:
            x_list[i] = x_list[i][: split_size[i][0], : split_size[i][1]]
        else:
            raise ValueError(
                "Support only for 2-dimensional unbinded tensor. \
                    Split size for more dimensions provided"
            )
    return x_list


def padded_to_packed(
    x: torch.Tensor,
    split_size: Union[list, tuple, None] = None,
    pad_value: Union[float, int, None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a packed tensor
    of shape:
     - (sum(Mi), K) where (Mi, K) are the dimensions of
        each of the tensors in the batch and Mi is specified by split_size(i)
     - (N*M, K) if split_size is None

    Support only for 3-dimensional input tensor and 1-dimensional split size.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.
      pad_value: optional value to use to filter the padded values in the input
        tensor.

    Only one of split_size or pad_value should be provided, or both can be None.

    Returns:
      x_packed: a packed tensor.
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")

    N, M, D = x.shape

    if split_size is not None and pad_value is not None:
        raise ValueError("Only one of split_size or pad_value should be provided.")

    x_packed = x.reshape(-1, D)  # flatten padded

    if pad_value is None and split_size is None:
        return x_packed

    # Convert to packed using pad value
    if pad_value is not None:
        mask = x_packed.ne(pad_value).any(-1)
        x_packed = x_packed[mask]
        return x_packed

    # Convert to packed using split sizes
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    if not all(isinstance(i, int) for i in split_size):
        raise ValueError(
            "Support only 1-dimensional unbinded tensor. \
                Split size for more dimensions provided"
        )

    padded_to_packed_idx = torch.cat(
        [
            torch.arange(v, dtype=torch.int64, device=x.device) + i * M
            for (i, v) in enumerate(split_size)
        ],
        dim=0,
    )

    return x_packed[padded_to_packed_idx]


def _pad_texture_maps(images: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H, W, 3)

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W, 3)
    """
    tex_maps = []
    max_H = 0
    max_W = 0
    for im in images:
        h, w, _3 = im.shape
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = (max_H, max_W)

    for i, image in enumerate(tex_maps):
        if image.shape[:2] != max_shape:
            image_BCHW = image.permute(2, 0, 1)[None]
            new_image_BCHW = interpolate(
                image_BCHW, size=max_shape, mode="bilinear", align_corners=False
            )
            tex_maps[i] = new_image_BCHW[0].permute(1, 2, 0)
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, 3)
    return tex_maps


def _extend_tensor(input_tensor: torch.Tensor, N: int) -> torch.Tensor:
    """
    Extend a tensor `input_tensor` with ndim > 2, `N` times along the batch
    dimension. This is done in the following sequence of steps (where `B` is
    the batch dimension):

    .. code-block:: python

        input_tensor (B, ...)
        -> add leading empty dimension (1, B, ...)
        -> expand (N, B, ...)
        -> reshape (N * B, ...)

    Args:
        input_tensor: torch.Tensor with ndim > 2 representing a batched input.
        N: number of times to extend each element of the batch.
    """
    if input_tensor.ndim < 2:
        raise ValueError("Input tensor must have ndimensions >= 2.")
    B = input_tensor.shape[0]
    non_batch_dims = tuple(input_tensor.shape[1:])
    constant_dims = (-1,) * input_tensor.ndim  # these dims are not expanded.
    return (
        input_tensor.clone()[None, ...]
        .expand(N, *constant_dims)
        .transpose(0, 1)
        .reshape(N * B, *non_batch_dims)
    )


class Textures(object):
    def __init__(
        self,
        maps: Union[List, torch.Tensor, None] = None,
        faces_uvs: Optional[torch.Tensor] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        verts_rgb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, 3)] or a padded tensor of shape (N, H, W, 3).
            faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
                vertex in the face. Padding value is assumed to be -1.
            verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
            verts_rgb: (N, V, 3) tensor giving the rgb color per vertex. Padding
                value is assumed to be -1.

        Note: only the padded representation of the textures is stored
        and the packed/list representations are computed on the fly and
        not cached.
        """
        if faces_uvs is not None and faces_uvs.ndim != 3:
            msg = "Expected faces_uvs to be of shape (N, F, 3); got %r"
            raise ValueError(msg % repr(faces_uvs.shape))
        if verts_uvs is not None and verts_uvs.ndim != 3:
            msg = "Expected verts_uvs to be of shape (N, V, 2); got %r"
            raise ValueError(msg % repr(verts_uvs.shape))
        if verts_rgb is not None and verts_rgb.ndim != 3:
            msg = "Expected verts_rgb to be of shape (N, V, 3); got %r"
            raise ValueError(msg % repr(verts_rgb.shape))
        if maps is not None:
            if torch.is_tensor(maps) and maps.ndim != 4:
                msg = "Expected maps to be of shape (N, H, W, 3); got %r"
                raise ValueError(msg % repr(maps.shape))
            elif isinstance(maps, list):
                maps = _pad_texture_maps(maps)
            if faces_uvs is None or verts_uvs is None:
                msg = "To use maps, faces_uvs and verts_uvs are required"
                raise ValueError(msg)

        self._faces_uvs_padded = faces_uvs
        self._verts_uvs_padded = verts_uvs
        self._verts_rgb_padded = verts_rgb
        self._maps_padded = maps

        # The number of faces/verts for each mesh is
        # set inside the Meshes object when textures is
        # passed into the Meshes constructor.
        self._num_faces_per_mesh = None
        self._num_verts_per_mesh = None

    def clone(self):
        other = self.__class__()
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device):
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))
        return self

    def __getitem__(self, index):
        other = self.__class__()
        for key in dir(self):
            value = getattr(self, key)
            if torch.is_tensor(value):
                if isinstance(index, int):
                    setattr(other, key, value[index][None])
                else:
                    setattr(other, key, value[index])
        return other

    def faces_uvs_padded(self) -> torch.Tensor:
        return self._faces_uvs_padded

    def faces_uvs_list(self) -> Union[List[torch.Tensor], None]:
        if self._faces_uvs_padded is None:
            return None
        return padded_to_list(
            self._faces_uvs_padded,
            split_size=self._num_faces_per_mesh,
        )

    def faces_uvs_packed(self) -> Union[torch.Tensor, None]:
        if self._faces_uvs_padded is None:
            return None
        return padded_to_packed(
            self._faces_uvs_padded,
            split_size=self._num_faces_per_mesh,
        )

    def verts_uvs_padded(self) -> Union[torch.Tensor, None]:
        return self._verts_uvs_padded

    def verts_uvs_list(self) -> Union[List[torch.Tensor], None]:
        if self._verts_uvs_padded is None:
            return None
        # Vertices shared between multiple faces
        # may have a different uv coordinate for
        # each face so the num_verts_uvs_per_mesh
        # may be different from num_verts_per_mesh.
        # Therefore don't use any split_size.
        return padded_to_list(self._verts_uvs_padded)

    def verts_uvs_packed(self) -> Union[torch.Tensor, None]:
        if self._verts_uvs_padded is None:
            return None
        # Vertices shared between multiple faces
        # may have a different uv coordinate for
        # each face so the num_verts_uvs_per_mesh
        # may be different from num_verts_per_mesh.
        # Therefore don't use any split_size.
        return padded_to_packed(self._verts_uvs_padded)

    def verts_rgb_padded(self) -> Union[torch.Tensor, None]:
        return self._verts_rgb_padded

    def verts_rgb_list(self) -> Union[List[torch.Tensor], None]:
        if self._verts_rgb_padded is None:
            return None
        return padded_to_list(
            self._verts_rgb_padded,
            split_size=self._num_verts_per_mesh,
        )

    def verts_rgb_packed(self) -> Union[torch.Tensor, None]:
        if self._verts_rgb_padded is None:
            return None
        return padded_to_packed(
            self._verts_rgb_padded,
            split_size=self._num_verts_per_mesh,
        )

    # Currently only the padded maps are used.
    def maps_padded(self) -> Union[torch.Tensor, None]:
        return self._maps_padded

    def extend(self, N: int) -> "Textures":
        """
        Create new Textures class which contains each input texture N times

        Args:
            N: number of new copies of each texture.

        Returns:
            new Textures object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        if all(
            v is not None
            for v in [self._faces_uvs_padded, self._verts_uvs_padded, self._maps_padded]
        ):
            new_verts_uvs = _extend_tensor(self._verts_uvs_padded, N)
            new_faces_uvs = _extend_tensor(self._faces_uvs_padded, N)
            new_maps = _extend_tensor(self._maps_padded, N)
            return self.__class__(
                verts_uvs=new_verts_uvs, faces_uvs=new_faces_uvs, maps=new_maps
            )
        elif self._verts_rgb_padded is not None:
            new_verts_rgb = _extend_tensor(self._verts_rgb_padded, N)
            return self.__class__(verts_rgb=new_verts_rgb)
        else:
            msg = "Either vertex colors or texture maps are required."
            raise ValueError(msg)
