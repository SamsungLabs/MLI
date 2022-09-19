from numbers import Number
from typing import Sequence, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ###################### ROTATION MATRIX <---> QUATERNION ######################
# This part is borrowed from Kornia:
# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py
# ##############################################################################

def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor,
                                  eps: float = 1e-8,
                                  ) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (x, y, z, w) format.
    Args:
        rotation_matrix (torch.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        torch.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor,
                           denominator: torch.Tensor,
                           ) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(
        *rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qx, qy, qz, qw], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22),
                          cond_1(),
                          where_2)

    quaternion: torch.Tensor = torch.where(trace > 0., trace_positive_cond(), where_1)
    return normalize_quaternion(quaternion)


def normalize_quaternion(quaternion: torch.Tensor,
                         eps: float = 1e-7,
                         ) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-7.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    Example:
        >>> quaternion = torch.tensor([1., 0., 1., 0.])
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = torch.tensor([0., 0., 1., 0.])
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.)

    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1).reshape(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix

# ##############################################################################


def slerp_unit_vectors(start: torch.Tensor,
                       end: torch.Tensor,
                       timestamp: Union[float, Sequence[float], torch.Tensor],
                       quaternions: bool = True,
                       ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Implementation based on https://en.wikipedia.org/wiki/Slerp
    The last axis of `start` and `end` is treated as the vector coordinates.
    Vectors are assumed normalized.

    Args:
        start:
        end:
        timestamp: examples:
            - 0.75
            - (0.25, 0.5, 0.75)
            - torch.tensor([0.25, 0.5, 0.75])
        quaternions: if True, vectors of opposite directions are treated as equivalent ones (as they correspond
            to the same rotation matrix)

    """
    if isinstance(timestamp, Number):
        single_timestamp = True
        timestamp = torch.tensor([timestamp], dtype=torch.float, device=start.device)
    else:
        single_timestamp = False
        if torch.is_tensor(timestamp):
            assert timestamp.ndim == 1
            timestamp = timestamp.to(start.device)
        else:
            timestamp = torch.tensor(timestamp, dtype=torch.float, device=start.device)

    assert start.shape == end.shape

    # vectors are assumed to have unit norm
    # start, end: B x XYZW
    dot = torch.sum(start * end, dim=-1, keepdim=True)  # B x 1
    if quaternions:
        start = torch.where(dot.lt(0).expand_as(start),
                            -start,
                            start)
        dot[dot < 0] *= -1

    angle = torch.acos(dot)  # B x 1
    sin_angle = torch.sin(angle)
    start_sin_angle = torch.sin(angle * (1 - timestamp))  # B x T
    end_sin_angle = torch.sin(angle * timestamp)

    # B x XYZW x T
    slerp = (start.unsqueeze(-1) * start_sin_angle.unsqueeze(-2)
             + end.unsqueeze(-1) * end_sin_angle.unsqueeze(-2)
             ) / sin_angle.unsqueeze(-1)

    dot_threshold = 0.9995
    if dot.gt(dot_threshold).any():
        lerp = start.unsqueeze(-1) * (1 - timestamp) + end.unsqueeze(-1) * timestamp
        slerp = torch.where(dot.gt(dot_threshold).unsqueeze(-1).expand_as(lerp),
                            lerp,
                            slerp)

    slerp = F.normalize(slerp, dim=-2)
    slerp = slerp.unbind(-1)
    if single_timestamp:
        slerp = slerp[0]
    return slerp


def interpolate_rotation_matrices(start: torch.Tensor,
                                  end: torch.Tensor,
                                  timestamp: Union[float, Sequence[float], torch.Tensor],
                                  ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Args:
        start: ... x 3 x 3
        end:   ... x 3 x 3
        timestamp: examples:
            - 0.75
            - (0.25, 0.5, 0.75)
            - torch.tensor([0.25, 0.5, 0.75])
            - tensor of shape ... x n_steps
    """
    batch_shape = start.shape[:-2]
    if isinstance(timestamp, Number):
        single_timestamp = True
        shared_timestamps = None
        timestamp = torch.tensor([timestamp], dtype=torch.float, device=start.device)
    else:
        single_timestamp = False
        if torch.is_tensor(timestamp):
            shared_timestamps = timestamp.ndim == 1
            timestamp = timestamp.to(start.device)
        else:
            timestamp = torch.tensor(timestamp, dtype=torch.float, device=start.device)
            shared_timestamps = True

    weights = torch.stack([1 - timestamp, timestamp], dim=-1)  # n_steps x 2
    weights = weights.expand(*batch_shape, -1, -1)  # ... x n_steps x 2
    # ... x n_steps x 1 x 3 x 3
    start = start[..., None, None, :, :].expand(*batch_shape, timestamp.shape[-1], 1, -1, -1)
    end = end[..., None, None, :, :].expand(*batch_shape, timestamp.shape[-1], 1, -1, -1)

    # ... x n_steps x 3 x 3
    interpolated_matrices = average_rotation_matrices(
        torch.cat([start, end], dim=-3),
        weights=weights,
        keepdim=False,
    )

    if single_timestamp:
        return interpolated_matrices.squeeze(-3)
    elif not shared_timestamps:
        return interpolated_matrices
    else:
        return interpolated_matrices.unbind(dim=-3)


def average_rotation_matrices(rotation_matrices: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              keepdim: bool = False,
                              ) -> torch.Tensor:
    """
        Args:
            rotation_matrices: ... x N x 3 x 3
            weights: ... x N
            keepdim:
        Returns:
            torch.Tensor: ... x 3 x 3
    """
    batch_shape = rotation_matrices.shape[:-3]
    samples = rotation_matrices.shape[-3]
    rotation_matrices = rotation_matrices.reshape(-1, 3, 3)
    rotation_quaternions = rotation_matrix_to_quaternion(rotation_matrices)
    rotation_quaternions = rotation_quaternions.reshape(-1, samples, 4)
    if weights is not None:
        weights = weights.reshape(-1, samples)
    avg_quaternion = average_quaternions(rotation_quaternions, weights=weights)
    avg_quaternion = avg_quaternion.reshape(-1, 4)
    avg_matrix = quaternion_to_rotation_matrix(avg_quaternion).reshape(*batch_shape, 3, 3)
    return avg_matrix.unsqueeze(-3) if keepdim else avg_matrix


def average_quaternions(quaternions: torch.Tensor,
                        weights: Optional[torch.Tensor] = None,
                        keepdim: bool = False,
                        ):
    """
    Calculate average quaternion with the method described in the paper
    Markley, F Landis, Yang Cheng, John L Crassidis, and Yaakov Oshman. “Averaging Quaternions.”
    https://doi.org/10.2514/1.28949.

    Args:
        quaternions: ... x N x 4 - tensor and contains the quaternions to average in the rows.
            The quaternions are arranged as (x,y,z,w), with w being the scalar
        weights: ... x N - weight of each quaternion
        keepdim:

    Returns:
            torch.Tensor: ... x 4
    """

    # Number of quaternions to average
    samples = quaternions.shape[-2]
    batch_shape = quaternions.shape[:-2]
    quaternions = quaternions.reshape(-1, samples, 4)

    if weights is None:
        weights = 1. / samples
    else:
        assert weights.shape[-1] == samples, \
            f'number of weights {weights.shape[-1]} does not equal to number of quaternions {samples}'
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        weights = weights.reshape(-1, samples, 1).to(quaternions.device)

    # B x 4 X 4
    mat_a = torch.einsum('bni,bnj->bij', [quaternions * weights, quaternions])
    # compute eigenvalues and -vectors
    _, eigen_vectors = torch.symeig(mat_a, eigenvectors=True)
    max_value_eigenvector = eigen_vectors[..., -1].reshape(*batch_shape, 4)

    return max_value_eigenvector.unsqueeze(-2) if keepdim else max_value_eigenvector


def quanterion_mult(q: torch.Tensor, r: torch.Tensor):
    """
      Multiply quaternion(s) q with quaternion(s) r.
      Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
      Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def quanterion_between_two_vectors(vector_a: torch.Tensor, vector_b: torch.Tensor):
    """
    Quaternion between two vectors

    Args:
        vector_a: N x 3
        vector_b: N x 3
    Returns:
            torch.Tensor: N x 3 x 3
    """
    vector_a_norm = vector_a / torch.linalg.norm(vector_a, dim=1)
    vector_b_norm = vector_b / torch.linalg.norm(vector_b, dim=1)
    half = vector_a_norm + vector_b_norm
    half = half / torch.linalg.norm(half, dim=1)
    w = torch.sum(vector_a_norm * half, dim=1)
    xyz = torch.cross(vector_a_norm, half, dim=1)
    quaternion = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return quaternion


def rotation_matrix_between_two_vectors(vector_a: torch.Tensor, vector_b: torch.Tensor):
    """
    Calculate rotation matrix between two vectors

    Args:
        vector_a: N x 3
        vector_b: N x 3
    Returns:
            torch.Tensor: N x 3 x 3
    """
    vector_a_norm = vector_a / torch.linalg.norm(vector_a, dim=1)
    vector_b_norm = vector_b / torch.linalg.norm(vector_b, dim=1)
    v = torch.cross(vector_a_norm, vector_b_norm, dim=1)
    cos = torch.sum(vector_a_norm * vector_b_norm, dim=1)
    sin = torch.linalg.norm(v, dim=1)
    kmat = torch.zeros([vector_a_norm.shape[0], 3, 3])
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rotation_matrix = torch.eye(3) + kmat + torch.sum(kmat * kmat, dim=(1, 2)) * ((1 - cos) / (sin ** 2))
    return rotation_matrix
