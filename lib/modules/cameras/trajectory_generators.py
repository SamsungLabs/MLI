from typing import Union, List, Tuple

import numpy as np
import torch

from lib.utils.geometry import quanterion_between_two_vectors, rotation_matrix_to_quaternion, \
    quaternion_to_rotation_matrix, quanterion_mult


def generate_cross_poses(ref_pose: torch.Tensor,
                         offset: Union[int, float] = 5,
                         direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                         direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                         focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                         scale_y: Union[int, float] = 0.5,
                         num_frames: int = 30,
                         ) -> List[torch.Tensor]:
    """
    Function generates poses for smooth video rendering
    Args:
        ref_pose: pose of the mpi reference camera
        offset: maximum offset the camera can shift along x axis
        direction_x: direction of the horizontal shift, camera coordinates
        direction_y: direction of the vertical shift, camera coordinates
        focal: focal values for rescaling the offsets
        scale_y: ratio between maximum vertical and horizontal shifts. Smaller vertical shift looks more plausible
        num_frames: number of frames to generate

    Returns:

    """
    offsets = np.linspace(-offset, offset, num_frames // 2)
    render_poses = []
    for idx, o in enumerate(offsets):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets[::-1]):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal + \
                              np.sin(np.pi * idx / (0.5 * num_frames)) * direction_y * offset * scale_y / focal
        render_poses.append(new_pose)
    for idx, o in enumerate(offsets[::-1]):
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * o / focal - \
                              np.sin(np.pi * idx / (0.5 * num_frames)) * direction_y * offset * scale_y / focal
        render_poses.append(new_pose)

    return render_poses


def generate_spiral_poses(ref_pose: torch.Tensor,
                          offset: Union[int, float] = 5,
                          depth_limits: Tuple[float, float] = (-0.1, 0.2),
                          direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                          direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                          direction_z: torch.Tensor = torch.tensor([0, 0, 1], dtype=torch.float),
                          focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                          scale_y: Union[int, float] = 1,
                          num_frames: int = 30,
                          ) -> List[torch.Tensor]:
    timestamps = np.linspace(-2, 2, num_frames)
    min_depth, max_depth = depth_limits
    render_poses = []
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * offset / focal * np.cos(np.pi * t)
        new_pose[0, :3, 3] += direction_y * offset / focal * np.sin(np.pi * t) * scale_y
        new_pose[0, :3, 3] -= direction_z * ((0.5 * np.cos(np.pi * t / 2) + 0.5) * (max_depth - min_depth) + min_depth)
        render_poses.append(new_pose)
    return render_poses


def generate_spiral_poses_centered(ref_pose: torch.Tensor,
                                   offset: Union[int, float] = 5,
                                   depth_limits: Tuple[float, float] = (-0.5, 0.5),
                                   direction_x: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                                   direction_y: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float),
                                   direction_z: torch.Tensor = torch.tensor([0, 0, 1], dtype=torch.float),
                                   focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                                   scale_y: Union[int, float] = 1,
                                   num_frames: int = 30,
                                   ) -> List[torch.Tensor]:
    focus_point = torch.tensor([[0.0, 0.0, 1.0]]) * 4
    timestamps = np.linspace(-2, 2, num_frames)
    min_depth, max_depth = depth_limits
    render_poses = []
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction_x * offset / focal * np.cos(np.pi * t)
        new_pose[0, :3, 3] += direction_y * offset / focal * np.sin(np.pi * t) * scale_y
        new_pose[0, :3, 3] -= direction_z * ((0.5 * np.cos(np.pi * t / 2) + 0.5) * (max_depth - min_depth) + min_depth)

        dir_vector = focus_point - new_pose[0, :3, 3]
        quanterion = quanterion_between_two_vectors(torch.tensor([[0.0, 0.0, 1.0]]), dir_vector)
        quanterion_ref = rotation_matrix_to_quaternion(ref_pose[0, :3, :3])
        new_pose[0, :3, :3] = quaternion_to_rotation_matrix(quanterion_mult(quanterion_ref, quanterion))

        render_poses.append(new_pose)
    return render_poses


def generate_along_direction(ref_pose: torch.Tensor,
                             limits: Tuple[float, float],
                             direction: torch.Tensor = torch.tensor([1, 0, 0], dtype=torch.float),
                             focal: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float),
                             num_frames: int = 30,
                             ) -> List[torch.Tensor]:
    timestamps = np.linspace(-1, 1, num_frames)
    render_poses = []
    min_val, max_val = limits
    for t in timestamps:
        new_pose = ref_pose.clone()
        new_pose[0, :3, 3] += direction / focal * ((0.5 * np.cos(np.pi * t) + 0.5) * (max_val - min_val) + min_val)
        render_poses.append(new_pose)
    return render_poses


def normalize(x):
    return x / torch.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], axis=1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    return viewmatrix(vec2, up, center)


def generate_spiral_poses_with_focus(ref_pose,
                                     min_depth,
                                     max_depth,
                                     num_frames,
                                     offset=0.2,
                                     num_rotations=2,
                                     zrate=0.5):
    # translation = ref_pose[0, :3, -1:]
    # rotation = ref_pose[0, :3, :3]
    # translation = -rotation.transpose(-1, -2) @ translation
    # c2w = torch.cat([rotation, translation], axis=1)
    #
    # ## Get spiral
    # # Get average pose
    # up = c2w[:3, 1]
    #
    # # Find a reasonable "focus depth" for this dataset
    # close_depth, inf_depth = min_depth * .9, max_depth * 5.
    # dt = .75
    # mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    # focal = mean_dz
    # render_poses = []
    #
    # for theta in np.linspace(0., 2. * np.pi * num_rotations, num_frames + 1)[:-1]:
    #     c = c2w[:3, :4] @ torch.tensor([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.],
    #                                    dtype=torch.float) * offset
    #     z = normalize(c - c2w[:3, :4] @ torch.tensor([0, 0, -focal, 1.], dtype=torch.float))
    #     render_poses.append(viewmatrix(z, up, c))
    # render_poses = torch.stack(render_poses, axis=0)
    # render_poses = torch.cat([render_poses[:, :3, :3], -render_poses[:, :3, :3] @ render_poses[:, :3, -1:]], dim=-1)
    #
    # return render_poses

    translation = ref_pose[0, :3, -1:]
    rotation = ref_pose[0, :3, :3]
    translation = -rotation.transpose(-1, -2) @ translation
    c2w = torch.cat([torch.eye(3), translation * 0], axis=1)

    up = c2w[:3, 1]

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = min_depth * .9, max_depth * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    zdelta = close_depth * .2

    c2w_path = c2w

    render_poses = []

    for theta in np.linspace(0., 2. * np.pi * num_rotations, num_frames + 1)[:-1]:
        c = c2w[:3, :4] @ torch.tensor([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.],
                                       dtype=torch.float) * offset
        z = normalize(c - c2w[:3, :4] @ torch.tensor([0, 0, -focal, 1.], dtype=torch.float))
        render_poses.append(viewmatrix(z, up, c))

    render_poses = torch.stack(render_poses, axis=0)
    render_poses = torch.cat([rotation @ render_poses[:, :3, :3],
                              -render_poses[:, :3, :3] @ (render_poses[:, :3, -1:] + translation)],
                             dim=-1)

    return render_poses
