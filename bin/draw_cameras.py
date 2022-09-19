import argparse
from itertools import combinations, product

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # unused, but necessary import statement for 3d plots

import numpy as np
import torch

from lib.modules.cameras import CameraPinhole
from lib.utils.io import load_krt, load_pose

DEFAULT_POSE_PATH = None
DEFAULT_KRT_PATH = 'volumetric_data/data/axis_multiview/scene_0/time_0/KRT.txt'


def main():
    parser = argparse.ArgumentParser('Draw cameras for the scene.')
    parser.add_argument('--krt-path', type=str, default=DEFAULT_KRT_PATH,
                        help='Path to the KRT file.')
    parser.add_argument('--pose-path', type=str, default=DEFAULT_POSE_PATH,
                        help='Path to the pose file.')
    parser.add_argument('--world-scale', type=float, default=0.2,
                        help='World scale multiplier.')
    parser.add_argument('--orientation', type=str, default=False,
                        help='Orientation modification: metashape | pyrender')
    parser.add_argument('--n-cameras', type=int, default=10000,
                        help='How many cameras to show')
    opts = parser.parse_args()

    if opts.pose_path is None:
        world_center = None
    else:
        pose = load_pose(opts.pose_path)
        world_center = pose[:, -1:]
    krt = load_krt(opts.krt_path,
                   world_center=world_center,
                   world_scale=opts.world_scale,
                   change_orientation=opts.orientation,
                   )

    cam_names = list(krt.keys())
    extrinsics = torch.from_numpy(np.stack([krt[cam]['extrin'] for cam in cam_names])).float()
    intrinsics = torch.from_numpy(np.stack([krt[cam]['intrin'] for cam in cam_names])).float()

    n_cameras = min(opts.n_cameras, len(extrinsics))
    idx = torch.from_numpy(
        np.random.choice(len(extrinsics), n_cameras, replace=False)
    ).long()
    cameras = CameraPinhole(extrinsics=extrinsics[idx],
                            intrinsics=intrinsics[idx],
                            )

    scale = 0.2
    cam_poses = cameras.world_position.numpy()
    shift_z = cameras.world_view_direction(axis='z').numpy()
    shift_x = cameras.world_view_direction(axis='x').numpy() * scale
    shift_y = cameras.world_view_direction(axis='y').numpy() * scale

    ax = plt.axes(projection='3d')

    # draw camera Z axis
    ax.quiver(*cam_poses.T, *shift_z.T, colors=['b'] * 3)
    ax.scatter3D(*cam_poses.T, c='g')
    ax.scatter3D(*(cam_poses + shift_z).T, c='r')

    # draw camera X and Y axis
    ax.quiver(*cam_poses.T, *shift_x.T, colors=['r'] * 3)
    ax.quiver(*cam_poses.T, *shift_y.T, colors=['g'] * 3)

    # draw camera names
    for name, pose in zip(cam_names, cam_poses):
        ax.text(*pose, name, color='black')

    # draw unit cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="m")

    # draw world coordinate system
    ax.quiver(0, 0, 0, 2, 0, 0, colors=['r'] * 3)
    ax.quiver(0, 0, 0, 0, 2, 0, colors=['g'] * 3)
    ax.quiver(0, 0, 0, 0, 0, 2, colors=['b'] * 3)

    plt.show()


if __name__ == '__main__':
    main()
