import torch

from lib.modules.cameras.utils import average_extrinsics
from lib.modules.cameras import CameraMultiple


def average_cam(cameras: CameraMultiple):
    """
            Average cameras on last dim.
            Args:
                cameras: ... x N cameras

            Returns:
                cameras: ... x 1 cameras

            """
    poses_extrinsics = cameras.get_extrinsics()
    poses_intrinsics = cameras.get_intrinsics()
    new_extrinsic = average_extrinsics(poses_extrinsics)
    new_intrinsic = poses_intrinsics[..., [0], :, :].clone()
    new_cameras = CameraMultiple(extrinsics=new_extrinsic,
                                 intrinsics=new_intrinsic,
                                 images_sizes=cameras.images_size,
                                 )
    return new_cameras

def first_cam(cameras: CameraMultiple):
    """
            Average cameras on last dim.
            Args:
                cameras: ... x N cameras

            Returns:
                cameras: ... x 1 cameras

            """
    poses_extrinsics = cameras.get_extrinsics()
    poses_intrinsics = cameras.get_intrinsics()
    new_cameras = CameraMultiple(extrinsics=poses_extrinsics[..., [0], :, :].clone(),
                                 intrinsics=poses_intrinsics[..., [0], :, :].clone(),
                                 images_sizes=cameras.images_size,
                                 )
    return new_cameras

# class AverageCam:
#     def __call__(self,
#                  cameras: CameraMultiple
#                  ):
#         """
#         Average cameras on last dim.
#         Args:
#             cameras: ... x N cameras
#
#         Returns:
#             cameras: ... x 1 cameras
#
#         """
#         poses_extrinsics = cameras.get_extrinsics()
#         poses_intrinsics = cameras.get_intrinsics()
#         new_extrinsic = average_extrinsics(poses_extrinsics)
#         new_intrinsic = poses_intrinsics[..., [0], :, :].clone()
#         new_cameras = CameraMultiple(extrinsics=new_extrinsic,
#                                      intrinsics=new_intrinsic,
#                                      images_sizes=cameras.images_size,
#                                      )
#         return new_cameras
