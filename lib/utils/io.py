from copy import deepcopy
import logging
import os
import sys
from typing import List, Union, Tuple, NamedTuple
import struct

import numpy as np
from PIL import Image
import cv2

try:
    from pyntcloud import PyntCloud
except:
    pass
from .coord_conversion import (change_coordinate_system_orientation_metashape,
                               change_coordinate_system_orientation_pyrender,
                               )

if sys.version_info.major < 3 or sys.version_info.minor < 7:  # Python < 3.7
    import oyaml as yaml
else:
    import yaml

logger = logging.getLogger(__name__)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def load_krt(path: str,
             world_center: List[float] = None,
             world_scale: float = None,
             image_rescale_ratio: float = None,
             change_orientation: str = None,
             image_size: Union[int, Tuple[int, int]] = None,
             center_crop: Union[int, Tuple[int, int]] = None,
             ) -> dict:
    """
    Load KRT file containing intrinsic and extrinsic parameters.

    Args:
        path (str): Path to KRT file.
        world_center (List[float]): Scene center in global coordinates.
        world_scale (float): World coordinates scale.
        image_rescale_ratio (float): Rescale image factor, if < 1 down-sample, else up-sample.
        change_orientation (str): Coordinate system format which used in krt file, None if system have normal format.
            Supported: pyrender | metashape | None .
            Detailed described in :func:`~coord_conversion.change_coordinate_system_orientation_metashape`
        image_size (Union[int, Tuple[int, int]]): Image sizes in pixels,
            if image sizes square you can pass scalar if not Tuple.
        center_crop (Union[int, Tuple[int]]): Center crop sizes in pixels,
            if crop sizes square you can pass scalar if not Tuple.

    Returns:
        Dictionary: dict with cameras parameters in normal coordinate system.

    """
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            dist = np.array([float(x) for x in f.readline().split()])
            extrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])

            if change_orientation:
                if change_orientation == 'pyrender':
                    extrin = change_coordinate_system_orientation_pyrender(extrin)
                elif change_orientation == 'metashape':
                    extrin = change_coordinate_system_orientation_metashape(extrin)
                else:
                    logger.info(f'Unsupported coordinates transformation: {change_orientation}')

            if center_crop is not None and image_size is not None:
                if np.isscalar(center_crop):
                    center_crop = [center_crop, center_crop]
                if np.isscalar(image_size):
                    image_size = [image_size, image_size]

                shift = ((np.array(image_size) - np.array(center_crop)) / 2).astype(np.uint)

                intrin[0, -1] = intrin[0, -1] - shift[1]
                intrin[1, -1] = intrin[1, -1] - shift[0]

            if image_rescale_ratio is not None:
                intrin = intrin * image_rescale_ratio

            if world_center is not None:
                extrin[:, -1:] = extrin[:, -1:] + extrin[:3, :3] @ world_center

            if world_scale is not None:
                extrin[:, -1:] = extrin[:, -1:] * world_scale

            f.readline()

            cameras[name[:-1]] = {
                'intrin': intrin.astype(np.float32),
                'dist': dist.astype(np.float32),
                'extrin': extrin.astype(np.float32),
            }

    return cameras


def load_pose(path):
    return np.genfromtxt(path, dtype=np.float32, skip_footer=2)


def load_roles(path):
    with open(path, 'r') as stream:
        split = yaml.load(stream, Loader=yaml.SafeLoader)
    split = {k: v for k, v in split.items() if k.startswith('scene_')}
    for scene in split.values():
        for time in scene.values():
            for role, cameras in time.items():
                time[role] = list(map(str, cameras))

    return split


def save_mpi_binary(mpi: np.ndarray,
                    focal: float,
                    basedir: str,
                    tag: str,
                    save_images: bool = False,
                    save_video: bool = False,
                    ) -> None:
    """
    Args:
        mpi: n_planes x RGBA x H x W, RGB values in range [-1, 1], alpha values in range [0, 1]
        focal: focal distance (w.r.t. Y axis)
        save_images: if True, save png images in addition to the binary file
        save_video: if True, video in addition to the binary file
        basedir: directory to save files
        tag: files tag
    """
    n_planes, channels, height, width = mpi.shape
    if channels != 4:
        logger.warning(f'Expected to get RGBA layers, but obtained {channels} channels, '
                       f'for visualizes will used first 3 as RGB and last as alpha')
        mpi = np.concatenate([mpi[:, :3], mpi[:, -1:]], axis=1)
    mpi = np.concatenate([mpi[:, :-1] * 0.5 + 0.5, mpi[:, -1:]], axis=1)
    mpi = mpi.transpose((0, 2, 3, 1))
    mpi_alpha_blended = mpi[:, :, :, :-1] * mpi[:, :, :, -1:]
    mpi = (255 * np.clip(mpi, 0, 1)).astype(np.uint8)
    mpi_alpha_blended = (255 * np.clip(mpi_alpha_blended, 0, 1)).astype(np.uint8)

    os.makedirs(basedir, exist_ok=True)
    with open(os.path.join(basedir, 'metadata.txt'), 'w') as f:
        f.write(f'1 {width} {height} {n_planes}\n')

    os.makedirs(os.path.join(basedir, tag,), exist_ok=True)
    with open(os.path.join(basedir, tag, 'metadata.txt'), 'w') as f:
        f.write(f'{height} {width} {n_planes} {focal:.6f}\n')

        # extrinsics in OpenGL camera format (and transposed -> 4x3 instead of 3x4)
        f.write('0 -1 0\n')
        f.write('1 0 0\n')
        f.write('0 0 -1\n')
        f.write('0 0 0\n')

        # far and near plane
        f.write('100 1\n')

    with open(os.path.join(basedir, tag, 'mpi.b'), 'wb') as f:
        f.write(mpi.tobytes())
    if save_images or save_video:
        os.makedirs(os.path.join(basedir, tag, 'layers'), exist_ok=True)
    if save_images:
        for i, layer in enumerate(mpi):
            Image.fromarray(layer).save(os.path.join(basedir, tag, 'layers', f'layer_{i:03d}.png'))
    if save_video:
        video = cv2.VideoWriter(os.path.join(basedir, 'mpi_' + tag + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        for i, layer in enumerate(mpi_alpha_blended):
            video.write(cv2.cvtColor(layer, cv2.COLOR_RGB2BGR))
        video.release()


def load_colmap_binary_cloud(filename: str) -> np.ndarray:
    """
    See
    https://github.com/colmap/colmap/blob/d84ccf885547c6e91c58b2e5dab8a6a34e636be3/scripts/python/read_write_model.py#L128
    """
    list_points = []

    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    with open(filename, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)

            list_points.append(xyz)
    return np.stack(list_points, axis=0)


def load_colmap_fused_ply(path_to_fused_ply):
    point_cloud = PyntCloud.from_file(path_to_fused_ply)
    xyz_arr = point_cloud.points.loc[:, ["x", "y", "z"]].to_numpy()
    return xyz_arr


def load_colmap_bin_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def save_pointcloud_to_ply(filename, xyz_points, rgb_points=None):
    """
    Creates a .pkl file of the point clouds generated

    Args:
        filename:
        xyz_points:
        rgb_points: from 0 to 255

    """

    assert xyz_points.shape[1] == 3, 'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, \
        'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    xyz_points = xyz_points.astype(float)
    rgb_points = rgb_points.astype(np.uint8)
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",
                                        xyz_points[i, 0],
                                        xyz_points[i, 1],
                                        xyz_points[i, 2],
                                        rgb_points[i, 0].tostring(),
                                        rgb_points[i, 1].tostring(),
                                        rgb_points[i, 2].tostring())))
    fid.close()


class ModifiedConfig(NamedTuple):
    status: bool
    config: dict


def modify_mpi_config_to_produce_mesh(config: dict,
                                      faces_per_pixel: int = 64,
                                      resolution: int = 256,
                                      n_groups: int = 8,
                                      n_samples: int = 20,
                                      sigma: float = 0.1,
                                      ) -> ModifiedConfig:
    trainer_type = config['trainer']
    gen_type = config['models']['gen']['architecture']
    if not (trainer_type == 'TrainerStereoMagnification' and gen_type == 'GeneratorStereoMagnification'):
        return ModifiedConfig(False, config)

    new_config = deepcopy(config)
    new_config['trainer'] = 'TrainerDeformedPSV'
    gen_config = new_config['models']['gen']
    gen_config['architecture'] = 'GeneratorStereoMagnification2Mesh'
    gen_config['n_groups'] = n_groups
    gen_config['n_samples'] = n_samples
    gen_config['sigma'] = sigma
    gen_config['modules']['rasterizer'] = dict(architecture='RasterizerMesh',
                                               faces_per_pixel=faces_per_pixel,
                                               image_size=resolution,
                                               perspective_correct=True,
                                               )
    gen_config['modules']['shader'] = dict(architecture='MPIShader')
    for dset in ['render', 'val']:
        dset_config = new_config['dataloaders'][dset]['dataset']
        if dset_config['type'] == 'JoinedDataset':
            for dset_part_config in dset_config['concatenated_datasets']:
                dset_part_config['relative_intrinsics'] = True
        else:
            dset_config['relative_intrinsics'] = True

    return ModifiedConfig(True, new_config)
