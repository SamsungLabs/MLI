"""
RealEstate10k dataset should have the following structure:
| root/:
|---- videos/
    |---- scene_id1
        |---- frame1.jpg
        |---- frame2.jpg
    |---- scene_id2
        |---- frame1.jpg
        |---- frame2.jpg
|---- scene_id1.txt
|---- scene_id2.txt
"""

__all__ = ['ImageFolder']

import os
import logging
import math
from glob import glob
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from lib.utils.coord_conversion import coords_pixel_to_film

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ImageFolder(Dataset):
    def __init__(self,
                 root: str,
                 image_size: Tuple[int, int],
                 crop_size: Optional[Tuple[int, int]] = None,
                 crop_mode: str = 'center',
                 relative_intrinsics: bool = False,
                 focal_length: int = 647,
                 ):
        """

        Args:
            root: root folder
            image_size: height, width for resize
            crop_size: height, width for crop after resize
            crop_mode: center | random. The selected mode is applied to all the images.
            relative_intrinsics: enable relative intrinsics format
        Returns:
            dict of camera types with dicts of torch.Tensor for each camera, stacked along dim=0
        """
        super().__init__()
        self.transforms = [transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           ]
        self.transforms = transforms.Compose(self.transforms)
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]
        else:
            self.image_size = image_size
        self.path = root
        self.focal_length = focal_length
        self.scenes_data = {}
        self.images_paths = glob(os.path.join(self.path, '*.jpg'))
        self.relative_intrinsics = relative_intrinsics
        self.crop_mode = crop_mode
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        for image_path in tqdm(self.images_paths):
            scene_id = image_path.split('/')[-1].strip()
            image = Image.open(image_path)
            image = self.transforms(image)
            _, w, h = image.shape
            (focal_length_x,
             focal_length_y,
             principal_point_x,
             principal_point_y) = (w / focal_length, h / focal_length, 0.500000000, 0.500000000)

            intrinsics = np.array([[focal_length_x, 0, principal_point_x],
                                   [0, focal_length_y, principal_point_y],
                                   [0, 0, 1]])

            extrinsics = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)

            self.scenes_data[scene_id] = {'images_path': image_path,
                                          'extrinsics': extrinsics,
                                          'intrinsics': intrinsics,
                                          'image_size': [w, h]}

    def __len__(self):
        return len(self.scenes_data)

    def __getitem__(self, idx):
        scene_id = list(self.scenes_data.keys())[idx]

        data = self._read_frame(scene_id)

        time_id = 0
        output = {
            'time_id': time_id,
            'scene_time': f'{scene_id}//{time_id}',
            'scene_id': scene_id,
            'reference': data,
            'initial': data,
            'novel': data,
        }

        return output

    def _transform_extrinsics(self,
                              extrinsics: torch.Tensor,
                              reference_extrinsics: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.set_reference_as_origin:
            return extrinsics, reference_extrinsics

        inverse_reference_extrinsics = torch.inverse(
            reference_extrinsics.expand_as(extrinsics)[..., :3]
        )
        transformed_extrinsics = extrinsics.clone()
        transformed_extrinsics[..., :3] = torch.einsum('...ij,...jk->...ik',
                                                       [extrinsics[..., :3], inverse_reference_extrinsics])
        transformed_extrinsics[..., 3] = extrinsics[..., 3] - reference_extrinsics[..., 3]
        return (transformed_extrinsics,
                torch.cat([torch.eye(3, 3), torch.zeros(3, 1)], dim=1).expand_as(reference_extrinsics),
                )

    @staticmethod
    def _relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic * scaling

    @staticmethod
    def _absolute_intrinsic_to_relative(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
        scaling = torch.tensor([width, height, 1.]).view(-1, 1)
        return intrinsic / scaling

    def _read_frame(self,
                    scene_id: str,
                    ) -> Dict[str, torch.Tensor]:

        image_path: str = self.scenes_data[scene_id]['images_path']
        try:
            with Image.open(image_path) as img:
                current_image = self.transforms(img)
        except OSError as e:
            logger.error(f'Possibly, image file is broken: {image_path}')
            raise e

        current_extr = torch.from_numpy(self.scenes_data[scene_id]['extrinsics']).float()
        relative_intr = torch.from_numpy(self.scenes_data[scene_id]['intrinsics']).float()

        if not self.relative_intrinsics or self.crop_size is not None:
            absolute_intr = self._relative_intrinsic_to_absolute(*self.scenes_data[scene_id]['image_size'],
                                                                 relative_intr)

        if self.crop_size is not None:
            current_image, absolute_intr = self._crop_data(current_image, absolute_intr)

        if self.relative_intrinsics:
            if self.crop_size is not None:
                current_intr = self._absolute_intrinsic_to_relative(*self.crop_size, absolute_intr)
            else:
                current_intr = relative_intr
        else:
            current_intr = absolute_intr

        images = current_image.unsqueeze(0)
        extrinsics = current_extr.unsqueeze(0)
        intrinsics = current_intr.unsqueeze(0)

        return {
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

    def _crop_data(self, image: torch.Tensor, intrinsic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random crop the image and transform the absolute intrinsics

        Args:
            image: C x H x W
            intrinsic: 3 x 3

        Returns:
            cropped_image: C x H_crop x W_crop
            cropped_intrinsic: 3 x 3
        """
        _, height, width = image.shape
        crop_height, crop_width = self.crop_size
        if self.crop_mode == 'random':
            crop_x = np.random.randint(0, width - crop_width + 1)
            crop_y = np.random.randint(0, height - crop_height + 1)
        elif self.crop_mode == 'center':
            crop_x = math.floor((width - crop_width) / 2)
            crop_y = math.floor((height - crop_height) / 2)
        else:
            raise ValueError(f'Unknown crop_mode={self.crop_mode}')
        cropped_image = image[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float).view(1, 1, -1)
        film_coords = coords_pixel_to_film(pixel_coords, intrinsic.unsqueeze(0))[0, 0]
        new_principal_point = - film_coords * torch.diagonal(intrinsic[:-1, :-1], dim1=0, dim2=1)
        cropped_intrinsic = intrinsic.clone()
        cropped_intrinsic[:-1, -1] = new_principal_point

        return cropped_image, cropped_intrinsic
