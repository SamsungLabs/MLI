__all__ = ['SurfacesBase']

from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch

from lib.modules.cameras import CameraPinhole


class SurfacesBase:
    SURFCACES_MODES: Tuple[str, ...] = (
        'disparity',
        'depth',
        'disparity-random',
        'depth-random',
        'disparity-quasirandom',
        'depth-quasirandom',
    )

    @abstractmethod
    def set_position(self, *args, **kwargs) -> None:
        """Set the position of surfaces in the space"""
        pass

    def _build_surfaces(self,
                        n_surfaces: Optional[int],
                        min_distance: Optional[float],
                        max_distance: Optional[float],
                        mode: str = 'disparity',
                        device: Union[str, torch.device] = 'cpu',
                        ) -> Optional[torch.Tensor]:
        if n_surfaces is None or min_distance is None or max_distance is None:
            return None

        assert mode in self.SURFCACES_MODES, f'Invalid mode={mode}'
        if mode == 'disparity':
            disparities = torch.linspace(1 / max_distance, 1 / min_distance, n_surfaces, device=device)
            return disparities.reciprocal().contiguous()  # n_surfaces

        elif mode == 'depth':
            return torch.linspace(max_distance, min_distance, n_surfaces, device=device)

        elif mode.split('-')[-1] == 'random':
            uniform: torch.Tensor = torch.rand(n_surfaces, device=device)
            if mode == 'depth-random':
                out = uniform * (max_distance - min_distance) + min_distance
                return out.sort().values.flip(0)
            elif mode == 'disparity-random':
                out = uniform * (1 / min_distance - 1 / max_distance) + 1 / max_distance
                return out.reciprocal().sort().values.flip(0)

        elif mode.split('-')[-1] == 'quasirandom':
            uniform = torch.linspace(1/n_surfaces, 1, n_surfaces, device=device) + torch.rand(1, device=device)
            uniform = uniform - torch.floor(uniform)
            if mode == 'depth-quasirandom':
                out = uniform * (max_distance - min_distance) + min_distance
                return out.sort().values.flip(0)
            elif mode == 'disparity-quasirandom':
                out = uniform * (1 / min_distance - 1 / max_distance) + 1 / max_distance
                return out.reciprocal().sort().values.flip(0)

    @abstractmethod
    def project_on(self,
                   source_features: torch.Tensor,
                   source_camera: CameraPinhole,
                   reference_pixel_coords: torch.Tensor,
                   ) -> torch.Tensor:
        """
        Project features on the surfaces.

        Args:
            source_features: B x n_features x H_feat x W_feat
            source_camera: B cameras
            reference_pixel_coords: B x H_ref x W_ref x UV
        """
        pass

    @abstractmethod
    def look_at(self,
                reference_features: torch.Tensor,
                novel_camera: CameraPinhole,
                novel_pixel_coords: torch.Tensor,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Render the surfaces for the given cameras.

        Args:
            reference_features: B x n_surfaces x n_features x H_ref x W_ref
            novel_pixel_coords: B x H_novel x W_novel x UV
            novel_camera: B cameras
        """
        pass

    @property
    @abstractmethod
    def n_intersections(self):
        pass

    @abstractmethod
    def find_intersection(self,
                          velocity: torch.Tensor,
                          start_point: torch.Tensor,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Find intersections of rays with the surfaces.

        Args:
            velocity: B x H x W x XYZ
            start_point: B x XYZ

        Returns:
            intersection: B x n_intersections x H x W x XYZ, world coordinates of intersection points
            time: B x n_intersections x H x W, timestamps of intersection
            mask: b x n_intersections x H x W, flag whether the intersection exists (or some fake values are returned
            instead)
        """
        pass

    @abstractmethod
    def depths(self, normalize: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def disparities(self, normalize: bool = False) -> torch.Tensor:
        pass
