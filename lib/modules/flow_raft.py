__all__ = ['FlowRAFT']

import argparse
import sys
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from lib.utils.base import get_grid

sys.path.append(os.path.join(os.path.dirname(__file__), 'raft/core'))
from lib.modules.raft.core.raft import RAFT


class FlowRAFT(nn.Module):
    def __init__(self,
                 path: str,
                 small: bool = False,
                 mixed_precision: bool = False,
                 alternate_corr: bool = False):
        """
        Flow estimator which uses pretrained RAFT net.

        Args:
            path: path to RAFT weights
        """
        super().__init__()

        raft_args = argparse.Namespace()
        raft_args.small = small
        raft_args.mixed_precision = mixed_precision
        raft_args.alternate_corr = alternate_corr

        model = torch.nn.DataParallel(RAFT(raft_args))
        model.load_state_dict(torch.load(path))
        self.model = model.module

    @staticmethod
    def warp_tensor_with_flow(flow_target_source: torch.Tensor,
                              source_tensor: torch.Tensor) -> torch.Tensor:
        """
        Warps a tensor using a backward flow.

        Args:
            flow_target_source: B x 2 x H x W
            source_tensor: B x C x H x W

        Returns:
            warped_tensor: B x C x H x W

        """
        grid = get_grid(1,
                        flow_target_source.shape[2],
                        flow_target_source.shape[3],
                        device=flow_target_source.device).permute(0, 3, 1, 2)

        flow = grid + flow_target_source
        flow[:, 0, :, :] = flow[:, 0, :, :] / (flow_target_source.shape[3] - 1) * 2 - 1
        flow[:, 1, :, :] = flow[:, 1, :, :] / (flow_target_source.shape[2] - 1) * 2 - 1
        warped_tensor = torch.nn.functional.grid_sample(source_tensor, flow.permute(0, 2, 3, 1))

        return warped_tensor

    @staticmethod
    def _calculate_flow_diff(flow1: torch.Tensor,
                             flow2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates difference between flow1 and flow2 warped with flow1. (On flow1 view)
        Used for detecting inconsistency between flow1 and flow2.
        Args:
            flow1:
            flow2:

        Returns:
            flows_diff : B x 2 x H x W
            flow_reproject_tensor : B x 2 x H x W  (flow2 warped by flow1)
        """

        flow2_warped_by_flow1 = FlowRAFT.warp_tensor_with_flow(flow1, flow2)

        return flow2_warped_by_flow1 + flow1, flow2_warped_by_flow1

    def forward(self,
                image_a: torch.Tensor,
                image_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes flow between two images.
        Args:
            images : B x 3 x H x W normalized from -1 to 1.

        Returns:
            flow : B x 2 x H x W (in pixels)
        """

        return self.model.forward(image_a, image_b, iters=20, test_mode=True)[1]

    def get_flow_displacement(self,
                              image_a: torch.Tensor,
                              image_b: torch.Tensor,
                              two_way: bool = False):
        """
        Computes two flow tensors (a->b - flow_ab and b->a - flow_ba).
        Then warps (backward) a flow_ba with flow_ab, and compute pixel difference between flow_ba and flow_ba_warped.

        Args:
            two_way: If two_way is True, repeats procedure in inverse order.
            image_a: B x 3 x H x W normalized from -1 to 1.
            image_b: B x 3 x H x W normalized from -1 to 1.

        Returns:
            displacement: B x 2 x H x W
            or
            displacements: (B x 2 x H x W, B x 2 x H x W)
        """
        flow_ab_tensor = self.forward(image_a, image_b)
        flow_ba_tensor = self.forward(image_b, image_a)

        displacement_ab, _ = self._calculate_flow_diff(flow_ab_tensor, flow_ba_tensor)

        if two_way:
            displacement_ba, _ = self._calculate_flow_diff(flow_ba_tensor, flow_ab_tensor)
            return displacement_ab, displacement_ba
        else:
            return displacement_ab
