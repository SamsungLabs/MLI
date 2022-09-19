__all__ = ['gen_quad_planes', 'gen_hex_planes']

from typing import Tuple

import numpy as np
import torch


def gen_quad_planes(height: int,
                    width: int,
                    n_planes: int = 1,
                    align_corners: bool = True,
                    only_cloud: bool = False,
                    device: torch.device = "cpu",
                    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    """
    Generate n 2d mesh planes.
    Mesh planes its a mesh surface with topology as in the picture below, each vertex is a pixel on source image.
    If two vertex is neighbours, they matched with pixels which also neighbours on the image.

    Layer topology:
      #-----------+ X
      |(0,0) _______ _______ _______ (0,3)
      |     |\      |\      |\      |
      |     |  \    |  \    |  \    |
      |     |    \  |    \  |    \  |
      +(1,0)|_____ \|_____ \|_____ \|(1,3)
      Y     |\      |\      |\      |
            |  \    |  \    |  \    |
            |    \  |    \  |    \  |
       (2,0)|_____ \|_____ \|_____ \|(2,3)
            |\      |\      |\      |
            |  \    |  \    |  \    |
            |    \  |    \  |    \  |
            |_____ \|_____ \|_____ \|
        (3,0)     (3,1)   (3,2)      (3,3)

    Args:
        height: height in num vertices
        width: width in num vertices
        n_planes: num planes
        align_corners: if True, 0 and 1 coordinates are attributed to centers of corner pixels.
            Otherwise, to their corners.
        device: pytorch device
    Returns:
        verts: 2D vertices
        faces: faces
        verts_uvs: UV coordinated of vertices
    """
    # TODO check quad code
    quade_faces = torch.tensor([[0, width, 1 + width],
                                [0, 1 + width, 1]], device=device)

    """
    Generate vertices grid
    """
    delta_x = 0 if align_corners else 0.5 / width
    delta_y = 0 if align_corners else 0.5 / height
    xs = torch.linspace(delta_x, 1 - delta_x, width, device=device)
    ys = torch.linspace(delta_y, 1 - delta_y, height, device=device)
    x_verts = xs.view(1, width).repeat(height, 1)
    y_verts = ys.view(height, 1).repeat(1, width)
    verts = torch.cat((x_verts, y_verts), 0).view(2, -1).permute(1, 0)
    if only_cloud:
        verts = verts.repeat(n_planes, 1)
        return verts.unsqueeze(0), None, None
    """
    Generate faces. 
    * First, we generate a list of base vertex for each quad. 
    * Then we add to them the tensor with the relative coordinates of the quad, 
        thus we obtain the desired topology.

    Quad topology:
    Base vertex ---> 0 _______ 3
                      |\      |
                      |  \    |
                      |    \  |
                      |_____ \|
                     1         2
    """
    faces_base_vertices_ids = torch.arange(0, width * (height - 1), 1, device=device)
    faces_base_vertices_ids = faces_base_vertices_ids[np.arange(width * (height - 1)) % width != width - 1]

    faces = (quade_faces + faces_base_vertices_ids.unsqueeze(1).unsqueeze(1)).reshape(-1, 3)
    num_verts = len(verts)

    """
    Repeat plane n_planes times.
    * First, repeat vertices and faces. 
    * Then shift vertices indices for every plane.
    """
    verts = verts.repeat(n_planes, 1)
    faces_shift = (torch.arange(0, n_planes, 1, device=device) * num_verts).unsqueeze(1).unsqueeze(1)
    faces = faces.unsqueeze(0).repeat(n_planes, 1, 1) + faces_shift
    faces = faces.reshape(-1, 3)

    """
    Compute verts_uvs.
    We concatenate texture UV-maps for all the `n_planes` planes along the 0th axis and obtain a united texture for the
    layered mesh.
    """
    offset = 1 / (n_planes * height - 1)
    verts_uvs_flipped = verts.clone()  # n_planes*H*W x 2
    arange = torch.repeat_interleave(torch.arange(n_planes, device=verts_uvs_flipped.device, dtype=torch.float),
                                     repeats=height * width,
                                     dim=0)
    scale = (1 - (n_planes - 1) * offset) / n_planes  # scale * n_planes + offset * (n_planes - 1) == 1
    shift = arange * (offset + scale)
    verts_uvs_flipped[:, 1] = verts_uvs_flipped[:, 1] * scale + shift

    # revert the vertical axis
    verts_uvs_flipped[:, 1] = 1 - verts_uvs_flipped[:, 1]
    verts_uvs = verts_uvs_flipped

    return verts.unsqueeze(0), faces.unsqueeze(0), verts_uvs.unsqueeze(0)


def gen_hex_planes(height: int,
                   width: int,
                   n_planes: int = 1,
                   align_corners: bool = True,
                   only_cloud: bool = False,
                   device: torch.device = "cpu",
                   ):
    """
    Generate n 2d mesh planes with hex structure.
    Uses for construct mesh from view depth.
    Args:
        height: height in num vertices
        width: width in num vertices
        n_planes: num planes
        align_corners: if True, 0 and 1 coordinates are attributed to centers of corner pixels.
            Otherwise, to their corners.
        device: pytorch device
    Returns:
        verts: 2D vertices
        faces: faces
        verts_uvs: UV coordinated of vertices
    """

    """
    Generate faces pattern:
         0__________1________2
         /\        /\        /
        /  \      /  \      /
       /    \    /    \    / 
      /      \  /      \  /
     3________4/________5/
     \        /\        /\
      \      /  \      /  \
       \    /    \    /    \
        \  /      \  /      \
         6/________7/________8
    """

    grid_width = width
    grid_height = height

    faces_up_row1 = torch.tensor([0, grid_width, grid_width + 1], device=device, )[None, :]
    faces_down_row1 = torch.tensor([grid_width + 1, 1, 0], device=device)[None, :]
    faces_up_row2 = torch.tensor([1, grid_width, grid_width + 1], device=device)[None, :]
    faces_down_row2 = torch.tensor([grid_width, 1, 0], device=device)[None, :]
    grid_width_steps = torch.arange(0, grid_width - 1, 1, device=device)[:, None]

    row_1_up_faces = grid_width_steps + faces_up_row1
    row_1_down_faces = grid_width_steps + faces_down_row1
    row_2_up_faces = grid_width_steps + faces_up_row2 + grid_width
    row_2_down_faces = grid_width_steps + faces_down_row2 + grid_width

    """
    Generate vertices grid
    """
    col_w = 1.0
    row_h = col_w * np.sqrt(3) / 2

    x_pos_row1 = torch.arange(0.5, (grid_width) * col_w, col_w, device=device, dtype=torch.float)
    y_pos_row1 = torch.zeros_like(x_pos_row1)
    verts_row1 = torch.stack([x_pos_row1, y_pos_row1])
    verts_row2 = torch.stack([x_pos_row1, y_pos_row1]) + torch.tensor([-0.5, row_h],
                                                                      device=device,
                                                                      dtype=torch.float)[:, None]

    verts_pattern = torch.cat([verts_row1, verts_row2], dim=1)[None, :]
    verts_pattern = verts_pattern.repeat((grid_height + 1) // 2, 1, 1)

    shift_y = torch.arange(0, (grid_height + 1) // 2, 1.0, device=device)[None, :] * row_h * 2
    shift_x = torch.zeros_like(shift_y)
    shift = torch.cat([shift_x, shift_y], axis=0)[:, :, None].permute(1, 0, 2)

    verts = shift + verts_pattern
    verts = verts.permute(0, 2, 1).reshape([-1, 2])

    if grid_height % 2 == 1:
        verts = verts[:-grid_width]

    verts = verts / torch.max(verts, dim=0, keepdim=True)[0]

    if only_cloud:
        verts = verts.repeat(n_planes, 1)
        return verts.unsqueeze(0), None, None

    """
    Generate faces. 
    * First, we generate a faces pattern  
    * Then we repeat it and add index shift
    """
    faces_row = torch.cat([row_1_up_faces, row_1_down_faces, row_2_up_faces, row_2_down_faces], dim=0)[None, :]
    faces_row = faces_row.repeat((grid_height) // 2, 1, 1)
    shift_faces = torch.arange(0, (grid_height) // 2, 1, device=device)[:, None, None] * grid_width * 2
    faces = (faces_row + shift_faces).reshape([-1, 3])
    if grid_height % 2 == 0:
        faces = faces[:-(grid_width - 1) * 2]
    num_verts = len(verts)

    """
    Repeat plane n_planes times.
    * First, repeat vertices and faces. 
    * Then shift vertices indices for every plane.
    """
    verts = verts.repeat(n_planes, 1)
    faces_shift = (torch.arange(0, n_planes, 1, device=device) * num_verts).unsqueeze(1).unsqueeze(1)
    faces = faces.unsqueeze(0).repeat(n_planes, 1, 1) + faces_shift
    faces = faces.reshape(-1, 3)

    return verts.unsqueeze(0), faces.unsqueeze(0), None
