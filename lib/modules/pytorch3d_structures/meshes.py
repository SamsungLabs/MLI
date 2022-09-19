# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch

from . import struct_utils


class Meshes:
    """
    This class provides functions for working with batches of triangulated
    meshes with varying numbers of faces and vertices, and converting between
    representations.

    Within Meshes, there are three different representations of the faces and
    verts data:

    List
      - only used for input as a starting point to convert to other representations.
    Padded
      - has specific batch dimension.
    Packed
      - no batch dimension.
      - has auxiliary variables used to index into the padded representation.

    Example:

    Input list of verts V_n = [[V_1], [V_2], ... , [V_N]]
    where V_1, ... , V_N are the number of verts in each mesh and N is the
    number of meshes.

    Input list of faces F_n = [[F_1], [F_2], ... , [F_N]]
    where F_1, ... , F_N are the number of faces in each mesh.

    # SPHINX IGNORE
     List                      | Padded                  | Packed
    ---------------------------|-------------------------|------------------------
    [[V_1], ... , [V_N]]       | size = (N, max(V_n), 3) |  size = (sum(V_n), 3)
                               |                         |
    Example for verts:         |                         |
                               |                         |
    V_1 = 3, V_2 = 4, V_3 = 5  | size = (3, 5, 3)        |  size = (12, 3)
                               |                         |
    List([                     | tensor([                |  tensor([
      [                        |     [                   |    [0.1, 0.3, 0.5],
        [0.1, 0.3, 0.5],       |       [0.1, 0.3, 0.5],  |    [0.5, 0.2, 0.1],
        [0.5, 0.2, 0.1],       |       [0.5, 0.2, 0.1],  |    [0.6, 0.8, 0.7],
        [0.6, 0.8, 0.7],       |       [0.6, 0.8, 0.7],  |    [0.1, 0.3, 0.3],
      ],                       |       [0,    0,    0],  |    [0.6, 0.7, 0.8],
      [                        |       [0,    0,    0],  |    [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.3],       |     ],                  |    [0.1, 0.5, 0.3],
        [0.6, 0.7, 0.8],       |     [                   |    [0.7, 0.3, 0.6],
        [0.2, 0.3, 0.4],       |       [0.1, 0.3, 0.3],  |    [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.3],       |       [0.6, 0.7, 0.8],  |    [0.9, 0.5, 0.2],
      ],                       |       [0.2, 0.3, 0.4],  |    [0.2, 0.3, 0.4],
      [                        |       [0.1, 0.5, 0.3],  |    [0.9, 0.3, 0.8],
        [0.7, 0.3, 0.6],       |       [0,    0,    0],  |  ])
        [0.2, 0.4, 0.8],       |     ],                  |
        [0.9, 0.5, 0.2],       |     [                   |
        [0.2, 0.3, 0.4],       |       [0.7, 0.3, 0.6],  |
        [0.9, 0.3, 0.8],       |       [0.2, 0.4, 0.8],  |
      ]                        |       [0.9, 0.5, 0.2],  |
    ])                         |       [0.2, 0.3, 0.4],  |
                               |       [0.9, 0.3, 0.8],  |
                               |     ]                   |
                               |  ])                     |
    Example for faces:         |                         |
                               |                         |
    F_1 = 1, F_2 = 2, F_3 = 7  | size = (3, 7, 3)        | size = (10, 3)
                               |                         |
    List([                     | tensor([                | tensor([
      [                        |     [                   |    [ 0,  1,  2],
        [0, 1, 2],             |       [0,   1,  2],     |    [ 3,  4,  5],
      ],                       |       [-1, -1, -1],     |    [ 4,  5,  6],
      [                        |       [-1, -1, -1]      |    [ 8,  9,  7],
        [0, 1, 2],             |       [-1, -1, -1]      |    [ 7,  8, 10],
        [1, 2, 3],             |       [-1, -1, -1]      |    [ 9, 10,  8],
      ],                       |       [-1, -1, -1],     |    [11, 10,  9],
      [                        |       [-1, -1, -1],     |    [11,  7,  8],
        [1, 2, 0],             |     ],                  |    [11, 10,  8],
        [0, 1, 3],             |     [                   |    [11,  9,  8],
        [2, 3, 1],             |       [0,   1,  2],     |  ])
        [4, 3, 2],             |       [1,   2,  3],     |
        [4, 0, 1],             |       [-1, -1, -1],     |
        [4, 3, 1],             |       [-1, -1, -1],     |
        [4, 2, 1],             |       [-1, -1, -1],     |
      ],                       |       [-1, -1, -1],     |
    ])                         |       [-1, -1, -1],     |
                               |     ],                  |
                               |     [                   |
                               |       [1,   2,  0],     |
                               |       [0,   1,  3],     |
                               |       [2,   3,  1],     |
                               |       [4,   3,  2],     |
                               |       [4,   0,  1],     |
                               |       [4,   3,  1],     |
                               |       [4,   2,  1],     |
                               |     ]                   |
                               |   ])                    |
    -----------------------------------------------------------------------------

    Auxiliary variables for packed representation

    Name                           |   Size              |  Example from above
    -------------------------------|---------------------|-----------------------
                                   |                     |
    verts_packed_to_mesh_idx       |  size = (sum(V_n))  |   tensor([
                                   |                     |     0, 0, 0, 1, 1, 1,
                                   |                     |     1, 2, 2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (12)
                                   |                     |
    mesh_to_verts_packed_first_idx |  size = (N)         |   tensor([0, 3, 7])
                                   |                     |   size = (3)
                                   |                     |
    num_verts_per_mesh             |  size = (N)         |   tensor([3, 4, 5])
                                   |                     |   size = (3)
                                   |                     |
    faces_packed_to_mesh_idx       |  size = (sum(F_n))  |   tensor([
                                   |                     |     0, 1, 1, 2, 2, 2,
                                   |                     |     2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (10)
                                   |                     |
    mesh_to_faces_packed_first_idx |  size = (N)         |   tensor([0, 1, 3])
                                   |                     |   size = (3)
                                   |                     |
    num_faces_per_mesh             |  size = (N)         |   tensor([1, 2, 7])
                                   |                     |   size = (3)
                                   |                     |
    verts_padded_to_packed_idx     |  size = (sum(V_n))  |  tensor([
                                   |                     |     0, 1, 2, 5, 6, 7,
                                   |                     |     8, 10, 11, 12, 13,
                                   |                     |     14
                                   |                     |  )]
                                   |                     |  size = (12)
    -----------------------------------------------------------------------------
    # SPHINX IGNORE

    From the faces, edges are computed and have packed and padded
    representations with auxiliary variables.

    E_n = [[E_1], ... , [E_N]]
    where E_1, ... , E_N are the number of unique edges in each mesh.
    Total number of unique edges = sum(E_n)

    # SPHINX IGNORE
    Name                           |   Size                  | Example from above
    -------------------------------|-------------------------|----------------------
                                   |                         |
    edges_packed                   | size = (sum(E_n), 2)    |  tensor([
                                   |                         |     [0, 1],
                                   |                         |     [0, 2],
                                   |                         |     [1, 2],
                                   |                         |       ...
                                   |                         |     [10, 11],
                                   |                         |   )]
                                   |                         |   size = (18, 2)
                                   |                         |
    num_edges_per_mesh             | size = (N)              |  tensor([3, 5, 10])
                                   |                         |  size = (3)
                                   |                         |
    edges_packed_to_mesh_idx       | size = (sum(E_n))       |  tensor([
                                   |                         |    0, 0, 0,
                                   |                         |     . . .
                                   |                         |    2, 2, 2
                                   |                         |   ])
                                   |                         |   size = (18)
                                   |                         |
    faces_packed_to_edges_packed   | size = (sum(F_n), 3)    |  tensor([
                                   |                         |    [2,   1,  0],
                                   |                         |    [5,   4,  3],
                                   |                         |       .  .  .
                                   |                         |    [12, 14, 16],
                                   |                         |   ])
                                   |                         |   size = (10, 3)
                                   |                         |
    mesh_to_edges_packed_first_idx | size = (N)              |  tensor([0, 3, 8])
                                   |                         |  size = (3)
    ----------------------------------------------------------------------------
    # SPHINX IGNORE
    """

    _INTERNAL_TENSORS = [
        "_verts_packed",
        "_verts_packed_to_mesh_idx",
        "_mesh_to_verts_packed_first_idx",
        "_verts_padded",
        "_num_verts_per_mesh",
        "_faces_packed",
        "_faces_packed_to_mesh_idx",
        "_mesh_to_faces_packed_first_idx",
        "_faces_padded",
        "_faces_areas_packed",
        "_verts_normals_packed",
        "_faces_normals_packed",
        "_num_faces_per_mesh",
        "_edges_packed",
        "_edges_packed_to_mesh_idx",
        "_mesh_to_edges_packed_first_idx",
        "_faces_packed_to_edges_packed",
        "_num_edges_per_mesh",
        "_verts_padded_to_packed_idx",
        "_laplacian_packed",
        "valid",
        "equisized",
    ]

    def __init__(
        self,
        verts,
        faces,
        textures=None,
        *,
        verts_normals=None,
    ) -> None:
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.
            verts_normals:
                Optional. Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the normals of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  They should be padded with fill value of 0 so they all have
                  the same number of vertices.
                Note that modifying the mesh later, e.g. with offset_verts_,
                can cause these normals to be forgotten and normals to be recalculated
                based on the new vertex positions.

        Refer to comments above for descriptions of List and Padded representations.
        """
        self.device = torch.device("cpu")
        if textures is not None and not hasattr(textures, "sample_textures"):
            msg = "Expected textures to be an instance of type TexturesBase; got %r"
            raise ValueError(msg % type(textures))

        self.textures = textures

        # Indicates whether the meshes in the list/batch have the same number
        # of faces and vertices.
        self.equisized = False

        # Boolean indicator for each mesh in the batch
        # True if mesh has non zero number of verts and face, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of meshes)
        self._V = 0  # (max) number of vertices per mesh
        self._F = 0  # (max) number of faces per mesh

        # List of Tensors of verts and faces.
        self._verts_list = None
        self._faces_list = None

        # Packed representation for verts.
        self._verts_packed = None  # (sum(V_n), 3)
        self._verts_packed_to_mesh_idx = None  # sum(V_n)

        # Index to convert verts from flattened padded to packed
        self._verts_padded_to_packed_idx = None  # N * max_V

        # Index of each mesh's first vert in the packed verts.
        # Assumes packing is sequential.
        self._mesh_to_verts_packed_first_idx = None  # N

        # Packed representation for faces.
        self._faces_packed = None  # (sum(F_n), 3)
        self._faces_packed_to_mesh_idx = None  # sum(F_n)

        # Index of each mesh's first face in packed faces.
        # Assumes packing is sequential.
        self._mesh_to_faces_packed_first_idx = None  # N

        # Packed representation of edges sorted by index of the first vertex
        # in the edge. Edges can be shared between faces in a mesh.
        self._edges_packed = None  # (sum(E_n), 2)

        # Map from packed edges to corresponding mesh index.
        self._edges_packed_to_mesh_idx = None  # sum(E_n)
        self._num_edges_per_mesh = None  # N
        self._mesh_to_edges_packed_first_idx = None  # N

        # Map from packed faces to packed edges. This represents the index of
        # the edge opposite the vertex for each vertex in the face. E.g.
        #
        #         v0
        #         /\
        #        /  \
        #    e1 /    \ e2
        #      /      \
        #     /________\
        #   v2    e0   v1
        #
        # Face (v0, v1, v2) => Edges (e0, e1, e2)
        self._faces_packed_to_edges_packed = None  # (sum(F_n), 3)

        # Padded representation of verts.
        self._verts_padded = None  # (N, max(V_n), 3)
        self._num_verts_per_mesh = None  # N

        # Padded representation of faces.
        self._faces_padded = None  # (N, max(F_n), 3)
        self._num_faces_per_mesh = None  # N

        # Face areas
        self._faces_areas_packed = None

        # Normals
        self._verts_normals_packed = None
        self._faces_normals_packed = None

        # Packed representation of Laplacian Matrix
        self._laplacian_packed = None

        # Identify type of verts and faces.
        if isinstance(verts, list) and isinstance(faces, list):
            self._verts_list = verts
            self._faces_list = [
                f[f.gt(-1).all(1)].to(torch.int64) if len(f) > 0 else f for f in faces
            ]
            self._N = len(self._verts_list)
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                self.device = self._verts_list[0].device
                if not (
                    all(v.device == self.device for v in verts)
                    and all(f.device == self.device for f in faces)
                ):
                    raise ValueError(
                        "All Verts and Faces tensors should be on same device."
                    )
                self._num_verts_per_mesh = torch.tensor(
                    [len(v) for v in self._verts_list], device=self.device
                )
                self._V = int(self._num_verts_per_mesh.max())
                self._num_faces_per_mesh = torch.tensor(
                    [len(f) for f in self._faces_list], device=self.device
                )
                self._F = int(self._num_faces_per_mesh.max())
                self.valid = torch.tensor(
                    [
                        len(v) > 0 and len(f) > 0
                        for (v, f) in zip(self._verts_list, self._faces_list)
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                if (len(self._num_verts_per_mesh.unique()) == 1) and (
                    len(self._num_faces_per_mesh.unique()) == 1
                ):
                    self.equisized = True

        elif torch.is_tensor(verts) and torch.is_tensor(faces):
            if verts.size(2) != 3 or faces.size(2) != 3:
                raise ValueError("Verts or Faces tensors have incorrect dimensions.")
            self._verts_padded = verts
            self._faces_padded = faces.to(torch.int64)
            self._N = self._verts_padded.shape[0]
            self._V = self._verts_padded.shape[1]

            if verts.device != faces.device:
                msg = "Verts and Faces tensors should be on same device. \n Got {} and {}."
                raise ValueError(msg.format(verts.device, faces.device))

            self.device = self._verts_padded.device
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                # Check that padded faces - which have value -1 - are at the
                # end of the tensors
                faces_not_padded = self._faces_padded.gt(-1).all(2)
                self._num_faces_per_mesh = faces_not_padded.sum(1)
                if (faces_not_padded[:, :-1] < faces_not_padded[:, 1:]).any():
                    raise ValueError("Padding of faces must be at the end")

                # NOTE that we don't check for the ordering of padded verts
                # as long as the faces index correspond to the right vertices.

                self.valid = self._num_faces_per_mesh > 0
                self._F = int(self._num_faces_per_mesh.max())
                if len(self._num_faces_per_mesh.unique()) == 1:
                    self.equisized = True

                self._num_verts_per_mesh = torch.full(
                    size=(self._N,),
                    fill_value=self._V,
                    dtype=torch.int64,
                    device=self.device,
                )

        else:
            raise ValueError(
                "Verts and Faces must be either a list or a tensor with \
                    shape (batch_size, N, 3) where N is either the maximum \
                       number of verts or faces respectively."
            )

        if self.isempty():
            self._num_verts_per_mesh = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._num_faces_per_mesh = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )

        # Set the num verts/faces on the textures if present.
        if textures is not None:
            shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
            if not shape_ok:
                msg = "Textures do not match the dimensions of Meshes."
                raise ValueError(msg)

            self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
            self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
            self.textures.valid = self.valid

        if verts_normals is not None:
            self._set_verts_normals(verts_normals)

    def _set_verts_normals(self, verts_normals) -> None:
        if isinstance(verts_normals, list):
            if len(verts_normals) != self._N:
                raise ValueError("Invalid verts_normals input")

            for item, n_verts in zip(verts_normals, self._num_verts_per_mesh):
                if (
                    not isinstance(item, torch.Tensor)
                    or item.ndim != 2
                    or item.shape[1] != 3
                    or item.shape[0] != n_verts
                ):
                    raise ValueError("Invalid verts_normals input")
            self._verts_normals_packed = torch.cat(verts_normals, 0)
        elif torch.is_tensor(verts_normals):
            if (
                verts_normals.ndim != 3
                or verts_normals.size(2) != 3
                or verts_normals.size(0) != self._N
            ):
                raise ValueError("Vertex normals tensor has incorrect dimensions.")
            self._verts_normals_packed = struct_utils.padded_to_packed(
                verts_normals, split_size=self._num_verts_per_mesh.tolist()
            )
        else:
            raise ValueError("verts_normals must be a list or tensor")

    def __len__(self) -> int:
        return self._N

    def __getitem__(
        self, index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor]
    ) -> "Meshes":
        """
        Args:
            index: Specifying the index of the mesh to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            Meshes object with selected meshes. The mesh tensors are not cloned.
        """
        if isinstance(index, (int, slice)):
            verts = self.verts_list()[index]
            faces = self.faces_list()[index]
        elif isinstance(index, list):
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        else:
            raise IndexError(index)

        textures = None if self.textures is None else self.textures[index]

        if torch.is_tensor(verts) and torch.is_tensor(faces):
            return self.__class__(verts=[verts], faces=[faces], textures=textures)
        elif isinstance(verts, list) and isinstance(faces, list):
            return self.__class__(verts=verts, faces=faces, textures=textures)
        else:
            raise ValueError("(verts, faces) not defined correctly")

    def isempty(self) -> bool:
        """
        Checks whether any mesh is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def verts_list(self):
        """
        Get the list representation of the vertices.

        Returns:
            list of tensors of vertices of shape (V_n, 3).
        """
        if self._verts_list is None:
            assert (
                self._verts_padded is not None
            ), "verts_padded is required to compute verts_list."
            self._verts_list = struct_utils.padded_to_list(
                self._verts_padded, self.num_verts_per_mesh().tolist()
            )
        return self._verts_list

    def faces_list(self):
        """
        Get the list representation of the faces.

        Returns:
            list of tensors of faces of shape (F_n, 3).
        """
        if self._faces_list is None:
            assert (
                self._faces_padded is not None
            ), "faces_padded is required to compute faces_list."
            self._faces_list = struct_utils.padded_to_list(
                self._faces_padded, self.num_faces_per_mesh().tolist()
            )
        return self._faces_list

    def verts_packed(self):
        """
        Get the packed representation of the vertices.

        Returns:
            tensor of vertices of shape (sum(V_n), 3).
        """
        self._compute_packed()
        return self._verts_packed

    def verts_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as verts_packed.
        verts_packed_to_mesh_idx[i] gives the index of the mesh which contains
        verts_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._verts_packed_to_mesh_idx

    def mesh_to_verts_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first vertex of the ith mesh is verts_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._mesh_to_verts_packed_first_idx

    def num_verts_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of vertices in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_verts_per_mesh

    def faces_packed(self):
        """
        Get the packed representation of the faces.
        Faces are given by the indices of the three vertices in verts_packed.

        Returns:
            tensor of faces of shape (sum(F_n), 3).
        """
        self._compute_packed()
        return self._faces_packed

    def faces_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as faces_packed.
        faces_packed_to_mesh_idx[i] gives the index of the mesh which contains
        faces_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._faces_packed_to_mesh_idx

    def mesh_to_faces_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first face of the ith mesh is faces_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._mesh_to_faces_packed_first_idx

    def verts_padded(self):
        """
        Get the padded representation of the vertices.

        Returns:
            tensor of vertices of shape (N, max(V_n), 3).
        """
        self._compute_padded()
        return self._verts_padded

    def faces_padded(self):
        """
        Get the padded representation of the faces.

        Returns:
            tensor of faces of shape (N, max(F_n), 3).
        """
        self._compute_padded()
        return self._faces_padded

    def num_faces_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of faces in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_faces_per_mesh

    def edges_packed(self):
        """
        Get the packed representation of the edges.

        Returns:
            tensor of edges of shape (sum(E_n), 2).
        """
        self._compute_edges_packed()
        return self._edges_packed

    def edges_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as edges_packed.
        edges_packed_to_mesh_idx[i] gives the index of the mesh which contains
        edges_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_edges_packed()
        return self._edges_packed_to_mesh_idx

    def mesh_to_edges_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first edge of the ith mesh is edges_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_edges_packed()
        return self._mesh_to_edges_packed_first_idx

    def faces_packed_to_edges_packed(self):
        """
        Get the packed representation of the faces in terms of edges.
        Faces are given by the indices of the three edges in
        the packed representation of the edges.

        Returns:
            tensor of faces of shape (sum(F_n), 3).
        """
        self._compute_edges_packed()
        return self._faces_packed_to_edges_packed

    def num_edges_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of edges in each mesh.

        Returns:
            1D tensor of sizes.
        """
        self._compute_edges_packed()
        return self._num_edges_per_mesh

    def verts_padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of vertices
        such that verts_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = verts_padded().reshape(-1, 3)
            verts_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._verts_padded_to_packed_idx is not None:
            return self._verts_padded_to_packed_idx

        self._verts_padded_to_packed_idx = torch.cat(
            [
                torch.arange(v, dtype=torch.int64, device=self.device) + i * self._V
                for (i, v) in enumerate(self.num_verts_per_mesh())
            ],
            dim=0,
        )
        return self._verts_padded_to_packed_idx

    def has_verts_normals(self) -> bool:
        """
        Check whether vertex normals are already present.
        """
        return self._verts_normals_packed is not None

    def verts_normals_packed(self):
        """
        Get the packed representation of the vertex normals.

        Returns:
            tensor of normals of shape (sum(V_n), 3).
        """
        self._compute_vertex_normals()
        return self._verts_normals_packed

    def verts_normals_list(self):
        """
        Get the list representation of the vertex normals.

        Returns:
            list of tensors of normals of shape (V_n, 3).
        """
        if self.isempty():
            return [
                torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ] * self._N
        verts_normals_packed = self.verts_normals_packed()
        split_size = self.num_verts_per_mesh().tolist()
        return struct_utils.packed_to_list(verts_normals_packed, split_size)

    def verts_normals_padded(self):
        """
        Get the padded representation of the vertex normals.

        Returns:
            tensor of normals of shape (N, max(V_n), 3).
        """
        if self.isempty():
            return torch.zeros((self._N, 0, 3), dtype=torch.float32, device=self.device)
        verts_normals_list = self.verts_normals_list()
        return struct_utils.list_to_padded(
            verts_normals_list, (self._V, 3), pad_value=0.0, equisized=self.equisized
        )

    def faces_normals_packed(self):
        """
        Get the packed representation of the face normals.

        Returns:
            tensor of normals of shape (sum(F_n), 3).
        """
        self._compute_face_areas_normals()
        return self._faces_normals_packed

    def faces_normals_list(self):
        """
        Get the list representation of the face normals.

        Returns:
            list of tensors of normals of shape (F_n, 3).
        """
        if self.isempty():
            return [
                torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ] * self._N
        faces_normals_packed = self.faces_normals_packed()
        split_size = self.num_faces_per_mesh().tolist()
        return struct_utils.packed_to_list(faces_normals_packed, split_size)

    def faces_normals_padded(self):
        """
        Get the padded representation of the face normals.

        Returns:
            tensor of normals of shape (N, max(F_n), 3).
        """
        if self.isempty():
            return torch.zeros((self._N, 0, 3), dtype=torch.float32, device=self.device)
        faces_normals_list = self.faces_normals_list()
        return struct_utils.list_to_padded(
            faces_normals_list, (self._F, 3), pad_value=0.0, equisized=self.equisized
        )

    def faces_areas_packed(self):
        """
        Get the packed representation of the face areas.

        Returns:
            tensor of areas of shape (sum(F_n),).
        """
        self._compute_face_areas_normals()
        return self._faces_areas_packed

    def laplacian_packed(self):
        self._compute_laplacian_packed()
        return self._laplacian_packed

    def _compute_face_areas_normals(self, refresh: bool = False):
        """
        Compute the area and normal of each face in faces_packed.
        The convention of a normal for a face consisting of verts [v0, v1, v2]
        is normal = (v1 - v0) x (v2 - v0)

        Args:
            refresh: Set to True to force recomputation of face areas.
                     Default: False.
        """
        from ..ops.mesh_face_areas_normals import mesh_face_areas_normals

        if not (
            refresh
            or any(
                v is None
                for v in [self._faces_areas_packed, self._faces_normals_packed]
            )
        ):
            return
        faces_packed = self.faces_packed()
        verts_packed = self.verts_packed()
        face_areas, face_normals = mesh_face_areas_normals(verts_packed, faces_packed)
        self._faces_areas_packed = face_areas
        self._faces_normals_packed = face_normals

    def _compute_vertex_normals(self, refresh: bool = False):
        """Computes the packed version of vertex normals from the packed verts
        and faces. This assumes verts are shared between faces. The normal for
        a vertex is computed as the sum of the normals of all the faces it is
        part of weighed by the face areas.

        Args:
            refresh: Set to True to force recomputation of vertex normals.
                Default: False.
        """
        if not (refresh or any(v is None for v in [self._verts_normals_packed])):
            return

        if self.isempty():
            self._verts_normals_packed = torch.zeros(
                (self._N, 3), dtype=torch.int64, device=self.device
            )
        else:
            faces_packed = self.faces_packed()
            verts_packed = self.verts_packed()
            verts_normals = torch.zeros_like(verts_packed)
            vertices_faces = verts_packed[faces_packed]

            faces_normals = torch.cross(
                vertices_faces[:, 2] - vertices_faces[:, 1],
                vertices_faces[:, 0] - vertices_faces[:, 1],
                dim=1,
            )

            # NOTE: this is already applying the area weighting as the magnitude
            # of the cross product is 2 x area of the triangle.
            verts_normals = verts_normals.index_add(
                0, faces_packed[:, 0], faces_normals
            )
            verts_normals = verts_normals.index_add(
                0, faces_packed[:, 1], faces_normals
            )
            verts_normals = verts_normals.index_add(
                0, faces_packed[:, 2], faces_normals
            )

            self._verts_normals_packed = torch.nn.functional.normalize(
                verts_normals, eps=1e-6, dim=1
            )

    def _compute_padded(self, refresh: bool = False):
        """
        Computes the padded version of meshes from verts_list and faces_list.
        """
        if not (
            refresh or any(v is None for v in [self._verts_padded, self._faces_padded])
        ):
            return

        verts_list = self.verts_list()
        faces_list = self.faces_list()
        assert (
            faces_list is not None and verts_list is not None
        ), "faces_list and verts_list arguments are required"

        if self.isempty():
            self._faces_padded = torch.zeros(
                (self._N, 0, 3), dtype=torch.int64, device=self.device
            )
            self._verts_padded = torch.zeros(
                (self._N, 0, 3), dtype=torch.float32, device=self.device
            )
        else:
            self._faces_padded = struct_utils.list_to_padded(
                faces_list, (self._F, 3), pad_value=-1.0, equisized=self.equisized
            )
            self._verts_padded = struct_utils.list_to_padded(
                verts_list, (self._V, 3), pad_value=0.0, equisized=self.equisized
            )

    # TODO(nikhilar) Improve performance of _compute_packed.
    def _compute_packed(self, refresh: bool = False):
        """
        Computes the packed version of the meshes from verts_list and faces_list
        and sets the values of auxiliary tensors.

        Args:
            refresh: Set to True to force recomputation of packed representations.
                Default: False.
        """

        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._verts_packed,
                    self._verts_packed_to_mesh_idx,
                    self._mesh_to_verts_packed_first_idx,
                    self._faces_packed,
                    self._faces_packed_to_mesh_idx,
                    self._mesh_to_faces_packed_first_idx,
                ]
            )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for verts_list and faces_list.
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        if self.isempty():
            self._verts_packed = torch.zeros(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self._verts_packed_to_mesh_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._mesh_to_verts_packed_first_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._num_verts_per_mesh = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._faces_packed = -(
                torch.ones((0, 3), dtype=torch.int64, device=self.device)
            )
            self._faces_packed_to_mesh_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._mesh_to_faces_packed_first_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._num_faces_per_mesh = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            return

        verts_list_to_packed = struct_utils.list_to_packed(verts_list)
        self._verts_packed = verts_list_to_packed[0]
        if not torch.allclose(self.num_verts_per_mesh(), verts_list_to_packed[1]):
            raise ValueError("The number of verts per mesh should be consistent.")
        self._mesh_to_verts_packed_first_idx = verts_list_to_packed[2]
        self._verts_packed_to_mesh_idx = verts_list_to_packed[3]

        faces_list_to_packed = struct_utils.list_to_packed(faces_list)
        faces_packed = faces_list_to_packed[0]
        if not torch.allclose(self.num_faces_per_mesh(), faces_list_to_packed[1]):
            raise ValueError("The number of faces per mesh should be consistent.")
        self._mesh_to_faces_packed_first_idx = faces_list_to_packed[2]
        self._faces_packed_to_mesh_idx = faces_list_to_packed[3]

        faces_packed_offset = self._mesh_to_verts_packed_first_idx[
            self._faces_packed_to_mesh_idx
        ]
        self._faces_packed = faces_packed + faces_packed_offset.view(-1, 1)

    def _compute_edges_packed(self, refresh: bool = False):
        """
        Computes edges in packed form from the packed version of faces and verts.
        """
        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._edges_packed,
                    self._faces_packed_to_mesh_idx,
                    self._edges_packed_to_mesh_idx,
                    self._num_edges_per_mesh,
                    self._mesh_to_edges_packed_first_idx,
                ]
            )
        ):
            return

        if self.isempty():
            self._edges_packed = torch.full(
                (0, 2), fill_value=-1, dtype=torch.int64, device=self.device
            )
            self._edges_packed_to_mesh_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            return

        faces = self.faces_packed()
        F = faces.shape[0]
        v0, v1, v2 = faces.chunk(3, dim=1)
        e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
        e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
        e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

        # All edges including duplicates.
        edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
        edge_to_mesh = torch.cat(
            [
                self._faces_packed_to_mesh_idx,
                self._faces_packed_to_mesh_idx,
                self._faces_packed_to_mesh_idx,
            ],
            dim=0,
        )  # sum(F_n)*3

        # Sort the edges in increasing vertex order to remove duplicates as
        # the same edge may appear in different orientations in different faces.
        # i.e. rows in edges after sorting will be of the form (v0, v1) where v1 > v0.
        # This sorting does not change the order in dim=0.
        edges, _ = edges.sort(dim=1)

        # Remove duplicate edges: convert each edge (v0, v1) into an
        # integer hash = V * v0 + v1; this allows us to use the scalar version of
        # unique which is much faster than edges.unique(dim=1) which is very slow.
        # After finding the unique elements reconstruct the vertex indices as:
        # (v0, v1) = (hash / V, hash % V)
        # The inverse maps from unique_edges back to edges:
        # unique_edges[inverse_idxs] == edges
        # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]

        V = self._verts_packed.shape[0]
        edges_hash = V * edges[:, 0] + edges[:, 1]
        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

        # Find indices of unique elements.
        # TODO (nikhilar) remove following 4 lines when torch.unique has support
        # for returning unique indices
        sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
        unique_mask = torch.ones(
            edges_hash.shape[0], dtype=torch.bool, device=self.device
        )
        unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        unique_idx = sort_idx[unique_mask]

        self._edges_packed = torch.stack([u // V, u % V], dim=1)
        self._edges_packed_to_mesh_idx = edge_to_mesh[unique_idx]

        self._faces_packed_to_edges_packed = inverse_idxs.reshape(3, F).t()

        # Compute number of edges per mesh
        num_edges_per_mesh = torch.zeros(self._N, dtype=torch.int32, device=self.device)
        ones = torch.ones(1, dtype=torch.int32, device=self.device).expand(
            self._edges_packed_to_mesh_idx.shape
        )
        num_edges_per_mesh = num_edges_per_mesh.scatter_add_(
            0, self._edges_packed_to_mesh_idx, ones
        )
        self._num_edges_per_mesh = num_edges_per_mesh

        # Compute first idx for each mesh in edges_packed
        mesh_to_edges_packed_first_idx = torch.zeros(
            self._N, dtype=torch.int64, device=self.device
        )
        num_edges_cumsum = num_edges_per_mesh.cumsum(dim=0)
        mesh_to_edges_packed_first_idx[1:] = num_edges_cumsum[:-1].clone()

        self._mesh_to_edges_packed_first_idx = mesh_to_edges_packed_first_idx

    def _compute_laplacian_packed(self, refresh: bool = False):
        """
        Computes the laplacian in packed form.
        The definition of the laplacian is
        L[i, j] =    -1       , if i == j
        L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        L[i, j] =    0        , otherwise
        where deg(i) is the degree of the i-th vertex in the graph

        Returns:
            Sparse FloatTensor of shape (V, V) where V = sum(V_n)

        """
        from ..ops import laplacian

        if not (refresh or self._laplacian_packed is None):
            return

        if self.isempty():
            self._laplacian_packed = torch.zeros(
                (0, 0), dtype=torch.float32, device=self.device
            ).to_sparse()
            return

        verts_packed = self.verts_packed()  # (sum(V_n), 3)
        edges_packed = self.edges_packed()  # (sum(E_n), 3)

        self._laplacian_packed = laplacian(verts_packed, edges_packed)

    def clone(self):
        """
        Deep copy of Meshes object. All internal tensors are cloned individually.

        Returns:
            new Meshes object.
        """
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.clone() for v in verts_list]
        new_faces_list = [f.clone() for f in faces_list]
        other = self.__class__(verts=new_verts_list, faces=new_faces_list)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())

        # Textures is not a tensor but has a clone method
        if self.textures is not None:
            other.textures = self.textures.clone()
        return other

    def detach(self):
        """
        Detach Meshes object. All internal tensors are detached individually.

        Returns:
            new Meshes object.
        """
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.detach() for v in verts_list]
        new_faces_list = [f.detach() for f in faces_list]
        other = self.__class__(verts=new_verts_list, faces=new_faces_list)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())

        # Textures is not a tensor but has a detach method
        if self.textures is not None:
            other.textures = self.textures.detach()
        return other

    def to(self, device, copy: bool = False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            Meshes object.
        """

        device_ = torch.device(device) if isinstance(device, str) else device
        if device_.type == "cuda" and device_.index is None:
            # If cuda but with no index, then the current cuda device is indicated.
            # In that case, we fix to that device
            device_ = torch.device(f"cuda:{torch.cuda.current_device()}")

        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        if other._N > 0:
            other._verts_list = [v.to(device_) for v in other._verts_list]
            other._faces_list = [f.to(device_) for f in other._faces_list]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device_))
        if self.textures is not None:
            other.textures = other.textures.to(device_)
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def get_mesh_verts_faces(self, index: int):
        """
        Get tensors for a single mesh from the list representation.

        Args:
            index: Integer in the range [0, N).

        Returns:
            verts: Tensor of shape (V, 3).
            faces: LongTensor of shape (F, 3).
        """
        if not isinstance(index, int):
            raise ValueError("Mesh index must be an integer.")
        if index < 0 or index > self._N:
            raise ValueError(
                "Mesh index must be in the range [0, N) where \
            N is the number of meshes in the batch."
            )
        verts = self.verts_list()
        faces = self.faces_list()
        return verts[index], faces[index]

    # TODO(nikhilar) Move function to a utils file.
    def split(self, split_sizes: list):
        """
        Splits Meshes object of size N into a list of Meshes objects of
        size len(split_sizes), where the i-th Meshes object is of size split_sizes[i].
        Similar to torch.split().

        Args:
            split_sizes: List of integer sizes of Meshes objects to be returned.

        Returns:
            list[Meshes].
        """
        if not all(isinstance(x, int) for x in split_sizes):
            raise ValueError("Value of split_sizes must be a list of integers.")
        meshlist = []
        curi = 0
        for i in split_sizes:
            meshlist.append(self[curi : curi + i])
            curi += i
        return meshlist

    def offset_verts_(self, vert_offsets_packed):
        """
        Add an offset to the vertices of this Meshes. In place operation.
        If normals are present they may be recalculated.

        Args:
            vert_offsets_packed: A Tensor of shape (3,) or the same shape as
                                self.verts_packed, giving offsets to be added
                                to all vertices.
        Returns:
            self.
        """
        verts_packed = self.verts_packed()
        if vert_offsets_packed.shape == (3,):
            update_normals = False
            vert_offsets_packed = vert_offsets_packed.expand_as(verts_packed)
        else:
            update_normals = True
        if vert_offsets_packed.shape != verts_packed.shape:
            raise ValueError("Verts offsets must have dimension (all_v, 3).")
        # update verts packed
        self._verts_packed = verts_packed + vert_offsets_packed
        new_verts_list = list(
            self._verts_packed.split(self.num_verts_per_mesh().tolist(), 0)
        )
        # update verts list
        # Note that since _compute_packed() has been executed, verts_list
        # cannot be None even if not provided during construction.
        self._verts_list = new_verts_list

        # update verts padded
        if self._verts_padded is not None:
            for i, verts in enumerate(new_verts_list):
                if len(verts) > 0:
                    self._verts_padded[i, : verts.shape[0], :] = verts

        # update face areas and normals and vertex normals
        # only if the original attributes are present
        if update_normals and any(
            v is not None
            for v in [self._faces_areas_packed, self._faces_normals_packed]
        ):
            self._compute_face_areas_normals(refresh=True)
        if update_normals and self._verts_normals_packed is not None:
            self._compute_vertex_normals(refresh=True)

        return self

    # TODO(nikhilar) Move out of place operator to a utils file.
    def offset_verts(self, vert_offsets_packed):
        """
        Out of place offset_verts.

        Args:
            vert_offsets_packed: A Tensor of the same shape as self.verts_packed
                giving offsets to be added to all vertices.
        Returns:
            new Meshes object.
        """
        new_mesh = self.clone()
        return new_mesh.offset_verts_(vert_offsets_packed)

    def scale_verts_(self, scale):
        """
        Multiply the vertices of this Meshes object by a scalar value.
        In place operation.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            self.
        """
        if not torch.is_tensor(scale):
            scale = torch.full((len(self),), scale, device=self.device)
        new_verts_list = []
        verts_list = self.verts_list()
        for i, old_verts in enumerate(verts_list):
            new_verts_list.append(scale[i] * old_verts)
        # update list
        self._verts_list = new_verts_list
        # update packed
        if self._verts_packed is not None:
            self._verts_packed = torch.cat(new_verts_list, dim=0)
        # update padded
        if self._verts_padded is not None:
            for i, verts in enumerate(self._verts_list):
                if len(verts) > 0:
                    self._verts_padded[i, : verts.shape[0], :] = verts

        # update face areas and normals
        # only if the original attributes are computed
        if any(
            v is not None
            for v in [self._faces_areas_packed, self._faces_normals_packed]
        ):
            self._compute_face_areas_normals(refresh=True)
        return self

    def scale_verts(self, scale):
        """
        Out of place scale_verts.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            new Meshes object.
        """
        new_mesh = self.clone()
        return new_mesh.scale_verts_(scale)

    def update_padded(self, new_verts_padded):
        """
        This function allows for an update of verts_padded without having to
        explicitly convert it to the list representation for heterogeneous batches.
        Returns a Meshes structure with updated padded tensors and copies of the
        auxiliary tensors at construction time.
        It updates self._verts_padded with new_verts_padded, and does a
        shallow copy of (faces_padded, faces_list, num_verts_per_mesh, num_faces_per_mesh).
        If packed representations are computed in self, they are updated as well.

        Args:
            new_points_padded: FloatTensor of shape (N, V, 3)

        Returns:
            Meshes with updated padded representations
        """

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError("new values must have the same batch dimension.")
            if x.shape[1] != size[1]:
                raise ValueError("new values must have the same number of points.")
            if x.shape[2] != size[2]:
                raise ValueError("new values must have the same dimension.")

        check_shapes(new_verts_padded, [self._N, self._V, 3])

        new = self.__class__(verts=new_verts_padded, faces=self.faces_padded())

        if new._N != self._N or new._V != self._V or new._F != self._F:
            raise ValueError("Inconsistent sizes after construction.")

        # overwrite the equisized flag
        new.equisized = self.equisized

        # overwrite textures if any
        new.textures = self.textures

        # copy auxiliary tensors
        copy_tensors = ["_num_verts_per_mesh", "_num_faces_per_mesh", "valid"]

        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)  # shallow copy

        # shallow copy of faces_list if any, st new.faces_list()
        # does not re-compute from _faces_padded
        new._faces_list = self._faces_list

        # update verts/faces packed if they are computed in self
        if self._verts_packed is not None:
            copy_tensors = [
                "_faces_packed",
                "_verts_packed_to_mesh_idx",
                "_faces_packed_to_mesh_idx",
                "_mesh_to_verts_packed_first_idx",
                "_mesh_to_faces_packed_first_idx",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy
            # update verts_packed
            pad_to_packed = self.verts_padded_to_packed_idx()
            new_verts_packed = new_verts_padded.reshape(-1, 3)[pad_to_packed, :]
            new._verts_packed = new_verts_packed
            new._verts_padded_to_packed_idx = pad_to_packed

        # update edges packed if they are computed in self
        if self._edges_packed is not None:
            copy_tensors = [
                "_edges_packed",
                "_edges_packed_to_mesh_idx",
                "_mesh_to_edges_packed_first_idx",
                "_faces_packed_to_edges_packed",
                "_num_edges_per_mesh",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy

        # update laplacian if it is compute in self
        if self._laplacian_packed is not None:
            new._laplacian_packed = self._laplacian_packed

        assert new._verts_list is None
        assert new._verts_normals_packed is None
        assert new._faces_normals_packed is None
        assert new._faces_areas_packed is None

        return new

    # TODO(nikhilar) Move function to utils file.
    def get_bounding_boxes(self):
        """
        Compute an axis-aligned bounding box for each mesh in this Meshes object.

        Returns:
            bboxes: Tensor of shape (N, 3, 2) where bbox[i, j] gives the
            min and max values of mesh i along the jth coordinate axis.
        """
        all_mins, all_maxes = [], []
        for verts in self.verts_list():
            cur_mins = verts.min(dim=0)[0]  # (3,)
            cur_maxes = verts.max(dim=0)[0]  # (3,)
            all_mins.append(cur_mins)
            all_maxes.append(cur_maxes)
        all_mins = torch.stack(all_mins, dim=0)  # (N, 3)
        all_maxes = torch.stack(all_maxes, dim=0)  # (N, 3)
        bboxes = torch.stack([all_mins, all_maxes], dim=2)
        return bboxes

    def extend(self, N: int):
        """
        Create new Meshes class which contains each input mesh N times

        Args:
            N: number of new copies of each mesh.

        Returns:
            new Meshes object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")
        new_verts_list, new_faces_list = [], []
        for verts, faces in zip(self.verts_list(), self.faces_list()):
            new_verts_list.extend(verts.clone() for _ in range(N))
            new_faces_list.extend(faces.clone() for _ in range(N))

        tex = None
        if self.textures is not None:
            tex = self.textures.extend(N)

        return self.__class__(verts=new_verts_list, faces=new_faces_list, textures=tex)

    def sample_textures(self, fragments):
        if self.textures is not None:

            # Check dimensions of textures match that of meshes
            shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
            if not shape_ok:
                msg = "Textures do not match the dimensions of Meshes."
                raise ValueError(msg)

            # Pass in faces packed. If the textures are defined per
            # vertex, the face indices are needed in order to interpolate
            # the vertex attributes across the face.
            return self.textures.sample_textures(
                fragments, faces_packed=self.faces_packed()
            )
        else:
            raise ValueError("Meshes does not have textures")

    def submeshes(
        self,
        face_indices: Union[
            List[List[torch.LongTensor]], List[torch.LongTensor], torch.LongTensor
        ],
    ) -> "Meshes":
        """
        Split a batch of meshes into a batch of submeshes.

        The return value is a Meshes object representing
            [mesh restricted to only faces indexed by selected_faces
            for mesh, selected_faces_list in zip(self, face_indices)
            for faces in selected_faces_list]

        Args:
          face_indices:
            Let the original mesh have verts_list() of length N.
            Can be either
              - List of lists of LongTensors. The n-th element is a list of length
              num_submeshes_n (empty lists are allowed). The k-th element of the n-th
              sublist is a LongTensor of length num_faces_submesh_n_k.
              - List of LongTensors. The n-th element is a (possibly empty) LongTensor
                of shape (num_submeshes_n, num_faces_n).
              - A LongTensor of shape (N, num_submeshes_per_mesh, num_faces_per_submesh)
                where all meshes in the batch will have the same number of submeshes.
                This will result in an output Meshes object with batch size equal to
                N * num_submeshes_per_mesh.

        Returns:
          Meshes object of length `sum(len(ids) for ids in face_indices)`.

        Submeshing only works with no textures or with the TexturesVertex texture.

        Example 1:

        If `meshes` has batch size 1, and `face_indices` is a 1D LongTensor,
        then `meshes.submeshes([[face_indices]]) and
        `meshes.submeshes(face_indices[None, None])` both produce a Meshes of length 1,
        containing a single submesh with a subset of `meshes`' faces, whose indices are
        specified by `face_indices`.

        Example 2:

        Take a Meshes object `cubes` with 4 meshes, each a translated cube. Then:
            * len(cubes) is 4, len(cubes.verts_list()) is 4, len(cubes.faces_list()) 4,
            * [cube_verts.size for cube_verts in cubes.verts_list()] is [8, 8, 8, 8],
            * [cube_faces.size for cube_faces in cubes.faces_list()] if [6, 6, 6, 6],

        Now let front_facet, top_and_bottom, all_facets be LongTensors of
        sizes (2), (4), and (12), each picking up a number of facets of a cube by
        specifying the appropriate triangular faces.

        Then let `subcubes = cubes.submeshes([[front_facet, top_and_bottom], [],
                                              [all_facets], []])`.
            * len(subcubes) is 3.
            * subcubes[0] is the front facet of the cube contained in cubes[0].
            * subcubes[1] is a mesh containing the (disconnected) top and bottom facets
              of cubes[0].
            * subcubes[2] is cubes[2].
            * There are no submeshes of cubes[1] and cubes[3] in subcubes.
            * subcubes[0] and subcubes[1] are not watertight. subcubes[2] is.
        """
        if len(face_indices) != len(self):
            raise ValueError(
                "You must specify exactly one set of submeshes"
                " for each mesh in this Meshes object."
            )

        sub_verts = []
        sub_verts_ids = []
        sub_faces = []
        sub_face_ids = []

        for face_ids_per_mesh, faces, verts in zip(
            face_indices, self.faces_list(), self.verts_list()
        ):
            sub_verts_ids.append([])
            sub_face_ids.append([])
            for submesh_face_ids in face_ids_per_mesh:
                faces_to_keep = faces[submesh_face_ids]
                sub_face_ids[-1].append(faces_to_keep)

                # Say we are keeping two faces from a mesh with six vertices:
                # faces_to_keep = [[0, 6, 4],
                #                  [0, 2, 6]]
                # Then we want verts_to_keep to contain only vertices [0, 2, 4, 6]:
                vertex_ids_to_keep = torch.unique(faces_to_keep, sorted=True)
                sub_verts.append(verts[vertex_ids_to_keep])
                sub_verts_ids[-1].append(vertex_ids_to_keep)

                # Now, convert faces_to_keep to use the new vertex ids.
                # In our example, instead of
                # [[0, 6, 4],
                #  [0, 2, 6]]
                # we want faces_to_keep to be
                # [[0, 3, 2],
                #  [0, 1, 3]],
                # as each point id got reduced to its sort rank.
                _, ids_of_unique_ids_in_sorted = torch.unique(
                    faces_to_keep, return_inverse=True
                )
                sub_faces.append(ids_of_unique_ids_in_sorted)

        return self.__class__(
            verts=sub_verts,
            faces=sub_faces,
            textures=(
                self.textures.submeshes(sub_verts_ids, sub_face_ids)
                if self.textures
                else None
            ),
        )


def join_meshes_as_batch(meshes: List[Meshes], include_textures: bool = True) -> Meshes:
    """
    Merge multiple Meshes objects, i.e. concatenate the meshes objects. They
    must all be on the same device. If include_textures is true, they must all
    be compatible, either all or none having textures, and all the Textures
    objects being the same type. If include_textures is False, textures are
    ignored.

    If the textures are TexturesAtlas then being the same type includes having
    the same resolution. If they are TexturesUV then it includes having the same
    align_corners and padding_mode.

    Args:
        meshes: list of meshes.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        new Meshes object containing all the meshes from all the inputs.
    """
    if isinstance(meshes, Meshes):
        # Meshes objects can be iterated and produce single Meshes. We avoid
        # letting join_meshes_as_batch(mesh1, mesh2) silently do the wrong thing.
        raise ValueError("Wrong first argument to join_meshes_as_batch.")
    verts = [v for mesh in meshes for v in mesh.verts_list()]
    faces = [f for mesh in meshes for f in mesh.faces_list()]
    if len(meshes) == 0 or not include_textures:
        return Meshes(verts=verts, faces=faces)

    if meshes[0].textures is None:
        if any(mesh.textures is not None for mesh in meshes):
            raise ValueError("Inconsistent textures in join_meshes_as_batch.")
        return Meshes(verts=verts, faces=faces)

    if any(mesh.textures is None for mesh in meshes):
        raise ValueError("Inconsistent textures in join_meshes_as_batch.")

    # Now we know there are multiple meshes and they have textures to merge.
    all_textures = [mesh.textures for mesh in meshes]
    first = all_textures[0]
    tex_types_same = all(type(tex) == type(first) for tex in all_textures)

    if not tex_types_same:
        raise ValueError("All meshes in the batch must have the same type of texture.")

    tex = first.join_batch(all_textures[1:])
    return Meshes(verts=verts, faces=faces, textures=tex)


def join_meshes_as_scene(
    meshes: Union[Meshes, List[Meshes]], include_textures: bool = True
) -> Meshes:
    """
    Joins a batch of meshes in the form of a Meshes object or a list of Meshes
    objects as a single mesh. If the input is a list, the Meshes objects in the
    list must all be on the same device. Unless include_textures is False, the
    meshes must all have the same type of texture or must all not have textures.

    If textures are included, then the textures are joined as a single scene in
    addition to the meshes. For this, texture types have an appropriate method
    called join_scene which joins mesh textures into a single texture.
    If the textures are TexturesAtlas then they must have the same resolution.
    If they are TexturesUV then they must have the same align_corners and
    padding_mode. Values in verts_uvs outside [0, 1] will not
    be respected.

    Args:
        meshes: Meshes object that contains a batch of meshes, or a list of
                    Meshes objects.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        new Meshes object containing a single mesh
    """
    if not isinstance(include_textures, (bool, int)):
        # We want to avoid letting join_meshes_as_scene(mesh1, mesh2) silently
        # do the wrong thing.
        raise ValueError(
            f"include_textures argument cannot be {type(include_textures)}"
        )
    if isinstance(meshes, List):
        meshes = join_meshes_as_batch(meshes, include_textures=include_textures)

    if len(meshes) == 1:
        return meshes
    verts = meshes.verts_packed()  # (sum(V_n), 3)
    # Offset automatically done by faces_packed
    faces = meshes.faces_packed()  # (sum(F_n), 3)
    textures = None

    if include_textures and meshes.textures is not None:
        textures = meshes.textures.join_scene()

    mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures)
    return mesh
