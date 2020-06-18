# Copyright 2020 Daniel J. Tait
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Base mesh class for FEM solving. """
import abc
import six
import tensorflow as tf
import numpy as np
from typing import Tuple

__all__ = ['BaseMesh', ]


@six.add_metaclass(abc.ABCMeta)
class BaseMesh(tf.Module):
    """ Abstract base class for meshes.

    """
    def __init__(self,
                 nodes,
                 elements,
                 boundary_elements,
                 dtype=None,
                 name='base_mesh'):
        """ Creates a mesh instance

        Args:
            nodes: A float `Tensor` like of shape `[n_nodes, spatial_dimension]`, the
              vertices defining the mesh. `spatial_dimension` is the dimension of the
              Euclidean space in which the mesh is embedded.
            elements: An integer `Tensor` of shape `[n_elements, element_dimension]`,
              where `element_dimension` is the number of elements in each mesh.
            boundary_elements: An integer `Tensor` of shape
              `[n_boundary_elements, boundary_element_dimension]` giving the boundary
              elements of the mesh used for applying boundary conditions.
            dtype: (optional) A `tf.DType` giving the data-type of the mesh nodes.
              Default, None then dtype is inferred from `nodes`.
            name: (optional) A string giving the name of the mesh module.
              Default, `base_mesh`.
        """
        super(BaseMesh, self).__init__(name=name)
        self._nodes = tf.convert_to_tensor(
            nodes,
            dtype=dtype,
        )
        if dtype is not None:
            self.cast_nodes(dtype)

        self._elements = tf.convert_to_tensor(
            elements,
            dtype=tf.int32
        )

        self._boundary_elements = tf.convert_to_tensor(
                boundary_elements,
                dtype=tf.int32
        )

        # compute the indices of boundary nodes
        self._boundary_node_indices = self._get_boundary_node_indices_from_boundary_faces()

        # compute the interior indices
        self._interior_node_indices = self._get_interior_node_indices()

    @property
    def n_nodes(self):
        return tf.shape(self.nodes)[-2]

    @property
    def dtype(self):
        """ dtype of the mesh nodes. """
        return self.nodes.dtype

    @property
    def nodes(self):
        """ Nodes of the mesh. """
        return self._nodes

    @property
    def elements(self):
        """ Elements of the mesh. """
        return self._elements

    @property
    def n_elements(self):
        return tf.shape(self.elements)[-2]

    @property
    def boundary_elements(self):
        """ Boundary elements of the mesh. """
        return self._boundary_elements

    @property
    def boundary_node_indices(self):
        """ Indices of the boundary nodes of the mesh. """
        return self._boundary_node_indices

    @property
    def interior_node_indices(self):
        """ Indices of the interior nodes of the mesh. """
        return self._interior_node_indices

    @property
    def interior_nodes(self):
        return tf.gather(self.nodes, self.interior_node_indices)

    @property
    def boundary_nodes(self):
        return tf.gather(self.nodes, self.boundary_node_indices)

    def cast_nodes(self, dtype):
        """ Cast the node tensors to a new dtype. """
        self._nodes = tf.cast(self._nodes, dtype)

    def get_tensor_repr(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        nodes = self.nodes
        elements = self.elements
        boundary_elements = self.boundary_elements
        bnd_node_indices = self.boundary_node_indices
        node_types = tf.scatter_nd(self.boundary_node_indices[..., tf.newaxis],
                                   tf.ones_like(bnd_node_indices),
                                   shape=[self.n_nodes, ])

        return nodes, elements, boundary_elements, node_types

    def _get_boundary_node_indices_from_boundary_faces(self):
        """ Gets the indices of boundary nodes from the boundary faces.

        Returns:
            boundary_node_indices: An `integer` tensor of shape `[n_boundary_nodes]`
              formed by taking the unique elements from `self.boundary_elements`
        """
        bnd_nodes = tf.numpy_function(
            lambda x: np.unique(x).astype(np.int32),
            [self.boundary_elements, ], Tout=tf.int32)
        return bnd_nodes

    def _get_interior_node_indices(self):
        """ Gets the interior indices of the mesh.

        Returns:
            interior_node_indices: An `integer` tensor of shape `[n_interior_nodes]`
              representing indices of `self.nodes` which are in the interior.
        """
        def diff_nodes(n_nodes, bnd_node_indices):
            return np.setdiff1d(np.arange(n_nodes), bnd_node_indices).astype(np.int32)

        return tf.numpy_function(diff_nodes,
                                 [self.n_nodes, self.boundary_node_indices],
                                 Tout=tf.int32)
