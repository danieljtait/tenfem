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
import tensorflow as tf
import tenfem
from tenfem.reference_elements import TriangleElement


def mesh_from_tensor_repr(mesh_tensor_repr, mesh_element):
    """ Utility function to build a mesh from its tensor representation.

    Args:
        mesh_tensor_repr: A length four tuple of `Tensor`s giving a
          representation of the mesh in Tensor form. This tensor is of the
          form `(nodes, elements, boundary_elements, node_types)`.
        mesh_element: A `tenfem.fem.reference_element.Element` object giving
          information on the type of mesh to be assemble from the
          `mesh_tensor_repr`.

    Returns:
        mesh: A `tenfem.mesh` mesh object of element type determined
          by `mesh_element`.
    """
    nodes, elements, node_types = mesh_tensor_repr

    is_int_node = node_types == 0
    n_nodes = tf.shape(nodes)[-2]

    interior_node_indices = tf.gather(tf.range(n_nodes), tf.where(is_int_node)[:, 0])
    boundary_node_indices = tf.gather(tf.range(n_nodes), tf.where(tf.math.logical_not(is_int_node))[:, 0])

    boundary_elements =

    if isinstance(mesh_element, TriangleElement):
        return tenfem.mesh.TriangleMesh(
            nodes,
            elements,
            dtype=nodes.dtype)

    else:
        raise NotImplementedError
