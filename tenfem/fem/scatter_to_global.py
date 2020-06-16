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
""" Functions to scatter local element matrices and tensors to global. """
import tensorflow as tf
import tenfem


def scatter_matrix_to_global(local_values: tf.Tensor,
                             elements: tf.Tensor,
                             n_nodes: int) -> tf.Tensor:
    """ Scatters elements of a local tensor to the global.

    Args:
        local_values: A rank 4 float tensor of shape
          `[batch_size, num_elements, element_dim, element_dim]`
          giving batches of local stiffness matrices defined over the given mesh.
        elements: An rank 3 integer tensor giving the elements of the mesh of shape
          `[batch_size, num_elements, element_dim]`.
        n_nodes: integer, the number of nodes of the output matrix.

    Returns:
        global_matrix: A rank 3 float tensor of shape [batch_size, n_nodes, n_nodes]
          obtained by scattering the local matrix into the global matrix according to
          row-wise Cartesian products of mesh.element indices.
    """
    # ToDo: work out shape checking
    # assert tf.rank(local_values) == 4
    indices = tenfem.fem.indexing_utils.get_batched_element_indices(elements)
    updates = tf.reshape(local_values, [-1])
    batch_size = tf.shape(local_values)[0]
    shape = [batch_size, n_nodes, n_nodes]
    return tf.scatter_nd(indices, updates, shape)


def scatter_vector_to_global(local_values: tf.Tensor,
                             elements: tf.Tensor,
                             n_nodes: int) -> tf.Tensor:
    """ Scatters local vector values to global load vector.

    Args:
        local_values: A float `Tensor` of shape
          `[batch_size, n_elements, element_dim]` giving the values
          of the local load vector for each shape function.
        elements: An rank 3 integer tensor giving the elements of the mesh of shape
          `[batch_size, num_elements, element_dim]`.
        n_nodes: integer, the number of nodes of the output matrix.

    Returns:
        global_load_vector: A float `Tensor` of shape
         `[batch_size, n_nodes, 1]` giving the values of the local
         load vector for each node in the mesh.
    """
    indices = tenfem.fem.indexing_utils.get_batched_vector_element_indices(
        elements)
    updates = tf.reshape(local_values, [-1])
    batch_size = tf.shape(local_values)[0]
    shape = [batch_size, n_nodes]
    return tf.scatter_nd(indices, updates, shape)[..., tf.newaxis]
