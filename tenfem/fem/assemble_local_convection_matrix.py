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
""" Assembly of the local elements of the stiffness matrix. """
import tensorflow as tf
import tenfem


def _move_first_axis_to_last(arr):
    """ Moves the first axis of a tensor to the last.

    Args:
        arr: An array like of arbitrary shape.

    Returns:
        swapped_arr: An array such that `swapped_arr[..., i] == arr[i, ...]`

    """
    arr_rank = tf.rank(arr)
    perm = tf.concat((tf.range(1, arr_rank), tf.zeros([1], dtype=tf.int32)), axis=0)
    return tf.transpose(arr, perm=perm)


def assemble_local_convection_matrix(transport_vector_field: tf.Tensor,
                                     mesh: tenfem.mesh.BaseMesh,
                                     element: tenfem.reference_elements.BaseReferenceElement) -> tf.Tensor:
    """ Function assemble the local convection matrix over a mesh.

    Args:
        transport_vector_field: A float `Tensor` of shape
          `[..., n_elements, element_dim, 2]` giving the values
          of the transport vector field at each of the mesh quadrature
          nodes.
        mesh: A `tenfem.mesh.BaseMesh` object representing the finite
          element mesh of the domain.
        element: A `tenfem.reference_element.BaseReferenceElement`
          object describing the elements of the mesh.

    Returns:
        local_convection_matrix: A float `Tensor` of shape
          `[..., n_elements, element_dim, element_dim]`
          giving the local values of the convection matrix over
          elements.

    """
    element_nodes = tf.gather(mesh.nodes, mesh.elements)

    wts, quad_nodes = element.get_quadrature_nodes_and_weights()
    shape_fn, pf_shape_fn_grad, jac_det = element.isomap(element_nodes, quad_nodes)

    wxarea = jac_det * wts / 2

    # swap shape fun grads to agree with transport_vector_field
    pf_shape_fn_grad = _move_first_axis_to_last(pf_shape_fn_grad)

    # inner product with the transport vector field and the shape function gradients
    local_convec = tf.reduce_sum(transport_vector_field[..., tf.newaxis, :]
                                 * pf_shape_fn_grad, axis=-1)
    local_convec = (local_convec[..., tf.newaxis]
                    * shape_fn[..., tf.newaxis, :])
    return tf.reduce_sum(local_convec * wxarea[..., tf.newaxis, tf.newaxis], axis=-3)
