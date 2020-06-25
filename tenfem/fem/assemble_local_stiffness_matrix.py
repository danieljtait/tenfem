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


def assemble_local_stiffness_matrix(scalar_diffusion_coefficient: tf.Tensor,
                                    mesh: tenfem.mesh.BaseMesh,
                                    element: tenfem.reference_elements.BaseReferenceElement) -> tf.Tensor:
    """ Function to assemble the local stiffness matrix over a mesh.

    ToDo(dan): Implement tensor valued diffusion coefficients.

    Args:
        scalar_diffusion_coefficient: A float `Tensor` of shape
          `[..., n_elements, element_dim] giving the values of the scalar
            diffusion coefficient.
        mesh: A `tenfem.mesh.BaseMesh` object representing the finite element
          mesh of the domain.
        element: A `tenfem.reference_element` object describing the elements of the mesh.

    Returns:
        local_stiffness_matrix: A `Tensor` of shape `[..., n_elements, element_dim, element_dim]`
          giving the local values of the stiffness matrix tensor over elements.
    """
    element_nodes = tf.gather(mesh.nodes, mesh.elements)

    wts, quad_nodes = element.get_quadrature_nodes_and_weights()
    _, pf_shape_fn_grad, jac_det = element.isomap(element_nodes, quad_nodes)

    wxarea = jac_det * wts / 2
    ak_q = tf.reduce_sum(pf_shape_fn_grad[..., tf.newaxis] * pf_shape_fn_grad[..., tf.newaxis, :], axis=0)

    # ak_q.shape = [n_elements, len(wts), element_dim, element_dim]

    # scalar diff coeff should be shape mesh.get_quadrature_nodes()[..., 0] = [n_elements, len(wts)]
    ak_q = ak_q * scalar_diffusion_coefficient[..., tf.newaxis, tf.newaxis]

    return tf.reduce_sum(ak_q * wxarea[..., tf.newaxis, tf.newaxis], axis=-3)

