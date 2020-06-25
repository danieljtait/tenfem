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
""" Assemble the local load vector. """
import tensorflow as tf
import tenfem


def assemble_local_load_vector(source: tf.Tensor,
                               mesh: tenfem.mesh.BaseMesh,
                               element) -> tf.Tensor:
    """ Assembles the load vector from the source function values at quadrature nodes.

    Args:
        source: A float `Tensor` of shape `[..., n_elements, n_quadrature_nodes]`
          giving the values of the scalar source function at the
          `n_quadrature_nodes` of each element.
        mesh: A `tenfem.fem.BaseMesh` object specifying the finite element mesh of
          the region.
        element: A `tenfem.fem.reference_element.BaseReferenceElement` object
          describing the elements of the mesh.

    Returns:
        local_load_vector: A `Tensor` of shape `[..., n_elements, element_dim]` giving
          the values of the local load vectors.
    """
    element_nodes = tf.gather(mesh.nodes, mesh.elements)

    wts, quad_nodes = element.get_quadrature_nodes_and_weights()
    shape_fn_vals, _, jac_det = element.isomap(element_nodes, quad_nodes)

    wxarea = jac_det * wts / 2
    bk_q = source[..., tf.newaxis] * shape_fn_vals
    bk_q = tf.reduce_sum(bk_q * wxarea[..., tf.newaxis], axis=-2)

    return bk_q
