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
""" Layer to assemble the convection matrix. """
import tenfem
from typing import Callable
from tenfem.layers import BaseFEMLayer
import tensorflow as tf


class AssembleConvectionMatrix(BaseFEMLayer):
    """ tf.keras Layer for assembling the convection matrix. """

    def __init__(self,
                 transport_vector_field: Callable,
                 name: str = 'assemble_convection_matrix',
                 *args, **kwargs):
        """ Create an AssembleConvectionMatrix layer. """
        super(AssembleConvectionMatrix, self).__init__(name=name, *args, **kwargs)
        self._transport_vector_field = transport_vector_field

    @property
    def transport_vector_field(self):
        """ Transport vector field. """
        return self._transport_vector_field

    def call(self, mesh_tensor_repr):
        mesh = tenfem.mesh.utils.mesh_from_tensor_repr(mesh_tensor_repr,
                                                       self.reference_element)
        element = self.reference_element

        # shape [mesh.n_elements, element_dim, spatial_dim]
        mesh_quadrature_nodes = element.get_quadrature_nodes(mesh)

        element_dim = tf.shape(mesh_quadrature_nodes)[-2]
        spatial_dim = tf.shape(mesh.nodes)[-1]

        # evaluate the diffusion coefficient at the quadrature nodes
        flat_mesh_quadrature_nodes = tf.reshape(mesh_quadrature_nodes, [-1, spatial_dim])
        transport_vector_field_vals = tf.reshape(
            self.transport_vector_field(flat_mesh_quadrature_nodes),
            [-1, mesh.n_elements, element_dim, 2])

        local_convection_mat = tenfem.fem.assemble_local_convection_matrix(
            transport_vector_field_vals, mesh, element)
        local_convection_mat = tf.reshape(local_convection_mat, [-1, mesh.n_elements, element_dim, element_dim])

        batch_size = tf.shape(local_convection_mat)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        global_convection_mat = tenfem.fem.scatter_matrix_to_global(
            local_convection_mat, elements, mesh.n_nodes)

        return global_convection_mat
