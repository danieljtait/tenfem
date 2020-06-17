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
""" Layer to assemble the stiffness matrix. """
from typing import Callable
import tensorflow as tf
import tenfem
from tenfem.layers import BaseFEMLayer


class AssembleStiffnessMatrix(BaseFEMLayer):
    """ tf.keras Layer for assembling the stiffness matrix. """
    def __init__(self,
                 diffusion_coefficient: Callable,
                 name: str = 'assemble_stiffness_matrix'):
        """ Create an AssembleStiffnessMatrix layer. """
        super(AssembleStiffnessMatrix, self).__init__(name=name)
        self._diffusion_coefficient = diffusion_coefficient

    @property
    def diffusion_coefficient(self):
        """ Diffusion coefficient. """
        return self._diffusion_coefficient

    def call(self, mesh_tensor_repr):
        mesh = tenfem.mesh.utils.mesh_from_tensor_repr(mesh_tensor_repr)
        element = self.reference_element

        # shape [mesh.n_elements, element_dim, spatial_dim]
        mesh_quadrature_nodes = element.get_quadrature_nodes(mesh)

        element_dim = tf.shape(mesh_quadrature_nodes)[-2]
        spatial_dim = tf.shape(mesh.nodes)[-1]

        # evaluate the diffusion coefficient at the quadrature nodes
        flat_mesh_quadrature_nodes = tf.reshape(mesh_quadrature_nodes, [-1, spatial_dim])
        diffusion_coeff_vals = tf.reshape(
            self.diffusion_coefficient(flat_mesh_quadrature_nodes),
            [-1, mesh.n_elements, element_dim])

        local_stiffness_mat = tenfem.fem.assemble_local_stiffness_matrix(
            diffusion_coeff_vals, mesh, element)
        local_stiffness_mat = tf.reshape(local_stiffness_mat,
                                         [-1, mesh.n_elements, element_dim, element_dim])

        batch_size = tf.shape(local_stiffness_mat)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        global_stiffness_mat = tenfem.fem.scatter_matrix_to_global(
            local_stiffness_mat, elements, mesh.n_nodes)

        return global_stiffness_mat
