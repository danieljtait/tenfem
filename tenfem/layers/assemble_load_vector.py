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
""" Layer to assemble the load vector. """
from typing import Callable
import tensorflow as tf
import tenfem
from tenfem.layers import BaseFEMLayer


class AssembleLoadVector(BaseFEMLayer):
    """ tf.keras Layer for assembling the stiffness matrix """
    def __init__(self,
                 source: Callable,
                 name: str = 'assemble_load_vector',
                 *args, **kwargs):
        """ Create and AssembleLoadVector layer. """
        super(AssembleLoadVector, self).__init__(name=name, *args, **kwargs)
        self._source = source

    @property
    def source(self):
        """ Source function. """
        return self._source

    def call(self, mesh_tensor_repr):
        mesh = tenfem.mesh.utils.mesh_from_tensor_repr(mesh_tensor_repr,
                                                       self.reference_element)
        element = self.reference_element

        # shape [mesh.n_elements, element_dim, spatial_dim]
        mesh_quadrature_nodes = element.get_quadrature_nodes(mesh)
        element_dim = tf.shape(mesh_quadrature_nodes)[-2]
        spatial_dim = tf.shape(mesh.nodes)[-1]

        # evaluate the source at the quadrature node
        flat_mesh_quadrature_nodes = tf.reshape(mesh_quadrature_nodes, [-1, spatial_dim])

        source_vals = tf.reshape(
            self.source(flat_mesh_quadrature_nodes),
            [-1, mesh.n_elements, element_dim])

        local_load_vector = tenfem.fem.assemble_local_load_vector(
            source_vals, mesh, element)
        local_load_vector = tf.reshape(local_load_vector,
                                       [-1, mesh.n_elements, element_dim])

        batch_size = tf.shape(local_load_vector)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        load_vector = tenfem.fem.scatter_vector_to_global(
            local_load_vector, elements, mesh.n_nodes)

        return load_vector
