# Copyright 2020 Daniel J. Tait.
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
""" Layers to provide meshes. """
import tensorflow as tf
from tenfem.layers import BaseFEMLayer


class MeshProvider(tf.keras.layers.Layer):
    """ Layer to provide a mesh. """
    def __init__(self,
                 mesh,
                 reference_element,
                 padding_element=-1,
                 return_precond_matrix=False,
                 name='mesh_provider'):
        """ Create a MeshProvider instance. """
        super(MeshProvider, self).__init__(reference_element=reference_element,
                                           name=name)

        element_dim = tf.shape(mesh.elements)[-1]
        # shape checking of elements
        if element_dim != reference_element.element_dim:
            raise ValueError(
                ''.join(('mesh.elements is a Tensor of shape {}.'.format(tf.shape(mesh.elements)),
                         'but reference_element.element_dim == {}'.format(reference_element.element_dim))))

        self.mesh = mesh
        self.padding_element = tf.constant(padding_element)
        self.n_nodes = self.mesh.n_nodes
        self.n_elements = self.mesh.n_elements

        self.return_precond_matrix = return_precond_matrix
        if self.return_precond_matrix:
            raise NotImplementedError
            # self._build_precond_matrix()

    # def _build_precond_matrix(self):
    #     self.precond_matrix = tenfem.fem.StiffnessMatrixAssembler.assemble(
    #        tf.ones([1]), self.mesh)

    def call(self, inputs):
        mesh_tensor_repr = self.mesh.get_tensor_repr()
        if self.return_precond_matrix:
            return mesh_tensor_repr, self.precond_matrix
        else:
            return mesh_tensor_repr