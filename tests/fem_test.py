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
""" Tests for fem module """
from absl.testing import absltest

import tenfem
import tensorflow as tf

import numpy as np


class AssembleLocalStiffnessMatrixTest(absltest.TestCase):

    def test_assemble_local_stiffness_matrix(self):
        element = tenfem.reference_elements.TriangleElement(degree=1)
        mesh = tenfem.mesh.examples.square(2, 2)
        element_dim = tf.shape(mesh.elements)[-1]

        batch_shape = [3, 1, 4]
        diff_coeff = tf.ones(batch_shape + [mesh.n_elements, element_dim])

        local_stiffness_mat = tenfem.fem.assemble_local_stiffness_matrix(
            diff_coeff, mesh, element)

        self.assertEqual(batch_shape, tf.shape(local_stiffness_mat).numpy()[:3].tolist())

        local_stiffness_mat = tf.reshape(local_stiffness_mat, [-1, mesh.n_elements, element_dim, element_dim])
        batch_size = tf.shape(local_stiffness_mat)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        global_stiffness_mat = tenfem.fem.scatter_matrix_to_global(
            local_stiffness_mat, elements, mesh.n_nodes)

    def test_assemble_local_load_vector(self):
        element = tenfem.reference_elements.TriangleElement(degree=1)
        mesh = tenfem.mesh.examples.square(2, 2)
        element_dim = tf.shape(mesh.elements)[-1]

        batch_shape = [3, 1, 4]
        source = tf.ones(batch_shape + [mesh.n_elements, element_dim])

        local_load_vector = tenfem.fem.assemble_local_load_vector(
            source, mesh, element)

        self.assertEqual(batch_shape, tf.shape(local_load_vector).numpy()[:3].tolist())
        local_load_vector = tf.reshape(local_load_vector, [-1, mesh.n_elements, element_dim])
        batch_size = tf.shape(local_load_vector)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        global_load_vector = tenfem.fem.scatter_vector_to_global(
            local_load_vector, elements, mesh.n_nodes)

        self.assertEqual(tf.shape(global_load_vector).numpy().tolist(),
                         [batch_size.numpy(), mesh.n_nodes, 1])


if __name__ == '__main__':
    absltest.main()

