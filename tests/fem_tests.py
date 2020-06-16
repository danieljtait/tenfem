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
        diff_coeff = tf.ones([3, 1, 4, mesh.n_elements, element_dim])

        local_stiffness_mat = tenfem.fem.assemble_local_stiffness_matrix(
            diff_coeff, mesh, element)

        print(local_stiffness_mat.shape)

if __name__ == '__main__':
    absltest.main()

