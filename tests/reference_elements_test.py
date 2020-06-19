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
""" Tests for mesh module """
from absl.testing import absltest

import tenfem
import tensorflow as tf

import numpy as np


class TriangleElementTest(absltest.TestCase):

    def test_triangle_element_init(self):
        tri_elem_clz = tenfem.reference_elements.TriangleElement
        self.assertRaises(NotImplementedError, lambda: tri_elem_clz(degree=3))

        element = tri_elem_clz(degree=1)
        self.assertEqual(element.dtype, np.float32)
        self.assertEqual(element.element_dim, 3)

        element = tri_elem_clz(degree=2, dtype=tf.float64)
        self.assertEqual(element.dtype, np.float64)
        self.assertEqual(element.element_dim, 6)

    def test_quadrature_nodes(self):
        element = tenfem.reference_elements.TriangleElement(degree=1)
        mesh = tenfem.mesh.examples.square(2, 2)
        quad_nodes = element.get_quadrature_nodes(mesh)
        np.testing.assert_allclose(tf.shape(quad_nodes), [mesh.n_elements, 3, 2])


if __name__ == '__main__':
    absltest.main()
