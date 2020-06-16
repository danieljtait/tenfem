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
from absl.testing import parameterized

import tenfem
import tensorflow as tf

import numpy as np


class BaseMeshTest(absltest.TestCase):

    def test_base_mesh_init(self):
        nodes = np.array([[0., 0.], [0., 1.], [1., 1.], [0., 1.]])
        elems = np.array([[0, 1, 2], [2, 3, 0]]).astype(np.int32)
        bnd_elems = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).astype(np.int32)

        mesh = tenfem.mesh.BaseMesh(nodes, elems, bnd_elems)
        self.assertEqual(mesh.dtype, np.float64)
        mesh.cast_nodes(tf.float32)
        self.assertEqual(mesh.dtype, np.float32)

        bnd_node_inds = mesh.boundary_node_indices.numpy()
        self.assertEmpty(np.setdiff1d(bnd_node_inds, [0, 1, 2, 3]))

    def test_square_triangle_mesh_example(self):
        nx, ny = (3, 3)
        tri_mesh = tenfem.mesh.examples.square(nx, ny)
        self.assertEqual(tri_mesh.n_nodes, nx * ny)

        int_nodes = tri_mesh.interior_nodes
        np.testing.assert_allclose(int_nodes, np.array([[0.5, 0.5], ]))


if __name__ == '__main__':
    absltest.main()
