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
        elems = np.array([[0, 1, 2], [2, 3, 0]])
        bnd_elems = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        mesh = tenfem.mesh.BaseMesh(nodes, elems, bnd_elems)
        self.assertIsInstance(mesh.dtype, tf.float64)
