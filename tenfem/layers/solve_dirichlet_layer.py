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
""" Layer to solve the Dirichlet problem. """
import tensorflow as tf
import tenfem
from tenfem.layers import BaseFEMLayer


class SolveDirichletProblem(BaseFEMLayer):
    def __init__(self,
                 boundary_condition,
                 name='solve_dirichlet_problem'):
        super(SolveDirichletProblem, self).__init__(
            name=name)
        self._boundary_condition = boundary_condition

    @property
    def boundary_condition(self):
        return self._boundary_condition

    def call(self, inputs):
        stiffness_matrix, load, mesh_tensor_repr = inputs

        nodes = mesh_tensor_repr[0]
        node_types = mesh_tensor_repr[-1]

        boundary_values = self.boundary_condition(
            tf.gather_nd(nodes, tf.where(node_types == 1)))

        return tenfem.fem.solve_dirichlet_form_linear_system(
            stiffness_matrix[0], load[0], node_types, boundary_values)