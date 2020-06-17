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
        mesh = tenfem.mesh.mesh_from_tensor_repr(mesh_tensor_repr)

