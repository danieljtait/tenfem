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
""" Layers to implement the FEM method. """
from .base_fem_layer import BaseFEMLayer
from .assemble_stiffness_matrix import AssembleStiffnessMatrix
from .assemble_load_vector import AssembleLoadVector
from .mesh_provider import MeshProvider
from .solve_dirichlet_layer import SolveDirichletProblem
from .triangle_mesh_interpolater_layer import TriangleMeshInterpolator
from .linear_elliptic_operator import LinearEllipticOperator
