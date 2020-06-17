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
import tenfem
from tenfem.reference_elements import TriangleElement


def mesh_from_tensor_repr(mesh_tensor_repr, mesh_element):
    """ Utility function to build a mesh from its tensor representation.

    ToDo: Add functionality to handle masked nodes in `node_types`

    Args:
        mesh_tensor_repr: A length four tuple of `Tensor`s giving a
          representation of the mesh in Tensor form. This tensor is of the
          form `(nodes, elements, boundary_elements, node_types)`.
        mesh_element: A `tenfem.fem.reference_element.Element` object giving
          information on the type of mesh to be assemble from the
          `mesh_tensor_repr`.

    Returns:
        mesh: A `tenfem.mesh` mesh object of element type determined
          by `mesh_element`.
    """
    nodes, elements, boundary_elements, node_types = mesh_tensor_repr

    if isinstance(mesh_element, TriangleElement):
        mesh_clz = tenfem.mesh.TriangleMesh
    else:
        mesh_clz = tenfem.mesh.BaseMesh

    return mesh_clz(nodes,
                    elements,
                    boundary_elements,
                    dtype=nodes.dtype)
