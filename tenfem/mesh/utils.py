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
import tensorflow as tf
import numpy as np
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


def orientate_triangle_elements(elements):
    """ Orientate the elements on a TriangleMesh. """
    element_dim = tf.shape(elements)[-1]
    if element_dim == 3:
        # on the linear element we assume that the elements are already
        # orientated
        return elements
    elif element_dim == 6:
        elem = elements
        elem = tf.concat((elem[..., 0, tf.newaxis],
                          elem[..., 5, tf.newaxis],
                          elem[..., 1, tf.newaxis],
                          elem[..., 3, tf.newaxis],
                          elem[..., 2, tf.newaxis],
                          elem[..., 4, tf.newaxis]), axis=-1)
        return elem


# `boundary_elements` should be all those points on the boundary, w
def get_boundary_elements(elements, node_types, reference_element) -> tf.Tensor:
    """ Returns the boundary elements on a triangle mesh. """
    # form the edges
    if isinstance(reference_element, TriangleElement):
        orientated_elems = orientate_triangle_elements(elements)
    else:
        raise NotImplementedError('Currently only supported for TriangleElements')

    closed_elems = tf.concat((orientated_elems, orientated_elems[..., :1]), axis=-1)
    edges = tf.concat((closed_elems[..., :-1, tf.newaxis], closed_elems[..., 1:, tf.newaxis]), axis=-1)

    valid_nodes = node_types >= 0

    n_nodes = tf.reduce_sum(tf.where(valid_nodes, tf.ones_like(node_types), tf.zeros_like(node_types)))
    # is_bnd = np.isin(edges.numpy(), square_mesh.boundary_node_indices.numpy())

    boundary_node_indices = tf.gather(tf.range(n_nodes),
                                      tf.where(node_types == 1))
    is_bnd = tf.numpy_function(lambda x, y: np.isin(x, y),
                               [edges, boundary_node_indices], Tout=tf.bool)
    is_bnd_edge = tf.math.logical_and(is_bnd[..., 0], is_bnd[..., 1])
    bnd_edges = tf.gather_nd(edges, tf.where(is_bnd_edge))
    return bnd_edges