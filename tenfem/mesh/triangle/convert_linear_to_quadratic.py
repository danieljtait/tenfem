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
""" Convert a triangle mesh to one that can support degree 2 polynomial elements. """
import tensorflow as tf
import numpy as np
import tenfem


def _convert_linear_to_quadratic(nodes, elements, boundary_elements):
    """ numpy implementation. """

    def is_in_edge_set(i, j, edge_set):
        """ Utility function to check if key in set with permutation. """
        try:
            _ = edge_set[(i, j)]
            return True
        except KeyError:
            try:
                _ = edge_set[(j, i)]
                return True
            except KeyError:
                return False

    # create a hash-table for edges on the boundary
    bnd_edge_set = {(x[0], x[1]): True for x in boundary_elements}
    edge_set = {}

    node_id = nodes.shape[-2]  # start counting from n_nodes
    new_nodes = []
    new_elements = []
    new_boundary_elements = []

    for elem in elements:
        new_elem_nodes = []
        for cnt, start in enumerate(elem):
            if cnt == 2:
                stop = elem[0]
            else:
                stop = elem[cnt + 1]

            if is_in_edge_set(start, stop, edge_set):
                try:
                    split_node_id = edge_set[(start, stop)]
                except KeyError:
                    split_node_id = edge_set[(stop, start)]
            else:
                # We need to create a new node
                edge_set[(start, stop)] = node_id
                split_node_id = node_id
                node_id += 1

                # and get the coordinates of the new node
                node_coords = 0.5 * (nodes[start] + nodes[stop])
                new_nodes.append(node_coords)

            new_edge = [start, split_node_id]

            if is_in_edge_set(start, stop, bnd_edge_set):
                new_boundary_elements.extend(([start, split_node_id],
                                              [split_node_id, stop]))
            new_elem_nodes.append(split_node_id)
        a, b, c = new_elem_nodes
        new_elem = elem.tolist() + [b, c, a]
        new_elements.append(new_elem)

    new_nodes = np.row_stack((nodes, new_nodes)).astype(nodes.dtype)
    new_elements = np.array(new_elements).astype(np.int32)
    new_boundary_elements = np.array(new_boundary_elements).astype(np.int32)

    return new_nodes, new_elements, new_boundary_elements


def convert_linear_to_quadratic(mesh):

    nodes, elements, boundary_elements = tf.numpy_function(
        _convert_linear_to_quadratic,
        [mesh.nodes, mesh.elements, mesh.boundary_elements],
        Tout=[mesh.dtype, tf.int32, tf.int32])

    return tenfem.mesh.TriangleMesh(nodes,
                                    elements,
                                    boundary_elements,
                                    dtype=mesh.dtype)
