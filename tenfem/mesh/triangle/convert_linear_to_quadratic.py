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
import itertools
import tensorflow as tf
import numpy as np


def build_adjacency_matrix(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """ Builds the adjacency matrix for a set of triangular elements.

    Note these are numpy implementations of functions which won't
    need to be differentiated through.

    Args:
        nodes: Array like shape `[n_nodes, 2]`.
        elements: Array like of shape `[n_nodes, 3]`.

    Returns:
        adjacenvy_matrix: Array like of shape `[n_nodes, n_nodes]` giving
          the mesh adjacency structure determined by nodes and their edges.

    """
    n_nodes = nodes.shape[0]
    i, j, k = elements.T
    adj = np.zeros([n_nodes]*2)
    adj[j, k] = -1.
    adj[i, k] = -1.
    adj[i, j] = -1.
    adj = -1. * ((adj + adj.T) < 0)
    return adj


def _linear_trimesh_to_quadratic(nodes, elements, boundary_node_indices):
    """ Convert a mesh designed for linear elements to one suited for quadratic. """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    adj = build_adjacency_matrix(nodes, elements)

    # process the adjacency matrix to find edges
    triu_adj = np.triu(adj)  # use only the upper triangle
    edge_dict = {i: np.where(triu_adj[i] < 0)[0] for i in range(n_nodes)}

    edge_node_indices = np.asarray(
        list(itertools.chain.from_iterable(
            [zip([key] * len(item), item) for key, item in edge_dict.items()])))

    # reassemble the adjacency matrix with new numbering, this is used to count unique edges.
    adj = np.zeros([n_nodes] * 2).astype(np.int32)
    adj[edge_node_indices[:, 0], edge_node_indices[:, 1]] = range(len(edge_node_indices))
    adj = adj + adj.T

    # collect those edges which start and finish at a boundary node
    new_bnd_node_edges = np.logical_and(
        np.isin(edge_node_indices[:, 0], boundary_node_indices),
        np.isin(edge_node_indices[:, 1], boundary_node_indices))

    new_boundary_node_indices = adj[edge_node_indices[new_bnd_node_edges, 0],
                                    edge_node_indices[new_bnd_node_edges, 1]]
    new_boundary_node_indices += n_nodes  # start the counting at n_nodes

    # start finding edge mid points to create new nodes
    # change edges to be the new nodes -- indexing starts at n_nodes + 1
    edges = np.zeros([n_elements, 3]).astype(np.int32)
    for k in range(n_elements):
        edges[k, :] = [adj[elements[k, 1], elements[k, 2]],
                       adj[elements[k, 0], elements[k, 2]],
                       adj[elements[k, 0], elements[k, 1]]]
    edges = edges + n_nodes
    num_edges = edge_node_indices.shape[0]

    i, j, k = elements.T  # unpack elements
    new_nodes = np.zeros([n_nodes + num_edges, 2])
    new_nodes[:n_nodes] = nodes.copy()

    xcoord0 = 0.5 * (nodes[j, 0] + nodes[k, 0])
    ycoord0 = 0.5 * (nodes[j, 1] + nodes[k, 1])
    new_nodes[edges[..., 0], 0] = xcoord0
    new_nodes[edges[..., 0], 1] = ycoord0

    xcoord1 = 0.5 * (nodes[i, 0] + nodes[k, 0])
    ycoord1 = 0.5 * (nodes[i, 1] + nodes[k, 1])
    new_nodes[edges[..., 1], 0] = xcoord1
    new_nodes[edges[..., 1], 1] = ycoord1

    xcoord2 = 0.5 * (nodes[i, 0] + nodes[j, 0])
    ycoord2 = 0.5 * (nodes[i, 1] + nodes[j, 1])
    new_nodes[edges[..., 2], 0] = xcoord2
    new_nodes[edges[..., 2], 1] = ycoord2

    return (new_nodes.astype(np.float32),
            np.column_stack((elements, edges)),
            new_bnd_node_edges)


def convert_linear_to_quadratic(mesh):
    node_dtype = mesh.dtype

    nodes, elements, boundary_elements = tf.numpy_function(
        _linear_trimesh_to_quadratic,
        [mesh.nodes, mesh.elements, mesh.boundary_node_indices],
        Tout=[node_dtype, tf.int32, tf.int32])

    return nodes, elements, boundary_elements
