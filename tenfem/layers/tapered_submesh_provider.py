# Copyright 2020 Daniel J. Tait.
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
""" Layers to provide tapered sub-meshes. """
import tensorflow as tf
import numpy as np
from tenfem.layers import MeshProvider
from tenfem.mesh.utils import get_boundary_elements


class TaperedSubmeshProvider(MeshProvider):
    """ Provides a submesh using mini-patch tapering. """
    def __init__(self,
                 mesh,
                 reference_element,
                 threshold,
                 name='tapered_submesh_provider'):
        """ Create a TaperedSubmeshProvider instance.

        Args:
            mesh: A `tenfem.mesh.BaseMesh` instance giving the mesh of
              the whole domain.
            reference_element: A `tenfem.reference_elements.BaseReferenceElement`
              describing the type of element in the mesh.
            threshold: A float giving the radius of the each of the tapered
              neighborhoods.
        """
        super(TaperedSubmeshProvider, self).__init__(mesh,
                                                     reference_element,
                                                     name=name)
        self.threshold = tf.constant(threshold)

    def _build_tapered_neighbors(self):
        def _compute_tapered_neighbors(x, threshold, boundary_nodes):
            """ Numpy implementation of a function to compute the neighbours.
            Args:
                x: nodes
                threshold: tapered radius
                boundary_nodes: boudnary nodes of larger mesh

            Returns:
                tapered_neighbors: list of [..., ]

            """
            r = np.sum(x*x, axis=1)[:, None]
            pwd = r - 2*x.dot(x.T) + r.T
            n_nodes = np.shape(x)[-2]
            enumerated_inds = np.arange(n_nodes).astype(np.int32)
            tapered_neighbors = [enumerated_inds[pwd[i] < threshold] for i in enumerated_inds]
            # need to remove the boundary nodes of the master mesh from tapered neighbours
            tapered_neighbors = [np.setdiff1d(item, boundary_nodes)
                                 for item in tapered_neighbors]
            return tapered_neighbors

        self.tapered_neighbours = tf.numpy_function(
            _compute_tapered_neighbors,
            [self.mesh.nodes, self.threshold, self.mesh.boundary_node_indices],
            Tout=tf.int32)
        self.tapered_neighbours = tf.ragged.constant(
            [x.numpy() for x in self.tapered_neighbours])

    def _build_mesh_neighborhood(self):
        """ Compute the FEM neighbor structure implied by the mesh. """

        def maybe_add_neighbor(i, j, neighbors):
            """ Utility function to maybe add neighbors. """
            try:
                nei = neighbors[i]
                if j not in nei:
                    nei.append(j)
            except KeyError:
                neighbors[i] = [j]

        def _compute_mesh_neighbours(elems, n_nodes):
            """ numpy implementation. """
            neighbors = {}
            for e in elems:
                for j in range(elems.shape[1]):
                    nxt = e[j + 1] if j < (elems.shape[1] - 1) else e[0]
                    maybe_add_neighbor(e[j], nxt, neighbors)
                    maybe_add_neighbor(nxt, e[j], neighbors)
            nei_list = [np.array(list(neighbors[i])).astype(np.int32)
                        for i in range(n_nodes)]
            return nei_list

        self.mesh_neighbours = tf.numpy_function(
            _compute_mesh_neighbours,
            [self.mesh.elements, self.mesh.n_nodes],
            Tout=tf.int32)

        # convert to ragged tensor
        self.mesh_neighbours = tf.ragged.constant([x.numpy() for x in self.mesh_neighbours])


    def get_submesh(self, vertex):
        # get the interior nodes and boundary nodes
        interior_nodes = tf.gather(self.tapered_neighbours, vertex)
        boundary_nodes = tf.gather(self.mesh_neighbours, interior_nodes)
        boundary_nodes = boundary_nodes.flat_values
        boundary_nodes = tf.sets.difference(boundary_nodes[None, ...], interior_nodes[None, ...]).values

        def _get_elements(int_nodes, elements):
            """ numpy implementation of finding the elements used. """
            is_mini_patch_element = np.any(np.isin(elements, int_nodes), axis=1)
            return elements[is_mini_patch_element]

        mini_patch_elements = tf.numpy_function(_get_elements,
                                                [interior_nodes, self.mesh.elements],
                                                Tout=tf.int32)

        node_indices = tf.concat((interior_nodes, boundary_nodes), axis=0)

        def _gather_and_remap_inds(node_indices, elements):
            node_map = {item: i for i, item in enumerate(node_indices)}
            mapped_elems = np.array(
                list(map(lambda i: node_map[i], elements.flatten())))
            mapped_elems = np.reshape(mapped_elems, elements.shape).astype(np.int32)
            return mapped_elems

        # re-base the node indexes
        mini_patch_elements = tf.numpy_function(_gather_and_remap_inds,
                                                [node_indices, mini_patch_elements],
                                                Tout=tf.int32)
        mini_patch_nodes = tf.gather(self.mesh.nodes, node_indices)

        n_int_nodes = tf.shape(interior_nodes)[0]
        n_bnd_nodes = tf.shape(boundary_nodes)[0]

        node_types = tf.concat([tf.zeros(n_int_nodes, dtype=tf.int32),
                                tf.ones(n_bnd_nodes, dtype=tf.int32)], axis=0)

        boundary_elements = get_boundary_elements(mini_patch_elements,
                                                  node_types,
                                                  self.reference_element)

        if self.return_precond_matrix:
            precond_matrix = self.precond_matrix
            precond_submesh = tf.gather(
                tf.gather(precond_matrix, node_indices, axis=1), node_indices, axis=0)
            return mini_patch_nodes, mini_patch_elements, node_types, boundary_elements, precond_submesh
        else:
            return mini_patch_nodes, mini_patch_elements, node_types, boundary_elements

    def call(self, inputs):
        """ Returns the tensor representation of a tapered sub-mesh. """
        vertex = tf.random.uniform(shape=[], maxval=self.mesh.n_nodes, dtype=tf.int32)
        if self.return_precond_matrix:
            nodes, elements, node_types, precond_matrix = self.get_submesh(vertex)
            mesh_tensor_repr = (nodes, elements, node_types)
            return mesh_tensor_repr, precond_matrix
        else:
            mesh_tensor_repr = self.get_submesh(vertex)
            return mesh_tensor_repr
