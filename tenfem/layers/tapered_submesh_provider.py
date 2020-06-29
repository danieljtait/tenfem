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


class TaperedSubmeshProvider(MeshProvider):
    """ Provides a submesh using mini-patch tapering. """
    def __init__(self,
                 mesh,
                 reference_element,
                 threshold,
                 name='tapered_submesh_provider'):
        """ Create a TaperedSubmeshProvider instance. """
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