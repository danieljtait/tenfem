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
from .shape_functions import p1_shape_fn, p2_shape_fn
from .gaussian_quadrature import gauss_quad_nodes_and_weights


class TriangleElement(object):
    """ Reference element for a triangle mesh. """
    def __init__(self, degree: int, dtype: tf.DType = tf.float32):
        """

        Args:
            degree: A python integer giving the degree of the polynomials on this
              function space.
            dtype: (optional) A `tf.DType` giving the data-type of the mesh nodes.
              Default, None then dtype defaults to `tf.float32`.

        Raises:
            NotImplementedError: If `degree` is not in [1, 2]. Only linear and
              quadratic shape functions currently implemented.
        """
        self._degree = degree
        self._dtype = dtype

        if self.degree == 1:
            self._shape_fn = p1_shape_fn
            self._quadrature_order = tf.constant(2, dtype=tf.int32)
        elif self.degree == 2:
            self._shape_fn = p2_shape_fn
            self._quadrature_order = tf.constant(4, dtype=tf.int32)
        else:
            raise NotImplementedError(''.join(
                ('Only degree p in [1, 2] polynomials currently supported',
                 'for TriangleElement')))

    @property
    def degree(self):
        return self._degree

    @property
    def quadrature_order(self):
        """ The number of nodes for Gaussian quadrature. """
        return self._quadrature_order

    @property
    def dtype(self):
        """ Data-type of the element. """
        return self._dtype

    @property
    def shape_function(self):
        """ Shape functions on the element. """
        return self._shape_fn

    def get_quadrature_nodes(self, mesh):
        """ Get the gaussian quadrature nodes of the mesh.

        Args:
            mesh: A `tenfem.mesh.BaseMesh` object giving the mesh we want to find the
              quadrature nodes on.

        Returns:
            quadrature_nodes: A float `tf.Tensor` of shape
              `[mesh.n_elements, n_quadrature_nodes, mesh.spatial_dimension]`
              giving the coordinates of the quadrature nodes on the mesh.

        """
        shape_fn = self.shape_function
        _, quad_nodes = gauss_quad_nodes_and_weights(self.quadrature_order, dtype=self.dtype)
        element_nodes = tf.gather(mesh.nodes, mesh.elements)
        shape_fn_vals, shape_fn_grads = shape_fn(quad_nodes[..., 0], quad_nodes[..., 1])
        return tf.reduce_sum(element_nodes[..., tf.newaxis, :, :]
                             * shape_fn_vals[..., tf.newaxis], axis=-2)
