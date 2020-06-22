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
from tenfem.reference_elements import BaseReferenceElement
from .shape_functions import p1_shape_fn, p2_shape_fn
from .gaussian_quadrature import gauss_quad_nodes_and_weights


class TriangleElement(BaseReferenceElement):
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
        super(TriangleElement, self).__init__(name='triangle_element')
        self._degree = degree
        self._dtype = dtype

        # Todo: Investigate why the `get_quadrature_nodes` function
        #  breaks when quadrature order is a tensor -- something to
        #  do with the conditionals.
        if self.degree == 1:
            self._shape_fn = p1_shape_fn
            self._quadrature_order = 2
            self._element_dim = tf.constant(3, dtype=tf.int32)
        elif self.degree == 2:
            self._shape_fn = p2_shape_fn
            self._quadrature_order = 4
            self._element_dim = tf.constant(6, dtype=tf.int32)
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
        print(element_nodes.shape, shape_fn_vals.shape)
        return tf.reduce_sum(element_nodes[..., tf.newaxis, :, :]
                             * shape_fn_vals[..., tf.newaxis], axis=-2)

    def get_quadrature_nodes_and_weights(self):
        """ The nodes and weights for Gaussian quadrature on a triangle element.

        Returns:
            weights: A float `Tensor` like giving the weights of the
              gaussian quadrature rule of `self.quadrature_order`, with data-type equal
              to `dtype`.
            nodes: A float `Tensor` giving the nodes of the Gaussian
              quadrature rule of shape [len(weights), 2], with data-type
              equal to `dtype`.

        Raises:
            NotImplementedError: If order is not in [1, 2, 3, 4].
        """
        return gauss_quad_nodes_and_weights(self.quadrature_order, self.dtype)

    def isomap(self, nodes, canonical_coordinates):
        """ Transform from canonical coordinates to the elements given by nodes.

        ToDo: Add tests to isomap

        Args:
            nodes: A batched tensor of shape [..., element_dim, 2]
            canonical_coordinates: A tensor of shape [n, 2] where p is the number of shape
              functions for the given element.

        Returns:
            shape_fn_vals: The values of the shape functions at the canonical coordinates. This
              will be a tensor of shape [p, element_dim]
            pushfwd_shape_fn_grad: The gradient of shape functions in the physical coordinates. This
              will be a tensor of shape `[2, ..., n, ]`
            jacobian_det: A float `Tensor` of shape `[..., n]` giving the jacobian
              determinant of the coordinate transform at each canonical coordinate.
        """
        r = canonical_coordinates[..., 0]
        s = canonical_coordinates[..., 1]
        shape_fn_vals, shape_fn_grad = self.shape_function(r, s)

        x = nodes[..., 0][..., tf.newaxis, :]
        y = nodes[..., 1][..., tf.newaxis, :]

        j11 = tf.reduce_sum(shape_fn_grad[0] * x, axis=-1)
        j12 = tf.reduce_sum(shape_fn_grad[0] * y, axis=-1)
        j21 = tf.reduce_sum(shape_fn_grad[1] * x, axis=-1)
        j22 = tf.reduce_sum(shape_fn_grad[1] * y, axis=-1)

        jacobian_det = j11 * j22 - j12 * j21

        dSdx = (j22[..., tf.newaxis] * shape_fn_grad[0]
                - j12[..., tf.newaxis] * shape_fn_grad[1]) / jacobian_det[..., tf.newaxis]
        dSdy = (-j21[..., tf.newaxis] * shape_fn_grad[0]
                + j11[..., tf.newaxis] * shape_fn_grad[1]) / jacobian_det[..., tf.newaxis]

        pushfwd_shape_fn_grad = tf.concat((dSdx[tf.newaxis, ...],
                                           dSdy[tf.newaxis, ...]), axis=0)

        return shape_fn_vals, pushfwd_shape_fn_grad, jacobian_det
