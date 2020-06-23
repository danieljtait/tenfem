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
""" Interval elements.  """
import tensorflow as tf
from tenfem.mesh import BaseMesh, IntervalMesh
from tenfem.reference_elements import BaseReferenceElement
from .linear_element_shape_functions import p1_shape_fn
from .gaussian_quadrature import gauss_quad_nodes_and_weights
from typing import Tuple


__all__ = ['IntervalElement', ]


class IntervalElement(BaseReferenceElement):
    """ Reference element for an interval mesh. """
    def __init__(self,
                 degree: int,
                 quadrature_order: int = 2,
                 dtype: tf.DType = tf.float32):
        """ Creates in IntervalElement instance.

        Args:
            degree: A python integer giving the degree of the polynomials on this
              function space.
            quadrature_order: A python integer giving the order of the Gaussian
              quadrature of integrals over this element.
            dtype: (optional) A `tf.DType` giving the data-type of the mesh nodes.
              Default, None then dtype defaults to `tf.float32`.

        Raises:
            NotImplementedError: If `degree` is not in [1, ]. Only linear
              shape functions currently implemented.
        """
        super(IntervalElement, self).__init__(name='interval_element')
        self._degree = degree
        self._dtype = dtype
        self._quadrature_order = quadrature_order

        if degree == 1:
            self._shape_fn = p1_shape_fn
        else:
            raise NotImplementedError('-'.join((
                'Currently only linear shape functions',
                'supported on IntervalElements')))

    def get_quadrature_nodes(self, mesh: BaseMesh) -> tf.Tensor:
        """ Alias into get_mesh_quadrature_nodes for compatability with layers. """
        return self.get_mesh_quadrature_nodes(mesh)[0]

    def get_mesh_quadrature_nodes(self,
                                  mesh: BaseMesh) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Get the gaussian quadrature nodes of the mesh,
        and the weights of the points.

        Args:
            mesh: A `tenfem.mesh.BaseMesh` object giving the mesh we want
              to find the quadrature nodes on.

        Returns:
            quadrature_nodes: A float `tf.Tensor` of shape
             `[mesh.n_elements, n_quadrature_nodes, mesh.spatial_dimension]`
             giving the coordinates of the quadrature nodes on the mesh.
        """
        shape_fn = self.shape_function
        weights, quad_nodes = self.get_quadrature_nodes_and_weights()

        element_nodes = tf.gather(mesh.nodes, mesh.elements)
        shape_fn_vals, _ = shape_fn(quad_nodes[:, 0])
        return tf.reduce_sum(element_nodes[..., tf.newaxis, :, :]
                             * shape_fn_vals[..., tf.newaxis], axis=-2), weights

    def get_quadrature_nodes_and_weights(self):
        """ The nodes and weights for Gaussian quadrature on an interval element.

        Returns:
            weights: A float `Tensor` like giving the weights of the
              gaussian quadrature rule of `self.quadrature_order`, with data-type equal
              to `dtype`.
            nodes: A float `Tensor` giving the nodes of the Gaussian
              quadrature rule of shape [len(weights), 1], with data-type
              equal to `dtype`.
        """
        return gauss_quad_nodes_and_weights(self.quadrature_order, dtype=self.dtype)

    def get_element_volumes(self, mesh):
        """ Returns the element volumes of an interval mesh.

        Args:
            mesh: A `tenfem.mesh.BaseMesh` object giving the mesh we want
              to find the quadrature nodes on.

        Returns:
            element_volumes: A scalar float `Tensor` of length
              `mesh.n_elements`.
        """
        if isinstance(mesh, IntervalMesh):
            nodes = tf.gather(mesh.nodes, mesh.elements)
            volumes = tf.abs(nodes[..., 1, 0] - nodes[..., 0, 0])
            return volumes
        else:
            raise ValueError('An IntervalElement reference element expects an interval mesh.')

    def quadrature(self, f, mesh):
        """ Perform quadrature of a function over the mesh.

        Args:
            f: A scalar callable, when evaluated at a mesh node of shape
              `[..., spatial_dim]` it should return a `Tensor` of shape
              `[..., ]` with the same `dtype` as `mesh.

        Returns
            integral: A quadrature approximation to the integral of
              f over the mesh.
        """
        quad_nodes, quad_weights = self.get_mesh_quadrature_nodes(mesh)
        f_at_nodes = f(quad_nodes)
        volumes = self.get_element_volumes(mesh)
        return 0.5 * tf.reduce_sum(f_at_nodes
                                   * quad_weights
                                   * volumes[..., tf.newaxis], axis=[-1, -2])

    def isomap(self, nodes, canonical_coordinates):
        """
        Args:
            nodes: A batched tensor of shape [..., element_dim, 1]
            canonical_coordinates: A tensor of shape [p, 1] where p is the number of shape
              functions for the given element.

        Returns:
            pushfwd_shape_fn_grad: The gradient of shape functions in
              the physical coordinate system. A tensor of shape
              `[1, n_elements, n, 2]`
        """
        jacobian_det = nodes[..., 1, 0] - nodes[..., 0, 0]
        n = tf.shape(canonical_coordinates)[-2]  # number of canonical coords. per element
        jacobian_det = jacobian_det[..., tf.newaxis] * tf.ones([n], dtype=self.dtype)

        shape_fn_vals, shape_fn_grad = self.shape_function(canonical_coordinates[..., 0])

        pushfwd_shape_fn_grad = shape_fn_grad / jacobian_det[..., tf.newaxis]
        pushfwd_shape_fn_grad = pushfwd_shape_fn_grad[tf.newaxis, ...]

        return shape_fn_vals, pushfwd_shape_fn_grad, jacobian_det
