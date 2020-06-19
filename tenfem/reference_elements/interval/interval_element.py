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
from tenfem.reference_elements import BaseReferenceElement


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

    def get_quadrature_nodes(self, mesh):
        """ Get the gaussian quadrature nodes of the mesh.

        Args:
            mesh: A `tenfem.mesh.BaseMesh` object giving the mesh we want
              to find the quadrature nodes on.

        Returns:
            quadrature_nodes: A float `tf.Tensor` of shape
             `[mesh.n_elements, n_quadrature_nodes, mesh.spatial_dimension]`
             giving the coordinates of the quadrature nodes on the mesh.
        """
        shape_fn = self.shape_function
        _, quad_nodes = gauss_quad_nodes_and_weights(self.quadrature_order, dtype=self.dtype)

        element_nodes = tf.gather(mesh.nodes, mesh.elements)
        shape_fn_vals, _ = shape_fn(quad_nodes)

        return tf.reduce_sum(element_nodes[..., tf.newaxis, :, :]
                             * shape_fn_vals[..., tf.newaxis], axis=-2)
