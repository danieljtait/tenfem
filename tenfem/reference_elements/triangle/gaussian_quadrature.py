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


def gauss_quad_nodes_and_weights(order: int, dtype: tf.DType = tf.float32):
    """ The nodes and weights for Gaussian quadrature on a triangle element.

    Args:
        order: Python integer, the number of quadrature nodes.
        dtype: A `tf.DType` object giving the data-type of the quadrature nodes
          and weights.

    Returns:
        weights: A float `Tensor` like giving the weights of the
          gaussian quadrature rule of `order`, with data-type equal
          to `dtype`.
        nodes: A float `Tensor` giving the nodes of the Gaussian
          quadrature rule of shape [len(weights), 2], with data-type
          equal to `dtype`.

    Raises:
        NotImplementedError: If order is not in [1, 2, 3, 4].
    """
    if order == 1:
        weights = tf.constant([1], dtype=dtype)
        nodes = tf.constant([[1. / 3],
                             [1. / 3]], dtype=dtype)
    elif order == 2:
        weights = tf.constant([1. / 3, 1. / 3, 1. / 3], dtype=dtype)
        nodes = tf.constant([[1. / 6, 2. / 3, 1. / 6],
                             [1. / 6, 1. / 6, 2. / 3]], dtype=dtype)
    elif order == 3:
        weights = tf.constant([-27. / 48, 25. / 48, 25. / 48, 25. / 48], dtype=dtype)
        nodes = tf.constant([[1. / 3, 0.2, 0.6, 0.2],
                             [1. / 3, 0.2, 0.2, 0.6]], dtype=dtype)
    elif order == 4:
        weights = tf.constant([
            0.223381589678011, 0.223381589678011, 0.223381589678011,
            0.109951743655322, 0.109951743655322, 0.109951743655322], dtype=dtype)
        nodes = tf.constant([
            [0.445948490915965, 0.445948490915965, 0.108103018168070, 0.091576213509771, 0.091576213509771,
             0.816847572980459],
            [0.445948490915965, 0.108103018168070, 0.445948490915965, 0.091576213509771, 0.816847572980459,
             0.091576213509771]],
            dtype=dtype)
    else:
        raise NotImplementedError('Gaussian quadrature of order {} not currently supported'.format(order))

    return weights, tf.transpose(nodes)