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
from typing import Tuple
import tensorflow as tf


def gauss_quad_nodes_and_weights(order: int,
                                 dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
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
        NotImplementedError: If order is not in [1, 2, 3].
    """
    if order == 1:
        weights = tf.constant([1], dtype=dtype)
        nodes = tf.constant([[0.5],], dtype=dtype)
    elif order == 2:
        weights = tf.constant([1., 1.], dtype=dtype)
        nodes = tf.constant([[0.21132486540518708, ],
                             [0.7886751345948129, ]], dtype=dtype)
    elif order == 3:
        weights = tf.constant([5./9, 8./9., 5./9], dtype=dtype)
        nodes = tf.constant([[0.1127016653792583, ]
                             [0.5, ],
                             [0.8872983346207417, ]], dtype=dtype)
    else:
        raise NotImplementedError('Order {} not currently implemented.'.format(order))

    return weights, nodes
