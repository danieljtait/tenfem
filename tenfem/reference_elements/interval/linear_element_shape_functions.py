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
""" Shape functions for linear elements. """
from typing import Tuple
import tensorflow as tf


def p1_shape_fn(r: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Shape function of the linear Lagrange polynomial on an interval element.

    Args:
        r: A float `Tensor` of shape `[..., n]` the canonical coordinate on the
          reference interval element `I = [0, 1]`.

    Returns:
        shape_fn_vals: A float `Tensor` of shape `[..., n, 2] with each
          `[..., n, i]` the evaluation of the ith shape function at coordinate
          `r[..., n]`.
        shape_fn_grads: A float `Tensor` giving the gradient with respect to
          the canincal coordinate of the shape function. A float tensor of
          shape `[1, ..., n, 2]`.
    """
    shape_fn_vals = tf.concat((1 - r[..., tf.newaxis], r[..., tf.newaxis]), axis=-1)
    shape_fn_grads = tf.concat((-tf.ones_like(r)[..., tf.newaxis],
                                tf.ones_like(r)[..., tf.newaxis]), axis=-1)[tf.newaxis, ...]

    return shape_fn_vals, shape_fn_grads
