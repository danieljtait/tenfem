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


def p1_shape_fn(r, s):
    """ Shape function of the linear Lagrange polynomial on a Triangle element.

    Args:
        r: A float `Tensor` of shape [..., n]` the first canonical coordinate on the reference triangle
        s: A float `Tensor` of shape [..., n]` the second canonical coordinate on the reference triangle.

    Returns:
        shape_fn: A float `Tensor` of shape `[..., n, 3]` with each `[..., n, i]` the evaluation of the
          ith shape function
        shape_fn_grad: The gradient with respect to the canoncial coordinates of the shape_fn, a Tensor
          of shape `[2, ..., n, 3]`.
    """
    s0 = (1. - r - s)[..., tf.newaxis]
    s1 = r[..., tf.newaxis]
    s2 = s[..., tf.newaxis]

    shape_fn = tf.concat((s0, s1, s2), axis=-1)
    dsdr = tf.concat([-tf.ones_like(s0), tf.ones_like(s1), tf.zeros_like(s2)], axis=-1)
    dsds = tf.concat([-tf.ones_like(s0), tf.zeros_like(s1), tf.ones_like(s2)], axis=-1)

    return shape_fn, tf.concat((dsdr[tf.newaxis, ...], dsds[tf.newaxis, ...]), axis=0)


def p2_shape_fn(r, s):
    """ Shape function of the quadratic Lagrange triangle.

    Args:
        r: A float `Tensor` of shape [..., n]` the first canonical coordinate on the reference triangle
        s: A float `Tensor` of shape [..., n]` the second canonical coordinate on the reference triangle.

    Returns:
        shape_fn: A float `Tensor` of shape `[..., n, 6]` with each `[..., n, i]` the evaluation of the
          ith shape function
        shape_fn_grad: The gradient with respect to the canonical coordinates of the shape_fn, a Tensor
          of shape `[2, ..., n, 6]`.
    """
    r = r[..., tf.newaxis]
    s = s[..., tf.newaxis]
    s0 = 1. - 3. * r - 3. * s + 2. * r ** 2 + 4. * r * s + 2. * s ** 2
    s1 = 2. * r ** 2 - r
    s2 = 2. * s ** 2 - s
    s3 = 4. * r * s
    s4 = 4. * s - 4. * r * s - 4. * s ** 2
    s5 = 4. * r - 4. * r ** 2 - 4. * r * s

    dsdr = tf.concat([-3 + 4 * r + 4 * s, 4 * r - 1, tf.zeros_like(r), 4 * s, -4 * s, 4 - 8 * r - 4 * s], axis=-1)
    dsds = tf.concat([-3 + 4 * r + 4 * s, tf.zeros_like(r), 4 * s - 1, 4 * r, 4 - 4 * r - 8 * s, -4 * r], axis=-1)

    shape_fn = tf.concat((s0, s1, s2, s3, s4, s5), axis=-1)
    return shape_fn, tf.concat((dsdr[tf.newaxis, ...], dsds[tf.newaxis, ...]), axis=0)
