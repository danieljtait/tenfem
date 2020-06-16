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


def get_batched_element_indices(elements: tf.Tensor,
                                validate_args: bool = False):
    """ Creates the complete set of local to global indices by batching down the final dimension.

    Returns:
        batched_element_indices: An integer tensor of shape
          `[num_batches*num_elements*element_dim**2, 1+element_dim]`.
    """
    if validate_args:
        if tf.rank(elements) != 3:
            raise ValueError('Batched element indices expected a 3D tensor',
                             'instead found a tensor of shape {}'.format(tf.shape(elements)))

    elements_shape = tf.shape(elements)
    batch_size = elements_shape[0]
    n_elements = elements_shape[1]
    element_dim = elements_shape[-1]

    i1 = tf.reshape(tf.repeat(elements, tf.repeat(element_dim, element_dim), axis=-1), (-1, 1))
    i2 = tf.reshape(tf.tile(elements, [1, 1, element_dim]), (-1, 1))

    batch_indices = tf.repeat(tf.range(batch_size), element_dim ** 2 * n_elements)[:, tf.newaxis]

    indices = tf.concat([batch_indices, i1, i2], axis=-1)
    return indices


def get_batched_vector_element_indices(elements):
    """ Get the indices for batch scattering of local vectors.

    Args:
        elements: An integer Tensor of shape
          `[batch_size, num_elements, element_dim]`

    Returns:
        batched_element_indices: An integer tensor of shape
          `[batch_size*num_elements*element_dim, 1+element_dim]`.
    """
    elements_shape = tf.shape(elements)
    batch_size = elements_shape[0]
    n_elements = elements_shape[1]
    element_dim = elements_shape[-1]

    element_indices = tf.reshape(elements, [-1, 1])
    batch_indices = tf.repeat(tf.range(batch_size), element_dim * n_elements)[:, tf.newaxis]

    indices = tf.concat([batch_indices, element_indices], axis=-1)
    return indices
