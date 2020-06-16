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
        elif self.degree == 2:
            self._shape_fn = p2_shape_fn
        else:
            raise NotImplementedError(''.join(
                ('Only degree p in [1, 2] polynomials currently supported',
                 'for TriangleElement')))

    @property
    def degree(self):
        return self._degree

    @property
    def dtype(self):
        return self._dtype
