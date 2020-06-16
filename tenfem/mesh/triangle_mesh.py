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
from tenfem.mesh import BaseMesh
import tensorflow as tf


__all__ = ['TriangleMesh', ]


class TriangleMesh(BaseMesh):
    """ Triangle mesh in R2. """

    @property
    def spatial_dimension(self):
        """ Spatial dimension of triangle mesh embedding. """
        return tf.constant(2, dtype=tf.int32)