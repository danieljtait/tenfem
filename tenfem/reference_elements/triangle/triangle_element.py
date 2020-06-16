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


class TriangleElement(object):
    """ Reference element for a triangle mesh. """

    def __init__(self, dtype):
        """

        Args:
            dtype: (optional) A `tf.DType` giving the data-type of the mesh nodes.
              Default, None then dtype is inferred from `nodes`.
        """
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype
